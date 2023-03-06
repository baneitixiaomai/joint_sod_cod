import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
from datetime import datetime
from model.ResNet_models import Saliency_feat_encoder, Cod_feat_encoder, Share_feat_decoder, Contrastive_module,DynDiscriminatorSNconv
from data import get_loader, get_loader_clipargue_randomselect ,get_loader_val
import torch.optim.lr_scheduler as lr_scheduler
from eval_func import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_iters', type=int, default=30000, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2e-5, help='learning rate')
parser.add_argument('--lr_dis', type=float, default=2e-5, help='learning rate of discriminator')
parser.add_argument('--lr_feat', type=float, default=1.2e-5, help='learning rate of contrastive learning')
parser.add_argument('--batchsize', type=int, default=22, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--beta1_dis', type=float, default=0.5,help='beta of Adam for descriptor')
parser.add_argument('--latent_dim', type=int, default=700, help='latent feature size')
parser.add_argument('--reduced_dim', type=int, default=32, help='reduced dimension size')
parser.add_argument('--cos_weight', type=float, default=5, help='weight of the cosine loss')
parser.add_argument('--beta_feat', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_fq', type=int, default=1, help='contrastive learning latent frequency')  
parser.add_argument('--sadw', type=float, default=1.0, help='adversial learning sod weigth')  
parser.add_argument('--cadw', type=float, default=1.0, help='adversial learning cod weigth')  

opt = parser.parse_args()


size_rates = [0.75,1,1.25]

print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
decay_step= opt.train_iters /3*2

sal_gen=Saliency_feat_encoder()
sal_gen.cuda()
sal_gen_optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, sal_gen.parameters()), opt.lr_gen, betas=[opt.beta1_gen, 0.999])
sal_gen_scheduler = lr_scheduler.StepLR(sal_gen_optimizer,step_size=decay_step,gamma = 0.1)

cod_gen=Cod_feat_encoder()
cod_gen.cuda()
cod_gen_optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, cod_gen.parameters()), opt.lr_gen, betas=[opt.beta1_gen, 0.999])
cod_gen_scheduler = lr_scheduler.StepLR(cod_gen_optimizer,step_size=decay_step,gamma = 0.1)

share_dec=Share_feat_decoder()
share_dec.cuda()
share_dec_optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, share_dec.parameters()), opt.lr_gen, betas=[opt.beta1_gen, 0.999])
share_dec_scheduler = lr_scheduler.StepLR(share_dec_optimizer,step_size=decay_step,gamma = 0.1)

dis_model_sod = DynDiscriminatorSNconv()
dis_model_sod.cuda()
dis_params_sod = dis_model_sod.parameters()
dis_optimizer_sod = torch.optim.Adam(dis_params_sod, opt.lr_dis, betas=[opt.beta1_dis, 0.999])
dis_model_scheduler_sod = lr_scheduler.StepLR(dis_optimizer_sod,step_size=1000,gamma = 0.8)  # 0313

dis_model_cod = DynDiscriminatorSNconv()
dis_model_cod.cuda()
dis_params_cod = dis_model_cod.parameters()
dis_optimizer_cod = torch.optim.Adam(dis_params_cod, opt.lr_dis, betas=[opt.beta1_dis, 0.999])
dis_model_scheduler_cod = lr_scheduler.StepLR(dis_optimizer_cod,step_size=2000,gamma = 0.8)  # 0313

latent_feat=Contrastive_module(opt.reduced_dim, opt.latent_dim)
latent_feat.cuda()
latent_feat.eval()
latent_feat_optimizer=torch.optim.Adam(latent_feat.parameters(), opt.lr_feat, betas=[opt.beta_feat, 0.999])
latent_feat_scheduler = lr_scheduler.StepLR(latent_feat_optimizer,step_size=2000,gamma = 0.95)

val_name='JPEGImages_select'
val_img_root = '/train_data/'+ val_name + '/'

traindata='duts+wmae'

image_root = '/train_data/'+traindata+'/img/'
gt_root = '/train_data/'+traindata+'/gt/'


cod_image_root = '/train_data/COD_train/Imgs/'
cod_gt_root = '/train_data/COD_train/GT/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
cod_train_loader = get_loader_clipargue_randomselect(cod_image_root, cod_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
val_loader = get_loader_val(val_img_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
sal_train_iter = iter(train_loader)
sal_it = 0
cod_train_iter = iter(cod_train_loader)
cod_it = 0
val_train_iter = iter(val_loader)
val_it = 0

CE = torch.nn.BCELoss()
MSE = torch.nn.MSELoss(size_average=True, reduce=True)
l1_loss = torch.nn.L1Loss(size_average = True, reduce = True)

def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



model_save_name = '.pth'
model_save_path = './models/'
print(' model_save_path = ', model_save_path)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

temp_save_path = model_save_path+'/temp/'
if not os.path.exists(temp_save_path):
    os.makedirs(temp_save_path)
pred_label = 0
gt_label = 1

print("go go go!!!!!!!!!!!")
for i in range(1,opt.train_iters+1):
    sal_gen_scheduler.step()
    cod_gen_scheduler.step()
    share_dec_scheduler.step()
    latent_feat_scheduler.step()
    dis_model_scheduler_sod.step()
    dis_model_scheduler_cod.step()
    sal_gen.train()
    cod_gen.train()
    share_dec.train()

    if i % opt.latent_fq == 0:
        if val_it >= len(val_loader):
            val_train_iter = iter(val_loader)
            val_it = 0
        val_pack = next(val_train_iter)
        val_imgs = val_pack
        val_imgs = Variable(val_imgs)
        val_imgs = val_imgs.cuda()
        val_it += 1

        sal_gen_optimizer.zero_grad()
        cod_gen_optimizer.zero_grad()
        share_dec_optimizer.zero_grad()
        x1s, x2s, x3s, x4s = sal_gen(val_imgs)
        _, sod_ref= share_dec.forward(x1s, x2s, x3s, x4s)
        sod_ref = sod_ref.detach()

        x1c, x2c, x3c, x4c = cod_gen(val_imgs)
        _, cod_ref= share_dec.forward(x1c, x2c, x3c, x4c)
        cod_ref = cod_ref.detach()
        con_loss = latent_feat(x1s, x2s, x3s, x4s,sod_ref, x1c, x2c, x3c, x4c,cod_ref)
        cur_loss = opt.cos_weight*con_loss
        cur_loss.backward()
        sal_gen_optimizer.step()
        cod_gen_optimizer.step()
        latent_feat_optimizer.step()

            
    if sal_it >= len(train_loader):
        sal_train_iter = iter(train_loader)
        sal_it = 0
    sal_pack = next(sal_train_iter)
    for rate in size_rates:
        sal_imgs, sal_gts  = sal_pack
        sal_imgs = Variable(sal_imgs)
        sal_gts = Variable(sal_gts)
        sal_imgs = sal_imgs.cuda()
        sal_gts = sal_gts.cuda()

        trainsize = int(opt.trainsize * rate /32) *32
        # print (trainsize,rate)
        if rate != 1:
            sal_imgs = F.upsample(sal_imgs,(trainsize,trainsize), mode='bilinear')
            sal_gts = F.upsample(sal_gts,(trainsize, trainsize), mode='bilinear')

        sal_gen_optimizer.zero_grad()
        share_dec_optimizer.zero_grad()
        x1,x2,x3,x4 = sal_gen.forward(sal_imgs)
        sal_init,sal_ref = share_dec.forward(x1,x2,x3,x4)

        ## train discriminator with sod
        dis1_input = torch.sigmoid(sal_init)
        dis2_input = torch.sigmoid(sal_ref)
        Dis_out1_1 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis1_input)),
                                (sal_init.shape[2], sal_init.shape[3]), mode='bilinear')
        Dis_out1_2 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis2_input)),
                                (sal_ref.shape[2], sal_ref.shape[3]), mode='bilinear')
        Dis_out1_3 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,sal_gts)),
                                (sal_ref.shape[2], sal_ref.shape[3]), mode='bilinear')
        sal_structure_loss1 = structure_loss(sal_init, sal_gts)
        sal_structure_loss2 = structure_loss(sal_ref, sal_gts)


        Dis_out1_3=Dis_out1_3.detach()
        loss_gen_adv1 = CE(Dis_out1_1, Dis_out1_3)
        loss_gen_adv2 = CE(Dis_out1_2, Dis_out1_3)
        loss_gen_adv_sod = 0.5 * (loss_gen_adv1 + loss_gen_adv2)
        s_loss=0.5*(sal_structure_loss1+sal_structure_loss2)
        sal_loss= s_loss  + opt.sadw*loss_gen_adv_sod
        sal_loss.backward()
        sal_gen_optimizer.step()
        share_dec_optimizer.step()      


        dis_optimizer_sod.zero_grad()
        # ## train descriptor
        sal_init = sal_init.detach()
        sal_ref = sal_ref.detach()
        dis3_input = torch.sigmoid(sal_init)
        dis4_input = torch.sigmoid(sal_ref)
        Dis_out2_1 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis3_input)),
                                (sal_init.shape[2], sal_init.shape[3]), mode='bilinear')
        Dis_out2_2 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis4_input)),
                                (sal_ref.shape[2], sal_ref.shape[3]), mode='bilinear')

        loss_dis1_1 = CE(Dis_out2_1, torch.abs(torch.sigmoid(sal_init)-sal_gts))   # # pred_label = 0
        loss_dis1_2 = CE(Dis_out2_2, torch.abs(torch.sigmoid(sal_ref)-sal_gts))
        loss_dis1 = 0.5 * (loss_dis1_1 + loss_dis1_2)
        dis5_input = sal_gts
        Dis_out3 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis5_input)), (sal_gts.shape[2], sal_gts.shape[3]),
                            mode='bilinear')
        loss_dis2 = CE(Dis_out3,  torch.abs(sal_gts-sal_gts))  # gt_label = 1
        loss_dis = 0.5 * (loss_dis1 + loss_dis2)
        loss_dis.backward()
        dis_optimizer_sod.step()

    sal_it += 1

    #######################################################################
    if cod_it >= len(cod_train_loader):
        cod_train_iter = iter(cod_train_loader)
        cod_it = 0
    cod_pack =  next(cod_train_iter)
    for rate in size_rates:
        cod_imgs, cod_gts = cod_pack
        cod_imgs = Variable(cod_imgs)
        cod_gts = Variable(cod_gts)
        cod_imgs = cod_imgs.cuda()
        cod_gts = cod_gts.cuda()

        trainsize = int(opt.trainsize * rate /32) *32
        if rate != 1:
            cod_imgs = F.upsample(cod_imgs,(trainsize,trainsize), mode='bilinear')
            cod_gts = F.upsample(cod_gts,(trainsize, trainsize), mode='bilinear')

        cod_gen_optimizer.zero_grad()
        share_dec_optimizer.zero_grad()
        x1,x2,x3,x4 = cod_gen.forward(cod_imgs)
        cod_init, cod_ref = share_dec.forward(x1,x2,x3,x4)


        ## train discriminator with sod
        dis1_input = torch.sigmoid(cod_init)
        dis2_input = torch.sigmoid(cod_ref)
        Dis_out1_1_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis1_input)),
                                (cod_init.shape[2], cod_init.shape[3]), mode='bilinear')
        Dis_out1_2_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis2_input)),
                                (cod_ref.shape[2], cod_ref.shape[3]), mode='bilinear')
        Dis_out1_3_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,cod_gts)),
                                (cod_ref.shape[2], cod_ref.shape[3]), mode='bilinear')
        
        # init loss
        cod_ce_loss1 = structure_loss(cod_init, cod_gts)
        cod_ce_loss2 = structure_loss(cod_ref, cod_gts)

        Dis_out1_3_cod=Dis_out1_3_cod.detach()
        loss_gen_adv1_cod = CE(Dis_out1_1_cod, Dis_out1_3_cod)
        loss_gen_adv2_cod = CE(Dis_out1_2_cod, Dis_out1_3_cod)
        loss_gen_adv_cod = 0.5 * (loss_gen_adv1_cod + loss_gen_adv2_cod)
        c_loss=0.5*(cod_ce_loss1+cod_ce_loss2)
        cod_loss= c_loss  + opt.cadw*loss_gen_adv_cod
        cod_loss.backward()
        cod_gen_optimizer.step()
        share_dec_optimizer.step()    
        

        dis_optimizer_cod.zero_grad()
        # ## train descriptor
        cod_init = cod_init.detach()
        cod_ref = cod_ref.detach()
        dis3_input_cod = torch.sigmoid(cod_init)
        dis4_input_cod = torch.sigmoid(cod_ref)
        Dis_out2_1_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis3_input_cod)),
                                (cod_init.shape[2], cod_init.shape[3]), mode='bilinear')
        Dis_out2_2_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis4_input_cod)),
                                (cod_ref.shape[2], cod_ref.shape[3]), mode='bilinear')

        loss_dis1_1_cod = CE(Dis_out2_1_cod, torch.abs(torch.sigmoid(cod_init)-cod_gts))   # # pred_label = 0
        loss_dis1_2_cod = CE(Dis_out2_2_cod, torch.abs(torch.sigmoid(cod_ref)-cod_gts))
        loss_dis1_cod = 0.5 * (loss_dis1_1_cod + loss_dis1_2_cod)
        dis5_input_cod = cod_gts
        Dis_out3_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis5_input_cod)), (cod_gts.shape[2], cod_gts.shape[3]),
                            mode='bilinear')
        loss_dis2_cod =  CE(Dis_out3_cod, torch.abs(cod_gts-cod_gts))
        loss_dis_cod = 0.5 * (loss_dis1_cod + loss_dis2_cod)
        loss_dis_cod.backward()
        dis_optimizer_cod.step()

    cod_it += 1
    if i % 10 == 0 or i == len(train_loader):
        print('{} Step [{:04d}/{:04d}], sal Loss: {:.4f}, cod Loss: {:.4f}, Contras Loss: {:.4f}'.
            format(datetime.now(), i, opt.train_iters, s_loss.data, c_loss.data, cur_loss.data))


    if i % opt.train_iters == 0:
        torch.save(sal_gen.state_dict(), model_save_path + 'sal_gen' + '_%d' % i +model_save_name)
        torch.save(cod_gen.state_dict(), model_save_path + 'cod_gen' + '_%d' % i +model_save_name)
        torch.save(share_dec.state_dict(), model_save_path + 'share_dec' + '_%d' % i +model_save_name)
        torch.save(dis_model_sod.state_dict(), model_save_path + 'dis_model_sod' + '_%d' % i +model_save_name)
        torch.save(dis_model_cod.state_dict(), model_save_path + 'dis_model_cod' + '_%d' % i +model_save_name)
        torch.save(latent_feat.state_dict(), model_save_path + 'latent_feat' + '_%d' % i +model_save_name)



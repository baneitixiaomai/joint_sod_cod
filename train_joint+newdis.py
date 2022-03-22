import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
from datetime import datetime
from model.ResNet_models import Saliency_feat_encoder, Cod_feat_encoder, Share_feat_decoder, FCDiscriminator, Compute_latent_feature,FCDiscriminatorsod,DynDiscriminatorSOD,DynDiscriminatorCOD
from data import get_loader, get_loader_cod,get_loader_val
from save_to_temp import *
import torch.optim.lr_scheduler as lr_scheduler
from scipy import misc
from utils import adjust_lr, AvgMeter
from data import test_dataset
from tqdm import tqdm
import cv2
import sys
from eval_func import *

parser = argparse.ArgumentParser()
# parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--train_iters', type=int, default=36000, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate')
# parser.add_argument('--lr_dis_sod', type=float, default=2e-5, help='learning rate')
# parser.add_argument('--lr_dis_cod', type=float, default=2e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=15, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('-beta1_dis', type=float, default=0.5,help='beta of Adam for descriptor')
parser.add_argument('--latent_dim', type=int, default=700, help='latent feature size')
parser.add_argument('--reduced_dim', type=int, default=32, help='reduced dimension size')
parser.add_argument('--lr_feat', type=float, default=1e-5, help='learning rate')
parser.add_argument('--cos_weight', type=float, default=1e-1, help='weight of the cosine loss')
parser.add_argument('-beta_feat', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()


sod_adweight=[0.01,0.03,0.07] # 0.05,
cod_adweight=[0.2]
lr_cod=[2e-5]
lr_sod=[2e-5]
for sodadw in sod_adweight:
    for codadw in cod_adweight:
        for lr_dis_cod in lr_cod:
            for lr_dis_sod in lr_sod:
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


                dis_model_sod = DynDiscriminatorSOD()
                dis_model_sod.cuda()
                dis_params_sod = dis_model_sod.parameters()
                dis_optimizer_sod = torch.optim.Adam(dis_params_sod, lr_dis_sod, betas=[opt.beta1_dis, 0.999])
                # dis_model_scheduler_sod = lr_scheduler.StepLR(dis_optimizer_sod,step_size=2000,gamma = 0.95)  # 0309
                dis_model_scheduler_sod = lr_scheduler.StepLR(dis_optimizer_sod,step_size=1000,gamma = 0.8)  # 0313


                dis_model_cod = DynDiscriminatorCOD()
                dis_model_cod.cuda()
                dis_params_cod = dis_model_cod.parameters()
                dis_optimizer_cod = torch.optim.Adam(dis_params_cod, lr_dis_cod, betas=[opt.beta1_dis, 0.999])
                # dis_model_scheduler_sod = lr_scheduler.StepLR(dis_optimizer_sod,step_size=2000,gamma = 0.95)  # 0309
                dis_model_scheduler_cod = lr_scheduler.StepLR(dis_optimizer_cod,step_size=2000,gamma = 0.8)  # 0313


                # traindata='DUTS_1203'
                traindata='duts+wmae'
                model_time = '_0320'


                image_root = '/home1/liaixuan/Trans_joint_0722/mode2_joint_init/tpami/data_interactive/'+traindata+'/img/'
                gt_root = '/home1/liaixuan/Trans_joint_0722/mode2_joint_init/tpami/data_interactive/'+traindata+'/gt/'


                cod_image_root = '/home1/liaixuan/dataset/COD/COD_train/Imgs/'
                cod_gt_root = '/home1/liaixuan/dataset/COD/COD_train/GT/'

                train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
                cod_train_loader = get_loader_cod(cod_image_root, cod_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

                sal_train_iter = iter(train_loader)
                sal_it = 0
                cod_train_iter = iter(cod_train_loader)
                cod_it = 0

                CE = torch.nn.BCELoss()
                MSE = torch.nn.MSELoss(size_average=True, reduce=True)
                l1_loss = torch.nn.L1Loss(size_average = True, reduce = True)
                cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

                size_rates = [1]

                def structure_loss(pred, mask):
                    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
                    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
                    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

                    pred  = torch.sigmoid(pred)
                    inter = ((pred*mask)*weit).sum(dim=(2,3))
                    union = ((pred+mask)*weit).sum(dim=(2,3))
                    wiou  = 1-(inter+1)/(union-inter+1)
                    return (wbce+wiou).mean()

                def make_Dis_label(label,gts):
                    D_label = np.ones(gts.shape)*label
                    D_label = Variable(torch.FloatTensor(D_label)).cuda()

                    return D_label


                ffff='/home2/users/liaixuan/tpami_2022/'
                folder_name='baseline'
                model_kind="joint+new_dis" +"_sodadw="+str(sodadw)+"_codadw="+str(codadw) +"_sodlr="+str(lr_dis_sod) +"_codlr="+str(lr_dis_cod)
                model_root0 = ffff+'experience_models_tpami/'+folder_name+'/'+model_kind+'/'

                model_suffix=""
                model_save_name = '.pth'

                model_info = model_kind+'_traindata_'+traindata + model_time 
                temp_save_path = ffff+'experience_temp_film_newdisgt/temp'+"_"+model_info +'/'
                if not os.path.exists(temp_save_path):
                    os.makedirs(temp_save_path)
                model_save_path = model_root0 + model_info + '/'
                print(' model_save_path = ', model_save_path)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                # labels for adversarial training
                pred_label = 0
                gt_label = 1

                print("go go go!!!!!!!!!!!")
                for i in range(1,opt.train_iters+1):
                    sal_gen_scheduler.step()
                    cod_gen_scheduler.step()
                    share_dec_scheduler.step()
                    dis_model_scheduler_sod.step()
                    dis_model_scheduler_cod.step()
                    sal_gen.train()
                    cod_gen.train()
                    share_dec.train()
                    # loss_record_sal = AvgMeter()
                    # loss_record_cod = AvgMeter()


                    # shared_optimizer.step() twice
                    if sal_it >= len(train_loader):
                        sal_train_iter = iter(train_loader)
                        sal_it = 0
                    sal_pack = sal_train_iter.next()
                    for rate in size_rates:
                        sal_imgs, sal_gts  = sal_pack
                        sal_imgs = Variable(sal_imgs)
                        sal_gts = Variable(sal_gts)
                        sal_imgs = sal_imgs.cuda()
                        sal_gts = sal_gts.cuda()


                        sal_gen_optimizer.zero_grad()
                        share_dec_optimizer.zero_grad()
                        x1,x2,x3,x4 = sal_gen.forward(sal_imgs)
                        sal_init,sal_ref = share_dec.forward(x1,x2,x3,x4)
                        # init loss
                        sal_structure_loss1 = structure_loss(sal_init,sal_gts)
                        sal_structure_loss2 = structure_loss(sal_ref, sal_gts)

                        ## train discriminator with sod
                        dis1_input = torch.sigmoid(sal_init)
                        dis2_input = torch.sigmoid(sal_ref)
                        Dis_out1_1 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis1_input)),
                                                (sal_init.shape[2], sal_init.shape[3]), mode='bilinear')
                        Dis_out1_2 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis2_input)),
                                                (sal_ref.shape[2], sal_ref.shape[3]), mode='bilinear')
                        Dis_out1_3 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,sal_gts)),
                                                (sal_ref.shape[2], sal_ref.shape[3]), mode='bilinear')
                        conf=2*torch.abs(Dis_out1_2-0.5)
                        Dis_out1_3=Dis_out1_3.detach()
                        loss_gen_adv1 = CE(Dis_out1_1, Dis_out1_3)
                        loss_gen_adv2 = CE(Dis_out1_2, Dis_out1_3)
                        loss_gen_adv_sod = 0.5 * (loss_gen_adv1 + loss_gen_adv2)
                        s_loss=0.5*(sal_structure_loss1+sal_structure_loss2)
                        sal_loss= s_loss  + sodadw*loss_gen_adv_sod
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
                        loss_dis1_1 = CE(Dis_out2_1, make_Dis_label(pred_label, sal_gts))
                        loss_dis1_2 = CE(Dis_out2_2, make_Dis_label(pred_label, sal_gts))
                        loss_dis1 = 0.5 * (loss_dis1_1 + loss_dis1_2)
                        dis5_input = sal_gts
                        Dis_out3 = F.upsample(torch.sigmoid(dis_model_sod.forward(sal_imgs,dis5_input)), (sal_gts.shape[2], sal_gts.shape[3]),
                                            mode='bilinear')
                        loss_dis2 = CE(Dis_out3, make_Dis_label(gt_label, sal_gts))
                        loss_dis = 0.5 * (loss_dis1 + loss_dis2)
                        loss_dis.backward()
                        dis_optimizer_sod.step()

                        if i % 200 == 0:
                            visualize_sal_prediction(torch.sigmoid(sal_ref), temp_save_path)
                            visualize_sal_gt(sal_gts, temp_save_path)
                            visualize_rgb(sal_imgs, temp_save_path,'_sod_img')
                            visualize_binary(conf, temp_save_path,'_sod_conf')
                            visualize_binary(Dis_out3, temp_save_path,'_sod_dis_pred4gt')
                            visualize_binary(Dis_out2_2, temp_save_path,'_sod_dis_pred4pred')


                        # if rate == 1:
                        #     loss_record_sal.update(sal_loss.data, opt.batchsize)
                    sal_it += 1

                    #######################################################################
                    # shared_optimizer.step()
                    if i%1 == 0:
                        if cod_it >= len(cod_train_loader):
                            cod_train_iter = iter(cod_train_loader)
                            cod_it = 0
                        cod_pack = cod_train_iter.next()
                        for rate in size_rates:
                            cod_imgs, cod_gts = cod_pack
                            cod_imgs = Variable(cod_imgs)
                            cod_gts = Variable(cod_gts)
                            cod_imgs = cod_imgs.cuda()
                            cod_gts = cod_gts.cuda()


                            cod_gen_optimizer.zero_grad()
                            share_dec_optimizer.zero_grad()
                            x1,x2,x3,x4 = cod_gen.forward(cod_imgs)
                            cod_init, cod_ref = share_dec.forward(x1,x2,x3,x4)
                            # init loss
                            cod_ce_loss1 = structure_loss(cod_init, cod_gts)
                            cod_ce_loss2 = structure_loss(cod_ref, cod_gts)


                            ## train discriminator with sod
                            dis1_input = torch.sigmoid(cod_init)
                            dis2_input = torch.sigmoid(cod_ref)
                            Dis_out1_1_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis1_input)),
                                                    (cod_init.shape[2], cod_init.shape[3]), mode='bilinear')
                            Dis_out1_2_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis2_input)),
                                                    (cod_ref.shape[2], cod_ref.shape[3]), mode='bilinear')
                            Dis_out1_3_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,cod_gts)),
                                                    (cod_ref.shape[2], cod_ref.shape[3]), mode='bilinear')
                            conf_cod=2*torch.abs(Dis_out1_2-0.5)
                            Dis_out1_3_cod=Dis_out1_3_cod.detach()
                            loss_gen_adv1_cod = CE(Dis_out1_1_cod, Dis_out1_3_cod)
                            loss_gen_adv2_cod = CE(Dis_out1_2_cod, Dis_out1_3_cod)
                            loss_gen_adv_cod = 0.5 * (loss_gen_adv1_cod + loss_gen_adv2_cod)
                            c_loss=0.5*(cod_ce_loss1+cod_ce_loss2)
                            cod_loss= c_loss  + codadw*loss_gen_adv_cod
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
                            loss_dis1_1_cod = CE(Dis_out2_1_cod, make_Dis_label(pred_label, cod_gts))
                            loss_dis1_2_cod = CE(Dis_out2_2_cod, make_Dis_label(pred_label, cod_gts))
                            loss_dis1_cod = 0.5 * (loss_dis1_1_cod + loss_dis1_2_cod)
                            dis5_input_cod = cod_gts
                            Dis_out3_cod = F.upsample(torch.sigmoid(dis_model_cod.forward(cod_imgs,dis5_input_cod)), (cod_gts.shape[2], cod_gts.shape[3]),
                                                mode='bilinear')
                            loss_dis2_cod = CE(Dis_out3_cod, make_Dis_label(gt_label, cod_gts))
                            loss_dis_cod = 0.5 * (loss_dis1_cod + loss_dis2_cod)
                            loss_dis_cod.backward()
                            dis_optimizer_cod.step()

                            if i % 200 == 0:
                                visualize_cod_prediction(torch.sigmoid(cod_ref), temp_save_path)
                                visualize_cod_gt(cod_gts, temp_save_path)
                                visualize_rgb(cod_imgs, temp_save_path,'_cod_img')
                                visualize_binary(conf_cod, temp_save_path,'_cod_conf')
                                visualize_binary(Dis_out3_cod, temp_save_path,'_cod_dis_pred4gt')
                                visualize_binary(Dis_out2_2_cod, temp_save_path,'_cod_dis_pred4pred')


                            # if rate == 1:
                            #     loss_record_cod.update(cod_loss.data, opt.batchsize)

                        cod_it += 1
                    if i % 10 == 0 or i == len(train_loader):
                        print('{} Step [{:04d}/{:04d}], sal Loss: {:.4f}, adv Loss: {:.4f}, total Loss: {:.4f} , dis train Loss: {:.4f}, cod Loss: {:.4f}, adv Loss: {:.4f}, total Loss: {:.4f}, dis train Loss: {:.4f}'.
                            format(datetime.now(), i, opt.train_iters, s_loss.data,loss_gen_adv_sod.data, sal_loss.data,loss_dis.data, c_loss.data,loss_gen_adv_cod.data, cod_loss.data,loss_dis_cod.data))


                    if i % opt.train_iters == 0:
                        torch.save(sal_gen.state_dict(), model_save_path + 'sal_gen' + '_%d' % i + model_suffix+model_save_name)
                        torch.save(cod_gen.state_dict(), model_save_path + 'cod_gen' + '_%d' % i + model_suffix+model_save_name)
                        torch.save(share_dec.state_dict(), model_save_path + 'share_dec' + '_%d' % i + model_suffix+model_save_name)
                        torch.save(dis_model_sod.state_dict(), model_save_path + 'dis_model_sod' + '_%d' % i + model_suffix+model_save_name)
                        torch.save(dis_model_cod.state_dict(), model_save_path + 'dis_model_cod' + '_%d' % i + model_suffix+model_save_name)







                sal_model = model_save_path + "sal_gen_" + str(opt.train_iters) + model_suffix + model_save_name
                cod_model = model_save_path + "cod_gen_" + str(opt.train_iters) + model_suffix + model_save_name
                sharedecoder_model=model_save_path + "share_dec_" + str(opt.train_iters) + model_suffix + model_save_name

                pre_root= "result_tpami/"+folder_name+'/'+model_kind +"/result_" + model_info+"/"
                print("\""+pre_root+"\"")

                sal_gen=Saliency_feat_encoder()
                sal_gen.load_state_dict(torch.load(sal_model))

                cod_gen = Cod_feat_encoder()
                cod_gen.load_state_dict(torch.load(cod_model))
                #
                shared_decoder = Share_feat_decoder()
                shared_decoder.load_state_dict(torch.load(sharedecoder_model))
                #
                sal_gen.cuda()
                sal_gen.eval()

                cod_gen.cuda()
                cod_gen.eval()

                shared_decoder.cuda()
                shared_decoder.eval()



                dataset_path = '/home1/liaixuan/dataset/COD/COD_test/'
                test_datasets = ['CAMO','CHAMELEON','COD10K',"NC4K"]

                for dataset in test_datasets:
                    save_path = pre_root + 'cod/' + dataset + '/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print (dataset)
                    image_root = dataset_path + dataset + '/Imgs/'
                    test_loader = test_dataset(image_root, opt.testsize)
                    for i in tqdm( range(test_loader.size)):

                        image, HH, WW, name = test_loader.load_data()
                        image = image.cuda()
                        x1,x2,x3,x4 = cod_gen.forward(image)
                        _,cod_pred = shared_decoder.forward(x1,x2,x3,x4)
                        res = cod_pred
                        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
                        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        cv2.imwrite(save_path+name, res)
                    pass


                dataset_path = '/home1/liaixuan/dataset/SOD_test/datasets/IMG/'
                # test_datasets = ['DUT','DUTS','THUR','HKU-IS', 'SOC']
                test_datasets = ['DUTS','ECSSD','DUT','HKU-IS','PASCAL','SOD']

                for dataset in test_datasets:
                    save_path =  pre_root +'sod/' + dataset + '/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    image_root = dataset_path + dataset + '/'
                    #image_root = '/home/jing-zhang/jing_file/fixation_dataset/train/SALICON/images/test/'
                    test_loader = test_dataset(image_root, opt.testsize)
                    print (dataset)
                    for i in tqdm( range(test_loader.size)):

                        image, HH, WW, name = test_loader.load_data()
                        image = image.cuda()
                        x1,x2,x3,x4 =  sal_gen.forward(image)
                        _,sal_pred = shared_decoder.forward(x1,x2,x3,x4)
                        res = sal_pred
                        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
                        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        cv2.imwrite(save_path+name, res)
                    pass



                result_txt= pre_root+'resultcod.txt'
                sys.stdout = Logger(result_txt)


                cod_gt_dir = '/home1/liaixuan/dataset/COD/COD_test/'
                pred_dir = pre_root + 'cod/'

                test_datasets = ['CAMO','CHAMELEON','COD10K',"NC4K"]
                print('[INFO]: Process in Path [{}]'.format(pred_dir))

                results_list = []
                columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE']

                for dataset in test_datasets:
                    print("[INFO]: Process {} dataset".format(dataset))
                    loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(cod_gt_dir, dataset, 'GT'))

                    def my_collate(batch):
                        data = [item[0] for item in batch]
                        target = [item[1] for item in batch]
                        return [data, target]
                    data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
                    
                    torch.cuda.synchronize(); start = time()
                    [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
                    torch.cuda.synchronize(); end = time()

                    print('[INFO] Time used: {:.4f}'.format(end - start))
                    measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
                    print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), columns=columns_pd).to_string(index=False, float_format="%.5f"))
                    results_list.append(measure_list)

                result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
                print(result_table.to_string(float_format="%.3f"))



                # result_txt= pre_root+'resultsod.txt'
                # sys.stdout = Logger(result_txt)

                # gt_dir='/home1/liaixuan/dataset/SOD_test/datasets/'
                # pred_dir = pre_root + 'sod/'
                # # test_datasets = ["NC4K"]
                # test_datasets = ['DUTS','ECSSD','DUT','HKU-IS','PASCAL']
                # print('[INFO]: Process in Path [{}]'.format(pred_dir))

                # results_list = []
                # columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE'] 

                # for dataset in test_datasets:
                #     print("[INFO]: Process {} dataset".format(dataset))
                #     loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(gt_dir, 'GT', dataset))

                #     def my_collate(batch):
                #         data = [item[0] for item in batch]
                #         target = [item[1] for item in batch]
                #         return [data, target]
                #     data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
                    
                #     torch.cuda.synchronize(); start = time()
                #     [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
                #     torch.cuda.synchronize(); end = time()

                #     print('[INFO] Time used: {:.4f}'.format(end - start))
                #     measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
                #     print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), columns=columns_pd).to_string(index=False, float_format="%.5f"))
                #     results_list.append(measure_list)

                # result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
                # print(result_table.to_string(float_format="%.3f"))


                # result_txt= pre_root+'resultsod.txt'
                # sys.stdout = Logger(result_txt)

                # # test_datasets = ["NC4K"]
                # test_datasets = ['SOD']
                # print('[INFO]: Process in Path [{}]'.format(pred_dir))

                # results_list = []
                # columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE'] 

                # for dataset in test_datasets:
                #     loader = eval_SODDataset(osp.join(pred_dir, dataset), osp.join(gt_dir, 'GT', dataset))
                #     def my_collate(batch):
                #         data = [item[0] for item in batch]
                #         target = [item[1] for item in batch]
                #         return [data, target]
                #     data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
                #     torch.cuda.synchronize(); start = time()
                #     [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
                #     torch.cuda.synchronize(); end = time()
                #     measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
                #     results_list.append(measure_list)

                # result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
                # print(result_table.to_string(float_format="%.3f"))
                        


                result_txt= pre_root+'resultsod.txt'
                sys.stdout = Logger(result_txt)

                gt_dir='/home1/liaixuan/dataset/SOD_test/datasets/'
                pred_dir = pre_root + 'sod/'
                # test_datasets = ["NC4K"]
                test_datasets = ['DUTS','ECSSD','DUT','HKU-IS','PASCAL','SOD']
                print('[INFO]: Process in Path [{}]'.format(pred_dir))

                results_list = []
                columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE'] 

                for dataset in test_datasets:
                    print("[INFO]: Process {} dataset".format(dataset))
                    if dataset != 'SOD':
                        loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(gt_dir, 'GT', dataset))
                    else:
                        loader = eval_SODDataset(osp.join(pred_dir, dataset), osp.join(gt_dir, 'GT', dataset))

                    def my_collate(batch):
                        data = [item[0] for item in batch]
                        target = [item[1] for item in batch]
                        return [data, target]
                    data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
                    
                    torch.cuda.synchronize(); start = time()
                    [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
                    torch.cuda.synchronize(); end = time()

                    print('[INFO] Time used: {:.4f}'.format(end - start))
                    measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
                    print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), columns=columns_pd).to_string(index=False, float_format="%.5f"))
                    results_list.append(measure_list)

                result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
                print(result_table.to_string(float_format="%.3f"))


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from model.ResNet_models import Saliency_feat_encoder, Cod_feat_encoder, Share_feat_decoder
from data import test_dataset
from tqdm import tqdm
import cv2
from eval_func import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_iters', type=int, default=30000, help='epoch number')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--cod_dataset_path', type=str, default='./test_data/COD_test/', help='cod image root')
parser.add_argument('--sod_dataset_path', type=str, default='./test_data/SOD_test/datasets/IMG/', help='sod image root')
parser.add_argument('--cod_gt_dir', type=str, default='./test_data/COD_test/', help='cod gt root')
parser.add_argument('--gt_dir', type=str, default='./test_data/SOD_test/datasets/', help='sod gt root')
opt = parser.parse_args()


model_save_path = './models/'
sal_model = model_save_path + "sal_gen_"+ str(opt.train_iters) +".pth" 
cod_model = model_save_path + "cod_gen_"+ str(opt.train_iters) +".pth" 
sharedecoder_model=model_save_path + "share_dec_"+ str(opt.train_iters) +".pth" 

pre_root= './results/'
print("\""+pre_root+"\"")
if not os.path.exists(pre_root):
    os.makedirs(pre_root)

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

def overleaf(k,f):
    num = []
    for kk in range (len(k)):
        if '0.' in k[kk]:
            cc = str(round(float(k[kk]), 3)).replace('0.','.')
            num.append(f'{cc:0<4}')
    print(' & '.join(num), file=f)


test_datasets = ['CAMO','CHAMELEON','COD10K',"NC4K"]

for dataset in test_datasets:
    save_path = pre_root + 'cod/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print (dataset)
    image_root = opt.cod_dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    for ii in tqdm( range(test_loader.size)):

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



test_datasets = ['DUTS','ECSSD','DUT','HKU-IS','PASCAL','SOD']

for dataset in test_datasets:
    save_path =  pre_root +'sod/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = opt.sod_dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    print (dataset)
    for ii in tqdm( range(test_loader.size)):

        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        x1,x2,x3,x4 =  sal_gen.forward(image)
        _,sal_pred= shared_decoder.forward(x1,x2,x3,x4)
        res = sal_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)
    pass



result_txt= pre_root+'result.txt'
f = open(result_txt, 'w+')


pred_dir = pre_root + 'cod/'
test_datasets = ['CAMO','CHAMELEON','COD10K',"NC4K"]

print('[INFO]: Process in Path [{}]'.format(pred_dir), file=f)

results_list = []
mm_list = []
columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE']

for dataset in test_datasets:
    print("[INFO]: Process {} dataset".format(dataset), file=f)
    loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(opt.cod_gt_dir, dataset, 'GT'))

    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return [data, target]
    data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
    
    torch.cuda.synchronize(); start = time()
    [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
    torch.cuda.synchronize(); end = time()

    print('[INFO] Time used: {:.4f}'.format(end - start), file=f)
    measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
    print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), columns=columns_pd).to_string(index=False, float_format="%.5f"), file=f)
    results_list.append(measure_list)
    for kk in range(4):
        # num = '.'+str(np.around(measure_list[kk], 3)).split(('.'))[-1]
        mm_list.append(str(measure_list[kk]))

result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
print(result_table.to_string(float_format="%.3f"), file=f)
overleaf(mm_list,f)


pred_dir = pre_root + 'sod/'
test_datasets = ['DUTS','ECSSD','DUT','HKU-IS','PASCAL','SOD']
print('[INFO]: Process in Path [{}]'.format(pred_dir) , file=f)

results_list = []
mm_list = []
columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE'] 

for dataset in test_datasets:
    print("[INFO]: Process {} dataset".format(dataset) , file=f)
    if dataset != 'SOD':
        loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(opt.gt_dir, 'GT', dataset))
    else:
        loader = eval_SODDataset(osp.join(pred_dir, dataset), osp.join(opt.gt_dir, 'GT', dataset))

    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return [data, target]
    data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
    
    torch.cuda.synchronize(); start = time()
    [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
    torch.cuda.synchronize(); end = time()

    measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
    results_list.append(measure_list)
    for kk in range(4):
        mm_list.append(str(measure_list[kk]))
result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
print(result_table.to_string(float_format="%.3f") , file=f)
overleaf(mm_list,f)
f.close()

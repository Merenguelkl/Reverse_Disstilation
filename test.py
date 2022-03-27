import torch
import torch.nn as nn
from data_loader import TestDataset
from torch.utils.data import DataLoader
import argparse
import os
from model import Encoder, OcbeAndDecoder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from torchvision.transforms import ToPILImage
from scipy.ndimage import gaussian_filter

def get_ano_map(feature1, feature2):
    mseloss = nn.MSELoss(reduction='none') #1*C*H*W
    mse = mseloss(feature1, feature2) #1*C*H*W
    mse = torch.mean(mse,dim=1) #1*H*W
    cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
    ano_map = torch.ones_like(cos)-cos
    loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
    return ano_map.unsqueeze(1), loss, mse.unsqueeze(1)

def test(obj_name, ckp_dir, data_dir, reshape_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init model
    encoder = Encoder()
    encoder.to(device)
    ocbe_decoder = OcbeAndDecoder()

    ocbe_decoder.load_state_dict(torch.load(str(ckp_dir), map_location='cpu'))
    ocbe_decoder.to(device)
    
    encoder.eval()
    ocbe_decoder.eval()
    
    test_dataset = TestDataset(root_dir=data_dir, obj_name=obj_name, resize_shape=reshape_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_loss_total = 0
    scores=[]
    labels=[]
    gt_list_px = []
    pr_list_px = []
    
    with torch.no_grad():
        for idx, sample_test in enumerate(test_loader):
            image, label, gt = sample_test["image"], sample_test["label"], sample_test["gt_mask"]
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            e_feature1, e_feature2, e_feature3 = encoder(image.to(device))
            d_feature1, d_feature2, d_feature3 = ocbe_decoder(e_feature1, e_feature2, e_feature3)
            
            ano_map1, loss1, mse1 = get_ano_map(e_feature1, d_feature1)
            ano_map2, loss2, mse2 = get_ano_map(e_feature2, d_feature2)
            ano_map3, loss3, mse3 = get_ano_map(e_feature3, d_feature3)

            # add mse to score
            # ano_map1 = ano_map1 + mse1
            # ano_map2 = ano_map2 + mse2
            # ano_map3 = ano_map3 + mse3
            
            
            ano_map1 = nn.functional.interpolate(ano_map1,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map2 = nn.functional.interpolate(ano_map2,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            ano_map3 = nn.functional.interpolate(ano_map3,size=(reshape_size, reshape_size), mode='bilinear', align_corners=True)
            s_al = (ano_map1 + ano_map2 + ano_map3).squeeze().cpu().numpy()
            
            s_al = gaussian_filter(s_al, sigma=4)
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(s_al.ravel())

            score = np.max(s_al.ravel().tolist())
            
            scores.append(score)
            labels.append(label.numpy().squeeze())
            
            loss = loss1.item() + loss2.item() + loss3.item()
            test_loss_total += loss
            
    auroc_img = round(roc_auc_score(np.array(labels), np.array(scores)), 3)
    auroc_pix = round(roc_auc_score(np.array(gt_list_px), np.array(pr_list_px)), 3)
    return test_loss_total, auroc_img, auroc_pix

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# _, auroc_img, auroc_pix= test(obj_name="bottle", ckp_dir="./checkpoints/WRes50/bottle_lr0.001_bs32_2022-03-26_08_01_37/epoch108.pth", data_dir="./datasets/mvtec/" ,reshape_size=256)
# print(auroc_pix)
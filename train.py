import torch
import torch.nn as nn
from data_loader import TrainDataset
from torch.utils.data import DataLoader
import argparse
import os
from model import Encoder, OcbeAndDecoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from loss import CosineLoss
from test import test


def train(obj_name, args):
    resize_shape=256
    print("start train {}".format(obj_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    cur_time = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    run_time = str(obj_name) + "_lr" + str(args.lr) + "_bs" + str(args.bs) + "_" + cur_time
    writer = SummaryWriter(log_dir="./logs/WRes50/"+ run_time +"/")
    os.mkdir("./checkpoints/WRes50/" + run_time)    
    
    # init model
    encoder = Encoder()
    ocbe_decoder = OcbeAndDecoder()
    encoder.to(device)
    ocbe_decoder.to(device)
    encoder.eval()
    
    # init dataloader
    train_dataset = TrainDataset(root_dir=args.data_path, obj_name=obj_name, resize_shape=resize_shape)
    train_loader = DataLoader(
                            train_dataset, 
                            batch_size=args.bs, 
                            shuffle=True,
                            drop_last=True,
                            num_workers=8,
                            persistent_workers=True,
                            pin_memory=True,
                            prefetch_factor=5
                            )
    
    # define loss and optimizer
    mse = nn.MSELoss()
    cos_similarity = CosineLoss()
    optimizer = torch.optim.Adam(ocbe_decoder.parameters(), betas=(0.5,0.999),lr=args.lr)
    
    auroc_img_best, img_step = 0, 0
    auroc_pix_best, pix_step = 0, 0
    
    # training 
    for step in tqdm(range(args.epochs), ascii=True):
        
        ocbe_decoder.train()
        train_loss_total = 0
        for idx, sample in enumerate(train_loader):
            images = sample["image"].to(device)
            
            e_feature1, e_feature2, e_feature3 = encoder(images)
            d_feature1, d_feature2, d_feature3 = ocbe_decoder(e_feature1, e_feature2, e_feature3)
            
            # loss1 = cos_similarity(e_feature1, d_feature1) + mse(e_feature1, d_feature1)
            # loss2 = cos_similarity(e_feature2, d_feature2) + mse(e_feature2, d_feature2)
            # loss3 = cos_similarity(e_feature3, d_feature3) + mse(e_feature3, d_feature3)
            
            loss1 = cos_similarity(e_feature1, d_feature1)
            loss2 = cos_similarity(e_feature2, d_feature2)
            loss3 = cos_similarity(e_feature3, d_feature3)
            
            loss = loss1 + loss2 + loss3
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
        
        writer.add_scalar("train_loss", train_loss_total, int(step))
        
        if (args.test_interval > 0) and (int(step) % args.test_interval == 0):
            ckp_path = str(args.checkpoint_path + "WRes50/" + run_time  +"/epoch" + str(step) + ".pth")
            torch.save(ocbe_decoder.state_dict(), ckp_path)
            test_loss, auroc_img, auroc_pix = test(obj_name=obj_name, ckp_dir=ckp_path, data_dir=args.data_path, reshape_size=resize_shape)
            writer.add_scalar("test_loss", test_loss, int(step))
            writer.add_scalar("auroc_img", auroc_img, int(step))
            writer.add_scalar("auroc_pix", auroc_pix, int(step))
            if auroc_img <= auroc_img_best and auroc_pix <= auroc_pix_best:
                os.remove(ckp_path)
                
            if auroc_img > auroc_img_best:
                auroc_img_best = auroc_img
                img_step = int(step)
            if auroc_pix > auroc_pix_best:
                auroc_pix_best = auroc_pix
                pix_step = int(step)
    
    return run_time, auroc_img_best, auroc_pix_best, img_step, pix_step
            
            
                
            
def write2txt(filename, content):
    f=open(filename,'a')
    f.write(str(content) + "\n")
    f.close()
     
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=False, default=16)
    parser.add_argument('--lr', action='store', type=float, required=False, default=0.005)
    parser.add_argument('--epochs', action='store', type=int, required=False, default=200)
    parser.add_argument('--gpu_id', action='store', type=int, required=False, default=0)
    parser.add_argument('--data_path', action='store', type=str, required=False, default="./datasets/mvtec/")
    parser.add_argument('--checkpoint_path', action='store', type=str, required=False, default="./checkpoints/")
    # parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_interval', action='store',type=int, required=False, default=5)
    
    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    obj_names = ['bottle',
             'cable',
             'capsule',
             'carpet',
             'grid',
             'hazelnut',
             'leather',
             'metal_nut',
             'pill',
             'screw',
             'tile',
             'toothbrush',
             'transistor',
             'wood',
             'zipper']
    
    log_txt_name = "./logs_txt/"+str("{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now()))+".txt"
    os.mknod(log_txt_name)
    
    write2txt(log_txt_name, "log title")
    if args.obj_id == -1:
        for obj_name in obj_names:
            model_name, auroc_img_best, auroc_pix_best, img_step, pix_step= train(obj_name, args)
            write2txt(log_txt_name, str(model_name) + " || auroc_img: " + str(auroc_img_best)+ " epoch:"+str(img_step)+ " || auroc_pix: " + str(auroc_pix_best)+" epoch:"+str(pix_step))
            
    else:
        obj_name = obj_names[int(args.obj_id)]
        model_name, auroc_img_best, auroc_pix_best, img_step, pix_step = train(obj_name, args)
        write2txt(log_txt_name, str(model_name) + " || auroc_img: " + str(auroc_img_best)+ " epoch:"+str(img_step)+ " || auroc_pix: " + str(auroc_pix_best)+" epoch:"+str(pix_step))
    
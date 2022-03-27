from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from pathlib import Path
from torchvision import transforms
import numpy as np
class TrainDataset(Dataset):
    def __init__(self, root_dir, obj_name, transform=None, resize_shape=None):
        self.root_dir = Path(root_dir)
        self.obj_name = obj_name
        self.resize_shape=resize_shape
        self.image_names = sorted(glob.glob(root_dir + self.obj_name + "/train/*/*.png"))
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            # self.transform.transforms.append(transforms.RandomHorizontalFlip())
            # self.transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
            self.transform.transforms.append(transforms.ToTensor())
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img = Image.open(str(self.image_names[idx])).convert("RGB")
        img = self.transform(img)
        return {"image":img}
    
class TestDataset(Dataset):
    def __init__(self, root_dir, obj_name, transform=None, resize_shape=None):
        self.root_dir = Path(root_dir)
        self.obj_name = obj_name
        self.resize_shape=resize_shape
        self.image_names = sorted(glob.glob(root_dir + self.obj_name + "/test/*/*.png"))
        self.gt_root = "./" + "datasets/mvtec/" + self.obj_name + "/ground_truth/"
        
        if transform is not None:
            self.transform = transform
        else:
            # image preprocess
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            self.transform.transforms.append(transforms.ToTensor())
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))
            # gt preprocess
            self.gt_transform = transforms.Compose([])
            self.gt_transform.transforms.append(transforms.Resize((self.resize_shape, self.resize_shape)))
            self.gt_transform.transforms.append(transforms.ToTensor())
            
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_path = str(self.image_names[idx])
        label = img_path.split("/")[-2]
        gt_path = self.gt_root + label + "/" + img_path.split("/")[-1][:3] + "_mask.png"
        img = Image.open(img_path).convert("RGB")
        label = img_path.split("/")[-2]
        img = self.transform(img)
        
        if label == "good":
            gt_img = np.array([0], dtype=np.float32)
            gt_pix = torch.zeros([1, self.resize_shape, self.resize_shape])
        else:
            gt_img = np.array([1], dtype=np.float32)
            gt_pix = self.gt_transform(Image.open(gt_path))
            
        return {"image":img, "label":gt_img, "gt_mask":gt_pix}
    
        # good : 0, anomaly : 1


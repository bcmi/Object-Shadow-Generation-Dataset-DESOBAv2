import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import torchvision.transforms as transforms
from skimage import color
from PIL import Image, ImageMorph
import torch
from skimage.morphology import dilation, square
import os

class TrainDataset(Dataset):
    def __init__(self, data_file_path, device = None):    
        self.device = torch.device("cuda:0") if device == None else device
        self.gpu_ids = [int(self.device.index)]
        self.data_root = data_file_path
        with open(os.path.join(self.data_root, 'train.txt'), 'r') as file:
            self.data = [line.rstrip('\n') for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pic_name = self.data[idx] + '.jpg'
        
        shadowfree_img_path = os.path.join(self.data_root, 'shadowfree_imgs', pic_name)
        object_mask_path = os.path.join(self.data_root, 'object_masks', pic_name)
        shadow_mask_path = os.path.join(self.data_root, 'shadow_masks', pic_name)
        shadow_img_path = os.path.join(self.data_root, 'shadow_imgs', pic_name)
        background_object_mask_path = os.path.join(self.data_root, 'background_object_masks', pic_name)
        background_shadow_mask_path = os.path.join(self.data_root, 'background_shadow_masks', pic_name)
        prompt = ''
        
        width, height = 512, 512
        width_mask, height_mask = 64, 64

        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        background_object_mask = cv2.imread(background_object_mask_path, cv2.IMREAD_GRAYSCALE)
        background_object_mask = cv2.resize(background_object_mask, (width, height))
        background_shadow_mask = cv2.imread(background_shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        background_shadow_mask = cv2.resize(background_shadow_mask, (width, height))
        shadow_img = cv2.imread(shadow_img_path)
        shadow_img = cv2.resize(shadow_img, (width, height))
        shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask = cv2.resize(shadow_mask, (width, height))

        dilated_shadow_mask = cv2.resize(shadow_mask, (width_mask, height_mask))
        kernal = np.ones((6,6), np.uint8)
        dilated_shadow_mask = cv2.dilate(dilated_shadow_mask, kernal, iterations=1)
        
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        source2 = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis], background_shadow_mask[:, :, np.newaxis]), axis=-1)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source2 = source2.astype(np.float32) / 255.0
        shadow_mask = shadow_mask.astype(np.float32) / 255.0
        dilated_shadow_mask = dilated_shadow_mask.astype(np.float32) / 255.0
        object_mask = object_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return dict(jpg=target, txt=prompt, hint=source, hint2=source2, shadowmask=shadow_mask, objectmask=object_mask, dilated_shadow_mask=dilated_shadow_mask)


class TestDataset(Dataset):
    def __init__(self, data_file_path, device = None, ifgt=False):    
        self.device = torch.device("cuda:0") if device == None else device
        self.gpu_ids = [int(self.device.index)]
        self.data_root = data_file_path
        with open(os.path.join(self.data_root, 'test.txt'), 'r') as file:
            self.data = [line.rstrip('\n') for line in file]
        self.ifgt = ifgt    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if '.' in self.data[idx]:
            pic_name = self.data[idx]
        else:
            pic_name = self.data[idx] + '.jpg'
        
        shadowfree_img_path = os.path.join(self.data_root, 'shadowfree_imgs', pic_name)
        object_mask_path = os.path.join(self.data_root, 'object_masks', pic_name)
        shadow_mask_path = os.path.join(self.data_root, 'shadow_masks', pic_name)
        if self.ifgt:
            shadow_img_path = os.path.join(self.data_root, 'shadow_imgs', pic_name)
        else:
            shadow_img_path = os.path.join(self.data_root, 'shadowfree_imgs', pic_name)
        background_object_mask_path = os.path.join(self.data_root, 'background_object_masks', pic_name)
        background_shadow_mask_path = os.path.join(self.data_root, 'background_shadow_masks', pic_name)
        prompt = ''
        
        width, height = 512, 512
        width_mask, height_mask = 64, 64

        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        background_object_mask = cv2.imread(background_object_mask_path, cv2.IMREAD_GRAYSCALE)
        background_object_mask = cv2.resize(background_object_mask, (width, height))
        background_shadow_mask = cv2.imread(background_shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        background_shadow_mask = cv2.resize(background_shadow_mask, (width, height))
        shadow_img = cv2.imread(shadow_img_path)
        shadow_img = cv2.resize(shadow_img, (width, height))
        shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask = cv2.resize(shadow_mask, (width, height))

        dilated_shadow_mask = cv2.resize(shadow_mask, (width_mask, height_mask))
        kernal = np.ones((6,6), np.uint8)
        dilated_shadow_mask = cv2.dilate(dilated_shadow_mask, kernal, iterations=1)
        
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        source2 = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis], background_shadow_mask[:, :, np.newaxis]), axis=-1)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source2 = source2.astype(np.float32) / 255.0
        shadow_mask = shadow_mask.astype(np.float32) / 255.0
        dilated_shadow_mask = dilated_shadow_mask.astype(np.float32) / 255.0
        object_mask = object_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        shadow_img_ = cv2.imread(shadow_img_path)
        shadow_img_ = cv2.resize(shadow_img_, (256, 256))
        target_ = cv2.cvtColor(shadow_img_, cv2.COLOR_BGR2RGB)
        target_ = (target_.astype(np.float32) / 127.5) - 1.0
        
        shadow_mask_ = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask_ = cv2.resize(shadow_mask_, (256, 256))
        shadow_mask_ = shadow_mask_.astype(np.float32) / 255.0
        
        shadowfree_img_ = cv2.imread(shadowfree_img_path)
        shadowfree_img_ = cv2.resize(shadowfree_img_, (256, 256))
        shadowfree_img_ = cv2.cvtColor(shadowfree_img_, cv2.COLOR_BGR2RGB)
        shadowfree_img_ = (shadowfree_img_.astype(np.float32) / 127.5) - 1.0
        
        background_shadow_mask_ = cv2.imread(background_shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        background_shadow_mask_ = cv2.resize(background_shadow_mask_, (256,256))
        background_shadow_mask_ = background_shadow_mask_.astype(np.float32) / 255.0

        background_object_mask_ = cv2.imread(background_object_mask_path, cv2.IMREAD_GRAYSCALE) 
        background_object_mask_ = cv2.resize(background_object_mask_, (256,256))
        background_object_mask_ = background_object_mask_.astype(np.float32) / 255.0

        object_mask_ = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask_ = cv2.resize(object_mask_, (256, 256))
        object_mask_ = object_mask_.astype(np.float32) / 255.0
        
        return dict(jpg=target, txt=prompt, hint=source, hint2=source2, shadowmask=shadow_mask, objectmask=object_mask, \
                        dilated_shadow_mask=dilated_shadow_mask, gt=target_, shadow_mask_ = shadow_mask_, \
                        img_name=pic_name, shadowfree_img_=shadowfree_img_, background_object_mask_=background_object_mask_,\
                        background_shadow_mask_=background_shadow_mask_, object_mask_=object_mask_)
    

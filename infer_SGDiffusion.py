import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import json
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import pickle
from PIL import Image
import os
from einops import rearrange
import torchvision
from torch.utils.data import DataLoader
from tutorial_dataset import TestDataset

from torch.optim import Adam
import torchvision

import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import torchvision.transforms as transforms
from skimage import color
from PIL import Image, ImageMorph
import torch
import fast_histogram
from skimage.morphology import dilation, square

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from collections import OrderedDict
import time
import util
import math

import ssim
from ber import *
import argparse

def trans_tensor2img(grid_, if_mask=False):
    if if_mask:
        grid =(grid_ > 0.5).float()
    else:
        grid = (grid_ + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    result_img = Image.fromarray(grid)
    return result_img



def get_args_parser():
    parser = argparse.ArgumentParser('test shadow diffusion', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint_path', default='data/ckpt/DESOBAv2.ckpt', type=str)
    parser.add_argument('--gpu_id', default=2, type=int)
    parser.add_argument('--test_dataset_path', default='data/desoba_v2', type=str)
    parser.add_argument('--save_dir', default='result', type=str)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cuda'), strict=False)
    model = model.cuda()
    model.eval()

    os.makedirs(os.path.join(args.save_dir,"gen_result"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gen_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadow_img"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadowfree_img"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadow_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_object_mask"), exist_ok=True)

    dataset_test = TestDataset(data_file_path=args.test_dataset_path, device=torch.device("cuda:"+str(args.gpu_id)))
    dataloader_test = DataLoader(dataset_test, num_workers=0, batch_size=1, shuffle=False)


    for step, batch in tqdm(enumerate(dataloader_test)):
        print(step, len(dataloader_test))
        gt = rearrange(batch['gt'], 'b h w c -> b c h w')
        shadowfree_img = rearrange(batch['shadowfree_img_'], 'b h w c -> b c h w')
        shadow_mask_ = batch['shadow_mask_'].unsqueeze(0)
        background_shadow_mask_ = batch['background_shadow_mask_'].unsqueeze(0)
        background_object_mask_ = batch['background_object_mask_'].unsqueeze(0) 
        object_mask_ = batch['object_mask_'].unsqueeze(0)
        img_name = batch['img_name'][0]
        pic_name, extension = os.path.splitext(img_name)    

        for key in batch.keys():
            if key == 'txt' or key == 'img_name':
                batch[key] *= 5
            elif key == 'objectmask':
                batch[key] = batch[key].repeat(5, 1, 1).cuda()
            else:
                batch[key] = batch[key].repeat(5, 1, 1, 1).cuda()
    
        images = model.log_images(batch, use_x_T=True)

        for k in images:
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                images[k] = torch.clamp(images[k], -1., 1.)
        img_to_save = []

        gt_img = trans_tensor2img(gt.squeeze(0))
        gt_object_mask_img = trans_tensor2img(object_mask_.squeeze(0),if_mask=True)
        resultlist = []
        resultlist.append(gt_img)
        resultlist.append(gt_object_mask_img)

        for i in range(5):
            result_img = F.interpolate(images['samples_cfg_scale_9.00'][i].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True)
            result_mask = F.interpolate(images['predicted_mask'][i].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True)
            
            result_img_pil = trans_tensor2img(result_img.squeeze(0))
            result_img_name = pic_name + '_' + str(i) + extension 
            result_img_pil.save(os.path.join(args.save_dir,'gen_result', result_img_name))

            result_mask_pil = trans_tensor2img(result_mask.squeeze(0),if_mask=True)
            result_mask_name = pic_name + '_' + str(i) + extension 
            result_mask_pil.save(os.path.join(args.save_dir,'gen_mask', result_mask_name))

            gt_img = trans_tensor2img(gt.squeeze(0))
            gt_img_name = pic_name + '_' + str(i) + extension
            gt_img.save(os.path.join(args.save_dir, "gt_shadow_img", gt_img_name))

            shadowfree_image = trans_tensor2img(shadowfree_img.squeeze(0))
            shadowfree_img_name = pic_name + '_' + str(i) + extension
            shadowfree_image.save(os.path.join(args.save_dir, "gt_shadowfree_img", shadowfree_img_name))

            gt_shadow_mask_img = trans_tensor2img(shadow_mask_.squeeze(0),if_mask=True)
            gt_shadow_mask_img_name = pic_name + '_' + str(i) + extension 
            gt_shadow_mask_img.save(os.path.join(args.save_dir,'gt_shadow_mask', gt_shadow_mask_img_name))

            gt_object_mask_img = trans_tensor2img(object_mask_.squeeze(0),if_mask=True)
            gt_object_mask_img_name = pic_name + '_' + str(i) + extension 
            gt_object_mask_img.save(os.path.join(args.save_dir,'gt_object_mask', gt_object_mask_img_name))

            
            

            





            


            
        




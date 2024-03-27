from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import json
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import pickle
from PIL import Image
import os
from einops import rearrange
import torchvision
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset

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
import networks
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


def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def concatenate_images_horizontally(image_paths, ifopen=True):
    lens = len(image_paths)
    patch =20
    # 读取所有图片
    if ifopen:
        images = [Image.open(path) for path in image_paths]
    else:
        images = image_paths

    # 确保所有图片的高度相同
    max_height = max(img.height for img in images) + 2 * patch
    total_width = sum(img.width for img in images) + (lens+1) * patch

    # 创建一个新的空白图片，用于放置所有的图片
    concatenated_image = Image.new('RGB', (total_width, max_height))

    # 将每张图片粘贴到新图片上
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset+patch, patch))
        x_offset += img.width

    return concatenated_image

gpu_id = 2
torch.cuda.set_device(gpu_id)
if __name__ == '__main__':
    print('11201920')
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('lightning_logs/version_31/checkpoints/epoch=67-step=314771.ckpt', location='cuda'), strict=False)
    model = model.cuda()
    model.eval()
    save_dir = 'test_result/test4'
    gt_save_dir = 'test_result/test_desobav2_5times/v2_test_gt'
    sf_save_dir = 'test_result/test_desobav2_5times/v2_test_sf'
    mask_save_dir = 'test_result/test_desobav2_5times/v2_test_mask'

    os.makedirs(os.path.join(save_dir,"result"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"mask"), exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)
    os.makedirs(sf_save_dir, exist_ok=True)

    os.makedirs(os.path.join(mask_save_dir,"background_shadow_mask"), exist_ok=True)
    os.makedirs(os.path.join(mask_save_dir,"background_object_mask"), exist_ok=True)
    os.makedirs(os.path.join(mask_save_dir,"gt_shadow_mask"), exist_ok=True)
    os.makedirs(os.path.join(mask_save_dir,"gt_object_mask"), exist_ok=True)

    dataset_test = MyDataset(data_file_list=['training/tset3.json'], device=torch.device("cuda:"+str(gpu_id)), iftest=True)
    dataloader_test = DataLoader(dataset_test, num_workers=0, batch_size=1, shuffle=False)

    RMSE = []
    shadowRMSE = []
    SSIM = []
    shadowSSIM = []
    GPSNR = []
    LPSNR = []
    GBer_fix = []
    LBer_fix = []
    GBer_ratio = []
    LBer_ratio = []

    count = 0

    for step, batch in tqdm(enumerate(dataloader_test)):
        print(step, len(dataloader_test))
        # count +=1
        # if step <18:
        #     continue
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

        min_global_ber = 0x3f3f3f3f

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
            result_img_pil.save(os.path.join(save_dir,'result', result_img_name))

            # a = result_mask.squeeze(0)
            result_mask_pil = trans_tensor2img(result_mask.squeeze(0),if_mask=True)
            result_mask_name = pic_name + '_' + str(i) + '.png' 
            result_mask_pil.save(os.path.join(save_dir,'mask', result_mask_name))

        #     resultlist.append(result_img_pil)
        #     resultlist.append(result_mask_pil)
        
        # total_result = concatenate_images_horizontally(resultlist, ifopen=False)
        # total_result_name = pic_name + extension
        # total_result.save(os.path.join(save_dir, total_result_name))
    #         gt_img = trans_tensor2img(gt.squeeze(0))
    #         gt_img_name = pic_name + '_' + str(i) + extension
    #         gt_img.save(os.path.join(gt_save_dir, gt_img_name))

    #         shadowfree_image = trans_tensor2img(shadowfree_img.squeeze(0))
    #         shadowfree_img_name = pic_name + '_' + str(i) + extension
    #         shadowfree_image.save(os.path.join(sf_save_dir, shadowfree_img_name))

    #         background_shadow_mask_img = trans_tensor2img(background_shadow_mask_.squeeze(0),if_mask=True)
    #         background_shadow_mask_img_name = pic_name + '_' + str(i) + '.png'
    #         background_shadow_mask_img.save(os.path.join(mask_save_dir,'background_shadow_mask', background_shadow_mask_img_name))

    #         background_object_mask_img = trans_tensor2img(background_object_mask_.squeeze(0),if_mask=True)
    #         background_object_mask_img_name = pic_name + '_' + str(i) + '.png'
    #         background_object_mask_img.save(os.path.join(mask_save_dir,'background_object_mask', background_object_mask_img_name))
            
    #         gt_shadow_mask_img = trans_tensor2img(shadow_mask_.squeeze(0),if_mask=True)
    #         gt_shadow_mask_img_name = pic_name + '_' + str(i) + '.png'
    #         gt_shadow_mask_img.save(os.path.join(mask_save_dir,'gt_shadow_mask', gt_shadow_mask_img_name))

    #         gt_object_mask_img = trans_tensor2img(object_mask_.squeeze(0),if_mask=True)
    #         gt_object_mask_img_name = pic_name + '_' + str(i) + '.png'
    #         gt_object_mask_img.save(os.path.join(mask_save_dir,'gt_object_mask', gt_object_mask_img_name))

    #         mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.], device=x.device))

    #         t = time.time()
    #         visual_ret = OrderedDict()
    #         shadowimg_gt = util.tensor2im(gt.squeeze(0)).astype(np.float32)
    #         prediction = util.tensor2im(result_img.squeeze(0)).astype(np.float32)
    #         shadow_mask_threshold = (shadow_mask_ > 0.5).float().cpu()
    #         mask = util.tensor2imonechannel(shadow_mask_threshold)

    #         # local_rmse_tmp = math.sqrt(mean_squared_error(shadowimg_gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255))
    #         pred_mask = util.tensor2imonechannel(result_mask)
    #         global_ber, local_ber = ber_fixshreshold(pred_mask, mask)
    #         if global_ber[0] < min_global_ber:
    #             min_local_rmse = global_ber[0]
                
    #             local_rmse = math.sqrt(mean_squared_error(shadowimg_gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255))
    #             global_rmse = math.sqrt(mean_squared_error(shadowimg_gt, prediction))
                
    #             gt_tensor = (gt/2 + 0.5) * 255
    #             prediction_tensor = (result_img/2 + 0.5) * 255
    #             mask_tensor = shadow_mask_threshold
    #             global_ssim = ssim.ssim(gt_tensor, prediction_tensor, window_size = 11, size_average = True)
    #             local_ssim = ssim.ssim(gt_tensor, prediction_tensor,mask=mask_tensor)

    #             global_psnr = mse2psnr(torch.tensor(mean_squared_error(shadowimg_gt / 255.0, prediction / 255.0)))
    #             local_psnr = mse2psnr(torch.tensor(mean_squared_error(shadowimg_gt / 255.0*(mask/255), prediction / 255.0*(mask/255))*256*256/np.sum(mask/255)))

    #             pred_mask = util.tensor2imonechannel(result_mask)
    #             global_ber, local_ber = ber_fixshreshold(pred_mask, mask)
    #             global_ratiober, local_ratiober = ber_ratioshreshold(pred_mask, mask)
            
    #     RMSE.append(global_rmse)
    #     shadowRMSE.append(local_rmse)
    #     SSIM.append(global_ssim)
    #     shadowSSIM.append(local_ssim)
    #     GPSNR.append(global_psnr)
    #     LPSNR.append(local_psnr)
    #     GBer_fix.append(global_ber)
    #     LBer_fix.append(local_ber)
    #     GBer_ratio.append(global_ratiober)
    #     LBer_ratio.append(local_ratiober)
        
    #     print("RMSE:",global_rmse)
    #     print("shadowRMSE:",local_rmse)
    #     print("SSIM:",global_ssim)
    #     print("shadowSSIM:",local_ssim)
    #     print("GPSNR:",global_psnr)
    #     print("LPSNR:",local_psnr)
    #     print("GBer_fix:",global_ber)
    #     print("LBer_fix:",local_ber)
    #     print("GBer_ratio:",global_ratiober)
    #     print("LBer_ratio:",local_ratiober)


    # RMSE = [num for num in RMSE if not math.isnan(num)]
    # shadowRMSE = [num for num in shadowRMSE if not math.isnan(num)]
    # SSIM = [num for num in SSIM if not math.isnan(num)]
    # shadowSSIM = [num for num in shadowSSIM if not math.isnan(num)]
    # GPSNR = [num for num in GPSNR if not math.isnan(num)]
    # LPSNR = [num for num in LPSNR if not math.isnan(num)]
    # GBer_fix = [num[0] for num in GBer_fix if not math.isnan(num[0])]
    # LBer_fix = [num[0] for num in LBer_fix if not math.isnan(num[0])]
    # GBer_ratio = [num[0] for num in GBer_ratio if not math.isnan(num[0])]
    # LBer_ratio = [num[0] for num in LBer_ratio if not math.isnan(num[0])]

    # final_RMSE = sum(RMSE) / len(RMSE)
    # final_shadowRMSE = sum(shadowRMSE) / len(shadowRMSE)
    # final_SSIM = sum(SSIM) / len(SSIM)
    # final_shadowSSIM = sum(shadowSSIM) / len(shadowSSIM)
    # final_GPSNR = sum(GPSNR) / len(GPSNR)
    # final_LPSNR = sum(LPSNR) / len(LPSNR)
    # final_GBer_fix = sum(GBer_fix) / len(GBer_fix)
    # final_LBer_fix = sum(LBer_fix) / len(LBer_fix)
    # final_GBer_ratio = sum(GBer_ratio) / len(GBer_ratio)
    # final_LBer_ratio = sum(LBer_ratio) / len(LBer_ratio)

    # import sys
    # output = sys.stdout
    # outputfile = open(os.path.join(save_dir, 'result.txt'),'w')
    # sys.stdout = outputfile

    # print('RMSE: ', final_RMSE)
    # print('shadowRMSE: ', final_shadowRMSE)
    # print('SSIM: ', final_SSIM)
    # print('shadowSSIM: ', final_shadowSSIM)
    # print('GPSNR: ', final_GPSNR)
    # print('LPSNR: ', final_LPSNR)
    # print('GBer_fix: ', final_GBer_fix)
    # print('LBer_fix: ', final_LBer_fix)
    # print('GBer_ratio: ', final_GBer_ratio)
    # print('LBer_ratio: ', final_LBer_ratio)

    # outputfile.close()
    # sys.stdout = output        

            
            

            





            


            
        




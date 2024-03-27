import numpy as np
import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
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


def load_image_as_tensor(image_path, if_mask=False):
    if if_mask:
        image = Image.open(image_path).convert('L')
        image_numpy = np.array(image)
        image_tensor = torch.from_numpy(image_numpy).unsqueeze(0).unsqueeze(0).float() / 255.0
        image_tensor = (image_tensor * 2.0) - 1.0
    else:
        image = Image.open(image_path).convert('RGB')
        image_numpy = np.array(image)
        image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor * 2.0) - 1.0
    return image_tensor

def get_args_parser():
    parser = argparse.ArgumentParser('eval shadow diffusion', add_help=False)
    parser.add_argument('--result_dataset_path', default='result', type=str)
    return parser

if __name__ == '__main__':
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

    parser = get_args_parser()
    args = parser.parse_args()
    
    eval_image_list = os.listdir(os.path.join(args.result_dataset_path, 'gen_result'))

    for image_name in eval_image_list: 
        gt = load_image_as_tensor(os.path.join(args.result_dataset_path, 'gt_shadow_img', image_name))  
        result_img = load_image_as_tensor(os.path.join(args.result_dataset_path, 'pp_result', image_name))
        shadow_mask_ = load_image_as_tensor(os.path.join(args.result_dataset_path, 'gt_shadow_mask', image_name), if_mask=True)   
        result_mask = load_image_as_tensor(os.path.join(args.result_dataset_path, 'gen_mask', image_name), if_mask=True) 
    
        mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.], device=x.device))

        t = time.time()
        visual_ret = OrderedDict()
        shadowimg_gt = util.tensor2im(gt).astype(np.float32)
        prediction = util.tensor2im(result_img.squeeze(0)).astype(np.float32)
        shadow_mask_threshold = (shadow_mask_ > 0.5).float().cpu()
        mask = util.tensor2imonechannel(shadow_mask_threshold)

        # local_rmse_tmp = math.sqrt(mean_squared_error(shadowimg_gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255))
        pred_mask = util.tensor2imonechannel(result_mask)
        global_ber, local_ber = ber_fixshreshold(pred_mask, mask)
        
        local_rmse = math.sqrt(mean_squared_error(shadowimg_gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255))
        global_rmse = math.sqrt(mean_squared_error(shadowimg_gt, prediction))
        
        gt_tensor = ((gt/2 + 0.5) * 255).unsqueeze(0)
        prediction_tensor = ((result_img/2 + 0.5) * 255).unsqueeze(0)
        mask_tensor = shadow_mask_threshold
        global_ssim = ssim.ssim(gt_tensor, prediction_tensor, window_size = 11, size_average = True)
        local_ssim = ssim.ssim(gt_tensor, prediction_tensor,mask=mask_tensor)

        global_psnr = mse2psnr(torch.tensor(mean_squared_error(shadowimg_gt / 255.0, prediction / 255.0)))
        local_psnr = mse2psnr(torch.tensor(mean_squared_error(shadowimg_gt / 255.0*(mask/255), prediction / 255.0*(mask/255))*256*256/np.sum(mask/255)))

        pred_mask = util.tensor2imonechannel(result_mask)
        global_ber, local_ber = ber_fixshreshold(pred_mask, mask)
        global_ratiober, local_ratiober = ber_ratioshreshold(pred_mask, mask)
        
        RMSE.append(global_rmse)
        shadowRMSE.append(local_rmse)
        SSIM.append(global_ssim)
        shadowSSIM.append(local_ssim)
        GPSNR.append(global_psnr)
        LPSNR.append(local_psnr)
        GBer_fix.append(global_ber)
        LBer_fix.append(local_ber)
        GBer_ratio.append(global_ratiober)
        LBer_ratio.append(local_ratiober)
        


    RMSE = [num for num in RMSE if not math.isnan(num)]
    shadowRMSE = [num for num in shadowRMSE if not math.isnan(num)]
    SSIM = [num for num in SSIM if not math.isnan(num)]
    shadowSSIM = [num for num in shadowSSIM if not math.isnan(num)]
    GPSNR = [num for num in GPSNR if not math.isnan(num)]
    LPSNR = [num for num in LPSNR if not math.isnan(num)]
    GBer_fix = [num[0] for num in GBer_fix if not math.isnan(num[0])]
    LBer_fix = [num[0] for num in LBer_fix if not math.isnan(num[0])]
    GBer_ratio = [num[0] for num in GBer_ratio if not math.isnan(num[0])]
    LBer_ratio = [num[0] for num in LBer_ratio if not math.isnan(num[0])]

    final_RMSE = sum(RMSE) / len(RMSE)
    final_shadowRMSE = sum(shadowRMSE) / len(shadowRMSE)
    final_SSIM = sum(SSIM) / len(SSIM)
    final_shadowSSIM = sum(shadowSSIM) / len(shadowSSIM)
    final_GPSNR = sum(GPSNR) / len(GPSNR)
    final_LPSNR = sum(LPSNR) / len(LPSNR)
    final_GBer_fix = sum(GBer_fix) / len(GBer_fix)
    final_LBer_fix = sum(LBer_fix) / len(LBer_fix)
    final_GBer_ratio = sum(GBer_ratio) / len(GBer_ratio)
    final_LBer_ratio = sum(LBer_ratio) / len(LBer_ratio)

    print('RMSE: ', final_RMSE)
    print('shadowRMSE: ', final_shadowRMSE)
    print('SSIM: ', final_SSIM)
    print('shadowSSIM: ', final_shadowSSIM)
    print('GPSNR: ', final_GPSNR)
    print('LPSNR: ', final_LPSNR)
    print('GBer_fix: ', final_GBer_fix)
    print('LBer_fix: ', final_LBer_fix)
    print('GBer_ratio: ', final_GBer_ratio)
    print('LBer_ratio: ', final_LBer_ratio)
            
            

            





            


            
        




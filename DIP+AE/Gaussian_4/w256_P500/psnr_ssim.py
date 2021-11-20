### This script will calcualte psnr and ssim
### draw plot and save them into a csv file
import pandas as pd
import glob
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim


def calcualte_PSNR(im_true, im_test):
    im_true = np.transpose(im_true,(1,2,0))
    im_test = np.transpose(im_test, (1, 2, 0))
    #print('The imagesize is PSNR')
    #print(im_true.shape)
    #print(im_true.shape)
    psnr_value = peak_signal_noise_ratio(im_true, im_test)
    return psnr_value


def calcualte_SSIM(im_true, im_test, multichannel): # multichannel=True for RGB images
    im_true = np.transpose(im_true, (1, 2, 0))
    im_test = np.transpose(im_test, (1, 2, 0))
    #print('The imagesize is SSIM ')
    #print(im_true.shape)
    #print(im_true.shape)
    ssim_value = compare_ssim(im_true, im_test, multichannel=multichannel, data_range=im_test.max() - im_test.min())
    return ssim_value





if __name__ == '__main__':
    pass
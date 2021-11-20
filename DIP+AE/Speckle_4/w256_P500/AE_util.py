import torch
import torch.utils.data as Data
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math
import skimage as sk
from skimage import io


###################################################################
################# Prepare AE Dataset into Batch Size #################
###################################################################
class AEDataset(Dataset):
    def __init__(self, clean_data, num, transform=None):
        self.transform = transform
        self.clean_data = clean_data
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample_clean = self.clean_data[idx,:]
        if self.transform:
            #sample_clean = self.transform(sample_clean)
            sample_clean = torch.from_numpy(sample_clean)
        return (sample_clean,idx)


#################  #################
def prepare_AEData(input_data, batch_size, num):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
    ])
    AE_dataset = AEDataset(input_data, num, transform=transform)

    ##### used to give different samples with different weights
    ##### in our experiemnts, we do not use it; so we comment it out
    ##### but you can try it to see whether you can further improve the performance
    # sample_nums = 0
    # weights = []
    # alpha = 0.9
    # fenmu = 0
    # for i in range(num,0, -1):
    #     fenmu += np.power(alpha, i)
    # for data, idx in AE_dataset:
    #     cur_weight = np.power(alpha, num-idx)/fenmu
    #     weights.append(cur_weight)
    #
    # add_w = np.sum(weights)
    #
    #
    # sample_nums = num
    #
    # # if num<2:
    # #     sample_nums = num
    # # else:
    # #     sample_nums = num -1
    # #
    # # batch_size = 1
    # sampler = WeightedRandomSampler(weights,
    #                                 num_samples=sample_nums,
    #                                 replacement=True)

    dataloader = DataLoader(AE_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
    return dataloader, AE_dataset



def draw_figures_AE(all_images, figure_name):
    #all_images = torch.cat([true_images, fake_images], dim=0)
    #plt.figure(figsize=(32, 32))
    plt.figure()
    plt.axis("off")
    plt.title("Real Images (1st), Corrupted Images (2nd), Reconstructed Images (3rd)")
    plt.imshow(np.transpose(vutils.make_grid(all_images, nrow=2, padding=2, normalize=True),(1,2,0)))
    plt.savefig(figure_name)
    plt.close()

if __name__ == '__main__':
    pass
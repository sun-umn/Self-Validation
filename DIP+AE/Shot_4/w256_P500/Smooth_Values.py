### This script conducts: 1) Smooth the curve;
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from util import *

####### Whether we want to further smooth the AE reconstruction error curve
####### In our paper, we do not use it.
####### Get Smoothed Values
def SmoothIt(input_values, alpha, adjust_tag, savefile):
    smooth_values = []
    for item in input_values:
        if len(smooth_values) == 0:
            smooth_values.append(item)
        else:
            previous_sv = smooth_values[-1]
            new_sv = alpha * previous_sv + (1 - alpha) * item
            if adjust_tag == "Y":
                t = len(smooth_values) + 1
                new_sv = new_sv / (1 - alpha ** t)
            smooth_values.append(new_sv)
    smooth_values = np.asarray(smooth_values).reshape(-1,1)
    save_code(smooth_values, savefile)
    return smooth_values


def plot_SV(smooth_values, alpha, figure_name):
    plt.plot(smooth_values)
    plt.title('Smoothed Err (alpha={})'.format(alpha))
    plt.ylabel('Reconstructed Err')
    plt.xlabel('Epoch Index')
    plt.savefig(figure_name)
    plt.close()

if __name__ == '__main__':
    root_dir_fig_noadj = '0_Smooth/figures/noadjust/'
    root_dir_fig_adj = '0_Smooth/figures/adjust/'
    root_dir_data_noadj = '0_Smooth/data_noadj'
    root_dir_data_adj = '0_Smooth/data_adj'

    for dir in [root_dir_fig_noadj,root_dir_fig_adj,root_dir_data_noadj,root_dir_data_adj]:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Do not need to create the folder!')

    hist_err_file = 'AE/hist_new_csv/2900.csv'
    hist_err_df = pd.read_csv(hist_err_file)
    hist_err = hist_err_df['Hist_Mean'].values

    alpha_list = np.arange(0, 1.1, 0.1)

    adjust_tag = "N"
    for alpha in alpha_list:
        savefile = os.path.join(root_dir_data_noadj,'smooth_{}.npz'.format(round(alpha,2)))
        SVs = SmoothIt(hist_err, alpha, adjust_tag,savefile)
        fig_alpha = round(alpha, 1)
        figure_name = os.path.join(root_dir_fig_noadj,'{}_{}.png'.format(fig_alpha, adjust_tag))
        plot_SV(SVs, fig_alpha, figure_name)


    adjust_tag = "Y"
    for alpha in alpha_list:
        savefile = os.path.join(root_dir_data_adj,'smooth_{}.npz'.format(round(alpha,2)))
        SVs = SmoothIt(hist_err, alpha, adjust_tag,savefile)
        fig_alpha = round(alpha, 1)
        figure_name = os.path.join(root_dir_fig_adj, '{}_{}.png'.format(fig_alpha, adjust_tag))
        plot_SV(SVs, fig_alpha, figure_name)



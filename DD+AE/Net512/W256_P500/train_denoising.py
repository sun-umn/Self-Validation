#----------------This is the main script for denoising-----------------#

from torch.autograd import Variable
from util import *
from tqdm import tqdm
import time
import torch.optim.lr_scheduler as lrs
import math
import pandas as pd
from psnr_ssim import *
import torch
import torch.optim
from AE_train import *
from Early_Stop import *
from include import *

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# This function will reconstruct images by DIP and call our AE to decide the early-stopping points
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def DD_train(learning_rate_model, OPTIMIZER, LOSS, net_width, clean_image_path, corrupted_image_path, max_epoch, print_step,gpu,earlystop,cur_image,start_detection,Keep_track,window_size):
    ######### Create floders to host our results (e.g., the models, the reconstructed images, the loss, and the PSNR/SSIM)
    dir_name = '{}/UntrainedNN_training/'.format(cur_image)
    dir_name_best = '{}/0_BEST_Results/'.format(cur_image)
    make_dir([dir_name, dir_name_best])

    ################## Using GPU when it is available ##################
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    print('#_______________________')
    print('#_______________________')
    print(torch.cuda.is_available())
    print('device info:')
    print(device)
    print('#_______________________')
    print('#_______________________')

    ############################ Settings from DD
    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    reg_noise_std = 0 # 0 for deep decoder #1. / 30.  # set to 1./20. for sigma=50
    img_size = 512
    width=net_width
    num_channels = [width] * 5
    upsample_first = True
    output_depth = 3

    net = decodernw(output_depth, num_channels_up=num_channels, upsample_first=upsample_first)
    net.to(device)

    #----------------------------- net input------------------------------#
    net_input = None
    if net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network
        totalupsample = 2**len(num_channels)
        width = int(512/totalupsample)
        height = int(512/totalupsample)
        shape = [1, num_channels[0], width, height]
        print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10
        net_input = net_input.to(device)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    ################## Get Dataset ##################
    train_loader = prepare_data(clean_image_path, corrupted_image_path)

    # define loss function & Opt
    l1_loss_func = L1_Func()
    l2_loss_func = L2_Func()


    ################## Define Optimizer ##################
    # RMSprop
    if OPTIMIZER == 'SGD':
        optimizer_net = torch.optim.SGD(net.parameters(),learning_rate_model,momentum=0.9)
    elif OPTIMIZER == 'Adam':
        optimizer_net = torch.optim.Adam(net.parameters(), learning_rate_model)
    else:
        assert False, "Optimizer is wrong!"


    ################## Start to Train ##################
    Hist_Err_Epoch = {'Epoch':[], 'AE_Err':[]}

    Collected_Output = []
    total_loss = []
    total_loss_epoch = []
    total_epoch = []

    train_psnr_corr = []
    train_psnr_rec = []

    train_ssim_corr = []
    train_ssim_rec = []

    ########## start to optimize ############
    for epoch in range(max_epoch):
        net.train()
        epoch_loss = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)
        for step, (X_clean, X_corrupted, idx) in enumerate(train_loader):
            ########### zero grad each time #############
            optimizer_net.zero_grad()

            ################## Get Training & Traget Dataset ##################
            X_clean, X_corrupted = X_clean.to(device), X_corrupted.to(device)

            ################## Train and Backpropagation ##################
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(net_input)

            if LOSS == "L1":
                print('**************')
                print('Using L1 loss')
                print('**************')
                loss = l1_loss_func(X_corrupted, out)  # L1
            elif LOSS == "MSE":
                print('**************')
                print('Using MSE')
                print('**************')
                loss = l2_loss_func(X_corrupted, out)  # MSE
            elif LOSS == 'Huber':
                print('**************')
                print('Using Huber')
                print('**************')
                delta = 0.05
                loss = Huber_Loss(X_corrupted, out, delta)
            else:
                assert False, "Loss function is wrong!"

            ############ backpropagation update ###########
            loss.backward()
            optimizer_net.step()

            epoch_loss.append(loss.item())
            progress.set_postfix({'loss': loss.item()})
            progress.update()

            ###### Collect the historical reconstructed images ##############
            ###### We will it to train our AE ################
            if epoch >= start_detection and Keep_track==True:
                cur_out = out.detach().cpu().numpy()
                cur_out = np.clip(cur_out, 0, 1)
                Collected_Output.append(cur_out)
                cur_collection_size = len(Collected_Output)
                if cur_collection_size >= window_size + 1:
                    Collected_Output.pop(0)

            ######### Here, we need to detour to train our AutoEncoder ###################
            if epoch >= start_detection and len(Collected_Output) == window_size and epoch % print_step == 0 and Keep_track == True:

                #### Get the feedback information from AE to decide whether we should stop
                AE_Err, AE_Stop, AE_Save = train_stopAE(gpu, Collected_Output, earlystop, epoch, cur_image)

                Hist_Err_Epoch['AE_Err'].append(AE_Err)
                Hist_Err_Epoch['Epoch'].append(epoch)

                ##### Should save the current best results
                if AE_Save == True:
                    # -----------------------Let's save our best data-----
                    code_file = os.path.join(dir_name_best, '{}_best_data.npz'.format(cur_image))
                    with torch.no_grad():
                        net.eval()
                        out = net(net_input_saved.to(device))
                        out = out.detach().cpu()

                        best_data = out.detach().cpu().numpy()
                        save_code(best_data, code_file)
                        print('Congrats! Best data has been saved successfully!')

                        model_file = os.path.join(dir_name_best, '{}_best_net.npz'.format(cur_image))
                        save_model(net.eval(), model_file)

                ##### Should stop now
                if AE_Stop == True:
                    # ------------------------ we must stop our training at this epoch---------------
                    # ----1) Let's retrieve our best data and get its performance
                    # ----2) Let's plot hist err
                    # ----3) Let's retrieve best epoch
                    hist_err = np.asarray(Hist_Err_Epoch['AE_Err']).flatten()
                    hist_epoch = np.asarray(Hist_Err_Epoch['Epoch']).flatten()
                    best_epoch = earlystop.get_best_info()
                    figure_name = os.path.join(dir_name_best, 'hist_err_{}.png'.format(best_epoch))
                    display_hist_err(hist_epoch, hist_err, best_epoch, figure_name)

                    hist_err_epoch_df = pd.DataFrame.from_dict(Hist_Err_Epoch)
                    hist_err_epoch_file = os.path.join(dir_name_best, 'hist_err_{}.csv'.format(best_epoch))
                    hist_err_epoch_df.to_csv(hist_err_epoch_file)

                    # retrieve best data and get its prediction info
                    final_best_data = np.load(os.path.join(dir_name_best, '{}_best_data.npz'.format(cur_image)))[
                                          'arr_0'][0, :]  # 3,512,512---0,1,2
                    final_best_data = np.transpose(final_best_data, (1, 2, 0))
                    final_best_data = np.clip(final_best_data, 0, 1)
                    clean_data = np.load(clean_image_path)['arr_0'] / 255.  # 512,512,3----0,1,2
                    clean_data = np.clip(clean_data, 0, 1)
                    # clean_data = np.transpose(clean_data,(2,0,1))
                    corrupted_data = np.load(corrupted_image_path)['arr_0'] / 255.
                    corrupted_data = np.clip(corrupted_data, 0, 1)
                    # corrupted_data = np.transpose(corrupted_data, (2, 0, 1))
                    corrupted_psnr = peak_signal_noise_ratio(clean_data, corrupted_data)
                    corrupted_ssim = compare_ssim(clean_data, corrupted_data, multichannel=True,
                                                  data_range=corrupted_data.max() - corrupted_data.min())
                    rec_psnr = peak_signal_noise_ratio(clean_data, final_best_data)
                    rec_ssim = compare_ssim(clean_data, final_best_data, multichannel=True,
                                            data_range=final_best_data.max() - final_best_data.min())
                    best_dict = {'Epoch': [best_epoch],
                                 'Corr_PSNR': [corrupted_psnr],
                                 'Corr_SSIM': [corrupted_ssim],
                                 'Stop_PSNR': [rec_psnr],
                                 'Stop_SSIM': [rec_ssim]}
                    best_df = pd.DataFrame.from_dict(best_dict)
                    best_file = os.path.join(dir_name_best, '{}_best_metrics.csv'.format(cur_image))
                    best_df.to_csv(best_file, index=False)
                    print('We must stop at this epoch! Exit!')
                    Keep_track = False

            ##### this case is we do not stop even we reach the maximum epoch we set
            ##### if Keep_track == True and epoch== max_epoch
            if (Keep_track == True) and (epoch == max_epoch - 1):
                # ------------------------ we must stop our training at this epoch---------------
                # ----1) Let's retrieve our best data and get its performance
                # ----2) Let's plot hist err
                # ----3) Let's retrieve best epoch
                hist_err = np.asarray(Hist_Err_Epoch['AE_Err']).flatten()
                hist_epoch = np.asarray(Hist_Err_Epoch['Epoch']).flatten()
                best_epoch = earlystop.get_best_info()
                figure_name = os.path.join(dir_name_best, 'hist_err_{}.png'.format(best_epoch))
                display_hist_err(hist_epoch, hist_err, best_epoch, figure_name)

                hist_err_epoch_df = pd.DataFrame.from_dict(Hist_Err_Epoch)
                hist_err_epoch_file = os.path.join(dir_name_best, 'hist_err_{}.csv'.format(best_epoch))
                hist_err_epoch_df.to_csv(hist_err_epoch_file)

                # retrieve best data and get its prediction info
                final_best_data = np.load(os.path.join(dir_name_best, '{}_best_data.npz'.format(cur_image)))[
                                      'arr_0'][0, :]  # 3,512,512---0,1,2
                final_best_data = np.transpose(final_best_data, (1, 2, 0))
                final_best_data = np.clip(final_best_data, 0, 1)
                clean_data = np.load(clean_image_path)['arr_0'] / 255.  # 512,512,3----0,1,2
                clean_data = np.clip(clean_data, 0, 1)
                # clean_data = np.transpose(clean_data,(2,0,1))
                corrupted_data = np.load(corrupted_image_path)['arr_0'] / 255.
                corrupted_data = np.clip(corrupted_data, 0, 1)
                # corrupted_data = np.transpose(corrupted_data, (2, 0, 1))
                corrupted_psnr = peak_signal_noise_ratio(clean_data, corrupted_data)
                corrupted_ssim = compare_ssim(clean_data, corrupted_data, multichannel=True,
                                              data_range=corrupted_data.max() - corrupted_data.min())
                rec_psnr = peak_signal_noise_ratio(clean_data, final_best_data)
                rec_ssim = compare_ssim(clean_data, final_best_data, multichannel=True,
                                        data_range=final_best_data.max() - final_best_data.min())
                best_dict = {'Epoch': [best_epoch],
                             'Corr_PSNR': [corrupted_psnr],
                             'Corr_SSIM': [corrupted_ssim],
                             'Stop_PSNR': [rec_psnr],
                             'Stop_SSIM': [rec_ssim]}
                best_df = pd.DataFrame.from_dict(best_dict)
                best_file = os.path.join(dir_name_best, '{}_best_metrics.csv'.format(cur_image))
                best_df.to_csv(best_file, index=False)
                print('We must stop at this epoch! Exit!')
                Keep_track = False

        progress.close()
        total_loss.append(np.mean(epoch_loss))
        total_loss_epoch.append(epoch)

        ############### after one epoch training, we can start to test our model
        if epoch%print_step == 0:
            figure_name = dir_name + '00_rec.png'
            train_mean_psnr_corr, train_mean_psnr_rec, train_mean_ssim_corr, train_mean_ssim_rec=DD_test(net, net_input_saved, train_loader, device, figure_name)
            train_psnr_corr.append(train_mean_psnr_corr)
            train_psnr_rec.append(train_mean_psnr_rec)
            train_ssim_corr.append(train_mean_ssim_corr)
            train_ssim_rec.append(train_mean_ssim_rec)

            total_epoch.append(epoch)

            ##### plot loss
            loss_file_name = dir_name + 'loss_{}.png'.format(epoch)
            #display_loss(total_loss, print_step, loss_file=loss_file_name) # uncomment this line to have the loss plot

            #### plost psnr
            if epoch == (max_epoch-1):
                loss_file_name = dir_name + 'psnr.png'
                display_psnr(train_psnr_rec, train_psnr_corr, print_step, loss_file=loss_file_name)

            #### after training, we can save the psnr values
            psnr_dict = {'epoch': total_epoch,
                         'train_psnr_corr': train_psnr_corr,
                         'train_psnr_rec': train_psnr_rec,
                         'train_ssim_corr': train_ssim_corr,
                         'train_ssim_rec': train_ssim_rec,
                         }
            psnr_df = pd.DataFrame.from_dict(psnr_dict)
            if epoch == (max_epoch - 1):
                psnr_file = dir_name + '00_psnr.csv'
                psnr_df.to_csv(psnr_file, index=False)

    ################### after we have finished all epochs we set, let's finally summarize all the results into csv files#####
    ############ save final results
    stop_epoch = earlystop.get_best_info()
    final_psnr_file = os.path.join(dir_name,'00_psnr.csv')
    final_psnr_df = pd.read_csv(final_psnr_file)
    ####### get stop record
    stop_record = final_psnr_df[final_psnr_df['epoch']==stop_epoch]
    stop_psnr = stop_record['train_psnr_rec'].values[0]
    stop_ssim = stop_record['train_ssim_rec'].values[0]
    ####### get peak record
    peak_psnr_df= final_psnr_df.sort_values(by=['train_psnr_rec'], ascending=False)
    peak_record = peak_psnr_df.iloc[0:1]
    peak_epoch = peak_record['epoch'].values[0]
    peak_psnr = peak_record['train_psnr_rec'].values[0]
    peak_ssim = peak_record['train_ssim_rec'].values[0]
    corr_psnr = peak_record['train_psnr_corr'].values[0]
    corr_ssim = peak_record['train_ssim_corr'].values[0]

    gap_psnr = peak_psnr - stop_psnr
    gap_ssim = peak_ssim - stop_ssim

    peak_ssim_BEST = peak_psnr_df['train_ssim_rec'].max()
    gap_ssim_BEST = peak_ssim_BEST - stop_ssim

    final_dict = {'Corr_PSNR': [corr_psnr],
                  'Corr_SSIM': [corr_ssim],
                  'Stop_Epoch': [stop_epoch],
                  'Stop_PSNR': [stop_psnr],
                  'Stop_SSIM': [stop_ssim],
                  'Peak_Epoch': [peak_epoch],
                  'Peak_PSNR': [peak_psnr],
                  'Peak_SSIM': [peak_ssim],
                  'GAP_PSNR': [gap_psnr],
                  'GAP_SSIM': [gap_ssim],
                  'GAP_SSIM2': [gap_ssim_BEST]}

    final_df = pd.DataFrame.from_dict(final_dict)
    final_file = os.path.join(dir_name_best, '{}_final_metrics.csv'.format(cur_image))
    final_df.to_csv(final_file, index=False)
    print('We must stop at this epoch! Exit!')

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# This function will test the model and the quality of the reconstructed images
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def DD_test(model,net_input, train_loader,device,figure_name):
    model.eval()
    with torch.no_grad():
        test_loader = iter(train_loader)
        test_clean_torch, test_corrupted_torch, idx = next(test_loader)
        true_image = test_clean_torch
        corrupted_image = test_corrupted_torch
        rec_image = model(net_input.to(device))
        rec_image = rec_image.detach().cpu()
        # draw the reconstructed image along with its clean and corrupted version
        all_images = torch.cat([ true_image, corrupted_image, rec_image], dim=0)
        draw_figures(all_images, figure_name=figure_name)

        ################# calculate PSNR & SSIM #################
        clean_torch, corr_torch = test_clean_torch, test_corrupted_torch
        X_clean = clean_torch[0:1, :]
        X_corrupted = corr_torch[0:1, :]

        psnr_corr = []
        psnr_rec = []
        ssim_corr = []
        ssim_rec = []

        cur_clean = X_clean[0]
        cur_corr = X_corrupted[0]
        cur_rec = rec_image[0]

        psnr_clean_img = cur_clean.cpu().numpy()
        psnr_clean_img = np.clip(psnr_clean_img, 0, 1)
        psnr_corrupted_img = cur_corr.cpu().numpy()
        psnr_corrupted_img = np.clip(psnr_corrupted_img, 0, 1)
        psnr_rec_ae_img = cur_rec.detach().cpu().numpy()
        psnr_rec_ae_img = np.clip(psnr_rec_ae_img, 0, 1)

        cur_psnr_corr = calcualte_PSNR(psnr_clean_img, psnr_corrupted_img)
        psnr_corr.append(cur_psnr_corr)

        cur_psnr_rec = calcualte_PSNR(psnr_clean_img, psnr_rec_ae_img)
        psnr_rec.append(cur_psnr_rec)

        cur_ssim_corr = calcualte_SSIM(psnr_clean_img, psnr_corrupted_img, multichannel=True)
        ssim_corr.append(cur_ssim_corr)

        cur_ssim_rec = calcualte_SSIM(psnr_clean_img, psnr_rec_ae_img, multichannel=True)
        ssim_rec.append(cur_ssim_rec)

        mean_psnr_corr = np.mean(psnr_corr)
        mean_psnr_rec = np.mean(psnr_rec)

        mean_ssim_corr = np.mean(ssim_corr)
        mean_ssim_rec = np.mean(ssim_rec)

        return mean_psnr_corr, mean_psnr_rec, mean_ssim_corr, mean_ssim_rec


if __name__ == '__main__':
    pass
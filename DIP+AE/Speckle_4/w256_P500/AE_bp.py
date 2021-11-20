from util import *
from psnr_ssim import *
from tqdm import tqdm
import pandas as pd
from AE_model import *
import glob
from track_rank import *
import seaborn as sns
import copy
from AE_util import *

def autoencoder_train(AE_learn_rate, max_epoch, Input_Data, batch_size, sample_num, gpu, print_step,out_epoch,cur_image):

    ###### Create folders to hold the results
    dir_name = '{}/AE/training_results/{}/'.format(cur_image, out_epoch)
    dir_model_name = '{}/AE/training_models/{}'.format(cur_image, out_epoch)
    dir_best_path = '{}/AE/bestNet/'.format(cur_image)
    dir_name_rank = '{}/AE/rank/'.format(cur_image)
    dir_name_idx = '{}/AE/idx/'.format(cur_image)

    make_dir([dir_name_rank,dir_name_idx])

    ################## Using GPU when it is available ##################
    if gpu=="MSI": # if we want to use MSI and multiple GPU
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    ################## Get Dataset ##################
    train_loader, train_dataset = prepare_AEData(Input_Data, batch_size, sample_num)

    ################ Create Autoencoder model ############################
    En_Net = EncoderNet()
    Encoder_file = os.path.join(dir_best_path,'best_Encoder.pt')
    ################## check if we have the best model, if so, we reload its weights
    if os.path.exists(Encoder_file):
        try:
            En_Net.load_state_dict(torch.load(Encoder_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 1!')
        except:
            En_Net = nn.DataParallel(En_Net)
            En_Net.load_state_dict(torch.load(Encoder_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 2!')
    En_Net.to(device)

    MLP_Net = DeepL()
    MLP_file = os.path.join(dir_best_path, 'best_MLP.pt')
    ################## check if we have the best model, if so, we reload its weights
    if os.path.exists(MLP_file):
        try:
            MLP_Net.load_state_dict(torch.load(MLP_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 1!')
        except:
            MLP_Net = nn.DataParallel(MLP_Net)
            MLP_Net.load_state_dict(torch.load(MLP_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 2!')
    MLP_Net.to(device)

    De_Net = DecoderNet()
    Decoder_file = os.path.join(dir_best_path, 'best_Decoder.pt')
    ################## check if we have the best model, if so, we reload its weights
    if os.path.exists(Decoder_file):
        try:
            De_Net.load_state_dict(torch.load(Decoder_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 1!')
        except:
            De_Net = nn.DataParallel(De_Net)
            De_Net.load_state_dict(torch.load(Decoder_file, map_location='cuda:{}'.format(gpu)))
            print('Loading EncoderNet models 2!')
    De_Net.to(device)


    ############### define loss function & Opt
    optimizer_En = torch.optim.Adam(En_Net.parameters(), lr= AE_learn_rate)
    optimizer_MLP = torch.optim.Adam(MLP_Net.parameters(), lr=AE_learn_rate)
    optimizer_De = torch.optim.Adam(De_Net.parameters(), lr=AE_learn_rate)

    l2_loss_func = L2_Func()


    ################## Start to Train AutoEncoder ##################
    total_loss = []
    total_epoch = []
    best_MSE = float('inf')
    selected_idx = []

    ########### Train our AE, in our experiment, we set max_epoch be 1
    ########### So the AE may not be converged but as we keep doing this,
    ########### the AE becomes better and better
    for epoch in range(max_epoch):
        print('')
        print('')
        print('###################### Start to Train AE ##########################')
        En_Net.train()
        MLP_Net.train()
        De_Net.train()
        epoch_loss = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)
        for step, (X_clean, idx) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer_En.zero_grad()
            optimizer_MLP.zero_grad()
            optimizer_De.zero_grad()

            ################## Get Training & Traget Dataset ##################
            X_clean = X_clean.to(device)
            selected_idx.extend(idx.numpy().reshape((-1, 1)))

            # forward + backward + optimize
            out_en = En_Net(X_clean)
            out_mlp = MLP_Net(out_en)
            rec_X = De_Net(out_mlp)
            loss = l2_loss_func(X_clean, rec_X)  # MSE loss
            #check_matrix = X_clean - rec_X
            #loss = get_L2(check_matrix) #L2 loss
            #loss = get_sparsity_measurement(check_matrix) # L1/L2 loss
            loss.backward()
            optimizer_En.step()
            optimizer_MLP.step()
            optimizer_De.step()

            epoch_loss.append(loss.data.cpu().numpy())
            progress.set_postfix({'loss': loss.data.cpu().numpy()})
            progress.update()

            ################## save the best AE model
            if best_MSE > loss.item():
                pass
                # best_MSE = loss.item()
                # save_model(En_Net.eval(),Encoder_file)
                # save_model(MLP_Net.eval(), MLP_file)
                # save_model(De_Net.eval(), Decoder_file)
                # best_En_Net = copy.deepcopy(En_Net)
                # best_MLP = copy.deepcopy(MLP_Net)
                # best_De_Net = copy.deepcopy(De_Net)
            else:
                print('Falling back to previous checkpoint.')
                # for new_param, net_param in zip(best_En_Net.parameters(), En_Net.parameters()):
                #     net_param.data.copy_(new_param.cuda())
                #
                # for new_param, net_param in zip(best_MLP.parameters(), MLP_Net.parameters()):
                #     net_param.data.copy_(new_param.cuda())
                #
                # for new_param, net_param in zip(best_De_Net.parameters(), De_Net.parameters()):
                #     net_param.data.copy_(new_param.cuda())

        progress.close()
        total_loss.append(np.mean(epoch_loss))
        #scheduler.step()

        # print out the information each "print_step" steps
        if epoch%print_step == 0:
            best_MSE = autoencoder_test(best_MSE, Encoder_file, MLP_file, Decoder_file, En_Net, MLP_Net, De_Net, train_loader, device, dir_name, epoch, 'train')
            # train_psnr_rec.append(train_mean_psnr_rec)
            total_epoch.append(epoch)

            ##### plot loss (to save time, we disabled the I/O. You can enable it to obtain the loss curve)
            loss_file_name = dir_name + 'loss_{}.png'.format(epoch)
            #display_loss(total_loss, print_step, loss_file= loss_file_name)

            # #### plost psnr (to save time, we disabled the I/O. You can enable it to obtain the PSNR curve)
            # loss_file_name = dir_name + 'psnr_{}.png'.format(epoch)
            # display_psnr(train_psnr_rec, test_psnr_rec, print_step, loss_file=loss_file_name)

    # to save time, we disabled the I / O.You can enable it to obtain the rank curve
    #rec_psnr_dict, train_mean_psnr_rec = autoencoder_test(best_MSE, Encoder_file, MLP_file, Decoder_file, En_Net, MLP_Net, De_Net, train_loader, device, dir_name, max_epoch+1, 'test')
    figure_name = os.path.join(dir_name_rank,'rank_{}.png'.format(out_epoch))
    plot_rank(En_Net, MLP_Net, train_loader, figure_name, device)

    ############### save its idx to see what is the idx distribution
    try:
        selected_idx = np.asarray(selected_idx).reshape((-1,1))
        selected_idx_df = pd.DataFrame(selected_idx, columns=['idx'])
        idx_file_name = os.path.join(dir_name_idx,'00a_idx_{}.csv'.format(out_epoch))
        #selected_idx_df.to_csv(idx_file_name)

        #### draw its histogram
        sns.distplot(selected_idx, hist=True, kde=False)
        plt.title('idx histogram @{}'.format(out_epoch))
        figure_name = os.path.join(dir_name_idx,'01a_idx_{}.png'.format(out_epoch))
        #plt.savefig(figure_name)
        plt.close()
    except:
        print('Err in Coverting idx to numpy!')
    return

###### This function is used to test our AE
def autoencoder_test(best_MSE, Encoder_file, MLP_file, Decoder_file, En_Net, MLP_Net, De_Net,data_loader, device, dir_name, epoch, train_or_test):
    En_Net.eval()
    MLP_Net.eval()
    De_Net.eval()

    with torch.no_grad():
        ##### reconstructed several images to visually show its performance
        pick_loader = iter(data_loader)
        clean_torch,_ = next(pick_loader)
        X_clean = clean_torch[0:1, :]
        true_images = X_clean

        net_input = X_clean.to(device)
        out_en = En_Net(net_input)
        out_mlp = MLP_Net(out_en)
        rec_images = De_Net(out_mlp)
        rec_images = rec_images.detach().cpu()
        all_images = torch.cat([true_images, rec_images], dim=0)
        figure_name = os.path.join(dir_name + 'rec_{}_{}.png'.format(train_or_test, epoch))
        #draw_figures_AE(all_images, figure_name=figure_name)
        test_loss = []
        for step, (X_clean, idx) in enumerate(data_loader):
            X_clean = X_clean.to(device)
            out_en = En_Net(X_clean)
            out_mlp = MLP_Net(out_en)
            rec_X = De_Net(out_mlp)
            check_matrix = X_clean-rec_X
            cur_loss = get_L2(check_matrix)
            test_loss.append(cur_loss.item())

        final_test_loss = np.mean(test_loss)

        # if best_MSE > final_test_loss:
        #     best_MSE = final_test_loss
        #     save_model(En_Net.eval(), Encoder_file)
        #     save_model(MLP_Net.eval(), MLP_file)
        #     save_model(De_Net.eval(), Decoder_file)
        ######################### let's always save the most recent AE model
        best_MSE = final_test_loss
        save_model(En_Net.eval(), Encoder_file)
        save_model(MLP_Net.eval(), MLP_file)
        save_model(De_Net.eval(), Decoder_file)
        return best_MSE

if __name__ == '__main__':
    pass
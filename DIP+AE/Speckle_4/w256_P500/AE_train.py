####################### The script will train our AE model
from AE_bp import *
from AE_model import *
import os
from util import *
from Early_Stop import *

#---------------------------------------------------
#---------------------------------------------------
# This function is used to train our detection AE
#---------------------------------------------------
#---------------------------------------------------
def train_stopAE(gpu, train_dataset, earlystop, cur_untrained_epoch,cur_image):

    ##### Get the training samples and the new coming sample
    train_dataset = np.asarray(train_dataset).reshape((-1, 3, 512, 512))
    new_coming_sample = train_dataset[-1,:]

    ################ Parameters Settings ######################
    AE_learn_rate = 1e-3
    max_epoch = 1#1001
    print_step = 1#100
    save_flag = False

    ############ Create folders to hold the results
    ae_dir_name = '{}/AE/'.format(cur_image)
    #ae_dir_hist = '{}/AE/hist_new'.format(cur_image)
    #ae_dir_hist_csv = '{}/AE/hist_new_csv'.format(cur_image)
    #ae_dir_rank = '{}/AE/rank'.format(cur_image)
    dir_best_path = '{}/AE/bestNet/'.format(cur_image)

    make_dir([ae_dir_name, dir_best_path])
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    ########## if we set a large window size, then we go through these samples in batch mode
    train_num = train_dataset.shape[0]
    batch_size = train_num
    batch_size = 1
    if train_num >= 1 and train_num <= 32:
        batch_size = train_num
    elif train_num > 32:
        batch_size = 32
    else:
        assert False, 'AE training number is wrong!'


    ############ we should test our AE on the new coming next data
    ################## Get MSE Error for each sample ###########
    ################ Create Autoencoder model ############################
    En_Net = EncoderNet()
    Encoder_file = os.path.join(dir_best_path, 'best_Encoder.pt')
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
    En_Net.eval()

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
    MLP_Net.eval()

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
    De_Net.eval()

    ############# use our previous AE to check the new coming sample to see whether we should ES
    with torch.no_grad():
        cur_sample_np = new_coming_sample.reshape((-1, 3, 512, 512))
        cur_sample_torch = torch.from_numpy(cur_sample_np).to(device)
        out_en = En_Net(cur_sample_torch)
        out_mlp = MLP_Net(out_en)
        cur_rec_torch = De_Net(out_mlp)
        cur_matrix = cur_sample_torch - cur_rec_torch
        cur_Err = get_L2(cur_matrix).item()
        should_stop, should_save = earlystop.check_stop(cur_Err, cur_untrained_epoch)

    ################ if ES is false, then we need to update our AE with the new coming sample
    if should_stop == False:
        #------------------- The stop condition is not met;
        #------------------- We are going to train AE with our new augmented sample
        autoencoder_train(AE_learn_rate,
                            max_epoch,
                            train_dataset,
                            batch_size,
                            train_num,
                            gpu,
                            print_step,
                            cur_untrained_epoch,
                            cur_image)

    return cur_Err, should_stop, should_save
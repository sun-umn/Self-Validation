#------------------------- The start function for everything---------------------
from train_denoising import *
import time
import datetime

if __name__ == '__main__':
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    # Set random seed for reproducibility
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    manualSeed = 100
    manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    # Set up hyper-parameters
    # --------------------------------------------------------------------
    #--------------------------------------------------------------------
    gpu = 0
    patience = 500
    corr_level = 4
    corr_type = 'Impulse_Noise'
    corr_type_level = corr_type + '_' + str(corr_level)
    window_size = 256
    start_detection = 0
    print_step = 1
    max_epoch = 100001
    max_epoch = int(max_epoch / print_step) * print_step + 1

    # hyper-parameters for DIP/DD
    learning_rate_model = 0.01
    input_depth = 32
    OPTIMIZER = 'Adam'
    # LOSS = 'MSE'
    LOSS = 'L1'
    # LOSS = 'Huber'
    Keep_track = True

    image_list = ['Baboon', 'F16', 'House',
                  'kodim01', 'kodim02', 'kodim03',
                  'kodim12', 'Lena', 'Peppers']

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Processing images one by one
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    for cur_image in image_list:
        clean_image_path = '../../../0_Dataset/{}/Clean/{}_clean.npz'.format(corr_type, cur_image)
        corrupted_image_path = '../../../0_Dataset/{}/{}/{}_{}.npz'.format(corr_type, corr_type_level, cur_image,
                                                                           corr_type_level)

        # create our ES instance
        earlystop = EarlyStop(min_delta=0, patience=patience)

        # record the start time
        start = time.time()

        # call the train function to start the DIP/DD and Detection AE
        DIP_train(learning_rate_model,
                  OPTIMIZER,
                  LOSS,
                  input_depth,
                  clean_image_path,
                  corrupted_image_path,
                  max_epoch,
                  print_step,
                  gpu,
                  earlystop,
                  cur_image,
                  start_detection,
                  Keep_track,
                  window_size)

        # record the time we need to process one image
        end = time.time()
        used_time = end - start
        print('#-------------------------------------------')
        print('used_time= {}'.format(used_time))
        f = open('time.txt', 'a')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('#---------------------- Updated info by {} ----------------------'.format(datetime.datetime.now()))
        f.write('\n')
        f.write('\n')
        f.write(cur_image)
        f.write('-------------------time used={}'.format(used_time))
        f.write('\n')
        f.write('\n')
        f.write('#---------------------- End updated info by {} ----------------------'.format(datetime.datetime.now()))
        f.close()

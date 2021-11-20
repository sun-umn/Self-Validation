### This script conducts: Early-stop
############### This defines the script to stop our training

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from util import *

# min_delta: minimum change in the monitored quantity to qualify as an improvement,
# i.e. an absolute change of less than `min_delta`, will count as no improvement.
# Default: ``0.0``.

# patience: number of validation epochs with no improvement after which
# training will be stopped.


def plot_Err(Err, title, figure_name):
    plt.plot(Err)
    plt.title(title)
    plt.ylabel('Reconstructed Err')
    plt.xlabel('Epoch Index')
    plt.savefig(figure_name)
    plt.close()


class EarlyStop():
    def __init__(self,min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.should_save = False

    def check_stop(self, current, cur_epoch):
        input = current - self.min_delta

        if input < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.should_save = True
            self.wait_count = 0
            should_stop = False
        else:
            self.should_save = False
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience

        return should_stop, self.should_save

    def get_best_info(self):
        return self.best_epoch






import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

######## This function will get the rank of the representation
######## To save time for training, we disabled this function
######## But if you want to check the rank of the representation of our AE, please enable it
def plot_rank(enc, mlp, test_loader, figure_name, device):
    # load model ##########################################
    pass
    # enc.eval()
    # mlp.eval()
    # with torch.no_grad():
    #
    #     z = []
    #     for yi, _ in test_loader:
    #         yi = yi.to(device)
    #         z_hat = mlp(enc(yi))
    #         z.append(z_hat)
    #
    #     z = torch.cat(z, dim=0).data.detach().cpu().numpy()
    #     if z.shape[0] == 1:
    #         return
    #     c = np.cov(z, rowvar=False)
    #     u, d, v = np.linalg.svd(c)
    #
    #     d = d / d[0]
    #
    #     # idx = np.where(d>threshold)[0]
    #     # rank = len(np.where(d>threshold)[0])
    #     # return rank
    #
    #     code_dim = len(d)
    #
    #     plt.plot(range(code_dim), d)
    #
    #     plt.autoscale(enable=True, axis='y', tight=True)
    #     plt.autoscale(enable=True, axis='x', tight=True)
    #     plt.ylim(0, 0.5)
    #     plt.xlim(0, code_dim)
    #     plt.xlabel("Singular Value Rank")
    #     plt.ylabel("Singular Values")
    #     plt.title("Singular Values of Covariance Matrix")
    #     #plt.savefig(figure_name)
    #     plt.close()


if __name__ == '__main__':
    pass

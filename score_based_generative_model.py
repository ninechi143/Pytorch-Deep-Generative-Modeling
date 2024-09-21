# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt


"""
PyTorch implementation of Score-based Generative Model

introduction:  https://jmtomczak.github.io/blog/17/17_sbgms.html
source code reference:  https://github.com/jmtomczak/intro_dgm/blob/main/sbgms/sbgm_example.ipynb

paper:  https://arxiv.org/pdf/1907.05600
        https://yang-song.net/blog/2021/score/

"""

class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class SBGM(nn.Module):
    def __init__(self, snet, sigma, D, T):
        super(SBGM, self).__init__()

        print("SBGM by JT.")
        
        # sigma parameter
        self.sigma = torch.Tensor([sigma])
        
        # define the base distribution (multivariate Gaussian with the diagonal covariance)
        var = (1./(2.* torch.log(self.sigma))) * (self.sigma**2 - 1.)
        self.base = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(D), var * torch.eye(D))
        
        # score model
        self.snet = snet
        
        # time embedding (a single linear layer)
        self.time_embedding = nn.Sequential(nn.Linear(1, D), nn.Tanh())
        
        # other hyperparams
        self.D = D
        
        self.T = T
                
        self.EPS = 1.e-5
        
    def sigma_fun(self, t):
        # the sigma function (dependent on t), it is the std of the distribution
        return torch.sqrt((1./(2. * torch.log(self.sigma))) * (self.sigma**(2. * t) - 1.))

    def log_p_base(self, x):
        # the log-probability of the base distribition, p_1(x)
        log_p = self.base.log_prob(x)
        return log_p
    
    def sample_base(self, x_0):
        # sampling from the base distribution
        return self.base.rsample(sample_shape=torch.Size([x_0.shape[0]]))
        
    def sample_p_t(self, x_0, x_1, t):
        # sampling from p_0t(x_t|x_0)
        # x_0 ~ data, x_1 ~ noise
        x = x_0 + self.sigma_fun(t) * x_1
        
        return x
    
    def lambda_t(self, t):
        # the loss weighting
        return self.sigma_fun(t)**2
    
    def diffusion_coeff(self, t):
        # the diffusion coefficient in the SDE
        return self.sigma**t
    
    def forward(self, x_0, reduction='mean'):
        # =====
        # x_1 ~ the base distribiution
        x_1 = torch.randn_like(x_0)
        # t ~ Uniform(0, 1)
        t = torch.rand(size=(x_0.shape[0], 1))  * (1. - self.EPS) + self.EPS 
        
        # =====
        # sample from p_0t(x|x_0)
        x_t = self.sample_p_t(x_0, x_1, t)

        # =====
        # invert noise
        # NOTE: here we use the correspondence eps_theta(x,t) = -sigma*t score_theta(x,t)
        t_embd = self.time_embedding(t)
        x_pred = -self.sigma_fun(t) * self.snet(x_t + t_embd)

        # =====LOSS: Score Matching
        # NOTE: since x_pred is the predicted noise, and x_1 is noise, this corresponds to Noise Matching 
        #       (i.e., the loss used in diffusion-based models by Ho et al.)
        SM_loss = 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)
        
        if reduction == 'sum':
            loss = SM_loss.sum()
        else:
            loss = SM_loss.mean()

        return loss

    def sample(self, batch_size=64):
        # 1) sample x_0 ~ Normal(0,1/(2log sigma) * (sigma**2 - 1))
        x_t = self.sample_base(torch.empty(batch_size, self.D))
        
        # Apply Euler's method
        # NOTE: x_0 - data, x_1 - noise
        #       Therefore, we must use BACKWARD Euler's method! This results in the minus sign! 
        ts = torch.linspace(1., self.EPS, self.T)
        delta_t = ts[0] - ts[1]
        
        for t in ts[1:]:
            tt = torch.Tensor([t])
            u = 0.5 * self.diffusion_coeff(tt) * self.snet(x_t + self.time_embedding(tt))
            x_t = x_t - delta_t * u
        
        x_t = torch.tanh(x_t)
        return x_t
    
    def log_prob_proxy(self, x_0, reduction="mean"):
        # Calculate the proxy of the log-likelihood (see (Song et al., 2021))
        # NOTE: Here, we use a single sample per time step (this is done only for simplicity and speed);
        # To get a better estimate, we should sample more noise
        ts = torch.linspace(self.EPS, 1., self.T)

        for t in ts:
            # Sample noise
            x_1 = torch.randn_like(x_0)
            # Sample from p_0t(x_t|x_0)
            x_t = self.sample_p_t(x_0, x_1, t)
            # Predict noise
            t_embd = self.time_embedding(torch.Tensor([t]))
            x_pred = -self.snet(x_t + t_embd) * self.sigma_fun(t)
            # loss (proxy)          
            if t == self.EPS:
                proxy = 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)
            else:
                proxy = proxy + 0.5 * self.lambda_t(t) * torch.pow(x_pred + x_1, 2).mean(-1)
            
        if reduction == "mean":
            return proxy.mean()
        elif reduction == "sum":
            return proxy.sum()


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.log_prob_proxy(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    # plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.savefig(name+'_real_images.png')
    plt.close()


def samples_generated(name, data_loader, extra_name='', T=None):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()
    
    if T is not None:
        model_best.T = T

    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    # plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.savefig(name + '_generated_images' + extra_name + '.png',)
    plt.close()

def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('proxy')
    # plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.savefig(name + '_nll_val_curve.png')
    plt.close()


def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + f"{e:04d}")
            else:
                patience = patience + 1
        
        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val


def make_gif(dir_path = None):

    from pathlib import Path
    import cv2
    import imageio

    if not dir_path:
        png_list = list(Path(os.path.join(os.path.dirname(__file__))).rglob("*.png"))
    else:
        png_list = list(Path(dir_path).rglob("*.png"))

    fps = 10
    png_list = [i for i in png_list if "generated_image" in str(i)]
    png_list.sort()
    process = [cv2.imread(str(i)) for i in png_list]
    process += [process[-1] for _ in range(3 * fps)]
    if not dir_path:
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "SBGM_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "SBGM_training.gif") , process , fps = 10)
    # [os.remove(i) for i in png_list]


# def parse_args():
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default = "rings", choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
#     parser.add_argument('--model', default = "FCNet", choices=('FCNet', 'ConvNet'))
#     parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. default: 1e-4')
#     parser.add_argument('--stepsize', type=float, default=0.01, help='Langevin dynamics step size. default 0.01')
#     parser.add_argument('--n_step', type=int, default=200, help='The number of Langevin dynamics steps. default 200')
#     parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epoches. default 250')
#     parser.add_argument('--alpha', type=float, default=0.05, help='Regularizer coefficient. default 0.1')

#     args = parser.parse_args()
#     return args



def main():

    # args = parse_args()

    #----------------------------------------------------------------------------
    # data setup
    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)  # changing to [-1, 1]

    train_data = Digits(mode='train', transforms=transforms)
    val_data = Digits(mode='val', transforms=transforms)
    test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    #----------------------------------------------------------------------------
    # hyperparameter setup
    prob_path = "sbgm"

    D = 64   # input dimension

    M = 512  # the number of neurons in scale (s) and translation (t) nets

    T = 20

    sigma = 1.01

    lr = 1e-4 # learning rate
    num_epochs = 2000 # max. number of epochs
    max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    #----------------------------------------------------------------------------
    name = prob_path + '_' + str(T)
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)):    os.makedirs(result_dir, exist_ok = True)

    #----------------------------------------------------------------------------
    # training
    nnet = nn.Sequential(nn.Linear(D, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, M), nn.SiLU(),
                     nn.Linear(M, D), nn.Hardtanh(min_val=-3., max_val=3.))

    # Eventually, we initialize the full model
    model = SBGM(nnet, sigma=sigma, D=D, T=T)
    
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Training procedure
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                        training_loader=training_loader, val_loader=val_loader)


    #----------------------------------------------------------------------------
    # evaluation
    test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    samples_real(result_dir + name, test_loader)
    samples_generated(result_dir + name, test_loader, extra_name='FINAL')

    plot_curve(result_dir + name, nll_val)
    make_gif(result_dir)

    return

if __name__ == "__main__":
    main()
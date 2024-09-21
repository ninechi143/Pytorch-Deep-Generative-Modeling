# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

import torch
import torchvision
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt


"""
PyTorch implementation of Score-based Generative Model

introduction:  https://jmtomczak.github.io/blog/16/16_score_matching.html
source code reference: https://github.com/jmtomczak/intro_dgm/blob/main/sbgms/sm_example.ipynb

paper:  https://arxiv.org/pdf/1907.05600
        https://yang-song.net/blog/2021/score/

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ScoreMatching(nn.Module):
    def __init__(self, snet, alpha, sigma, eta, D, T):
        super(ScoreMatching, self).__init__()

        print('Score Matching by JT.')

        self.snet = snet
        
        # other hyperparams
        self.D = D
                
        self.sigma = sigma
        
        self.T = T
        
        self.alpha = alpha
        
        self.eta = eta
    
    def sample_base(self, x_1):
        # Uniform over [-1, 1]**D
        return (2. * torch.rand_like(x_1) - 1.).to(device)
    
    def langevine_dynamics(self, x):
        for t in range(self.T):
            x = x + self.alpha * self.snet(x) + self.eta * torch.randn_like(x).to(device)
        return x

    def forward(self, x, reduction='mean'):
        # =====Score Matching
        # sample noise
        epsilon = torch.randn_like(x).to(device)
        
        # =====
        # calculate the noisy data
        tilde_x = x + self.sigma * epsilon

        # =====
        # calculate the score model
        s = self.snet(tilde_x)
        
        # =====LOSS: the Score Matching Loss
        SM_loss = (1. / (2. * self.sigma)) * ((s + epsilon)**2.).sum(-1) # in order to keep the Langevine dynamics unchanged, we do not use \tilde{s} = -sigma * s but we use \tilde{s} = sigma * s
        
        # Final LOSS
        if reduction == 'sum':
            loss = SM_loss.sum()
        else:
            loss = SM_loss.mean()

        return loss

    def sample(self,  batch_size=64):
        # sample x_0
        x = self.sample_base(torch.empty(batch_size, self.D))
        
        # run langevine dynamics
        x = self.langevine_dynamics(x)
        
        x = torch.tanh(x)
        return x

def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        # test_data, test_target = test_batch
        test_data = test_batch
        test_data = test_data.to(device)
        loss_t = model_best.forward(test_data, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch[0].shape[0]
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
    x = next(iter(test_loader)).detach().cpu().numpy()
    x = np.clip(x, -1, 1)

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

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
    x = x.detach().cpu().numpy()
    x = np.clip(x, -1, 1)

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png',)
    plt.close()

def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('score matching loss')
    plt.savefig(name + '_sm_val_curve.png')
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
            # data, target = batch
            data = batch
            data = data.to(device)
            loss = model.forward(data)

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
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "ScM_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "ScM_training.gif") , process , fps = 10)
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
    # val_data = Digits(mode='val', transforms=transforms)
    test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    

    #----------------------------------------------------------------------------
    # hyperparameter setup
    D = 64   # input dimension
    # D = (1,28,28)

    M = 512  # the number of neurons in scale (s) and translation (t) nets

    alpha = 0.1
    sigma = 0.1
    eta = 0.05

    T = 100
    
    lr = 1e-4 # learning rate
    num_epochs = 1000 # max. number of epochs
    max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    #----------------------------------------------------------------------------
    name = 'sm' + '_' + str(T)
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)):    os.makedirs(result_dir, exist_ok=True)
    #----------------------------------------------------------------------------
    # training
    snet = nn.Sequential(nn.Linear(D, M), nn.SELU(),
                     nn.Linear(M, M), nn.SELU(),
                     nn.Linear(M, M), nn.SELU(),
                     nn.Linear(M, D), nn.Hardtanh(min_val=-4., max_val=4.))


    # Eventually, we initialize the full model
    model = ScoreMatching(snet=snet, alpha=alpha, sigma=sigma, eta=eta, T=T, D=D)
    model = model.to(device)

    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Training procedure
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                        training_loader=training_loader, val_loader=test_loader)
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
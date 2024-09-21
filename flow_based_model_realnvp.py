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
PyTorch implementation of Flow-based Model, RealNVP

introduction: https://jmtomczak.github.io/blog/3/3_flows.html
source code reference: https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb


before you look at this code, 
I want to point out that there are two ways in the coulpling function, meanwhile, it also has two ways when we calculate log_det_J

theoretically, we need to add torch.abs() when we obtain the det_J
however, here, both ways are correct when the form of coulpling layer is yb = exp(s) * xb + t
we can see that, since exp() is always non-negative, we don't need to use abs anymore, and such exp() will be cancelled out by the future's log()
in the loss term
to this end, our coupling layer can directly return the power term, i.e., s, without break anything.

don't get confused with it, it works only when we use yb = exp(s) * xb + t, 
otherwise, we always need to be careful of the abs operation to calculate log_det_J
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


class RealNVP(nn.Module):
    def __init__(self, nets, nett, num_flows, prior, D=2, dequantization=True):
        super(RealNVP, self).__init__()
        
        # Well, it's always good to brag about yourself.
        print('RealNVP by JT.')
        
        # We need to dequantize discrete data. This attribute is used during training to dequantize integer data.
        self.dequantization = dequantization
        
        # An object of a prior (here: torch.distribution of multivariate normal distribution)
        self.prior = prior
        # A module list for translation networks
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        # A module list for scale networks
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        # The number of transformations, in our equations it is denoted by K.
        self.num_flows = num_flows
        
        # The dimensionality of the input. It is used for sampling.
        self.D = D

    # This is the coupling layer, the core of the RealNVP model.
    def coupling(self, x, index, forward=True):
        # x: input, either images (for the first transformation) or outputs from the previous transformation
        # index: it determines the index of the transformation
        # forward: whether it is a pass from x to y (forward=True), or from y to x (forward=False)
        
        # We chunk the input into two parts: x_a, x_b
        (xa, xb) = torch.chunk(x, 2, 1)
        
        # We calculate s(xa), but without exp!
        s = self.s[index](xa)
        # We calculate t(xa)
        t = self.t[index](xa)
        
        # Calculate either the forward pass (x -> z) or the inverse pass (z -> x)
        # Note that we use the exp here!
        if forward:
            #yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            #xb = f(y)
            yb = torch.exp(s) * xb + t
        
        # We return the output y = [ya, yb], but also s for calculating the log-Jacobian-determinant
        #return torch.cat((xa, yb), 1), s # way1: directly return s because the original exp() operation will be cancelled out by log in the future
        return torch.cat((xa, yb), 1), torch.abs(torch.exp(s.sum(dim=1))) # way2: follows the original form

    # An implementation of the permutation layer
    def permute(self, x):
        # Simply flip the order.
        return x.flip(1)

    def f(self, x):
        # This is a function that calculates the full forward pass through the coupling+permutation layers.
        # We initialize the log-Jacobian-det
        log_det_J, z = x.new_zeros(x.shape[0]), x
        # We iterate through all layers
        for i in range(self.num_flows):
            # First, do coupling layer,
            z, s = self.coupling(z, i, forward=True)
            # then permute.
            z = self.permute(z)
            # To calculate the log-Jacobian-determinant of the sequence of transformations we sum over all of them.
            # As a result, we can simply accumulate individual log-Jacobian determinants.
            ## log_det_J = log_det_J - s.sum(dim=1)     # way1: since the original exp operation will cancel with log here, thus we directly calculate the power term
            log_det_J = log_det_J - torch.log(s + 1e-8) # way2: follows the original form
        # We return both z and the log-Jacobian-determinant, because we need z to feed in to the logarightm of the Norma;
        return z, log_det_J

    def f_inv(self, z):
        # The inverse path: from z to x.
        # We appply all transformations in the reversed order.
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)
        # Since we use this function for sampling, we don't need to return anything else than x.
        return x

    def forward(self, x, reduction='avg'):
        # This function is essential for PyTorch.
        # First, we calculate the forward part: from x to z, and also we need the log-Jacobian-determinant.
        z, log_det_J = self.f(x)
        # We can use either sum or average as the output.
        # Either way, we calculate the learning objective: self.prior.log_prob(z) + log_det_J.
        # NOTE: Mind the minus sign! We need it, because, by default, we consider the minimization problem,
        # but normally we look for the maximum likelihood estimate. Therefore, we use:
        # max F(x) <=> min -F(x)
        if reduction == 'sum':
            return -(self.prior.log_prob(z) + log_det_J).sum()
        else:
            return -(self.prior.log_prob(z) + log_det_J).mean()

    def sample(self, batchSize):
        # First, we sample from the prior, z ~ p(z) = Normal(z|0,1)
        z = self.prior.sample((batchSize, self.D))
        z = z[:, 0, :]
        # Second, we go from z to x.
        x = self.f_inv(z)
        return x.view(-1, self.D)


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        if hasattr(model_best, 'dequantization'):
            if model_best.dequantization:
                test_batch = test_batch + (1. - torch.rand(test_batch.shape))/2.
        loss_t = model_best.forward(test_batch, reduction='sum')
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

    plt.savefig(name + '_real_images.png')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
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
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + (1. - torch.rand(batch.shape))/2.
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
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "RealNVP_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "RealNVP_training.gif") , process , fps = 10)
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
    train_data = Digits(mode='train')
    val_data = Digits(mode='val')
    test_data = Digits(mode='test')

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    result_dir = './results/realnvp/'
    if not(os.path.exists(result_dir)): os.makedirs(result_dir, exist_ok=True)
    name = "realnvp"
    #----------------------------------------------------------------------------
    # hyperparameter setup
    D = 64   # input dimension
    M = 256  # the number of neurons in scale (s) and translation (t) nets

    lr = 3e-4 # learning rate
    num_epochs = 1000 # max. number of epochs
    max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
    #----------------------------------------------------------------------------
    # training
    # The number of invertible transformations
    num_flows = 8

    # scale (s) network
    nets = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                nn.Linear(M, M), nn.LeakyReLU(),
                                nn.Linear(M, D // 2), nn.Tanh())

    # translation (t) network
    nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                nn.Linear(M, M), nn.LeakyReLU(),
                                nn.Linear(M, D // 2))

    # Prior (a.k.a. the base distribution): Gaussian
    prior = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
    # Init RealNVP
    model = RealNVP(nets, nett, num_flows, prior, D=D, dequantization=True)
    # # Print the summary (like in Keras)
    # print(summary(model, torch.zeros(1, 64), show_input=False, show_hierarchical=False))

    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    
    # Training procedure
    nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)

    #----------------------------------------------------------------------------
    # evaluation
    test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(os.path.join(result_dir, 'realnvp_test_loss.txt'), "w")
    f.write(str(test_loss))
    f.close()

    samples_real(result_dir + name, test_loader)

    plot_curve(result_dir + name, nll_val)

    make_gif(result_dir)

    return

if __name__ == "__main__":
    main()
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
PyTorch implementation of Denosing Diffusion Probabilistic Model

introduction: https://jmtomczak.github.io/blog/10/10_ddgms_lvm_p2.html
source code reference: https://github.com/jmtomczak/intro_dgm/blob/main/ddgms/ddgm_example.ipynb


before you look at this code, 
I want to point out that there are some wrong codes in the source code ref. above,
where I have already commented out such wrong parts with the notation:#*********************, such that 
you can see the original src and the corresponding correct src.

Noting that the derivation of all theorem in website ref. is correct,
it is just that the author writes the wrong programming codes.
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


PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7

def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
#  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743

def log_min_exp(a, b, epsilon=1e-8):
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
    log(exp(a) - exp(b))
    c + log(exp(a-c) - exp(b-c))
    a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
      F.logsigmoid((x + 0.5 - mean) / scale),
      F.logsigmoid((x - 0.5 - mean) / scale))

    return logp

def log_integer_probability_standard(x):
    logp = log_min_exp(
      F.logsigmoid(x + 0.5),
      F.logsigmoid(x - 0.5))

    return logp


class DDGM(nn.Module):
    def __init__(self, p_dnns, decoder_net, beta, T, D):
        super(DDGM, self).__init__()

        print('DDGM by JT.')

        ## a list of sequentials; a single Sequential defines a DNN to parameterize a distribution p(z_i | z_i+1)
        # self.p_dnns = p_dnns # the original code in ref writes it but wrong, since optimizer with model.parameter() cannot recognize regular python list,
                               # it will leave these p_dnns unoptimized since it isn't seen by the optimizer
        self.p_dnns = nn.ModuleList(p_dnns) # instead, you need to use torch.nn.ModuleList

        self.decoder_net = decoder_net # the last DNN for p(x|z_1)

        # other params
        self.D = D # the dimensionality of the inputs (necessary for sampling!)

        self.T = T # the number of steps

        self.beta = torch.FloatTensor([beta]) # the fixed variance of diffusion
    
    # The reparameterization trick for the Gaussian distribution
    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    # The reparameterization trick for the Gaussian forward diffusion
    def reparameterization_gaussian_diffusion(self, x, i):
        return torch.sqrt(1. - self.beta) * x + torch.sqrt(self.beta) * torch.randn_like(x)

    def forward(self, x, reduction='avg'):

        """
                0     1     2     3     4
        x  ->  z1 -> z2 -> z3 -> z4 -> z5 = N(0, 1)      diffusion process
           <~     <~    <~    <~    <~                   reverse process
          decode  mu0   mu1   mu2   mu3

        zs = [x, z1, z2, z3, z4, z5]
        mus =     [mu0,mu1,mu2,mu3]
        """


        # =====
        # Forward Difussion
        # Please note that we just "wander" around in the space using Gaussian random walk.
        # We save all z's in a list
        # zs = [self.reparameterization_gaussian_diffusion(x, 0)] #***********************************************
        zs = [x, self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        # =====
        # Backward Diffusion
        # We start with the last z and proceed to x.
        # At each step, we calculate means and variances.
        mus = []
        log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):
            # h = self.p_dnns[i](zs[i+1]) ##***********************************************
            h = self.p_dnns[i](zs[i+2])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)

            # mus.append(mu_i) ##***********************************************
            # log_vars.append(log_var_i) ##***********************************************
            mus = [mu_i] + mus
            log_vars = [log_var_i] + log_vars
        
        # The last step: outputting the means for x.
        # NOTE: We assume the last distribution is Normal(x | tanh(NN(z_1)), 1)!
        # mu_x = self.decoder_net(zs[0]) ##***********************************************
        mu_x = self.decoder_net(zs[1])

        # =====ELBO
        ## RE
        # This is equivalent to - MSE(x, mu_x) + const
        RE = log_standard_normal(x - mu_x).sum(-1)

        ## KL: We need to go through all the levels of latents
        # actually there is no trainable parameters in this term, just like the term L_T in the DDPM paper
        # KL = log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-1], torch.log(self.beta) - log_standard_normal(zs[-1])).sum(-1)  ##***********************************************
        KL = (log_normal_diag(zs[-1], torch.sqrt(1. - self.beta) * zs[-2], torch.log(self.beta)) - log_standard_normal(zs[-1])).sum(-1)

        for i in range(len(mus)):
            # KL_i = (log_normal_diag(zs[i], torch.sqrt(1. - self.beta) * zs[i], torch.log(self.beta)) - log_normal_diag(zs[i], mus[i], log_vars[i])).sum(-1) ##***********************************************
            KL_i = (log_normal_diag(zs[i+1], torch.sqrt(1. - self.beta) * zs[i], torch.log(self.beta)) - log_normal_diag(zs[i+1], mus[i], log_vars[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()

        return loss
    
    # Sampling is the reverse diffusion with sampling at each step.
    def sample(self, batch_size=64):
        z = torch.randn([batch_size, self.D])
        for i in range(len(self.p_dnns) - 1, -1, -1):
            h = self.p_dnns[i](z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            z = self.reparameterization(torch.tanh(mu_i), log_var_i)

        mu_x = self.decoder_net(z)

        return mu_x
    
    # For sanity check, we also can sample from the forward diffusion.
    # The result should resemble a white noise.
    def sample_diffusion(self, x):
        zs = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            zs.append(self.reparameterization_gaussian_diffusion(zs[-1], i))

        return zs[-1]


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
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

    plt.savefig(name+'_real_images.png')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png')
    plt.close()

    
def samples_diffusion(name, data_loader, extra_name=''):
    x = next(iter(data_loader))

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    z = model_best.sample_diffusion(x)
    z = z.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(z[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_diffusion' + extra_name + '.png')
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
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "DDPM_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "DDPM_training.gif") , process , fps = 10)
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
    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)

    train_data = Digits(mode='train', transforms=transforms)
    val_data = Digits(mode='val', transforms=transforms)
    test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


    #----------------------------------------------------------------------------
    # hyperparameter setup
    D = 64   # input dimension

    M = 256  # the number of neurons in scale (s) and translation (t) nets

    T = 10

    beta = 0.3

    lr = 3e-4 # learning rate
    num_epochs = 1000 # max. number of epochs
    max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    name = 'ddpm' + '_' + str(T) + '_' + str(beta)
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)): os.makedirs(result_dir, exist_ok=True)
    #----------------------------------------------------------------------------
    # training
    p_dnns = [nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)) for _ in range(T-1)]

    decoder_net = nn.Sequential(nn.Linear(D, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, M*2), nn.LeakyReLU(),
                                nn.Linear(M*2, D), nn.Tanh())


    # Eventually, we initialize the full model
    model = DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D)

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

    plot_curve(result_dir + name, nll_val)

    samples_generated(result_dir + name, test_loader, extra_name='FINAL')
    samples_diffusion(result_dir + name, test_loader, extra_name='DIFFUSION')

    make_gif(result_dir)

    return

if __name__ == "__main__":
    main()
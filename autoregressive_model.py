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
PyTorch implementation of AutoRegressive Model (ARM)

introduction:  https://jmtomczak.github.io/blog/2/2_ARM.html
source code reference:  https://github.com/jmtomczak/intro_dgm/blob/main/arms/arm_example.ipynb

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


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, A=False, **kwargs):
        
        super(CausalConv1d, self).__init__()

        # The general idea is the following: We take the build-in PyTorch Conv1D. 
        # Then, we must pick a proper padding, because we must ensure the convolutional is causal.
        # Eventually, we must remove some final elements of the output, because we simply don't need them! 
        
        # Since CausalConv1D is still a convolution, we must define the kernel size, dilation and whether it is
        # option A (A=True) or option B (A=False). Remember that by playing with dilation we can enlarge
        # the size of the memory.
        
        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        
        self.padding = (kernel_size - 1) * dilation + A * 1

        # module:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=0,
                                      dilation=dilation,
                                      **kwargs)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)
        if self.A:
            return conv1d_out[:, :, :-1]
        else:
            return conv1d_out


EPS = 1e-5
def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

class ARM(nn.Module):
    def __init__(self, net, D=2, num_vals=256):
        super(ARM, self).__init__()
        
        # Remember, always credit the author, even if it's you ;)
        print('ARM by JT.')
        
        # This is a definition of a network. See the next cell.
        self.net = net
        # This is how many values a pixel can take.
        self.num_vals = num_vals
        # This is the problem dimentionality (the number of pixels)
        self.D = D
    
    # This function calculates the arm output.
    def f(self, x):
        # First, we apply causal convolutions.
        h = self.net(x.unsqueeze(1))
        # In channels, we have the number of values. Therefore, we change the order of dims.
        h = h.permute(0, 2, 1)
        # We apply softmax to calculate probabilities.
        p = torch.softmax(h, 2)
        return p
        
    # The forward pass calculates the log-probability of an image.
    def forward(self, x, reduction='avg'):
        if reduction == 'avg':
            return -(self.log_prob(x).mean())
        elif reduction == 'sum':
            return -(self.log_prob(x).sum())
        else:
            raise ValueError('reduction could be either `avg` or `sum`.')
    
    # This function calculates the log-probability (log-categorical). 
    # See the full code in the separate file for details.
    def log_prob(self, x):
        mu_d = self.f(x)
        log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)
        
        return log_p
    
    # This function implements sampling procedure. 
    def sample(self, batch_size):
        # As you can notice, we first initilize a tensor with zeros.
        x_new = torch.zeros((batch_size, self.D))
        
        # Then, iteratively, we sample a value for a pixel.
        for d in range(self.D):
            p = self.f(x_new)
            x_new_d = torch.multinomial(p[:, d, :], num_samples=1) # for multinomial, you can refer to https://blog.csdn.net/weixin_43301333/article/details/128236012 
            x_new[:, d] = x_new_d[:,0]

        return x_new

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
                    batch = batch + torch.rand(batch.shape)
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
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "ARM_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "ARM_training.gif") , process , fps = 10)
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


    #----------------------------------------------------------------------------
    # hyperparameter setup
    D = 64   # input dimension
    M = 256  # the number of neurons in scale (s) and translation (t) nets

    lr = 1e-4 # learning rate
    num_epochs = 1000 # max. number of epochs
    max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    name = 'arm'
    result_dir = 'results/arm/'
    if not(os.path.exists(result_dir)):     os.makedirs(result_dir, exist_ok=True)

    #----------------------------------------------------------------------------
    # training
    likelihood_type = 'categorical'

    num_vals = 17 # pixel value belongs to [0,1,2,3,...,16]

    kernel = 7

    net = nn.Sequential(
        CausalConv1d(in_channels=1, out_channels=M, dilation=1, kernel_size=kernel, A=True, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=M, out_channels=M, dilation=1, kernel_size=kernel, A=False, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=M, out_channels=M, dilation=1, kernel_size=kernel, A=False, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=M, out_channels=num_vals, dilation=1, kernel_size=kernel, A=False, bias=True))

    model = ARM(net, D=D, num_vals=num_vals)

    # Print the summary (like in Keras)
    # from pytorch_model_summary import summary
    # print(summary(model, torch.zeros(1, 64), show_input=False, show_hierarchical=False))

    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    # optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=lr)

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

    make_gif(result_dir)

    return

if __name__ == "__main__":
    main()
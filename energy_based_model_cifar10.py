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
PyTorch implementation of Energy-Based Model (EBM)

introduction:  https://jmtomczak.github.io/blog/11/11_energy_based_models.html
source code reference:  https://github.com/jmtomczak/intro_dgm/blob/main/ebms/ebm_example.ipynb


Joint Energy-based Model paper link:
https://arxiv.org/pdf/1912.03263

corresponding official source code of JEM:
https://github.com/wgrathwohl/JEM/blob/master/train_wrn_ebm.py



if training diverges, see:  https://github.com/wgrathwohl/JEM/issues/4

replay buffer is extremely useful, otherwise, we will need to set larger value for SGLD step.

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNIST(Dataset):

    def __init__(self, mode='train', transforms=None, seed = None):
        if mode == 'train':
            
            database = torchvision.datasets.MNIST(root = "./data" , 
                                            train = True,
                                            transform=None,
                                            download=True)
            self.data = database.data.numpy().astype(np.float32)
            self.targets = database.targets.numpy() # just single scalar, not one-hot encoding

        else:

            database = torchvision.datasets.MNIST(root = "./data" , 
                                            train = False,
                                            transform=None)

            self.data = database.data.numpy().astype(np.float32)
            self.targets = database.targets.numpy() # just single scalar, not one-hot encoding

        self.data = self.data[:, np.newaxis, :, :]

        print(self.data.shape, self.targets.shape)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return (sample_x, sample_y)




class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32).reshape([-1,1,8,8])
            self.targets = digits.target[:1000]
        # elif mode == 'val':
        #     self.data = digits.data[1000:1350].astype(np.float32)
        #     self.targets = digits.target[1000:1350]
        else:
            self.data = digits.data[1350:].astype(np.float32).reshape([-1,1,8,8])
            self.targets = digits.target[1350:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return (sample_x, sample_y)



class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print("torch.isfinite(x).all(): {}, min. {:.5f}, max. {:.5f}".format(
            torch.isfinite(x).all(), x.min(), x.max()))
        return x


class EBM(nn.Module):
    def __init__(self, energy_net, alpha, sigma, ld_steps, D):
        super(EBM, self).__init__()

        print('EBM by JT.')

        # the neural net used by the EBM
        self.energy_net = energy_net

        # the loss for classification
        self.nll = nn.NLLLoss(reduction='none')  # it requires log-softmax as input!!
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        # hyperparams
        self.D = D

        self.sigma = sigma

        self.alpha = alpha

        self.ld_steps = ld_steps

        self.replay_buffer = torch.FloatTensor(10000, *self.D).uniform_(-1, 1).to(device)

    def classify(self, x):
        f_xy = self.energy_net(x)
        y_pred = torch.softmax(f_xy, 1)
        return torch.argmax(y_pred, dim=1)

    def class_loss(self, f_xy, y):
        # - calculate logits (for classification)
        # y_pred = torch.softmax(f_xy, 1)
        # return self.nll(torch.log(y_pred + 1e-10), y) # y should be scalar vector, instead of one-hot encoding

        return self.cross_entropy(f_xy, y)

    def gen_loss(self, x, f_xy):
        # - sample using Langevine dynamics
        x_sample = self.sample(x=None, batch_size=x.shape[0])
    
        # - calculate f(x_sample)[y]
        f_x_sample_y = self.energy_net(x_sample)

        term_numerator = torch.logsumexp(f_xy, 1)
        term_denominator = torch.logsumexp(f_x_sample_y, 1)

        if torch.abs((term_numerator - term_denominator).mean()) >= 1e4:
            print(f"numer: {term_numerator.detach().sum().item()}, denomi: {term_denominator.detach().sum().item()}")
            # 1/0 # training diverges, model blows up !

        regularization = torch.square(f_x_sample_y.sum(axis = 1)) + torch.square(f_xy.sum(axis = 1))


        return -(term_numerator - term_denominator).mean()

    def forward(self, x, y, reduction='avg'):
        # =====
        # forward pass through the network
        # - calculate f(x)[y]
        # x.requires_grad = True
        f_xy = self.energy_net(x)

        # =====
        # discriminative part
        # - calculate the discriminative loss: the cross-entropy
        L_clf = self.class_loss(f_xy, y)

        # =====
        # generative part
        # - calculate the generative loss: E(x) - E(x_sample)
        L_gen = self.gen_loss(x, f_xy)


        # # =====
        # # gradient_penalty part
        # gradients = torch.autograd.grad(outputs = interpolate_logits, inputs = interpolate_images,
        #                                 grad_outputs = torch.ones(interpolate_logits.size()).to(self.device),
        #                                 create_graph = True, retain_graph = True, only_inputs=True)[0]

        # gradients = gradients.view(gradients.shape[0] , -1)
        # gradient_penalty = self.Lambda * ((gradients.norm(2 , 1) - 1) ** 2).mean()

        # gradient_penalty = torch.autograd.grad(f_xy.sum(), x, create_graph = True, retain_graph = True, only_inputs=True)[0].view(x.size(0), -1)
        # gradient_penalty = gradient_penalty.norm(2, -1)
        gradient_penalty = 0


        # =====
        # Final objective
        loss = (L_clf + L_gen + 0.01 * gradient_penalty).mean()


        return loss


    def energy_gradient(self, x):
        self.energy_net.eval()

        # copy original data that doesn't require grads!
        x_i = torch.FloatTensor(x.detach().cpu().data).to(device)
        x_i.requires_grad = True  # WE MUST ADD IT, otherwise autograd won't work

        # calculate the gradient
        x_i_grad = torch.autograd.grad(torch.logsumexp(self.energy_net(x_i), 1).sum(), [x_i], retain_graph=True)[0]

        self.energy_net.train()

        return x_i_grad

    def langevine_dynamics_step(self, x_old, alpha):
        # Calculate gradient wrt x_old
        grad_energy = self.energy_gradient(x_old)

        # Sample eta ~ Normal(0, alpha)
        epsilon = torch.randn_like(grad_energy).to(device) * self.sigma

        # New sample
        x_new = x_old + alpha * grad_energy + epsilon

        return x_new

    def sample(self, batch_size=64, x=None):
        # - 1) Sample from uniform
        # random_sample = 2. * torch.rand([batch_size, *self.D]).to(device) - 1.
        random_sample =  torch.FloatTensor(batch_size, *self.D).uniform_(-1, 1).to(device)

        buffer_size = len(self.replay_buffer)
        indexes = torch.randint(0, buffer_size, (batch_size,))
        buffer_sample = self.replay_buffer[indexes]

        choose_random = (torch.rand(batch_size) < 0.05).float()[:, None, None, None].to(device)
        x_sample = choose_random * random_sample + (1 - choose_random) * buffer_sample
        # x_sample = random_sample

        # - 2) run Langevine Dynamics
        for i in range(self.ld_steps):
            x_sample = self.langevine_dynamics_step(x_sample, alpha=self.alpha)

        x_sample = x_sample.detach()
        
        # update replay buffer
        if len(self.replay_buffer) > 0:
            self.replay_buffer[indexes] = x_sample

        return x_sample


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    loss_error = 0.
    loss_gen = 0.
    N = 0.
    for indx_batch, (test_batch, test_targets) in enumerate(test_loader):
        
        test_batch = test_batch.to(device)
        test_targets = test_targets.to(device)
        
        # hybrid loss
        loss_t = model_best.forward(test_batch, test_targets, reduction='sum')
        loss = loss + loss_t.item()
        
        # classification error
        y_pred = model_best.classify(test_batch)
        e = 1.*(y_pred == test_targets)
        loss_error = loss_error + (1. - e).sum().item()
        
        # # generative nll
        # f_xy_test = model_best.energy_net(test_batch)
        # loss_gen = loss_gen + model_best.gen_loss(test_batch, f_xy_test).sum()

        # the number of examples
        N = N + test_batch.shape[0]
    loss = loss / N
    loss_error = loss_error / N
    loss_gen = loss_gen / N

    if epoch is None:
        print(f'FINAL PERFORMANCE: nll={loss}, ce={loss_error}, gen_nll={loss_gen}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}, val ce={loss_error}, val gen_nll={loss_gen}')

    return loss, loss_error, loss_gen


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x, _ = next(iter(test_loader))
    x = x.detach().numpy()
    x = np.clip(x, -1, 1)
    x = 0.5 * (x + 1)
    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.transpose(x[i], (1,2,0 ))
        # ax.imshow(plottable_image, cmap='gray')
        ax.imshow(plottable_image, )
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
    x = x.detach().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = 0.5 * (x + 1)
    # print(f"model sample: {x.shape}")

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.transpose(x[i], (1,2,0 ))
        # ax.imshow(plottable_image, cmap='gray')
        ax.imshow(plottable_image, )
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png')
    plt.close()


def plot_curve(name, nll_val, file_name='_nll_val_curve.png', color='b-'):
    plt.plot(np.arange(len(nll_val)), nll_val, color, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + file_name)
    plt.close()


def training(name, max_patience, num_epochs, model, optimizer, scheduler, training_loader, val_loader, load_path):
    nll_val = []
    gen_val = []
    error_val = []
    best_nll = 1e8
    patience = 0

    update_step = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, (batch, targets) in tqdm(enumerate(training_loader), desc = f"Epoch: {e}", leave = False):

            if update_step < 1000:
                lr = 1e-4 * update_step / float(1000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            batch = batch.to(device)
            targets = targets.to(device)

            loss = model.forward(batch, targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            update_step += 1

        # Validation
        loss_e, error_e, gen_e = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_e)  # save for plotting
        gen_val.append(gen_e)  # save for plotting
        error_val.append(error_e)  # save for plotting

        scheduler.step(loss_e)

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_e
        else:
            if loss_e < best_nll:
                print('saved!')
                torch.save(model, name + ".model")
                best_nll = loss_e
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + f"{e:04d}")
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    error_val = np.asarray(error_val)
    gen_val = np.asarray(gen_val)

    return nll_val, error_val, gen_val


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
        imageio.mimsave(os.path.join(os.path.dirname(__file__), "EBM_mnist_training.gif") , process , fps = fps)
    else:
        imageio.mimsave(os.path.join(dir_path, "EBM_mnist_training.gif") , process , fps = 10)
    # [os.remove(i) for i in png_list]


def parse_args():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default = "rings", choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
    # parser.add_argument('--model', default = "FCNet", choices=('FCNet', 'ConvNet'))
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. default: 1e-4')
    # parser.add_argument('--stepsize', type=float, default=0.01, help='Langevin dynamics step size. default 0.01')
    # parser.add_argument('--n_step', type=int, default=200, help='The number of Langevin dynamics steps. default 200')
    # parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epoches. default 250')
    # parser.add_argument('--alpha', type=float, default=0.05, help='Regularizer coefficient. default 0.1')
    parser.add_argument("--load" , type=str , default = None , help = "load pretrained ckpt")

    args = parser.parse_args()
    return args



def main():

    args = parse_args()

    #----------------------------------------------------------------------------
    # data setup
    transforms_train = tt.Compose( [
                                # tt.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                # tt.Lambda(lambda x: torch.from_numpy(x).float()),
                                # tt.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                # tt.Lambda(lambda x: x + 0.02 * torch.randn_like(x))
                                tt.Pad(4, padding_mode="reflect"),
                                tt.RandomCrop(32),
                                tt.RandomHorizontalFlip(),
                                tt.ToTensor(),
                                tt.Normalize((.5, .5, .5), (.5, .5, .5)),
                                lambda x: x + 0.02 * torch.randn_like(x)
                                ])

    transforms_val  = tt.Compose( [
                                # tt.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                # tt.Lambda(lambda x: torch.from_numpy(x).float()),
                                # tt.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                tt.Pad(4, padding_mode="reflect"),
                                tt.RandomCrop(32),
                                tt.RandomHorizontalFlip(),
                                tt.ToTensor(),
                                tt.Normalize((.5, .5, .5), (.5, .5, .5)),
                                lambda x: x + 0.02 * torch.randn_like(x)
                                ])
    
    # train_data = MNIST(mode='train', transforms=transforms_train)
    # test_data = MNIST(mode='test', transforms=transforms_val)
    # train_data = Digits(mode='train', transforms=transforms_train)   
    # test_data = Digits(mode='test', transforms=transforms_val)
    train_data = torchvision.datasets.CIFAR10(root="./data", transform=transforms_train, download=True, train=True)
    test_data = torchvision.datasets.CIFAR10(root="./data", transform=transforms_val, download=True, train=False)


    training_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    #----------------------------------------------------------------------------
    # hyperparameter setup
    D = (3,32,32)  # input dimension, mnist is 1, 28, 28
    K = 10 # output dimension
    M = 512  # the number of neurons

    sigma = 0.01 # the noise level

    alpha = 1.  # the step-size for SGLD
    ld_steps = 60  # the number of steps of SGLD

    lr = 1e-4  # learning rate
    num_epochs = 1000  # max. number of epochs
    max_patience = 50  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
    #----------------------------------------------------------------------------
    name = 'ebm_mnist' + '_' + str(alpha) + '_' + str(sigma) + '_' + str(ld_steps)
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)):    os.makedirs(result_dir, exist_ok = True)

    #----------------------------------------------------------------------------
    # # training
    energy_net = nn.Sequential(
                        nn.Conv2d(D[0], 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
                        # nn.Flatten(start_dim = 1, end_dim = -1),
                        # nn.Linear(7*7*256, 128), nn.LeakyReLU(),
                        nn.AvgPool2d(8),
                        nn.Flatten(start_dim = 1, end_dim = -1),

                        nn.Linear(512, K),
                        # PrintLayer(),
               )

    # Eventually, we initialize the full model
    model = EBM(energy_net, alpha=alpha, sigma=sigma, ld_steps=ld_steps, D=D)   
    model = model.to(device)

    # optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=lr,)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min" , factor = 0.5, patience = 30,  min_lr = 5e-7)

    # Training procedure
    nll_val, error_val, gen_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs,
                                        model=model, optimizer=optimizer,scheduler = scheduler,
                                        training_loader=training_loader, val_loader=test_loader,
                                        load_path = args.load)
    
    #----------------------------------------------------------------------------
    # evaluation
    test_loss, test_error, test_gen = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write('NLL: ' + str(test_loss) + '\nCLS: ' + str(test_error) + '\nGEN NLL: ' + str(test_gen))
    f.close()

    samples_real(result_dir + name, test_loader)
    samples_generated(result_dir + name, test_loader)

    plot_curve(result_dir + name, nll_val)
    plot_curve(result_dir + name, error_val, file_name='_cls_val_curve.png', color='r-')
    plot_curve(result_dir + name, gen_val, file_name='_gen_val_curve.png', color='g-')

    make_gif(result_dir)

    return

if __name__ == "__main__":
    main()
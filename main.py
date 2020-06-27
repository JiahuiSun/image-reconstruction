import numpy as np
from matplotlib import pyplot as plt
import math, os, time
import matplotlib.pyplot as plt
from os.path import join as pjoin

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from utils import Logger


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='./saved')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dim', type=int, default=20)
args = parser.parse_args()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = 'cuda' if args.cuda else 'cpu'

lr = int(-np.log10(args.lr))
stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
logdir = pjoin(args.save_dir, f'lr{lr}', f'dim{args.dim}')
logger = Logger(pjoin(logdir, f'{stamp}.log'))
logger.write(f'\nTraining configs: {args}')
save_path = pjoin(logdir, f"{stamp}.pkl")
writer = SummaryWriter(log_dir=logdir)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

class VAE(nn.Module):
    def __init__(self, dim=20):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, dim)
        self.fc22 = nn.Linear(400, dim)
        self.fc3 = nn.Linear(dim, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    bce_loss = 0
    kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        bce_loss += BCE.item()
        kl_loss += KLD.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    writer.add_scalar('train/loss', train_loss/len(train_loader.dataset), epoch)
    writer.add_scalar('train/BCE_loss', bce_loss/len(train_loader.dataset), epoch)
    writer.add_scalar('train/KL_loss', kl_loss/len(train_loader.dataset), epoch)
    print('====> Epoch: {} loss: {:.4f} BCE loss: {:.4f} KL loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), bce_loss / len(train_loader.dataset), 
          kl_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                        'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__  == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(args.dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        sample = torch.randn(64, args.dim)
        sample = sample.to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), save_path)

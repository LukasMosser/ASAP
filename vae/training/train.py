import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import numpy as np

from asap.model import VAE
from asap.data import load_dataset as load_dataset

import os

parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden_size', type=int, default=2, metavar='N',
                    help='how big is z')
parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                    help='how big is linear around z')
parser.add_argument('--min_inline', type=int, default=1, metavar='N',
                    help='Minimum Inline Value')
parser.add_argument('--max_inline', type=int, default=1, metavar='N',
                    help='Minimum Inline Value')
parser.add_argument('--min_xline', type=int, default=1, metavar='N',
                    help='Minimum Xline Value')
parser.add_argument('--max_xline', type=int, default=1, metavar='N',
                    help='Minimum Xline Value')
parser.add_argument('--step_inline', type=int, default=1, metavar='N',
                    help='Step Inline Value')
parser.add_argument('--step_xline', type=int, default=1, metavar='N',
                    help='Step Xline Value')
parser.add_argument('--fname_near_stack', type=str, default=None, metavar='N',
                    help='File Name Near Stack')
parser.add_argument('--fname_far_stack', type=str, default=None, metavar='N',
                    help='File Name Far Stack')
parser.add_argument('--out_dir', type=str, default=None, metavar='N',
                    help='output directory')
parser.add_argument('--window_size', type=int, default=16, metavar='N',
                    help='Size of Window')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def process_whole_dataset(model, loader):
    model.eval()
    batches = []
    zs = []
    errors = []
    criterion_mse = nn.MSELoss(size_average=False, reduce=False)
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(loader):
                if args.cuda:
                    data = data.cuda()
                data = Variable(data)
                recon_batch, mu, logvar, z = model(data)
                BCE = criterion_mse(recon_batch.view(-1, 2, args.window_size), data.view(-1, 2, args.window_size))
                errors.append(BCE.cpu().numpy())
                batches.append(recon_batch.cpu().numpy())
                zs.append(z.cpu().numpy())
        batches = np.concatenate(batches, 0)
        zs = np.concatenate(zs, 0)
        errors = np.concatenate(errors, 0)
    
    return batches, zs, errors


near_stack_fname, far_stack_fname = os.path.expandvars(args.fname_near_stack), os.path.expandvars(args.fname_far_stack)

X_train, y_train, X_test, y_test, _, _ = load_dataset(near_stack_fname, far_stack_fname, args.window_size)

train_dset = TensorDataset(X_train, y_train)
test_dset = TensorDataset(X_test, y_test)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(train_dset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

criterion_mse = nn.MSELoss(size_average=False)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = criterion_mse(recon_x.view(-1, 2, args.window_size), x.view(-1, 2, args.window_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data.requires_grad = True
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        print(data.size())
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(test_loader):
                if args.cuda:
                    data = data.cuda()
                recon_batch, mu, logvar, _ = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
                if epoch == args.epochs and i == 0:
                    n = min(data.size(0), 8)

        test_loss /= len(test_loader.dataset)
        np.save(os.path.expandvars(args.out_dir)+"/test_out_"+str(epoch)+".npy", np.array([data.cpu().numpy(), recon_batch.cpu().numpy()]))
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss/len(test_loader.dataset)

losses = []
for epoch in range(1, args.epochs + 1):
    trainl = train(epoch)
    testl = test(epoch)
    losses.append([trainl, testl])
    if epoch % 1 == 0:
        print("saving")
        torch.save(model.state_dict(), os.path.expandvars(args.out_dir)+"/model_epoch_"+str(epoch)+".pth")

np.save(os.path.expandvars(args.out_dir)+"/losses.npy", np.array(losses))


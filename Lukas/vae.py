import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.model_selection import ShuffleSplit
import segypy 

from model import VAE
from data import load_segy
import os

parser = argparse.ArgumentParser(description='InSeis AutoEncoder')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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

near_stack = np.load(os.path.expandvars(args.fname_near_stack))#load_segy(os.path.expandvars(args.fname_near_stack), args.min_inline, args.max_inline, args.step_inline, args.min_xline, args.max_xline, args.step_xline)
far_stack = np.load(os.path.expandvars(args.fname_far_stack))#load_segy(os.path.expandvars(args.fname_far_stack), args.min_inline, args.max_inline, args.step_inline, args.min_xline, args.max_xline, args.step_xline)

traces = np.stack([near_stack, far_stack], 1)
traces = np.swapaxes(traces, axis1=1, axis2=2)
traces = np.swapaxes(traces, axis1=0, axis2=3)

training_traces = traces[::2, ::2]
test_traces = traces[1::2, 1::2]

def get_windows(trace, window_length):
    windows = []
    for i in range(window_length, trace.shape[1]-window_length):
        windows.append(trace[:, i:i+window_length])
    return np.array(windows)

windowed_training = np.array([get_windows(tr, args.window_size) for tr in training_traces.reshape(-1, training_traces.shape[2], training_traces.shape[3])])
windowed_test = np.array([get_windows(tr, args.window_size) for tr in test_traces.reshape(-1, test_traces.shape[2], test_traces.shape[3])])

windowed_training = windowed_training.reshape(-1, windowed_training.shape[2], windowed_training.shape[3])
windowed_test = windowed_test.reshape(-1, windowed_test.shape[2], windowed_test.shape[3])

for i in range(2):
    windowed_training[:, i] -= np.mean(training_traces[:, :, i])
    windowed_training[:, i] /= np.std(training_traces[:, :, i])
    windowed_test[:, i] -= np.mean(training_traces[:, :, i])
    windowed_test[:, i] /= np.std(training_traces[:, :, i])


X_train = torch.from_numpy(windowed_training).float()
y_train = torch.from_numpy(np.zeros((X_train.shape[0], 1))).float()

X_test = torch.from_numpy(windowed_test).float()
y_test = torch.from_numpy(np.zeros((X_test.shape[0], 1))).float()


train_dset = TensorDataset(X_train, y_train)
test_dset = TensorDataset(X_test, y_test)
all_dset = TensorDataset(X_test, y_test)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(train_dset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

all_loader = torch.utils.data.DataLoader(all_dset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

model = VAE(args)
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
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(test_loader):
                if args.cuda:
                    data = data.cuda()
                data = Variable(data, volatile=True)
                recon_batch, mu, logvar, _ = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
                if epoch == args.epochs and i == 0:
                    n = min(data.size(0), 8)

        test_loss /= len(test_loader.dataset)
        np.save(os.path.expandvars(args.out_dir)+"/test_out_"+str(epoch)+".npy", np.array([data.cpu().numpy(), recon_batch.cpu().numpy()]))
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def all_out(epoch):
    model.eval()
    batches = []
    zs = []
    errors = []
    criterion_mse = nn.MSELoss(size_average=False, reduce=False)
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(all_loader):
                if args.cuda:
                    data = data.cuda()
                data = Variable(data, volatile=True)
                recon_batch, mu, logvar, z = model(data)
                BCE = criterion_mse(recon_batch.view(-1, 2, args.window_size), data.view(-1, 2, args.window_size))
                errors.append(BCE.cpu().numpy())
                batches.append(recon_batch.cpu().numpy())
                zs.append(z.cpu().numpy())
        batches = np.concatenate(batches, 0)
        zs = np.concatenate(zs, 0)
        errors = np.concatenate(errors, 0)
        np.save(os.path.expandvars(args.out_dir)+"/out_all_epoch_"+str(epoch)+".npy", batches)
        np.save(os.path.expandvars(args.out_dir)+"/out_all_zs_epoch_"+str(epoch)+".npy", zs)
        np.save(os.path.expandvars(args.out_dir)+"/out_all_errors_epoch_"+str(epoch)+".npy", errors)

losses = []
for epoch in range(1, args.epochs + 1):
    tl = train(epoch)
    testl = test(epoch)
    if epoch == args.epochs:
        sample = Variable(torch.randn(64, args.hidden_size))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        np.save("sample.npy", sample.detach().numpy())
    losses.append([tl, testl])
    if epoch % 2 == 0:
        torch.save(model.state_dict(), os.path.expandvars(args.out_dir)+"/model_epoch_"+str(epoch)+".pth")
        all_out(epoch)
np.save(os.path.expandvars(args.out_dir)+"/losses.npy", np.array(losses))


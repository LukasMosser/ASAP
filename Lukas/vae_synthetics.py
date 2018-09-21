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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                    help='how big is z')
parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                    help='how big is linear around z')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

near_traces = np.load("./near_traces_64.npy")
far_traces =  np.load("./far_traces_64.npy")
print(near_traces.mean(), near_traces.std(), near_traces.min(), near_traces.max())
pseudos = np.load("pseudos2.npy")

print(near_traces.shape, far_traces.shape)
#real_traces = real_traces.reshape(-1, 64)
well_i, well_x = 38, 138
well_variance_near = np.mean(np.std(near_traces[well_i-2:well_i+1, well_x-2:well_x+1], 2))
well_variance_far = np.mean(np.std(far_traces[well_i-2:well_i+1, well_x-2:well_x+1], 2))

well_trace_near = near_traces[well_i, well_x]
well_trace_far = far_traces[well_i, well_x]

well_trace_near_abs_std = np.std(np.abs(well_trace_near))
well_trace_far_abs_std = np.std(np.abs(well_trace_far))

print("1", well_trace_near_abs_std/pseudos[:, 0, :].std(), well_variance_near)
print("2", well_trace_far_abs_std/pseudos[:, 1, :].std(), well_variance_far)
pseudos[:, 0, :] = pseudos[:, 0, :]*well_trace_near_abs_std/ pseudos[:, 0, :].std()
#print("1", well_trace_near_abs_std, pseudos[:, 0, :].std())#/ pseudos[:, 0, :].std())
pseudos[:, 1, :] = pseudos[:, 1, :]*well_trace_far_abs_std/pseudos[:, 1, :].std()
#/pseudos[:, 1, :].std())
near_traces /= well_variance_near
far_traces /= well_variance_far

pseudos[:, 0, :] /= well_variance_near
pseudos[:, 1, :] /= well_variance_far

X = torch.from_numpy(np.stack([pseudos[:, 0, :], pseudos[:, 1, :]], 1)).float()*5
print(X.mean(), X.std(), X.min(), X.max())
print(near_traces.mean(), near_traces.std(), near_traces.min(), near_traces.max())
y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()

all_dset = TensorDataset(X, y)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

all_loader = torch.utils.data.DataLoader(all_dset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024, 128)

        # Latent space
        self.fc21 = nn.Linear(128, 2)
        self.fc22 = nn.Linear(128, 2)

        # Decoder
        self.fc3 = nn.Linear(2, 128)
        self.fc4 = nn.Linear(128, 1024)
        self.deconv1 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(32, 2, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        #print(h3.size())
        out = self.relu(self.fc4(h3))
        #print(out.size())
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 32)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.conv5(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE()
if args.cuda:
    model.cuda()
model.load_state_dict(torch.load("./models/run_3/model_1_epoch_49.pth"))

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = criterion_mse(recon_x.view(-1, 2, 64), x.view(-1, 2, 64))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def all_out():
    model.eval()
    batches = []
    zs = []
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(all_loader):
                if args.cuda:
                    data = data.cuda()
                data = Variable(data, volatile=True)
                recon_batch, mu, logvar, z = model(data)
                batches.append(recon_batch.cpu().numpy())
                zs.append(z.cpu().numpy())
        batches = np.concatenate(batches, 0)
        zs = np.concatenate(zs, 0)
        np.save("./vae3/pseudos2_out.npy", batches)
        np.save("./vae3/pseudos2_out_zs.npy", zs)

all_out()


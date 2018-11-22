import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 128)

        # Latent space
        self.fc21 = nn.Linear(128, args.hidden_size)
        self.fc22 = nn.Linear(128, args.hidden_size)

        # Decoder
        self.fc3 = nn.Linear(args.hidden_size, 128)
        self.fc4 = nn.Linear(128, 256)
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

        out = out.view(out.size(0), -1)

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

        out = self.relu(self.fc4(h3))

        out = out.view(out.size(0), 32, 8)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.conv5(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

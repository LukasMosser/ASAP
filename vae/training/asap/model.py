import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


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
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()#Variable()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
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

"""
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 16)

        # Latent space
        self.fc21 = nn.Linear(16, 2)
        self.fc22 = nn.Linear(16, 2)

        # Decoder
        self.fc3 = nn.Linear(2, 16)
        self.fc4 = nn.Linear(16, 256)
        self.deconv1 = nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(8, 2, kernel_size=3, stride=1, padding=1)

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
            eps = std.data.new(std.size()).normal_()#Variable()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 8, 32)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.conv5(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 16)

        # Latent space
        self.fc21 = nn.Linear(16, 2)
        self.fc22 = nn.Linear(16, 2)

        # Decoder
        self.fc3 = nn.Linear(2, 16)
        self.fc4 = nn.Linear(16, 64)
        self.deconv1 = nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(8, 2, kernel_size=3, stride=1, padding=1)

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
            eps = std.data.new(std.size()).normal_()#Variable()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 8, 8)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.conv5(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z"""
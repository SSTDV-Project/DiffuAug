# Self-attention GAN implementation by Christian Cosgrove
# Based on the paper by Zhang et al.
# https://arxiv.org/abs/1805.08318

# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import torch

from DiffuAug.srcs.generation.sagan.models.spectral_normalization import SpectralNorm
from DiffuAug.srcs.generation.sagan.models.conditional_batch_norm import ConditionalBatchNorm2d
from DiffuAug.srcs.generation.sagan.models.self_attention import (SelfAttention, SelfAttentionPost)


channels = 1
leak = 0.1
num_classes = 2
w_g=2


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(z_dim, 256, 4, stride=1))
        self.bn1 = ConditionalBatchNorm2d(256, num_classes)
        
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)))
        self.bn2 = ConditionalBatchNorm2d(128, num_classes)
        
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)))
        self.bn3 = ConditionalBatchNorm2d(64, num_classes)
        
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=(1,1)))
        self.bn4 = ConditionalBatchNorm2d(32, num_classes)
        
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=(1,1)))
        self.bn5 = ConditionalBatchNorm2d(16, num_classes)
        
        self.conv6 = SpectralNorm(nn.Conv2d(16, channels, 3, stride=1, padding=(1,1)))

    def forward(self, z, label):

        x = z.view(-1, self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x, label)
        # print(f"generator before conv1 x.size: {x.size()}")
        
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, label)
        x = nn.ReLU()(x)
        # print(f"generator before conv2 x.size: {x.size()}")
        
        x = self.conv3(x)
        x = self.bn3(x, label)
        x = nn.ReLU()(x)
        # print(f"generator before conv3 x.size: {x.size()}")
        
        x = self.conv4(x)
        x = self.bn4(x, label)
        x = nn.ReLU()(x)
        # print(f"generator before conv4 x.size: {x.size()}")
        
        x = self.conv5(x)
        x = self.bn5(x, label)
        x = nn.ReLU()(x)
        # print(f"generator before conv5 x.size: {x.size()}")
        
        x = self.conv6(x)
        x = nn.Tanh()(x)
        # print(f"generator after last conv x.size: {x.size()}")

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(2,2)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))

        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))

        self.attention_size = 32
        self.att = SelfAttention(256, self.attention_size)
        self.att_post = SelfAttentionPost(256, self.attention_size)

        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)))
        self.conv8 = SpectralNorm(nn.Conv2d(256, 128, 3, stride=1, padding=(1,1)))
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, 3, stride=2, padding=(1,1)))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 64, 3, stride=1, padding=(1,1)))
        self.conv11 = SpectralNorm(nn.Conv2d(64, 32, 1, stride=1, padding=(1,1)))

        self.embed = SpectralNorm(nn.Linear(num_classes, w_g * w_g * 128))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x, c):
        # print('x shape', x.size())
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        # print(f"after conv5: {m.size()}")
        
        m = nn.LeakyReLU(leak)(self.conv6(m))
        # print(f"after conv6: {m.size()}")

        self.attention_output = self.att(m)

        m = self.att_post(m, self.attention_output)
        # print(f"after att_post: {m.size()}")

        m = nn.LeakyReLU(leak)(self.conv7(m))
        # print(f"after conv7: {m.size()}")
        
        m = nn.LeakyReLU(leak)(self.conv8(m))
        # print(f"after conv8: {m.size()}")
        
        m = nn.LeakyReLU(leak)(self.conv9(m))
        # print(f"after conv9: {m.size()}")

        m = nn.LeakyReLU(leak)(self.conv10(m))
        # print(f"after conv10: {m.size()}")
        
        m = nn.LeakyReLU(leak)(self.conv11(m))
        # print(f"after conv11: {m.size()}")
        
        # print(f"c.size: {c.size()}")
        # print(f"self.embed(c).size: {self.embed(c).size()}")
        
        m = m.contiguous().view(-1, w_g * w_g * 512)
        # print(f"after view: {m.size()}")

        return self.fc(m) + torch.bmm(m.view(-1, 1, w_g * w_g * 512), self.embed(c).view(-1, w_g * w_g * 512, 1))


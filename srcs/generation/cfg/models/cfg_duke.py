import os
import math
from pathlib import Path
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from DiffuAug.srcs.datasets.duke_for_generation import DukeDataset
from DiffuAug.srcs import utility

# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    @abstractmethod
    def foward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    
    def forward(self, x, t_emb, c_emb, mask):
        for layer in self:
            if(isinstance(layer, TimestepBlock)):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
        return x

def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Residual block
class Residual_block(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, class_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.class_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_channels, out_channels)  
        )
        
        
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, t, c, mask):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `c` has shape `[batch_size, class_dim]`
        `mask` has shape `[batch_size, ]`
        """
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_c = self.class_emb(c)*mask[:, None]
        # time과 class embedding을 더해서 사용
        h += (emb_t[:,:, None, None] + emb_c[:,:, None, None])
        h = self.conv2(h)
        
        return h + self.shortcut(x)
    

# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x
    

# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class UnetModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_channels=128,
                 out_channels=3,
                 num_res_blocks=2,
                 attention_resolutions=(8,16),
                 dropout=0,
                 channel_mult=(1,2,2,2),
                 conv_resample=True,
                 num_heads=4,
                 class_num=10
                ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.class_num = class_num
        
        #time embedding
        time_emb_dim = model_channels*4
        self.time_emb = nn.Sequential(
                nn.Linear(model_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        #class embedding
        class_emb_dim = model_channels
        self.class_emb = nn.Embedding(class_num, class_emb_dim)
        
        #down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [Residual_block(ch, model_channels*mult, time_emb_dim, class_emb_dim, dropout)]
                ch = model_channels*mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_channels.append(ch)
            if level != len(channel_mult)-1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_channels.append(ch)
                ds*=2
                
        #middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout), 
            AttentionBlock(ch, num_heads),
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout)
        )
        
        #up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult[::-1]):
            for i in range(num_res_blocks+1):
                layers = [
                    Residual_block(ch+down_block_channels.pop(), model_channels*mult,\
                                   time_emb_dim, class_emb_dim, dropout)]
                ch = model_channels*mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level!=len(channel_mult)-1 and i==num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
                
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps, c, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c: a 1-D batch of classes.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step and class embedding
        t_emb = self.time_emb(timestep_embedding(timesteps, dim=self.model_channels))
        c_emb = self.class_emb(c)
        
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, t_emb, c_emb, mask)
#             print(h.shape)
            hs.append(h)
        
        # middle stage
        h = self.middle_blocks(h, t_emb, c_emb, mask)
        
        # up stage
        for module in self.up_blocks:
#             print(h.shape, hs[-1].shape)
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t_emb, c_emb, mask)
        
        return self.out(h)


# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def sigmoid_beta_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas)/(betas.max()-betas.min())*(0.02-betas.min())/10
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        #self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, c, w, clip_denoised=True):
        device = next(model.parameters()).device
        batch_size = x_t.shape[0]
        # predict noise using model
        pred_noise_c = model(x_t, t, c, torch.ones(batch_size).int().to(device))
        pred_noise_none = model(x_t, t, c, torch.zeros(batch_size).int().to(device))
        pred_noise = (1+w)*pred_noise_c - w*pred_noise_none
        
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, c, w, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                c, w, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: ddpm reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape, n_class=10, w=2, mode='random', select_class_num=None, clip_denoised=True):
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # generate labels
        if mode == 'random':
            cur_y = torch.randint(0, n_class, (batch_size,)).to(device)
        elif mode == 'all':
            if batch_size%n_class!=0:
                batch_size = n_class
                print('change batch_size to', n_class)
            cur_y = torch.tensor([x for x in range(n_class)]*(batch_size//n_class), dtype=torch.long).to(device)
        elif mode == 'select':
            if select_class_num is None:
                raise ValueError('select_class_num should be given')
            cur_y = torch.full((batch_size,), select_class_num).to(device)
        else:
            cur_y = torch.ones(batch_size).long().to(device)*int(mode)
        
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), cur_y, w, clip_denoised)
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3, n_class=10, w=2, mode='random', select_class_num=None, clip_denoised=True):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), n_class, w, mode, select_class_num, clip_denoised)
    
    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        batch_size=8,
        channels=3,
        ddim_timesteps=50,
        n_class = 10,
        w = 2,
        mode= 'random',
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        select_class_num=None,
        clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            # c만큼의 간격을 건너 뛰며 ddim_timestep_seq를 np.array로 선언
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            # 1. 0을 시작점, self.timesteps * .8에 제곱근을 취한 값을 끝점으로 두고 ddim_timesteps개수 만큼의 같은 간격을 가지는 배열 생성
            # 2. 각 값에 제곱을 취한 후 int로 변환
            # self.timesteps = 500, ddim_timesteps = 50 기준 예시
            # -> [0.0, 0.4082, 0.8163, 1.2245, ... 18.7755, 19.1837, 19.5918, 20.0]
            # -> [0.0, 0.1667, 0.6667, 1.5, 2.6667, ... 352.6667, 368.1667, 384.0, 400.0]
            # -> [0, 0, 0, 1, ..., 352, 368, 384, 400]
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
        # 최종 알파값을 올바르게 얻기 위해 1을 추가 (샘플링 중에 첫 번째 스케일부터 데이터까지)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # 이전 시퀀스
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device

        # label 생성
        if mode == 'random':
            cur_y = torch.randint(0, n_class, (batch_size,)).to(device)
        elif mode == 'all':
            # batch size가 label 개수에 맞아 떨어지지 않을 경우 batch size를 label 개수로 변경
            if batch_size%n_class!=0:                
                batch_size = n_class
                print('change batch_size to', n_class)
            # label을 0부터 n_class-1까지 생성하는 것을 batch_size // n_class만큼 반복. batch size=40, n_class=10이라면 총 40개의 label을 생성
            cur_y = torch.tensor([x for x in range(n_class)] * (batch_size//n_class), dtype=torch.long).to(device)
        elif mode == 'select':
            if select_class_num is None:
                raise ValueError('select_class_num should be given')
            cur_y = torch.full((batch_size,), select_class_num).to(device)
        else:
            cur_y = torch.ones(batch_size).long().to(device)*int(mode)
        
        # 순수 노이즈로부터 샘플링 시작 (각 배치의 example마다)
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        # seq_img = [sample_img.cpu().numpy()] # 첫 번째 시작은 완전 랜덤 노이즈이므로 기본은 이렇게 선언되어있다.
        seq_img = [] # seq_img를 cur_y를 tuple로 감싸 저장하기 위해 빈 리스트로 선언

        # ddim_timesteps 단위로 건너뛰며 샘플링
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. 현재 및 이전 alpha_cumprod 가져오기
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. 모델을 사용해서 noise 예측하기
            pred_noise_c = model(sample_img, t, cur_y, torch.ones(batch_size).int().cuda())
            pred_noise_none = model(sample_img, t, cur_y, torch.zeros(batch_size).int().cuda())
            pred_noise = (1+w)*pred_noise_c - w*pred_noise_none # Classsifier-free-guidance
            
            # 3. x_0를 예측하는 DDIM 식이나, pred_noise에 classifier-free-guidance를 적용한 식을 사용하는 것이 인상적이다.
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            if mode == 'all' or mode == 'select':
                seq_img.append((sample_img.cpu().numpy(), cur_y.cpu().numpy()))
            
        if mode == 'all' or mode == 'select':
            return seq_img
        else:
            return sample_img.cpu().numpy()
    
    # compute train losses
    def train_losses(self, model, x_start, t, c, mask_c):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, c, mask_c)
        loss = F.mse_loss(noise, predicted_noise)
        return loss


def get_duke_dataloader(png_dir, train_batchsize=32, img_size=256, num_workers=8):
    img_transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
        ]
    )
    
    dataset = DukeDataset(
        data_dir=png_dir, 
        transform=img_transform,
        target_label="all",
        each_total=2600
        )
    print(len(dataset))
    
    train_lodaer = DataLoader(dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_workers)
    
    return train_lodaer


def train(cfg):
    # 기본 설정
    batch_size = cfg.params.batch_size
    timesteps = cfg.params.timesteps
    data_dir = cfg.paths.data_dir
    
    exp_path = cfg.paths.exp_path
    log_path = os.path.join(exp_path, cfg.paths.log_dir)
    model_save_root_path = os.path.join(exp_path, cfg.paths.model_dir)

    Path(exp_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(model_save_root_path).mkdir(parents=True, exist_ok=True)

    train_loader = get_duke_dataloader(
        png_dir=data_dir,
        train_batchsize=batch_size,
        img_size=cfg.params.img_size,
        num_workers=cfg.params.gpu_num * cfg.params.base_num_workers
    )

    # 모델 선언 및 Diffusion 연산 유틸 선언
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetModel(
        in_channels=cfg.params.channels,
        model_channels=cfg.params.init_channels,
        out_channels=cfg.params.channels,
        channel_mult=cfg.params.dim_mults,
        attention_resolutions=[],
        class_num=cfg.cfg_params.class_num
    )
    model.to(device)

    # train
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule='linear')
    epochs = cfg.params.epochs
    p_uncound = cfg.cfg_params.p_uncond
    len_data = len(train_loader)
    time_end = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for step, (images, labels) in enumerate(train_loader):     
            time_start = time_end
            
            optimizer.zero_grad()
            
            batch_size = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound>p_uncound).int().to(device)
            
            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)
            
            if step % 100 == 0:
                time_end = time.time()
                print("Epoch{}/{}\t  Step{}/{}\t Loss {:.4f}\t Time {:.2f}".format(epoch+1, epochs, step+1, len_data, loss.item(), time_end-time_start))
                
            loss.backward()
            optimizer.step()

        # 에포크 별 모델 학습 내용 저장
        states = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'step': 0,
        }
        
        model_save_path = os.path.join(model_save_root_path, f"{epoch}_fianl_ckpt.pth")
        utility.save_model(
            net=model,
            states=states, 
            save_path=model_save_path
            )
        
        # 특정 에포크 별 샘플링
        if epoch % cfg.params.sample_epoch_freq == 0:
            generate_img(
                cfg=cfg,
                model=model,
                cur_epoch=epoch
            )

def generate_img(cfg, model=None, model_path=None, cur_epoch=None):
    """ 이미지 생성 함수 호출부 """    
    if model is None and model_path is None:
        raise ValueError("model or model_path should be given")
    if model is not None and model_path is not None:
        raise ValueError("model and model_path should not be given at the same time")
    
    if model_path is not None:
        model = UnetModel(
            in_channels=cfg.params.channels,
            model_channels=cfg.params.init_channels,
            out_channels=cfg.params.channels,
            channel_mult=cfg.params.dim_mults,
            attention_resolutions=[],
            class_num=cfg.cfg_params.class_num
        )
    
        # 모델 로드    
        states = torch.load(model_path, map_location='cuda')
        model = model.to("cuda")
        model.load_state_dict(states['model_state'])
    
    # model_path가 없고, model이 있는 경우는 바로 get_sampling_img 함수 호출
    if cur_epoch is not None:
        get_sampling_img(cfg, model, cur_epoch)
    else:
        get_sampling_img(cfg, model)


def get_sampling_img(cfg, model, cur_epoch=None):
    # 샘플링 폴더 생성
    sampling_path = cfg.paths.sampling_path
    Path(sampling_path).mkdir(parents=True, exist_ok=True)
    
    # Diffusion 연산 유틸 선언
    gaussian_diffusion = GaussianDiffusion(timesteps=cfg.params.timesteps, beta_schedule='linear')
    
    # 설정 값 로드
    img_size = cfg.params.img_size
    n_class = cfg.cfg_params.class_num

    # DDPM 이미지 생성
    generated_images = gaussian_diffusion.sample(
        model=model,
        image_size=img_size,
        batch_size=4, 
        channels=1, 
        n_class=n_class, 
        w=2, 
        mode='random',
        clip_denoised=False
        )
    # DDIM 이미지 생성
    ddim_generated_images = gaussian_diffusion.ddim_sample(
        model=model,
        image_size=img_size,
        batch_size=4,
        channels=1,
        ddim_timesteps=50,
        n_class=n_class,
        w=2,
        mode='random',
        ddim_discr_method='quad',
        ddim_eta=0.0,
        clip_denoised=False)
    # Class별 DDIM 이미지 생성 
    gif_generated_images = gaussian_diffusion.ddim_sample(
        model=model, 
        image_size=img_size, 
        batch_size=4,
        channels=1,
        ddim_timesteps=100,
        n_class=n_class,
        w=2,
        mode='all',
        ddim_discr_method='quad',
        ddim_eta=0.0,
        clip_denoised=False
        )

    # DDPM 이미지 시각화
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    imgs = generated_images[-1].reshape(2, 2, img_size, img_size)
    for n_row in range(2):
        for n_col in range(2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    if cur_epoch is not None:
        plt.savefig(os.path.join(sampling_path, f"ddpm_generated_images_{cur_epoch}.png"))
    else:
        plt.savefig(os.path.join(sampling_path, "ddpm_generated_images.png"))
    
    # DDIM 이미지 시각화
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    imgs = ddim_generated_images.reshape(2, 2, img_size, img_size)
    for n_row in range(2):
        for n_col in range(2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    if cur_epoch is not None:
        plt.savefig(os.path.join(sampling_path, f"ddim_generated_images_{cur_epoch}.png"))
    else:
        plt.savefig(os.path.join(sampling_path, "ddim_generated_images.png"))

    # DDIM으로 생성된 클래스별 이미지 시각화 -> 0 1
    IMG_IDX = 0
    LABEL_IDX = 1
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(n_class, 2)
    
    imgs = gif_generated_images[-1][IMG_IDX].reshape(n_class, 2, img_size, img_size)
    labels = gif_generated_images[-1][LABEL_IDX].reshape(n_class, 2)
    print(imgs.shape)
    print(labels.shape)
    
    for n_row in range(n_class):
        for n_col in range(2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
            f_ax.text(0.5, -0.1, labels[n_row, n_col], color='red', fontsize=20, ha='left', va='top', transform=f_ax.transAxes)
            f_ax.axis("off")
    if cur_epoch is not None:
        plt.savefig(os.path.join(sampling_path, f"ddim_class_images_{cur_epoch}.png"))
    else:
        plt.savefig(os.path.join(sampling_path, "ddim_class_images.png"))
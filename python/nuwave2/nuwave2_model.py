#Some codes are adopted from 
#https://github.com/ivanvovk/WaveGrad
#https://github.com/lmnt-com/diffwave
#https://github.com/NVlabs/SPADE
#https://github.com/pkumivision/FFC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from math import sqrt, log

Linear = nn.Linear
silu = F.silu
relu = F.relu

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_channels = hparams.dpm.pos_emb_channels
        self.linear_scale = hparams.dpm.pos_emb_scale
        self.out_channels = hparams.arch.pos_emb_dim

        self.projection1 = Linear(self.n_channels, self.out_channels)
        self.projection2 = Linear(self.out_channels, self.out_channels)

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32).to(noise_level) * -emb)
        emb = self.linear_scale * noise_level.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.projection1(emb)
        emb = silu(emb)
        emb = self.projection2(emb)
        emb = silu(emb)
        return emb


class BSFT(nn.Module):
    def __init__(self, nhidden, out_channels):
        super().__init__()
        self.mlp_shared = nn.Conv1d(2, nhidden, kernel_size=3, padding=1)

        self.mlp_gamma = Conv1d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = Conv1d(nhidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x, band):
        # band: (B, 2, n_fft // 2 + 1)
        actv = silu(self.mlp_shared(band))

        gamma = self.mlp_gamma(actv).unsqueeze(-1)
        beta = self.mlp_beta(actv).unsqueeze(-1)

        # apply scale and bias
        out = x * (1 + gamma) + beta

        return out


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, filter_length=1024, hop_length=256, win_length=1024,
                 sampling_rate=48000):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        hann_window = torch.hann_window(win_length)
        self.register_buffer('hann_window', hann_window)

        self.conv_layer = Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                 kernel_size=1, padding=0, bias=False)
        self.bsft = BSFT(bsft_channels, out_channels * 2)

    def forward(self, x, band):
        batch = x.shape[0]

        x = x.view(-1, x.size()[-1])

        ffted = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=True, normalized=True, onesided=True, return_complex=False)
        ffted = ffted.permute(0, 3, 1, 2).contiguous()  # (BC, 2, n_fft/2+1, T)
        ffted = ffted.view((batch, -1,) + ffted.size()[2:])  # (B, 2C, n_fft/2+1, T)

        ffted = relu(self.bsft(ffted, band))  # (B, 2C, n_fft/2+1, T)
        ffted = self.conv_layer(ffted)

        ffted = ffted.view((-1, 2,) + ffted.size()[2:]).permute(0, 2, 3, 1).contiguous()  # (BC, n_fft/2+1, T, 2)

        output = torch.istft(ffted, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=True, normalized=True, onesided=True)
        output = output.view(batch, -1, x.size()[-1])
        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, **audio_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.conv1 = Conv1d(
            in_channels, out_channels // 2, kernel_size=1, bias=False)

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, bsft_channels, **audio_kwargs)

        self.conv2 = Conv1d(
            out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x, band):
        x = silu(self.conv1(x))
        output = self.fu(x, band)
        output = self.conv2(x + output)

        return output


class FFC(nn.Module): # STFC
    def __init__(self, in_channels, out_channels, bsft_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, padding=1,
                 **audio_kwargs):
        super(FFC, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        self.convl2l = Conv1d(in_cl, out_cl, kernel_size, padding=padding, bias=False)
        self.convl2g = Conv1d(in_cl, out_cg, kernel_size, padding=padding, bias=False)
        self.convg2l = Conv1d(in_cg, out_cl, kernel_size, padding=padding, bias=False)
        self.convg2g = SpectralTransform(in_cg, out_cg, bsft_channels, **audio_kwargs)

    def forward(self, x_l, x_g, band):
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g, band)

        return out_xl, out_xg


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, pos_emb_dim, bsft_channels, **audio_kwargs):
        super().__init__()
        self.ffc1 = FFC(residual_channels, 2*residual_channels, bsft_channels,
                               kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1, **audio_kwargs) # STFC

        self.diffusion_projection = Linear(pos_emb_dim, residual_channels)
        self.output_projection = Conv1d(residual_channels,
                                        2 * residual_channels, 1)

    def forward(self, x, band, noise_level):
        noise_level = self.diffusion_projection(noise_level).unsqueeze(-1)

        y = x + noise_level
        y_l, y_g = torch.split(y, [y.shape[1] - self.ffc1.global_in_num, self.ffc1.global_in_num], dim=1)
        y_l, y_g = self.ffc1(y_l, y_g, band) # STFC
        gate_l, filter_l = torch.chunk(y_l, 2, dim=1)
        gate_g, filter_g = torch.chunk(y_g, 2, dim=1)
        gate, filter = torch.cat((gate_l, gate_g), dim=1), torch.cat((filter_l, filter_g), dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class NuWave2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_projection = Conv1d(2, hparams.arch.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            hparams)
        audio_kwargs = dict(filter_length = hparams.audio.filter_length, hop_length = hparams.audio.hop_length,
                          win_length = hparams.audio.win_length, sampling_rate = hparams.audio.sampling_rate)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hparams.arch.residual_channels,
                          hparams.arch.pos_emb_dim,
                          hparams.arch.bsft_channels,
                          **audio_kwargs)
            for i in range(hparams.arch.residual_layers)
        ])
        self.len_res = len(self.residual_layers)
        self.skip_projection = Conv1d(hparams.arch.residual_channels,
                                      hparams.arch.residual_channels, 1)
        self.output_projection = Conv1d(hparams.arch.residual_channels, 1, 1)

    def forward(self, audio, audio_low, band, noise_level):
        x = torch.stack((audio, audio_low), dim=1)
        x = self.input_projection(x)
        x = silu(x)
        noise_level = self.diffusion_embedding(noise_level)
        band = F.one_hot(band).transpose(1, -1).float()

        #This way is more faster!
        #skip = []
        skip =0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, band, noise_level)
            #skip.append(skip_connection)
            skip += skip_connection

        #x = torch.sum(torch.stack(skip), dim=0) / sqrt(self.len_res)
        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x).squeeze(1)
        return x

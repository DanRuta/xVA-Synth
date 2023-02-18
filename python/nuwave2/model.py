import os
from math import atan, exp

import librosa
import numpy as np
import scipy
from scipy.signal import resample_poly
from omegaconf import OmegaConf as OC

import torch
import torch.nn as nn
import pytorch_lightning as pl

try:
    import sys
    sys.path.append(".")
    from resources.app.python.nuwave2.nuwave2_model import NuWave2 as model
except:
    try:
        from python.nuwave2.nuwave2_model import NuWave2 as model
    except:
        try:
            from nuwave2.nuwave2_model import NuWave2 as model
        except:
            from nuwave2_model import NuWave2 as model
class Diffusion(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = model(hparams)

        self.logsnr_min = hparams.logsnr.logsnr_min
        self.logsnr_max = hparams.logsnr.logsnr_max

        self.logsnr_b = atan(exp(-self.logsnr_max / 2))
        self.logsnr_a = atan(exp(-self.logsnr_min / 2)) - self.logsnr_b

    def snr(self, time):
        logsnr = - 2 * torch.log(torch.tan(self.logsnr_a * time + self.logsnr_b))
        norm_nlogsnr = (self.logsnr_max - logsnr) / (self.logsnr_max - self.logsnr_min)

        alpha_sq, sigma_sq = torch.sigmoid(logsnr), torch.sigmoid(-logsnr)
        return logsnr, norm_nlogsnr, alpha_sq, sigma_sq

    def forward(self, y, y_l, band, t, z=None):
        logsnr, norm_nlogsnr, alpha_sq, sigma_sq = self.snr(t)

        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z
        return noise, logsnr, (alpha_sq, sigma_sq)

    def denoise(self, y, y_l, band, t, h):
        noise, logsnr_t, (alpha_sq_t, sigma_sq_t) = self(y, y_l, band, t)

        f_t = - self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)
        g_t_sq = 2 * self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)

        dzt_det = (f_t * y - 0.5 * g_t_sq * (-noise / torch.sqrt(sigma_sq_t))) * h

        denoised = y - dzt_det
        return denoised

    def denoise_ddim(self, y, y_l, band, logsnr_t, logsnr_s, z=None):
        norm_nlogsnr = (self.logsnr_max - logsnr_t) / (self.logsnr_max - self.logsnr_min)

        alpha_sq_t, sigma_sq_t = torch.sigmoid(logsnr_t), torch.sigmoid(-logsnr_t)

        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z

        alpha_sq_s, sigma_sq_s = torch.sigmoid(logsnr_s), torch.sigmoid(-logsnr_s)

        pred = (y - torch.sqrt(sigma_sq_t) * noise) / torch.sqrt(alpha_sq_t)

        denoised = torch.sqrt(alpha_sq_s) * pred + torch.sqrt(sigma_sq_s) * noise
        return denoised, pred

    def diffusion(self, signal, noise, s, t=None):
        bsize = s.shape[0]

        time = s if t is None else torch.cat([s, t], dim=0)

        _, _, alpha_sq, sigma_sq = self.snr(time)
        if t is not None:
            alpha_sq_s, alpha_sq_t = alpha_sq[:bsize], alpha_sq[bsize:]
            sigma_sq_s, sigma_sq_t = sigma_sq[:bsize], sigma_sq[bsize:]

            alpha_sq_tbars = alpha_sq_t / alpha_sq_s
            sigma_sq_tbars = sigma_sq_t - alpha_sq_tbars * sigma_sq_s

            alpha_sq, sigma_sq = alpha_sq_tbars, sigma_sq_tbars

        alpha = torch.sqrt(alpha_sq)
        sigma = torch.sqrt(sigma_sq)

        noised = alpha.unsqueeze(-1) * signal + sigma.unsqueeze(-1) * noise
        return alpha, sigma, noised
class NuWave2(pl.LightningModule):
    def __init__(self, hparams, train=True):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = Diffusion(hparams)

        self.loss = nn.L1Loss()

    def forward(self, wav, wav_l, band, t):
        z = torch.randn(wav.shape, dtype=wav.dtype, device=wav.device)
        _, _, diffusion = self.model.diffusion(wav, z, t)

        estim, logsnr, _ = self.model(diffusion, wav_l, band, t)
        return estim, z, logsnr, wav, diffusion, logsnr

    def common_step(self, wav, wav_l, band, t):
        noise_estimation, z, logsnr, wav, wav_noisy, logsnr = self(wav, wav_l, band, t)

        loss = self.loss(noise_estimation, z)
        return loss, wav, wav_noisy, z, noise_estimation, logsnr

    def inference(self, wav_l, band, step, noise_schedule=None):
        with torch.no_grad():
            signal = torch.randn(wav_l.shape, dtype=wav_l.dtype, device=wav_l.device)
            signal_list = []
            if noise_schedule == None:
                h = (self.hparams.logsnr.logsnr_max - self.hparams.logsnr.logsnr_min) / step
            for i in range(step):
                if noise_schedule == None:
                    logsnr_t = (self.hparams.logsnr.logsnr_min + i * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                                                     device=signal.device)
                    logsnr_s = (self.hparams.logsnr.logsnr_min + (i+1) * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                                                     device=signal.device)
                    signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
                else:
                    logsnr_t = noise_schedule[i] * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                    if i == step-1:
                        logsnr_s = self.hparams.logsnr.logsnr_max * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                    else:
                        logsnr_s = noise_schedule[i+1] * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                    signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
                signal_list.append(signal)
            wav_recon = torch.clamp(signal, min=-1, max=1-torch.finfo(torch.float16).eps)
        return wav_recon, signal_list

    def training_step(self, batch, batch_idx):
        wav, wav_l, band = batch
        t = ((1 - torch.rand(1, dtype=wav.dtype, device=wav.device))
             + torch.arange(wav.shape[0], dtype=wav.dtype, device=wav.device)/wav.shape[0])%1
        loss, *_ = \
            self.common_step(wav, wav_l, band, t)

        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wav, wav_l, band = batch

        t = ((1 - torch.rand(1, dtype=wav.dtype, device=wav.device))
            + torch.arange(wav.shape[0], dtype=wav.dtype, device=wav.device) / wav.shape[0]) % 1
        loss, wav, wav_noisy, z, z_recon, logsnr = self.common_step(wav, wav_l, band, t)

        self.log('val/loss', loss, sync_dist=True)
        if batch_idx == 0:
            i = torch.randint(0, wav.shape[0], (1,)).item()
            logsnr_t, *_ = self.model.snr(t)
            _, wav_recon = self.model.denoise_ddim(wav_noisy[i].unsqueeze(0), wav_l[i].unsqueeze(0),
                                                   band[i].unsqueeze(0), logsnr_t[i].unsqueeze(0),
                                                   torch.tensor(self.hparams.logsnr.logsnr_min, device=logsnr_t.device).unsqueeze(0),
                                                   z_recon[i].unsqueeze(0))
            signal = torch.randn(wav.shape[-1], dtype=wav.dtype, device=wav.device).unsqueeze(0)
            h = 1/1000
            wav_l_i, band_i = wav_l[i].unsqueeze(0), band[i].unsqueeze(0)
            for step in range(1000):
                timestep = (1.0 - (step + 0.5) * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                       device=signal.device)
                signal = self.model.denoise(signal, wav_l_i, band_i, timestep, h)
                signal = signal.clamp(-10.0, 10.0)
            wav_recon_allstep = signal.clamp(-1.0, 1.0)
            z_error = z - z_recon
            self.trainer.logger.log_spectrogram(wav[i], wav_noisy[i], z_error[i],
                                                wav_recon_allstep[0], wav_recon[0], wav_l[i],
                                                t[i].item(), logsnr[i].item(),
                                                self.global_step)
            self.trainer.logger.log_audio(wav[i], wav_noisy[i], wav_recon[0], wav_recon_allstep[0], wav_l[i], self.current_epoch)

        return {
            'val_loss': loss,
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.train.lr,
                               eps=self.hparams.train.opt_eps,
                               betas=(self.hparams.train.beta1,
                                      self.hparams.train.beta2),
                               weight_decay=self.hparams.train.weight_decay)
        return opt

    def train_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_vctk_dataloader(self.hparams, 1)

    def test_dataloader(self, sr):
        return dataloader.create_vctk_dataloader(self.hparams, 2, sr)

# from utils.stft import STFTMag
class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    #x: [B,T] or [T]
    def forward(self, x):
        with torch.no_grad():
            T = x.shape[-1]
            stft = torch.stft(x,
                              self.nfft,
                              self.hop,
                              window=self.window,
                              )#return_complex=False)  #[B, F, TT,2]
            mag = torch.norm(stft, p=2, dim =-1) #[B, F, TT]
        return mag


class Nuwave2Model(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(Nuwave2Model, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.path = "./resources/app" if PROD else "."
        self.ckpt_path = None
        self.embeddings = []

        self.hparams = OC.load(f'{self.path}/python/nuwave2/hparameter.yaml')
        self.steps = 8
        self.noise_schedule = eval(self.hparams.dpm.infer_schedule)

        self.model = NuWave2(self.hparams)
        self.model.eval()
        stft = STFTMag()
        ckpt = torch.load(f'{self.path}/python/nuwave2/nuwave2_02_16_13_epoch=629.ckpt', map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'])

        self.sr = 22050

        highcut = self.sr // 2
        nyq = 0.5 * self.hparams.audio.sampling_rate
        self.hi = highcut / nyq

        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        self.ckpt_path = ckpt_path


    def sr_audio (self, in_path, out_path):

        wav, _ = librosa.load(in_path, sr=self.sr, mono=True)
        wav /= np.max(np.abs(wav))

        # upsample to the original sampling rate
        wav_l = resample_poly(wav, self.hparams.audio.sampling_rate, self.sr)
        wav_l = wav_l[:len(wav_l) - len(wav_l) % self.hparams.audio.hop_length]

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = torch.zeros(fft_size, dtype=torch.int64)
        band[:int(self.hi * fft_size)] = 1

        wav = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(self.device)
        band = band.unsqueeze(0).to(self.device)

        wav_recon, wav_list = self.model.inference(wav_l, band, self.steps, self.noise_schedule)

        wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)
        scipy.io.wavfile.write(out_path, self.hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())



    def set_device (self, device):
        self.device = device
        self.model = self.model.to(self.device)

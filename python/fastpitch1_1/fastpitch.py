# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import re

from typing import Optional

import torch
import traceback
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from python.common.layers import ConvReLUNorm
from python.common.utils import mask_from_lens
from python.fastpitch1_1.transformer import FFTransformer
from python.fastpitch1_1.attention import ConvAttention
from python.fastpitch1_1.alignment import b_mas, mas_width1

from python.common.utils import load_wav_to_torch
from librosa.filters import mel as librosa_mel_fn
from python.common.stft import STFT
from python.common.utils import mask_from_lens
from python.common.audio_processing import dynamic_range_compression, dynamic_range_decompression
from python.common.text.text_processing import TextProcessing

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

# For Speech-to-Speech
tp = TextProcessing("english_basic", ["english_cleaners"])
stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, 8000)


def regulate_len(durations, enc_out, pace: float = 1.0, mel_max_len: Optional[int] = None):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() * pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens

def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg

class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, n_symbols, padding_idx, symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads, in_fft_d_head, in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size, in_fft_output_size, p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb, out_fft_n_layers, out_fft_n_heads, out_fft_d_head, out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size, out_fft_output_size, p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb, dur_predictor_kernel_size, dur_predictor_filter_size, p_dur_predictor_dropout, dur_predictor_n_layers, pitch_predictor_kernel_size, pitch_predictor_filter_size, p_pitch_predictor_dropout, pitch_predictor_n_layers, pitch_embedding_kernel_size, energy_conditioning, energy_predictor_kernel_size, energy_predictor_filter_size, p_energy_predictor_dropout, energy_predictor_n_layers, energy_embedding_kernel_size, n_speakers, speaker_emb_weight, pitch_conditioning_formants=1, device=None):
        super(FastPitch, self).__init__()

        self.device = None

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads, d_model=symbols_embedding_dim, d_head=in_fft_d_head, d_inner=in_fft_conv1d_filter_size, kernel_size=in_fft_conv1d_kernel_size, dropout=p_in_fft_dropout, dropatt=p_in_fft_dropatt, dropemb=p_in_fft_dropemb, embed_input=True, d_embed=symbols_embedding_dim, n_embed=n_symbols, padding_idx=padding_idx)

        # if n_speakers > 1:
        #     self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        # else:
        #     self.speaker_emb = None
        self.speaker_emb = nn.Linear(256, symbols_embedding_dim)
        self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size, filter_size=dur_predictor_filter_size, kernel_size=dur_predictor_kernel_size, dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads, d_model=symbols_embedding_dim, d_head=out_fft_d_head, d_inner=out_fft_conv1d_filter_size, kernel_size=out_fft_conv1d_kernel_size, dropout=p_out_fft_dropout, dropatt=p_out_fft_dropatt, dropemb=p_out_fft_dropemb, embed_input=False, d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size, filter_size=pitch_predictor_filter_size, kernel_size=pitch_predictor_kernel_size, dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers, n_predictions=pitch_conditioning_formants
        )

        self.pitch_emb = nn.Conv1d(
            pitch_conditioning_formants, symbols_embedding_dim, kernel_size=pitch_embedding_kernel_size, padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        energy_conditioning = True
        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size, filter_size=energy_predictor_filter_size, kernel_size=energy_predictor_kernel_size, dropout=p_energy_predictor_dropout, n_layers=energy_predictor_n_layers, n_predictions=1
            )

            self.energy_emb = nn.Conv1d(1, symbols_embedding_dim, kernel_size=energy_embedding_kernel_size, padding=int((energy_embedding_kernel_size - 1) / 2))
        else:
            self.energy_predictor = None

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

        self.attention = ConvAttention(n_mel_channels, 0, symbols_embedding_dim, use_query_proj=True, align_query_enc_type='3xconv')






    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(self.device)

    def forward(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75):

        (inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense, speaker, attn_prior, audiopaths) = inputs

        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Alignment
        text_emb = self.encoder.word_emb(inputs)

        # make sure to do the alignments before folding
        attn_mask = mask_from_lens(input_lens)[..., None] == 0
        # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

        attn_soft, attn_logprob = self.attention(
            mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask, key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)

        attn_hard = self.binarize_attention_parallel(
            attn_soft, input_lens, mel_lens)

        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        dur_tgt = attn_hard_dur

        assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens))

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

        # Average pitch over characters
        pitch_tgt = average_pitch(pitch_dense, dur_tgt)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt)
        else:
            pitch_emb = self.pitch_emb(pitch_pred)
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Predict energy
        if self.energy_conditioning:
            energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)

            # Average energy over characters
            energy_tgt = average_pitch(energy_dense.unsqueeze(1), dur_tgt)
            energy_tgt = torch.log(1.0 + energy_tgt)

            energy_tgt = torch.clamp(energy_tgt, min=3.6, max=4.3)
            energy_emb = self.energy_emb(energy_tgt)
            energy_tgt = energy_tgt.squeeze(1)
            enc_out = enc_out + energy_emb.transpose(1, 2)
        else:
            energy_pred = None
            energy_tgt = None

        len_regulated, dec_lens = regulate_len(
            dur_tgt, enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob)

    def infer_using_vals (self, logger, plugin_manager, sequence, pace, enc_out, max_duration, enc_mask, dur_pred_existing, pitch_pred_existing, energy_pred_existing, old_sequence, new_sequence):

        start_index = None
        end_index = None

        # Calculate text splicing bounds, if needed
        if old_sequence is not None:

            old_sequence_np = old_sequence.cpu().detach().numpy()
            old_sequence_np = list(old_sequence_np[0])
            new_sequence_np = new_sequence.cpu().detach().numpy()
            new_sequence_np = list(new_sequence_np[0])


            # Get the index of the first changed value
            if old_sequence_np[0]==new_sequence_np[0]: # If the start of both sequences is the same, then the change is not at the start
                for i in range(len(old_sequence_np)):
                    if i<len(new_sequence_np):
                        if old_sequence_np[i]!=new_sequence_np[i]:
                            start_index = i-1
                            break
                    else:
                        start_index = i-1
                        break
                if start_index is None:
                    start_index = len(old_sequence_np)-1


            # Get the index of the last changed value
            old_sequence_np.reverse()
            new_sequence_np.reverse()
            if old_sequence_np[0]==new_sequence_np[0]: # If the end of both reversed sequences is the same, then the change is not at the end
                for i in range(len(old_sequence_np)):
                    if i<len(new_sequence_np):
                        if old_sequence_np[i]!=new_sequence_np[i]:
                            end_index = len(old_sequence_np)-1-i+1
                            break
                    else:
                        end_index = len(old_sequence_np)-1-i+1
                        break

            old_sequence_np.reverse()
            new_sequence_np.reverse()

        # Calculate its own pitch, duration, and energy vals if these were not already provided
        if (dur_pred_existing is None or dur_pred_existing.shape[1]==0) or old_sequence is not None:
            # Predict durations
            log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
            dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
            dur_pred = torch.clamp(dur_pred, 0.25)
        else:
            dur_pred = dur_pred_existing

        if (pitch_pred_existing is None or pitch_pred_existing.shape[1]==0) or old_sequence is not None:
            # Pitch over chars
            pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)
        else:
            pitch_pred = pitch_pred_existing.unsqueeze(1)

        energy_pred = energy_pred_existing

        # Splice/replace pitch/duration values from the old input if simulating only a partial re-generation
        if start_index is not None or end_index is not None:
            dur_pred_np = list(dur_pred.cpu().detach().numpy())[0]
            pitch_pred_np = list(pitch_pred.cpu().detach().numpy())[0][0]
            dur_pred_existing_np = list(dur_pred_existing.cpu().detach().numpy())[0]
            pitch_pred_existing_np = list(pitch_pred_existing.cpu().detach().numpy())[0]

            if start_index is not None: # Replace starting values

                for i in range(start_index+1):
                    dur_pred_np[i] = dur_pred_existing_np[i]
                    pitch_pred_np[i] = pitch_pred_existing_np[i]

            if end_index is not None: # Replace end values

                for i in range(len(old_sequence_np)-end_index):
                    dur_pred_np[-i-1] = dur_pred_existing_np[-i-1]
                    pitch_pred_np[-i-1] = pitch_pred_existing_np[-i-1]

            dur_pred = torch.tensor(dur_pred_np).to(self.device).unsqueeze(0)
            pitch_pred = torch.tensor(pitch_pred_np).to(self.device).unsqueeze(0).unsqueeze(0)

            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            energy_pred = self.energy_predictor(enc_out + pitch_emb, enc_mask).squeeze(-1)


        if plugin_manager is not None and len(plugin_manager.plugins["synth-line"]["mid"]):
            pitch_pred = pitch_pred.cpu().detach().numpy()
            plugin_data = {
                "duration": dur_pred.cpu().detach().numpy(),
                "pitch": pitch_pred.reshape((pitch_pred.shape[0],pitch_pred.shape[2])),
                "text": [val.split("|") for val in sequence],
                "is_fresh_synth": pitch_pred_existing is None and dur_pred_existing is None,
                "pluginsContext": plugin_manager.context
            }
            plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["mid"], event="mid synth-line", data=plugin_data)

            dur_pred = torch.tensor(plugin_data["duration"]).to(self.device)
            pitch_pred = torch.tensor(plugin_data["pitch"]).unsqueeze(1).to(self.device)

        pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        # Energy
        if self.energy_conditioning:
            if (energy_pred_existing is None or energy_pred_existing.shape[1]==0):
                energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
            else:
                # Splice/replace pitch/duration values from the old input if simulating only a partial re-generation
                if start_index is not None or end_index is not None:
                    energy_pred_np = list(energy_pred.cpu().detach().numpy())[0]
                    energy_pred_existing_np = list(energy_pred_existing.cpu().detach().numpy())[0]
                    if start_index is not None: # Replace starting values
                        for i in range(start_index+1):
                            energy_pred_np[i] = energy_pred_existing_np[i]

                    if end_index is not None: # Replace end values
                        for i in range(len(old_sequence_np)-end_index):
                            energy_pred_np[-i-1] = energy_pred_existing_np[-i-1]
                    energy_pred = torch.tensor(energy_pred_np).to(self.device).unsqueeze(0)
                    energy_pred = torch.clamp(energy_pred, min=3.6, max=4.3)


            if plugin_manager is not None and len(plugin_manager.plugins["synth-line"]["pre_energy"]):
                pitch_pred = pitch_pred.cpu().detach().numpy()
                plugin_data = {
                    "duration": dur_pred.cpu().detach().numpy(),
                    "pitch": pitch_pred.reshape((pitch_pred.shape[0],pitch_pred.shape[2])),
                    "energy": energy_pred.cpu().detach().numpy(),
                    "text": [val.split("|") for val in sequence], "is_fresh_synth": pitch_pred_existing is None and dur_pred_existing is None
                }
                plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["pre_energy"], event="pre_energy synth-line", data=plugin_data)

                pitch_pred = torch.tensor(plugin_data["pitch"]).unsqueeze(1).to(self.device)
                energy_pred = torch.tensor(plugin_data["energy"]).to(self.device)

            # Apply the energy
            energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        len_regulated, dec_lens = regulate_len(dur_pred, enc_out, pace, mel_max_len=None)
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        start_index = -1 if start_index is None else start_index
        end_index = -1 if end_index is None else end_index
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred, start_index, end_index

    def infer_advanced (self, logger, plugin_manager, cleaned_text, inputs, speaker_i, pace=1.0, pitch_data=None, max_duration=75, old_sequence=None):

        if speaker_i is not None and self.speaker_emb is not None:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker_i
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
            del speaker
        else:
            spk_emb = 0

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        if (pitch_data is not None) and ((pitch_data[0] is not None and len(pitch_data[0])) or (pitch_data[1] is not None and len(pitch_data[1]))):
            pitch_pred, dur_pred, energy_pred = pitch_data
            dur_pred = torch.tensor(dur_pred)
            dur_pred = dur_pred.view((1, dur_pred.shape[0])).float().to(self.device)
            pitch_pred = torch.tensor(pitch_pred)
            pitch_pred = pitch_pred.view((1, pitch_pred.shape[0])).float().to(self.device)
            energy_pred = torch.tensor(energy_pred)
            energy_pred = energy_pred.view((1, energy_pred.shape[0])).float().to(self.device)

            del spk_emb
            # Try using the provided pitch/duration data, but fall back to using its own, otherwise
            try:
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, dur_pred_existing=dur_pred, pitch_pred_existing=pitch_pred, energy_pred_existing=energy_pred, old_sequence=old_sequence, new_sequence=inputs)
            except:
                print(traceback.format_exc())
                logger.info(traceback.format_exc())
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, None, None, None, None, None)

        else:
            del spk_emb
            return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, None, None, None, None, None)


    def run_speech_to_speech (self, device, logger, models_manager, modelType, s2s_components, audiopath, in_text, text_to_sequence, sequence_to_text, model_instance):

        self.device = device
        max_wav_value = 32768
        modelType = modelType.lower().replace(".", "_").replace(" ", "")

        # Inference
        audio, sampling_rate = load_wav_to_torch(audiopath)
        audio_norm = audio / max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        melspec = melspec.to(device)
        mel = melspec

        text = in_text
        text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        text = model_instance.infer_arpabet_dict(text)


        sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        cleaned_text = sequence_to_text("english_basic", sequence)
        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.device)
        text = text.to(device)
        inputs = text


        # Input FFT
        spk_emb = 0
        max_duration = 75
        enc_out, enc_mask = models_manager.models(modelType).model.encoder(inputs, conditioning=spk_emb)
        mel_out, dec_lens, dur_pred, pitch_pred, energy_pred, start_index, end_index = models_manager.models(modelType).model.infer_using_vals(logger, None, cleaned_text, 1, enc_out, max_duration, enc_mask, None, None, None, None, None)

        dur_pred = dur_pred.cpu().detach().numpy()[0]
        pitch_pred = pitch_pred.cpu().detach().numpy()
        energy_pred = energy_pred.cpu().detach().numpy()



        # Compute the durations from the reference audio
        # ============
        mel_tgt = mel.unsqueeze(0)
        text_emb = self.encoder.word_emb(inputs).to(device)
        input_lens = torch.tensor([len(text[0])]).to(device)
        mel_lens = torch.tensor([mel.size(1)]).to(device)

        attn_prior = beta_binomial_prior_distribution(text.shape[1], mel.shape[1]).unsqueeze(0)
        attn_prior = attn_prior.to(device)
        attn_mask = mask_from_lens(input_lens)[..., None] == 0
        attn_soft, attn_logprob = self.attention(mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask, key_lens=input_lens, attn_prior=attn_prior)

        attn_hard = self.binarize_attention_parallel(attn_soft, input_lens, mel_lens)

        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        durs = attn_hard_dur
        durs = torch.clamp(durs, 0.25)
        durs = durs.cpu().detach().numpy()[0]

        # Apply configured interpolation
        if s2s_components["durations"]["enabled"]:
            durations_final = (durs*s2s_components["durations"]["strength"] + dur_pred*(1-s2s_components["durations"]["strength"]))
        else:
            durations_final = dur_pred
        # ============





        # Compute the pitch from the reference audio
        # ============
        mean = self.pitch_mean # None
        std = self.pitch_std # None

        pitch = estimate_pitch(audiopath, mel.size(-1), "praat", mean, std)
        # Average pitch over characters
        pitch_tgt = average_pitch(pitch.unsqueeze(0), attn_hard_dur)


        # Apply configured interpolation
        if s2s_components["pitch"]["enabled"]:
            pitch_final = (pitch_tgt*s2s_components["pitch"]["strength"] + pitch_pred*(1-s2s_components["pitch"]["strength"])).cpu().detach().numpy()
        else:
            pitch_final = np.array([[pitch_pred]])
        # ============




        # Compute the energy from the reference audio
        # ============
        # Average energy over characters
        energy_dense = torch.norm(mel.float(), dim=0, p=2)
        energy = average_pitch(energy_dense.unsqueeze(0).unsqueeze(0), attn_hard_dur)
        energy = torch.log(1.0 + energy)
        energy = torch.clamp(energy, min=3.6, max=4.3)
        energy_tgt = []
        try:
            energy_tgt = energy.squeeze().cpu().detach().numpy()
        except:
            logger.info(traceback.format_exc())

        # Apply configured interpolation
        if s2s_components["energy"]["enabled"]:
            energy_final = (energy_tgt*s2s_components["energy"]["strength"] + energy_pred*(1-s2s_components["energy"]["strength"]))
        else:
            energy_final = energy_pred
        # ============




        pitch_final = list(pitch_final.squeeze())
        # pitch_final = normalize_pitch_vectors(logger, pitch_final)
        # pitch_final = [max(-3, min(v, 3)) for v in pitch_final]
        durs_final = list(durations_final)
        energy_final = list(energy_final[0])

        return [cleaned_text, pitch_final, durs_final, energy_final]



import librosa
def normalize_pitch_vectors(logger, pitch_vecs):
    nonzeros = [v for v in pitch_vecs if v!=0.0]
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    for vi, v in enumerate(pitch_vecs):
        v -= mean
        v /= std
        pitch_vecs[vi] = v

    return pitch_vecs
def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None].to(pitch.device)
    pitch /= std[:, None].to(pitch.device)
    pitch[zeros] = 0.0
    return pitch
def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None, normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'praat':

        snd = parselmouth.Sound(wav)
        pitch_mel = snd.to_pitch(time_step=snd.duration / (mel_len + 3)
                                 ).selected_array['frequency']
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)

        if n_formants > 1:
            formant = snd.to_formant_burg(
                time_step=snd.duration / (mel_len + 3))
            formant_n_frames = formant.get_number_of_frames()
            assert np.abs(mel_len - formant_n_frames) <= 1.0

            formants_mel = np.zeros((formant_n_frames + 1, n_formants - 1))
            for i in range(1, formant_n_frames + 1):
                formants_mel[i] = np.asarray([
                    formant.get_value_at_time(
                        formant_number=f,
                        time=formant.get_time_from_frame_number(i))
                    for f in range(1, n_formants)
                ])

            pitch_mel = torch.cat(
                [pitch_mel, torch.from_numpy(formants_mel).permute(1, 0)],
                dim=0)

    elif method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel



from scipy.stats import betabinom
def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))



import parselmouth
import numpy as np
def maybe_pad(vec, l):
    assert np.abs(vec.shape[0] - l) <= 3
    vec = vec[:l]
    if vec.shape[0] < l:
        vec = np.pad(vec, pad_width=(0, l - vec.shape[0]))
    return vec
def calculate_pitch (fname, durs):
    mel_len = durs.sum()
    print(int(mel_len))
    durs_cum = np.cumsum(np.pad(durs, (1, 0), mode="constant"))
    snd = parselmouth.Sound(fname)

    pitch = snd.to_pitch(time_step=snd.duration / (mel_len + 3)).selected_array['frequency']

    assert np.abs(mel_len - pitch.shape[0]) <= 1.0

    # Average pitch over characters
    pitch_char = np.zeros((durs.shape[0],), dtype=np.float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    pitch_char = maybe_pad(pitch_char, len(durs))

    return pitch_char
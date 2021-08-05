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

import torch
import traceback
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence

from python.common.layers import ConvReLUNorm
from python.common.utils import mask_from_lens
from python.fastpitch.transformer import FFTransformer


def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None):
    """If target=None, then predicted durations are applied"""
    reps = torch.round(durations.float() * pace).long()
    dec_lens = reps.sum(dim=1)

    enc_rep = pad_sequence([torch.repeat_interleave(o, r, dim=0)
                            for o, r in zip(enc_out, reps)],
                           batch_first=True)
    if mel_max_len:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, device=None):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out.squeeze(-1)


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, max_seq_len, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size, n_speakers, speaker_emb_weight, device=None):
        super(FastPitch, self).__init__()
        del max_seq_len  # unused

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
            print(f'self.speaker_emb, {self.speaker_emb}')
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers
        )

        self.pitch_emb = nn.Conv1d(
            1, symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

    def forward(self, inputs, use_gt_durations=True, use_gt_pitch=True,
                pace=1.0, max_duration=75):
        inputs, _, mel_tgt, _, dur_tgt, _, pitch_tgt, speaker = inputs
        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        # Predict durations
        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        len_regulated, dec_lens = regulate_len(
            dur_tgt if use_gt_durations else dur_pred,
            enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        return mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred

    def infer(self, inputs, input_lens, pace=1.0, dur_tgt=None, pitch_tgt=None,
              pitch_transform=None, max_duration=75, speaker=0):
        del input_lens  # unused

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        # Predict durations
        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)

        if pitch_transform is not None:
            if self.pitch_std[0] == 0.0:
                # XXX LJSpeech-1.1 defaults
                mean, std = 218.14, 67.24
            else:
                mean, std = self.pitch_mean[0], self.pitch_std[0]
            pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)), mean, std)

        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1)).transpose(1, 2)
        else:
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1)).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred



    def infer_using_vals (self, logger, plugin_manager, sequence, pace, enc_out, max_duration, enc_mask, dur_pred_existing=None, pitch_pred_existing=None, old_sequence=None, new_sequence=None):

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


        # Calculate its own pitch and duration vals if these were not already provided
        if (dur_pred_existing is None or pitch_pred_existing is None) or old_sequence is not None:

            # Embedded for predictors
            pred_enc_out, pred_enc_mask = enc_out, enc_mask
            # Predict durations
            log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
            dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

            dur_pred = torch.clamp(dur_pred, 0.25)

            # Pitch over chars
            pitch_pred = self.pitch_predictor(enc_out, enc_mask)
        else:
            dur_pred = dur_pred_existing
            pitch_pred = pitch_pred_existing

        # Splice/replace pitch/duration values from the old input if simulating only a partial re-generation
        if start_index is not None or end_index is not None:
            dur_pred_np = list(dur_pred.cpu().detach().numpy())[0]
            pitch_pred_np = list(pitch_pred.cpu().detach().numpy())[0]
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
            pitch_pred = torch.tensor(pitch_pred_np).to(self.device).unsqueeze(0)

        if len(plugin_manager.plugins["synth-line"]["mid"]):
            plugin_data = {
                "duration": dur_pred.cpu().detach().numpy(),
                "pitch": pitch_pred.cpu().detach().numpy(),
                "text": sequence,
                "is_fresh_synth": pitch_pred_existing is None and dur_pred_existing is None
            }
            plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["mid"], event="mid synth-line", data=plugin_data)

            dur_pred = torch.tensor(plugin_data["duration"]).to(self.device)
            pitch_pred = torch.tensor(plugin_data["pitch"]).to(self.device)

        pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1)).transpose(1, 2)

        enc_out = enc_out + pitch_emb
        len_regulated, dec_lens = regulate_len(dur_pred, enc_out, pace, mel_max_len=None)
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        start_index = -1 if start_index is None else start_index
        end_index = -1 if end_index is None else end_index
        return mel_out, dec_lens, dur_pred, pitch_pred, start_index, end_index


    def infer_advanced (self, logger, plugin_manager, cleaned_text, inputs, speaker_i, pace=1.0, pitch_data=None, max_duration=75, old_sequence=None):

        if speaker_i is not None:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker_i
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
            del speaker
        else:
            spk_emb = 0

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        if pitch_data is not None and pitch_data[0] is not None and len(pitch_data[0]) and pitch_data[1] is not None and len(pitch_data[1]):
            pitch_pred, dur_pred = pitch_data
            dur_pred = torch.tensor(dur_pred)
            dur_pred = dur_pred.view((1, dur_pred.shape[0])).float().to(self.device)
            pitch_pred = torch.tensor(pitch_pred)
            pitch_pred = pitch_pred.view((1, pitch_pred.shape[0])).float().to(self.device)

            del spk_emb
            # Try using the provided pitch/duration data, but fall back to using its own, otherwise
            try:
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, dur_pred_existing=dur_pred, pitch_pred_existing=pitch_pred, old_sequence=old_sequence, new_sequence=inputs)
            except:
                print(traceback.format_exc())
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, None, None, None)

        else:
            del spk_emb
            return self.infer_using_vals(logger, plugin_manager, cleaned_text, pace, enc_out, max_duration, enc_mask, None, None, None)


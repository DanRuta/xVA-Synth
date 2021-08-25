# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from typing import Optional
from os.path import abspath, dirname

import torch

# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/'))
from python.fastpitch1_1.fastpitch import FastPitch as _FastPitch
# from python.model_fp import WaveGlow
from python.common.text.symbols import get_symbols, get_pad_idx

def parse_model_args(model_name, symbols_alphabet, parser, add_help=False):
    from python.fastpitch1_1.arg_parser import parse_fastpitch_args
    return parse_fastpitch_args(symbols_alphabet, parser, add_help)

def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, device, logger, uniform_initialize_bn_weight=False, forward_is_infer=False, jitable=False):
    model = None
    model_config["device"] = device

    if model_name == 'WaveGlow':
        if forward_is_infer:
            class WaveGlow__forward_is_infer(WaveGlow):
                def forward(self, spect, sigma=1.0):
                    return self.infer(spect, sigma)
            model = WaveGlow__forward_is_infer(**model_config, logger=logger)
        else:
            model = WaveGlow(**model_config, logger=logger)

    elif model_name == 'FastPitch':

        model_config["padding_idx"] = 0
        model_config["pitch_embedding_kernel_size"] = 3
        model_config["n_speakers"] = 5
        model_config["speaker_emb_weight"] = 1.0

        if forward_is_infer:

            class FastPitch__forward_is_infer(_FastPitch):
                def forward(self, inputs, input_lengths=None, pace: float = 1.0,
                            dur_tgt: Optional[torch.Tensor] = None,
                            pitch_tgt: Optional[torch.Tensor] = None,
                            pitch_transform=None, device=None):
                    return self.infer_advanced(inputs, input_lengths, pace=pace,
                                      dur_tgt=dur_tgt, pitch_tgt=pitch_tgt,
                                      pitch_transform=pitch_transform)

            model = FastPitch__forward_is_infer(**model_config)
        else:
            model = _FastPitch(**model_config)

    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    return model.to(device)


def get_model_config(model_name, args):
    if model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=len(get_symbols(args.symbol_set)),
            padding_idx=get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
        )
        return model_config

    else:
        raise NotImplementedError(model_name)

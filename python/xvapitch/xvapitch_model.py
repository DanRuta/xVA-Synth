import math
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv1d

from python.xvapitch.glow_tts import RelativePositionTransformer
from python.xvapitch.wavenet import WN
from python.xvapitch.hifigan import HifiganGenerator, DiscriminatorP
from python.xvapitch.sdp import StochasticDurationPredictor#, StochasticPredictor

from python.xvapitch.util import maximum_path, rand_segments, segment, sequence_mask, generate_path
from python.xvapitch.text import get_text_preprocessor, ALL_SYMBOLS, lang_names

# import cupy as cp
# from python.xvapitch.util import maximum_path_cupy, maximum_path_numba

class xVAPitch(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args.init_discriminator = True
        self.args.speaker_embedding_channels = 512
        self.args.use_spectral_norm_disriminator = False
        self.args.d_vector_dim = 512
        # self.args.embedded_language_dim = 4
        self.args.use_language_embedding = True
        self.args.detach_dp_input = True

        self.END2END = True
        # self.embedded_language_dim = 4
        # self.embedded_language_dim = 8

        self.embedded_language_dim = 12
        self.latent_size = 256

        # num_languages = 3
        num_languages = len(list(lang_names.keys()))
        # self.emb_l = nn.Embedding(self.args.num_languages, self.embedded_language_dim)
        self.emb_l = nn.Embedding(num_languages, self.embedded_language_dim)



        self.length_scale = 1.0
        self.noise_scale = 1.0

        self.inference_noise_scale = 0.333

        self.inference_noise_scale_dp = 0.333
        self.noise_scale_dp = 1.0
        self.max_inference_len = None
        self.spec_segment_size = 32


        self.text_encoder = TextEncoder(
            # 165,
            len(ALL_SYMBOLS),
            self.latent_size,#192,
            self.latent_size,#192,
            768,
            2,
            10,
            3,
            0.1,
            # language_emb_dim=4,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            513,
            self.latent_size,#+self.embedded_language_dim if self.args.flc else self.latent_size,#192,
            self.latent_size,#+self.embedded_language_dim if self.args.flc else self.latent_size,#192,
            kernel_size=5,
            dilation_rate=1,
            num_layers=16,
            cond_channels=self.args.d_vector_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.latent_size,#192,
            self.latent_size,#192,
            kernel_size=5,
            dilation_rate=1,
            num_layers=4,
            cond_channels=self.args.d_vector_dim,
            args=self.args
        )

        self.duration_predictor = StochasticDurationPredictor(
            self.latent_size,#192,
            self.latent_size,#192,
            3,
            0.5,
            4,
            cond_channels=self.args.d_vector_dim,
            language_emb_dim=self.embedded_language_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            self.latent_size,#192,
            1,
            "1",
            [[1,3,5],[1,3,5],[1,3,5]],
            [3,7,11],
            [16,16,4,4],
            512,
            [8,8,2,2],
            inference_padding=0,
            # cond_channels=self.args.d_vector_dim+self.embedded_language_dim if self.args.flc else self.args.d_vector_dim,
            cond_channels=self.args.d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        # self.disc = VitsDiscriminator(use_spectral_norm=False)



        self.pitch_predictor = RelativePositioningPitchEnergyEncoder(
            # 165,
            # len(ALL_SYMBOLS),
            out_channels=1,
            hidden_channels=self.latent_size+self.embedded_language_dim,#196,
            hidden_channels_ffn=768,
            num_heads=2,
            # num_layers=10,
            num_layers=3,
            kernel_size=3,
            dropout_p=0.1,
            # language_emb_dim=4,
            conditioning_emb_dim=self.args.d_vector_dim,
        )

        self.pitch_emb = nn.Conv1d(
            # 1, 384,
            # 1, 196,
            1,
            self.args.expanded_flow_dim if args.expanded_flow else self.latent_size,
            # pitch_conditioning_formants, symbols_embedding_dim,
            kernel_size=3,
            padding=int((3 - 1) / 2))


        # if args.energy:
        #     self.energy_predictor = RelativePositioningPitchEnergyEncoder(
        #         # 165,
        #         # len(ALL_SYMBOLS),
        #         out_channels=1,
        #         hidden_channels=196,
        #         hidden_channels_ffn=768,
        #         num_heads=2,
        #         # num_layers=10,
        #         num_layers=3,
        #         kernel_size=3,
        #         dropout_p=0.1,
        #         # language_emb_dim=4,
        #         conditioning_emb_dim=self.args.d_vector_dim,
        #     )

        #     if not self.args.ow_flow:
        #         self.energy_emb = nn.Conv1d(
        #             1,
        #             self.args.expanded_flow_dim if args.expanded_flow else 192,
        #             kernel_size=3,
        #             padding=int((3 - 1) / 2))



        self.TEMP_timing = []



    # def format_batch (self, batch):

    #     text_input = batch["text"]
    #     text_lengths = batch["text_lengths"]
    #     linear_input = batch["linear"]
    #     pitch_padded = batch["pitch_padded"]
    #     energy_padded = batch["energy_padded"]
    #     # mel_input = batch["mel"]
    #     mel_lengths = batch["mel_lengths"]
    #     mel_mask = batch["mel_mask"]
    #     # stop_targets = batch["stop_targets"]
    #     # item_idx = batch["item_idxs"]
    #     d_vectors = batch["d_vectors"]
    #     # speaker_ids = batch["speaker_ids"]
    #     waveform = batch["waveform"]
    #     language_ids = batch["language_ids"]
    #     max_text_length = torch.max(text_lengths.float())
    #     max_spec_length = torch.max(mel_lengths.float())

    #     # compute durations from attention masks
    #     durations = None

    #     # set stop targets wrt reduction factor
    #     # stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // 1, -1)
    #     # stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
    #     # stop_target_lengths = torch.divide(mel_lengths, 1).ceil_()

    #     return {
    #         "text_input": text_input,
    #         "text_lengths": text_lengths,
    #         # "mel_input": mel_input,
    #         "mel_lengths": mel_lengths,
    #         "mel_mask": mel_mask,
    #         "linear_input": linear_input,
    #         "pitch_padded": pitch_padded,
    #         "energy_padded": energy_padded,
    #         # "stop_targets": stop_targets,
    #         # "stop_target_lengths": stop_target_lengths,
    #         "durations": durations,
    #         # "speaker_ids": speaker_ids,
    #         "d_vectors": d_vectors,
    #         "max_text_length": float(max_text_length),
    #         "max_spec_length": float(max_spec_length),
    #         # "item_idx": item_idx,
    #         "waveform": waveform,
    #         "language_ids": language_ids,
    #     }


    # def forward(self, batch, optimizer_idx, y_disc_cache, wav_seg_disc_cache):

    #     if optimizer_idx == 0:
    #         text_input = batch["text_input"]
    #         text_lengths = batch["text_lengths"]
    #         pitch_padded = batch["pitch_padded"]
    #         energy_padded = batch["energy_padded"]
    #         mel_lengths = batch["mel_lengths"]
    #         mel_mask = batch["mel_mask"]
    #         linear_input = batch["linear_input"]
    #         d_vectors = batch["d_vectors"]
    #         language_ids = batch["language_ids"]
    #         waveform = batch["waveform"]

    #         # generator pass
    #         if self.args.hifi_only:
    #             outputs = self.train_hifi_only(
    #                 text_input,
    #                 text_lengths,
    #                 linear_input.transpose(1, 2),
    #                 mel_lengths,
    #                 pitch_padded,
    #                 energy_padded,
    #                 waveform.transpose(1, 2),
    #                 aux_input={"d_vectors": d_vectors, "language_ids": language_ids},
    #             )
    #         else:
    #             outputs = self.train_step(
    #                 text_input,
    #                 text_lengths,
    #                 linear_input.transpose(1, 2),
    #                 mel_lengths,
    #                 pitch_padded,
    #                 energy_padded,
    #                 waveform.transpose(1, 2),
    #                 aux_input={"d_vectors": d_vectors, "language_ids": language_ids},
    #             )

    #         del text_input, text_lengths, pitch_padded, energy_padded, mel_lengths, linear_input, d_vectors, language_ids, waveform

    #         # compute discriminator scores and features
    #         outputs["scores_disc_fake"], outputs["feats_disc_fake"], _, outputs["feats_disc_real"] = self.disc(
    #             outputs["model_outputs"], outputs["waveform_seg"]
    #         )

    #         new_outputs = {}

    #         if self.args.hifi_only:
    #             # compute losses
    #             loss_dict = self.criterion[optimizer_idx](
    #                 waveform_hat=outputs["model_outputs"].float(),
    #                 waveform=outputs["waveform_seg"].float(),
    #                 z_p=None,
    #                 logs_q=None,
    #                 m_p=None,
    #                 logs_p=None,
    #                 z_mask=mel_mask,
    #                 scores_disc_fake=outputs["scores_disc_fake"],
    #                 feats_disc_fake=outputs["feats_disc_fake"],
    #                 feats_disc_real=outputs["feats_disc_real"],
    #                 loss_duration=None,
    #                 # use_speaker_encoder_as_loss=False,
    #                 # gt_spk_emb=None,#outputs["gt_spk_emb"],
    #                 # syn_spk_emb=None,#outputs["syn_spk_emb"],
    #             )
    #             # print(f'hifi_debug, 6')
    #         else:
    #             # compute losses
    #             loss_dict = self.criterion[optimizer_idx](
    #                 waveform_hat=outputs["model_outputs"].float(),
    #                 waveform=outputs["waveform_seg"].float(),
    #                 z_p=outputs["z_p"].float(),
    #                 logs_q=outputs["logs_q"].float(),
    #                 m_p=outputs["m_p"].float(),
    #                 logs_p=outputs["logs_p"].float(),
    #                 # z_len=mel_lengths,
    #                 z_mask=mel_mask,
    #                 scores_disc_fake=outputs["scores_disc_fake"],
    #                 feats_disc_fake=outputs["feats_disc_fake"],
    #                 feats_disc_real=outputs["feats_disc_real"],
    #                 loss_duration=outputs["loss_duration"],
    #                 # use_speaker_encoder_as_loss=False,
    #                 # gt_spk_emb=None,#outputs["gt_spk_emb"],
    #                 # syn_spk_emb=None,#outputs["syn_spk_emb"],

    #                 mask=outputs["mask"],
    #                 pitch_pred=outputs["pitch_pred"],
    #                 pitch_tgt=outputs["pitch_tgt"],
    #                 energy_pred=outputs["energy_pred"],
    #                 energy_tgt=outputs["energy_tgt"],

    #                 # y_mask=None,#outputs["y_mask"],
    #                 pitch_flow=outputs["pitch_flow"],
    #                 energy_flow=outputs["energy_flow"],
    #                 z_p_pitch_pred=outputs["z_p_pitch_pred"],
    #                 z_p_energy_pred=outputs["z_p_energy_pred"],
    #                 z_p_pitch=outputs["z_p_pitch"],
    #                 z_p_energy=outputs["z_p_energy"],
    #             )

    #         new_outputs["model_outputs"] = outputs["model_outputs"]
    #         new_outputs["waveform_seg"] = outputs["waveform_seg"]
    #         del outputs
    #         outputs = new_outputs

    #     elif optimizer_idx == 1:
    #         # discriminator pass
    #         outputs = {}

    #         # compute scores and features
    #         outputs["scores_disc_fake"], _, outputs["scores_disc_real"], _ = self.disc(
    #             y_disc_cache, wav_seg_disc_cache
    #         )
    #         del _

    #         # compute loss
    #         loss_dict = self.criterion[optimizer_idx](
    #             outputs["scores_disc_real"],
    #             outputs["scores_disc_fake"],
    #         )

    #     return outputs, loss_dict



    def infer_get_lang_emb (self, language_id):

        aux_input = {
            # "d_vectors": embedding.unsqueeze(dim=0),
            "language_ids": language_id
        }

        sid, g, lid = self._set_cond_input(aux_input)
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        return lang_emb


    # def infer_advanced (self, logger, plugin_manager, cleaned_texts, text, lang_ids, speaker_embs, pace, pitch_data, old_sequence):
    def infer_advanced (self, logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace=1.0, editor_data=None, old_sequence=None, pitch_amp=None):

        if (editor_data is not None) and ((editor_data[0] is not None and len(editor_data[0])) or (editor_data[1] is not None and len(editor_data[1]))):
            pitch_pred, dur_pred, energy_pred = editor_data
            # TODO, use energy_pred
            dur_pred = torch.tensor(dur_pred)
            dur_pred = dur_pred.view((1, dur_pred.shape[0])).float().to(self.device)
            pitch_pred = torch.tensor(pitch_pred)
            pitch_pred = pitch_pred.view((1, pitch_pred.shape[0])).float().to(self.device)

            try:
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, dur_pred_existing=dur_pred, pitch_pred_existing=pitch_pred, old_sequence=old_sequence, new_sequence=text, pitch_amp=pitch_amp)
            except:
                print(traceback.format_exc())
                logger.info(traceback.format_exc())
                return self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, None, None, None, None, pitch_amp=pitch_amp)

        else:
            return self.infer_using_vals(logger, plugin_manager, cleaned_text, text, lang_embs, speaker_embs, pace, None, None, None, None, pitch_amp=pitch_amp)




    def infer_using_vals (self, logger, plugin_manager, cleaned_text, sequence, lang_embs, speaker_embs, pace, dur_pred_existing, pitch_pred_existing, old_sequence, new_sequence, pitch_amp=None):

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


        # cleaned_text is the actual text phonemes

        input_symbols = sequence
        # logger.info(f'input_symbols: {input_symbols}')
        x_lengths = torch.tensor(input_symbols.shape[1:2]).to(input_symbols.device)

        # TODO, store a bank of trained 31 language embeds, to use for interpolating
        # lang_emb = lang_embs
        lang_emb = self.emb_l(lang_embs).unsqueeze(-1)

        lang_emb_full = None # TODO
        x, x_emb, x_mask = self.text_encoder(input_symbols, x_lengths, lang_emb=lang_emb, stats=False, lang_emb_full=lang_emb_full)
        m_p, logs_p = self.text_encoder(x, x_lengths, lang_emb=lang_emb, stats=True, x_mask=x_mask)




        # Calculate its own pitch, and duration vals if these were not already provided
        if (dur_pred_existing is None or dur_pred_existing.shape[1]==0) or old_sequence is not None:
            # Predict durations
            # log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
            # dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
            # dur_pred = torch.clamp(dur_pred, 0.25)
            logw = self.duration_predictor(x, x_mask, g=speaker_embs, reverse=True, noise_scale=self.inference_noise_scale_dp, lang_emb=lang_emb)
            w = torch.exp(logw) * x_mask * self.length_scale
            w = w * 1.3 # The model seems to generate quite fast speech, so I'm gonna just globally adjust that
            w = w * pace
            w_ceil = w
            w_ceil = torch.ceil(w)
            dur_pred = w_ceil
        else:
            dur_pred = dur_pred_existing.unsqueeze(dim=0)
            dur_pred = dur_pred * pace

        # logger.info(f'pace: {pace}')
        # logger.info(f'dur_pred: {dur_pred.shape}')

        # y_lengths = torch.clamp_min(torch.sum(dur_pred, [1, 2]), 1).long()
        y_lengths = torch.clamp_min(torch.sum(torch.round(dur_pred), [1, 2]), 1).long()
        # logger.info(f'y_lengths: {y_lengths}')
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype)
        # logger.info(f'y_mask: {y_mask.shape}')

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)



        if (pitch_pred_existing is None or pitch_pred_existing.shape[1]==0) or old_sequence is not None:
            # Pitch over chars
            # pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)
            pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=speaker_embs, stats=False)

            # pitch_tgt = average_pitch(pitch_pred, w_ceil).detach()

        else:
            pitch_pred = pitch_pred_existing.unsqueeze(1)#.unsqueeze(1)

        # logger.info(f'm_p: {m_p.shape}')
        # logger.info(f'pitch_pred: {pitch_pred.shape}')
        # energy_pred = energy_pred_existing

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

            # pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            # energy_pred = self.energy_predictor(enc_out + pitch_emb, enc_mask).squeeze(-1)


        if pitch_amp is not None:
            # pitch_pred = pitch_pred * pitch_amp.squeeze(dim=1).unsqueeze(dim=0).unsqueeze(dim=0)
            # TEMP fix, do this properly
            for i in range(pitch_pred.shape[0]):
                pitch_pred[0] = pitch_pred[0] * pitch_amp.squeeze(dim=1)[i]


        if plugin_manager is not None and len(plugin_manager.plugins["synth-line"]["mid"]):
            pitch_pred = pitch_pred.cpu().detach().numpy()
            plugin_data = {
                "duration": dur_pred.cpu().detach().numpy(),
                "pitch": pitch_pred.reshape((pitch_pred.shape[0],pitch_pred.shape[2])),
                "text": [val.split("|") for val in cleaned_text],
                "is_fresh_synth": pitch_pred_existing is None and dur_pred_existing is None,

                "pluginsContext": plugin_manager.context
            }
            plugin_manager.run_plugins(plist=plugin_manager.plugins["synth-line"]["mid"], event="mid synth-line", data=plugin_data)

            dur_pred = torch.tensor(plugin_data["duration"]).to(self.device)
            pitch_pred = torch.tensor(plugin_data["pitch"]).unsqueeze(1).to(self.device)

        # pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
        # pitch_pred = self.expand_pitch_energy(pitch_pred, dur_pred)
        pitch_emb = self.pitch_emb(self.expand_pitch_energy(pitch_pred, dur_pred, logger=logger))
        # logger.info(f'pitch_emb: {pitch_emb.shape}')


        m_p += pitch_emb * self.args.pe_scaling

        # TODO, incorporate some sort of control for this
        self.inference_noise_scale = 0
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=speaker_embs, reverse=True)
        wav = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=speaker_embs)



        start_index = -1 if start_index is None else start_index
        end_index = -1 if end_index is None else end_index


        # TODO, make editable and actually doing something
        energy_pred = [1.0 for _ in range(pitch_pred.shape[-1])]
        energy_pred = torch.tensor(energy_pred)

        # logger.info(f'dur_pred: {dur_pred.shape}')
        # logger.info(f'pitch_pred: {pitch_pred.shape}')
        # logger.info(f'energy_pred: {energy_pred.shape}')

        return wav, dur_pred, pitch_pred, energy_pred, start_index, end_index






    def infer (self, input_symbols, lang_emb=None, embedding=None, durs_only=False, lang_emb_full=None, pacing=1):

        aux_input = {
            "d_vectors": embedding.unsqueeze(dim=0),
            "language_ids": lang_emb if lang_emb_full is None else None
        }

        _, g, lid = self._set_cond_input(aux_input)
        x_lengths = torch.tensor(input_symbols.shape[1:2]).to(input_symbols.device)

        if lang_emb_full is None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, x_emb, x_mask = self.text_encoder(input_symbols, x_lengths, lang_emb=lang_emb, stats=False, lang_emb_full=lang_emb_full)
        m_p, logs_p = self.text_encoder(x, x_lengths, lang_emb=lang_emb, stats=True, x_mask=x_mask)

        logw = self.duration_predictor(x, x_mask, g=g, reverse=True, noise_scale=self.inference_noise_scale_dp, lang_emb=lang_emb)

        w = torch.exp(logw) * x_mask * self.length_scale
        w = w * pacing
        w_ceil = torch.ceil(w)
        if durs_only:
            return w_ceil

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)


        # ================
        if self.args.ow_flow:
            # Average pitch over characters
            # mask = mask_from_lens(x_lengths, x.size(2))[..., None]# == 0

            pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)
            pitch_pred = self.expand_pitch_energy(pitch_pred, w_ceil)


            energy_pred = self.energy_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)
            energy_pred = self.expand_pitch_energy(energy_pred, w_ceil)
            energy_pred = torch.log(1.0 + energy_pred)
            energy_pred = energy_pred / 10

            # pitch_pred += 3
            # energy_pred += 3
            print(f'pitch_pred, {pitch_pred}')
            print(f'energy_pred, {energy_pred}')
            # m_p[:,:2,:] *= 0
            m_p[:,0,:] = pitch_pred.squeeze(dim=0)
            m_p[:,1,:] = energy_pred.squeeze(dim=0)
            print(f'm_p, {m_p}')

        else:
            if self.args.pitch:
                pitch_scaling = self.args.pe_scaling

                # Average pitch over characters
                # mask = mask_from_lens(x_lengths, x.size(2))[..., None]# == 0

                # if self.args.pitch_rpct:
                #     pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)
                # else:
                # print(f'mask, {mask}')
                # pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), mask).permute(0, 2, 1)
                pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)

                print(f'pitch_pred 0, {pitch_pred}')
                pitch_pred = self.expand_pitch_energy(pitch_pred, w_ceil)
                print(f'pitch_pred 1, {pitch_pred}')


                # pitch_pred *= 2
                # pitch_pred += -3
                pitch_pred += 20
                # pitch_pred += 10
                pitch_pred = self.pitch_emb(pitch_pred)
                print(f'pitch_pred 2, {pitch_pred}')

                if not self.args.expanded_flow:
                    m_p += pitch_pred * pitch_scaling

            if self.args.energy and not self.args.energy_sp:

                # Average pitch over characters
                # mask = mask_from_lens(x_lengths, x.size(2))[..., None]# == 0

                energy_scaling = self.args.pe_scaling# 0.25


                # energy_pred = self.energy_predictor(x.permute(0, 2, 1), mask).permute(0, 2, 1)
                print(f'x.permute(0, 2, 1), {x.permute(0, 2, 1)}')
                # if self.args.energy_rpct:
                #     energy_pred = self.energy_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)
                # else:
                    # energy_pred = self.energy_predictor(x.permute(0, 2, 1), mask).permute(0, 2, 1)
                energy_pred = self.energy_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)
                print(f'energy_pred 0, {energy_pred}')
                energy_pred = self.expand_pitch_energy(energy_pred, w_ceil)

                print(f'energy_pred 1, {energy_pred}')
                # energy_pred += 1
                print(f'energy_pred 1.5, {energy_pred}')
                # energy_pred = torch.log(1.0 + energy_pred)
                print(f'energy_pred 2, {energy_pred}')

                energy_pred = self.energy_emb(energy_pred)
                print(f'energy_pred 3, {energy_pred}')

                # print(f'm_p, {m_p}')
                # print(f'energy_pred, {energy_pred}')
                # m_p += energy_pred * energy_scaling
                if not self.args.expanded_flow:
                    m_p -= energy_pred * energy_scaling


        self.inference_noise_scale = 0
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale

        if self.args.expanded_flow:
            print(f'expanded_flow z_p 1, {z_p.shape}')
            print(f'expanded_flow pitch_pred, {pitch_pred.shape}')
            z_p = torch.cat([z_p, pitch_pred, energy_pred], dim=1)
            print(f'expanded_flow z_p 2, {z_p.shape}')
        # ================


        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)
        return o

    # def voice_conversion(self, y, y_lengths=None, speaker_cond_src=None, speaker_cond_tgt=None, spk1_emb=None, spk2_emb=None):
    def voice_conversion(self, y, y_lengths=None, spk1_emb=None, spk2_emb=None):

        # spk1_emb = spk1_emb.unsqueeze(0)
        # spk2_emb = spk2_emb.unsqueeze(0)
        # spk1_emb = F.normalize(spk1_emb, dim=1)
        # spk2_emb = F.normalize(spk2_emb, dim=1)

        # g_src = spk1_emb.unsqueeze(-1)
        # g_tgt = spk2_emb.unsqueeze(-1)

        if y_lengths is None:
            y_lengths = self.y_lengths_default

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=spk1_emb)
        # z_hat = z
        z_p = self.flow(z, y_mask, g=spk1_emb)
        z_hat = self.flow(z_p, y_mask, g=spk2_emb, reverse=True)

        o_hat = self.waveform_decoder(z_hat * y_mask, g=spk2_emb)
        return o_hat




    def train_hifi_only(self, x, x_lengths, y, y_lengths, pitch_padded, energy_padded, waveform, aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}):

        outputs = {}
        sid, g, lid = self._set_cond_input(aux_input)
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size)
        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * 256,
            self.spec_segment_size * 256,
        )

        gt_spk_emb, syn_spk_emb = None, None
        outputs.update(
            {
                "mel_pred": None,
                "model_outputs": o,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
            }
        )
        return outputs


    def train_step(self, x, x_lengths, y, y_lengths, pitch_padded, energy_padded, waveform, aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}):


        outputs = {}
        pitch_tgt = None
        pitch_pred = None
        energy_tgt = None
        energy_pred = None

        sid, g, lid = self._set_cond_input(aux_input)

        # language embedding
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        # posterior encoder - slow
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        del y

        # Encode the text inputs
        input_seq = x
        x, x_emb, x_mask = self.text_encoder(input_seq, x_lengths, lang_emb=lang_emb, stats=False)
        x_mask_d = x_mask.detach()
        del x_emb



        # --------------------------
        m_p, logs_p = self.text_encoder(x, x_lengths, lang_emb=lang_emb, stats=True, x_mask=x_mask)


        lang_emb = lang_emb.detach()
        z_p = self.flow(z, y_mask, g=g)

        pitch_flow = None
        energy_flow = None
        if self.args.ow_flow:
            pitch_flow = torch.narrow(z_p, 1, 0, 1)
            energy_flow = torch.narrow(z_p, 1, 1, 1)
            energy_flow = energy_flow * 10

            z_p_new = torch.zeros_like(z_p)
            z_p_new[:,2:,:] += torch.narrow(z_p, 1, 2, 190)
            z_p = z_p_new


        z_p_pitch = None
        z_p_energy = None
        z_p_pitch_pred = None
        z_p_energy_pred = None

        if self.args.pitch and not self.args.ow_flow:
            pitch_pred = self.pitch_emb(pitch_padded) * self.args.pe_scaling # <bs>,192,<560>
            z_p -= pitch_pred
            del pitch_pred

        if self.args.energy and not self.args.ow_flow:
            energy_padded = torch.log(1.0 + energy_padded)
            energy_pred = self.energy_emb(energy_padded) * self.args.pe_scaling * 0.01
            z_p += energy_pred

        # find the alignment path - slower
        attn_mask = torch.unsqueeze(x_mask_d, -1) * torch.unsqueeze(y_mask, 2)
        del y_mask
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p ** 2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4

            del logp1,logp2,logp3,logp4

            # [Original] numpy, 1gpu: ~0.128ms, 2gpu:
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            del logp

            # numba, ~0.147ms
            # logp = logp.cpu().detach().numpy()
            # attn_mask = attn_mask.squeeze(1).cpu().detach().numpy().astype(np.bool)
            # attn = maximum_path_numba(logp, attn_mask)
            # attn = torch.from_numpy(attn).to(device=logp2.device, dtype=logp2.dtype)
            # attn = attn.unsqueeze(1).detach()

            # cupy, 1gpu: ~0.257ms, 2gpu: ~0.63ms
            # with cp.cuda.Device(z_p.get_device()):
            #     attn = maximum_path_cupy(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()


        # duration predictor
        attn_durations = attn.sum(3)

        # x = x.detach()
        loss_duration_pred = self.duration_predictor(
            x.detach() if self.args.detach_dp_input else x,
            # x,
            x_mask,
            attn_durations,
            g=g.detach() if self.args.detach_dp_input and g is not None else g,
            # lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            lang_emb=lang_emb
        )
        loss_duration = loss_duration_pred / torch.sum(x_mask)
        del loss_duration_pred, lang_emb

        outputs["loss_duration"] = loss_duration


        w = attn_durations * x_mask
        del attn_durations, x_mask
        w_ceil = torch.ceil(w).squeeze(dim=1)



        # Encode and condition per-symbol pitch values
        mask = mask_from_lens(x_lengths, x.size(2))[..., None]# == 0

        if self.args.pitch:
            # Average pitch over characters
            with torch.no_grad():
                pitch_tgt = average_pitch(pitch_padded, w_ceil).detach()
                if self.args.ow_flow:
                    pitch_flow = average_pitch(pitch_flow, w_ceil).detach()

            pitch_pred = self.pitch_predictor(x.permute(0, 2, 1).detach(), x_lengths, speaker_emb=g, stats=False)
            # pitch_pred = self.pitch_predictor(x.permute(0, 2, 1), x_lengths, speaker_emb=g, stats=False)

        if self.args.energy:
            # Average energy over characters
            with torch.no_grad():
                energy_tgt = average_pitch(energy_padded, w_ceil).detach()
                energy_tgt = torch.log(1.0 + energy_tgt)

                if self.args.ow_flow:
                    energy_flow = average_pitch(energy_flow, w_ceil).detach()

            energy_pred = self.energy_predictor(x.permute(0, 2, 1).detach(), x_lengths, speaker_emb=g, stats=False)
            if self.args.ow_flow:
                energy_pred = torch.log(1.0 + energy_pred)
                energy_tgt = torch.log(1.0 + energy_tgt)

        del x

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])
        del attn

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size)
        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * 256,
            self.spec_segment_size * 256,
        )

        del y_lengths, z_slice, waveform
        # gt_spk_emb, syn_spk_emb = None, None
        outputs.update(
            {
                "mel_pred": None,
                "model_outputs": o,
                # "alignments": attn.squeeze(1),
                "z": z,
                "z_p": z_p,
                "m_p": m_p,
                "logs_p": logs_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                # "gt_spk_emb": gt_spk_emb,
                # "syn_spk_emb": syn_spk_emb,

                "pitch_tgt": pitch_tgt,
                "pitch_pred": pitch_pred,
                "energy_tgt": energy_tgt,
                "energy_pred": energy_pred,
                "mask": mask,
                # "y_mask": y_mask,
                "pitch_flow": pitch_flow,
                "energy_flow": energy_flow,
                "z_p_pitch_pred": z_p_pitch_pred,
                "z_p_energy_pred": z_p_energy_pred,
                "z_p_pitch": z_p_pitch,
                "z_p_energy": z_p_energy,
            }
        )
        return outputs





    def _set_cond_input (self, aux_input):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid = None, None, None
        # if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
        #     sid = aux_input["speaker_ids"]
        #     if sid.ndim == 0:
        #         sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        return sid, g, lid


    # Opposite of average_pitch; Repeat per-symbol values by durations, to get sequence-wide values
    def expand_pitch_energy (self, vals, durations, logger=None):

        vals = vals.view((vals.shape[0], vals.shape[2]))

        # max_dur = int(torch.ceil(durations).sum().item())
        max_dur = int(torch.round(durations).sum().item())
        # max_dur = int(durations.sum().item())
        # logger.info(f'max_dur: {int(durations.sum().item())}')
        # logger.info(f'max_dur: {max_dur}')

        if len(durations.shape)>2:
            durations = durations.view((durations.shape[0], durations.shape[2]))
            # expanded = torch.zeros((1,vals.shape[0], int(durations.sum(1).item()))).to(vals)
            # expanded = torch.zeros((1,vals.shape[0], math.ceil(durations.sum(1).item()))).to(vals)
            expanded = torch.zeros((1,vals.shape[0], max_dur)).to(vals)
        else:
            # expanded = torch.zeros((vals.shape[0], 1, int(durations.sum(0).item()))).to(vals)
            # expanded = torch.zeros((vals.shape[0], 1, math.ceil(durations.sum(0).item()))).to(vals)
            expanded = torch.zeros((vals.shape[0], 1, max_dur)).to(vals)

        # logger.info(f'vals: {vals.shape}')
        # logger.info(f'expanded: {expanded.shape}')
        # logger.info(f'durations: {durations}')
        # logger.info(f'durations: {durations.sum()}')

        for b in range(vals.shape[0]):
            b_vals = vals[b]
            b_durs = durations[b]
            # b_durs = torch.ceil(b_durs)
            expanded_vals = []

            for vi in range(b_vals.shape[0]):
                # for dur_i in range(math.ceil(b_durs[vi].item())):
                for dur_i in range(round(b_durs[vi].item())):
                    if len(durations.shape)>2:
                        expanded_vals.append(b_vals[vi])
                        # expanded_vals.append(int(b_vals[vi].item()))
                    else:
                        expanded_vals.append(b_vals[vi].unsqueeze(dim=0))
                        # expanded_vals.append(torch.floor(b_vals[vi]).unsqueeze(dim=0))

            expanded[b,:,:] += torch.tensor(expanded_vals).to(expanded)
        return expanded

    # def expand_pitch_energy (self, vals, durations, logger=None):

    #     vals = vals.view((vals.shape[0], vals.shape[2]))
    #     if len(durations.shape)>2:
    #         durations = durations.view((durations.shape[0], durations.shape[2]))
    #         expanded = torch.zeros((1,vals.shape[0], int(durations.sum(1).item()))).to(vals)
    #     else:
    #         expanded = torch.zeros((vals.shape[0], 1, int(durations.sum(0).item()))).to(vals)

    #     for b in range(vals.shape[0]):
    #         b_vals = vals[b]
    #         b_durs = durations[b]
    #         expanded_vals = []

    #         for vi in range(b_vals.shape[0]):
    #             for dur_i in range(int(b_durs[vi].item())):
    #                 if len(durations.shape)>2:
    #                     expanded_vals.append(b_vals[vi])
    #                 else:
    #                     expanded_vals.append(b_vals[vi].unsqueeze(dim=0))

    #         expanded[b,:,:] += torch.tensor(expanded_vals).to(expanded)
    #     return expanded


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

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg












class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int, # len(ALL_SYMBOLS)
        out_channels: int, # 192
        hidden_channels: int, # 192
        hidden_channels_ffn: int, # 768
        num_heads: int, # 2
        num_layers: int, # 10
        kernel_size: int, # 3
        dropout_p: float, # 0.1
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None, stats=False, x_mask=None, lang_emb_full=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """

        if stats:
            stats = self.proj(x) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)
            return m, logs
        else:
            x_emb = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

            # concat the lang emb in embedding chars
            if lang_emb is not None or lang_emb_full is not None:
                # x = torch.cat((x_emb, lang_emb.transpose(2, 1).expand(x_emb.size(0), x_emb.size(1), -1)), dim=-1)
                if lang_emb_full is None:
                    lang_emb_full = lang_emb.transpose(2, 1).expand(x_emb.size(0), x_emb.size(1), -1)
                x = torch.cat((x_emb, lang_emb_full), dim=-1)

            x = torch.transpose(x, 1, -1)  # [b, h, t]
            x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

            x = self.encoder(x * x_mask, x_mask)
            # stats = self.proj(x) * x_mask

            # m, logs = torch.split(stats, self.out_channels, dim=1)
            return x, x_emb, x_mask


        # x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # # concat the lang emb in embedding chars
        # if lang_emb is not None:
        #     x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        # x = torch.transpose(x, 1, -1)  # [b, h, t]
        # x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # x = self.encoder(x * x_mask, x_mask)
        # stats = self.proj(x) * x_mask

        # m, logs = torch.split(stats, self.out_channels, dim=1)
        # return x, m, logs, x_mask

# class PitchEnergyEncoder(nn.Module):
#     def __init__(
#         self,
#         # n_vocab: int, # len(ALL_SYMBOLS)
#         out_channels: int, # 192
#         hidden_channels: int, # 192
#         hidden_channels_ffn: int, # 768
#         num_heads: int, # 2
#         num_layers: int, # 10
#         kernel_size: int, # 3
#         dropout_p: float, # 0.1
#         conditioning_emb_dim: int = None,
#     ):
#         super().__init__()
#         self.out_channels = out_channels
#         self.hidden_channels = hidden_channels

#         # self.emb = nn.Embedding(n_vocab, hidden_channels)

#         # nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

#         if conditioning_emb_dim:
#             hidden_channels += conditioning_emb_dim

#         self.encoder = RelativePositionTransformer(
#             in_channels=hidden_channels,
#             out_channels=hidden_channels,
#             # out_channels=196,
#             hidden_channels=hidden_channels,
#             hidden_channels_ffn=hidden_channels_ffn,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             kernel_size=kernel_size,
#             dropout_p=dropout_p,
#             layer_norm_type="2",
#             rel_attn_window_size=4,
#         )

#         self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
#         # self.proj = nn.Conv1d(196, out_channels * 2, 1)

#     def forward(self, x, x_lengths, speaker_emb=None, stats=False, x_mask=None):
#         """
#         Shapes:
#             - x: :math:`[B, T]`
#             - x_length: :math:`[B]`
#         """

#         if stats:
#             stats = self.proj(x) * x_mask
#             m, logs = torch.split(stats, self.out_channels, dim=1)
#             return m, logs
#         else:
#             # x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

#             # concat the lang emb in embedding chars
#             if speaker_emb is not None:
#                 # print(f'x 1, {x.shape}')
#                 # print(f'speaker_emb, {speaker_emb.shape}')
#                 # print(f'speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1), {speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1).shape}')
#                 # print(f'speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), {speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1).shape}')
#                 x = torch.cat((x, speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
#                 # print(f'x 2, {x.shape}')

#             # x = x + x_emb
#             x = torch.transpose(x, 1, -1)  # [b, h, t]

#             # x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

#             # print(f'self.encoder, {self.encoder}')

#             # print(f'x 3, {x.shape}')
#             # print(f'x_mask, {x_mask.shape}')
#             x = self.encoder(x * x_mask, x_mask)
#             # print(f'x 4, {x.shape}')
#             # stats = self.proj(x) * x_mask

#             # m, logs = torch.split(stats, self.out_channels, dim=1)
#             return x#, x_mask

class RelativePositioningPitchEnergyEncoder(nn.Module):
    def __init__(
        self,
        # n_vocab: int, # len(ALL_SYMBOLS)
        out_channels: int, # 192
        hidden_channels: int, # 192
        hidden_channels_ffn: int, # 768
        num_heads: int, # 2
        num_layers: int, # 10
        kernel_size: int, # 3
        dropout_p: float, # 0.1
        conditioning_emb_dim: int = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # self.emb = nn.Embedding(n_vocab, hidden_channels)

        # nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if conditioning_emb_dim:
            hidden_channels += conditioning_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            # out_channels=hidden_channels,
            out_channels=1,
            # out_channels=196,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        # self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # self.proj = nn.Conv1d(196, out_channels * 2, 1)

    def forward(self, x, x_lengths=None, speaker_emb=None, stats=False, x_mask=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """

        # if stats:
        #     stats = self.proj(x) * x_mask
        #     m, logs = torch.split(stats, self.out_channels, dim=1)
        #     return m, logs
        # else:
        # x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if speaker_emb is not None:
            # print(f'x 1, {x.shape}')
            # print(f'speaker_emb, {speaker_emb.shape}')
            # print(f'speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1), {speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1).shape}')
            # print(f'speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), {speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1).shape}')
            x = torch.cat((x, speaker_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
            # print(f'x 2, {x.shape}')

        # print(f'rpct x 0', x, x.shape)

        # x = x + x_emb
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # print(f'rpct x 1', x, x.shape)

        # print(f'x_mask, {x_mask.shape}')
        # print(f'x_mask, {torch.transpose(x_mask, 1, -1).shape}')
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        # print(f'rpct x_mask', x_mask, x_mask.shape)

        # print(f'self.encoder, {self.encoder}')

        # print(f'x 3, {x.shape}')
        # print(f'new x_mask, {x_mask.shape}')
        x = self.encoder(x * x_mask, x_mask)
        # print(f'rpct x 2', x, x.shape)              #   This is zeros on init?
        # print(f'x 4, {x.shape}')
        # stats = self.proj(x) * x_mask
        # fddgd()

        # m, logs = torch.split(stats, self.out_channels, dim=1)
        return x#, x_mask


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
        args=None
    ):
        """Redisual Coupling blocks for VITS flow layers.

        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.args = args
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for flow_i in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    (192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else channels,
                    (192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    out_channels_override=(192+self.args.expanded_flow_dim+self.args.expanded_flow_dim) if flow_i==(num_flows-1) and self.args.expanded_flow else None,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for fi, flow in enumerate(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        out_channels_override=None,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)


        self.conv1d_projector = None
        if out_channels_override:
            self.conv1d_projector = nn.Conv1d(192, out_channels_override, 1)


    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if self.conv1d_projector is not None and not reverse:
            x = self.conv1d_projector(x)

        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x



class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x, x_hat=None):
        """
        Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats



def mask_from_lens(lens, max_len= None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask
def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    # print(f'durs_cums_ends, {durs_cums_ends.shape}')
    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
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
class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)

import os

import torch
import traceback

class ModelsManager(object):

    def __init__(self, logger, PROD, device="cpu"):
        super(ModelsManager, self).__init__()

        self.models_bank = {}
        self.logger = logger
        self.PROD = PROD
        self.device_label = device
        self.device = torch.device(device)

    def init_model (self, model_key, instance_index=0):
        model_key = model_key.lower()
        try:
            if model_key in list(self.models_bank.keys()) and instance_index in self.models_bank[model_key].keys() and self.models_bank[model_key][instance_index].isReady:
                return
            self.logger.info(f'ModelsManager: Initializing model: {model_key}')

            if model_key=="resemblyzer":
                from python.resemblyzer.model import ResemblyzerModel
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = ResemblyzerModel(self.logger, self.PROD, self.device, self)

            elif model_key=="hifigan":
                from python.hifigan.model import HiFi_GAN
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = HiFi_GAN(self.logger, self.PROD, self.device, self)

            elif model_key=="big_waveglow":
                from python.big_waveglow.model import BIG_WaveGlow
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = BIG_WaveGlow(self.logger, self.PROD, self.device, self)

            elif model_key=="256_waveglow":
                from python.waveglow.model import WaveGlow
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = WaveGlow(self.logger, self.PROD, self.device, self)

            elif model_key=="fastpitch":
                from python.fastpitch.model import FastPitch
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = FastPitch(self.logger, self.PROD, self.device, self)

            elif model_key=="fastpitch1_1":
                from python.fastpitch1_1.model import FastPitch1_1
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = FastPitch1_1(self.logger, self.PROD, self.device, self)

            elif model_key=="xvapitch":
                from python.xvapitch.model import xVAPitch
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = xVAPitch(self.logger, self.PROD, self.device, self)

            elif model_key=="s2s_fastpitch1_1":
                from python.fastpitch1_1.model import FastPitch1_1 as S2S_FastPitch1_1
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = S2S_FastPitch1_1(self.logger, self.PROD, self.device, self)

            elif model_key=="wav2vec2":
                from python.wav2vec2.model import Wav2Vec2
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = Wav2Vec2(self.logger, self.PROD, self.device, self)

            elif model_key=="speaker_rep":
                from python.xvapitch.speaker_rep.model import ResNetSpeakerEncoder
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = ResNetSpeakerEncoder(self.logger, self.PROD, self.device, self)

            elif model_key=="nuwave2":
                from python.nuwave2.model import Nuwave2Model
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = Nuwave2Model(self.logger, self.PROD, self.device, self)

            else:
                raise(f'Model not recognized: {model_key}')

            try:
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index].model = self.models_bank[model_key][instance_index].model.to(self.device)
            except:
                pass
            try:
                if model_key not in self.models_bank.keys():
                    self.models_bank[model_key] = {}
                self.models_bank[model_key][instance_index] = self.models_bank[model_key][instance_index].to(self.device)
            except:
                pass
        except:
            self.logger.info(traceback.format_exc())

    def load_model (self, model_key, ckpt_path, instance_index=0, **kwargs):

        if model_key not in self.models_bank.keys() or instance_index not in self.models_bank[model_key].keys():
            self.init_model(model_key, instance_index)

        if not os.path.exists(ckpt_path):
            return "ENOENT"

        if self.models_bank[model_key][instance_index].ckpt_path != ckpt_path:
            self.logger.info(f'ModelsManager: Loading model checkpoint: {model_key}, {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location="cpu")
            try:
                self.models_bank[model_key][instance_index].load_checkpoint(ckpt_path, ckpt, **kwargs)
            except:
                self.models_bank[model_key][instance_index].load_state_dict(ckpt_path, ckpt, **kwargs)

    def set_device (self, device, instance_index=0):
        if device=="gpu":
            device = "cuda"
        if self.device_label==device:
            return
        self.device_label = device
        self.device = torch.device(device)
        self.logger.info(f'ModelsManager: Changing device to: {device}')
        for model_key in list(self.models_bank.keys()):
            self.models_bank[model_key][instance_index].set_device(self.device)

    def models (self, key, instance_index=0):
        if key.lower() not in self.models_bank.keys() or instance_index not in self.models_bank[key.lower()].keys():
            self.init_model(key.lower(), instance_index=instance_index)
        return self.models_bank[key.lower()][instance_index]

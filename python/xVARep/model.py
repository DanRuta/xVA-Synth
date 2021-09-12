import os
from collections import OrderedDict

import torch
import torch.nn as nn

import librosa
import numpy as np
import pickle


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from resemblyzer import VoiceEncoder, preprocess_wav



def extract_features(encoder, file_name):
    wav = preprocess_wav(file_name)
    embed = encoder.embed_utterance(wav)
    return embed


class xVARep(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(xVARep, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.path = "./resources/app" if PROD else "."
        self.ckpt_path = None
        self.embeddings = []

        # self.logger.info("xVARep device")
        # self.logger.info(device)
        # self.logger.info(device=="cpu")
        # self.logger.info(str(device)=="cpu")
        self.model = VoiceEncoder(device)

        # layers = []

        # layers.append(("fc1", nn.Linear(193, 256)))
        # layers.append(("relu1", nn.LeakyReLU()))
        # layers.append(("d1", nn.Dropout(p=0.1)))

        # layers.append(("fc2", nn.Linear(256, 256)))

        # self.model = nn.Sequential(OrderedDict(layers))
        # self.model = self.model.to(self.device)

        # with open(f'{self.path}/python/xVARep/norm_stats.txt') as f:
        #     mean, std = f.read().split("\n")
        #     self.norm_stats = {}
        #     self.norm_stats["mean"] = [float(num) for num in mean.split(",")]
        #     self.norm_stats["std"] = [float(num) for num in std.split(",")]

        self.isReady = True


        if os.path.exists(f'{self.path}/python/xVARep/embs.pkl'):
            with open(f'{self.path}/python/xVARep/embs.pkl', "rb") as pklFile:
                self.embeddings = pickle.load(pklFile)
        else:
            self.embeddings = {}


    def load_state_dict (self, ckpt_path, sd):
        self.ckpt_path = ckpt_path


    # Get all the preview audios, and generate embeddings for them
    def compile_emb_bank (self, mappings, includeAllVoices, onlyInstalled):

        embeddings = {}

        voiceIds = []
        sampleWAVs = []
        voiceNames = []
        voiceGenders = []
        gameIds = []

        non_installed_voices = []

        for mapping in mappings.split("\n"):
            sampleWAV = mapping.split("=")[1]

            if len(sampleWAV):
                voiceIds.append(mapping.split("=")[0])
                sampleWAVs.append(sampleWAV)
                voiceNames.append(mapping.split("=")[2])
                voiceGenders.append(mapping.split("=")[3])
                gameIds.append(mapping.split("=")[4])
            else:
                if not onlyInstalled:
                    non_installed_voices.append({"voiceId": mapping.split("=")[0], "voiceName": mapping.split("=")[2], "gender": mapping.split("=")[3], "gameId": mapping.split("=")[4]})


        # Prepare audio data
        audio_feats_batch = {}
        todo_keys = []
        todo_names = []
        todo_genders = []
        todo_gameIDs = []

        for api, audio_path in enumerate(sampleWAVs):
            if voiceIds[api] not in self.embeddings.keys():
                audio_feats = extract_features(self.model, audio_path)
                audio_feats_batch[voiceIds[api]] = audio_feats

                todo_keys.append(voiceIds[api])
                todo_names.append(voiceNames[api])
                todo_genders.append(voiceGenders[api])
                todo_gameIDs.append(gameIds[api])


        todo_feats = [audio_feats_batch[key] for key in audio_feats_batch.keys()]
        self.logger.log(f'todo_keys, {todo_keys}')



        # Include all the embeddings for the installed voices with a preview audio path
        for api, audio_path in enumerate(sampleWAVs):
            if voiceIds[api] in self.embeddings.keys():
                embeddings[voiceIds[api]] = {}
                embeddings[voiceIds[api]]["emb"] = self.embeddings[voiceIds[api]]["emb"]
                embeddings[voiceIds[api]]["name"] = self.embeddings[voiceIds[api]]["name"]
                embeddings[voiceIds[api]]["gender"] = self.embeddings[voiceIds[api]]["gender"]
                embeddings[voiceIds[api]]["gameId"] = self.embeddings[voiceIds[api]]["gameId"]

        # Compute embedding using the model, for the installed voices without an embedding
        if len(todo_feats):
            embs = todo_feats
            for ei, emb in enumerate(embs):
                voiceId = todo_keys[ei]

                embeddings[voiceId] = {"emb": emb, "name": todo_names[ei], "gender": todo_genders[ei], "gameId": todo_gameIDs[ei]}
                self.embeddings[voiceId] = {"emb": emb, "name": todo_names[ei], "gender": todo_genders[ei], "gameId": todo_gameIDs[ei]}

        # Include the embeddings for the non-installed voices
        if includeAllVoices:
            for voiceId in list(self.embeddings.keys()):
                if voiceId not in embeddings.keys():
                    if voiceId in embeddings.keys():
                        self.logger.log(f'===== CONFLICT 2: {voiceId}')
                    embeddings[voiceId] = {}
                    embeddings[voiceId]["emb"] = self.embeddings[voiceId]["emb"]
                    embeddings[voiceId]["name"] = self.embeddings[voiceId]["name"]
                    embeddings[voiceId]["gender"] = self.embeddings[voiceId]["gender"]
                    embeddings[voiceId]["gameId"] = self.embeddings[voiceId]["gameId"]

        if not onlyInstalled:
            for voice in non_installed_voices:
                if voiceId in embeddings.keys():
                    self.logger.log(f'===== CONFLICT 3: {voice["voiceId"]}')
                embeddings[voice["voiceId"]] = {}
                embeddings[voice["voiceId"]]["emb"] = self.embeddings[voice["voiceId"]]["emb"]
                embeddings[voice["voiceId"]]["name"] = voice["voiceName"]
                embeddings[voice["voiceId"]]["gender"] = voice["gender"]
                embeddings[voice["voiceId"]]["gameId"] = voice["gameId"]

        with open(f'{self.path}/python/xVARep/embs.pkl', "wb") as pklFile:
            pickle.dump(self.embeddings, pklFile)

        return embeddings



    def reduce_data_dimension (self, mappings, includeAllVoices, onlyInstalled, algorithm):

        embeddings = self.compile_emb_bank(mappings, includeAllVoices, onlyInstalled)

        tsne_input_data = [embeddings[voiceId]["emb"] for voiceId in embeddings.keys()]

        if algorithm=="tsne":
            reduced_data = TSNE(n_components=3, random_state=0, perplexity=30).fit_transform(np.array(tsne_input_data))
        else: # pca
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(tsne_input_data)


        string_formatted_dict = []
        for vi, voiceId in enumerate(embeddings.keys()):
            formatted_string = f'{voiceId}={embeddings[voiceId]["name"]}={embeddings[voiceId]["gender"]}={embeddings[voiceId]["gameId"]}='
            formatted_string += ",".join([str(val*(500 if algorithm=="pca" else 1)) for val in reduced_data[vi]])
            string_formatted_dict.append(formatted_string)

        return "\n".join(string_formatted_dict)

    def set_device (self, device):
        self.device = device
        self.model = VoiceEncoder(device)

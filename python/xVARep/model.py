import os
from collections import OrderedDict

import torch
import torch.nn as nn

import librosa
import numpy as np
import pickle


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# https://towardsdatascience.com/how-to-build-a-neural-network-for-voice-classification-5e2810fe1efa
def extract_features(file_name):

    # Sets the name to be the path to where the file is in my computer
    # file_name = os.path.join(os.path.abspath('30_speakers_train')+'/'+str(files.file))
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    features = []
    for v in mfccs:
        features.append(v)
    for v in chroma:
        features.append(v)
    for v in mel:
        features.append(v)
    for v in contrast:
        features.append(v)
    for v in tonnetz:
        features.append(v)

    return features


import multiprocessing as mp
def mp_preprocess_extract_features(logger, norm_stats, sampleWAVs):
    workers = max(1, mp.cpu_count()-1)
    workers = min(len(sampleWAVs), workers)

    workItems = [[norm_stats, sampleWAV] for si, sampleWAV in enumerate(sampleWAVs)]

    logger.info("[mp embeddings feature extraction] workers: "+str(workers))

    pool = mp.Pool(workers)
    results = pool.map(processingTask, workItems)
    pool.close()
    pool.join()


    return results

def processingTask(data):
    norm_stats = data[0]
    wavPath = data[1]

    audio_feats = extract_features(wavPath)
    audio_feats = [(val-norm_stats["mean"][vi]) / norm_stats["std"][vi] for vi,val in enumerate(audio_feats)]
    return audio_feats


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

        layers = []

        layers.append(("fc1", nn.Linear(193, 256)))
        layers.append(("relu1", nn.LeakyReLU()))
        layers.append(("d1", nn.Dropout(p=0.1)))

        layers.append(("fc2", nn.Linear(256, 256)))

        self.model = nn.Sequential(OrderedDict(layers))
        self.model = self.model.to(self.device)

        with open(f'{self.path}/python/xVARep/norm_stats.txt') as f:
            mean, std = f.read().split("\n")
            self.norm_stats = {}
            self.norm_stats["mean"] = [float(num) for num in mean.split(",")]
            self.norm_stats["std"] = [float(num) for num in std.split(",")]

        self.isReady = True


        if os.path.exists(f'{self.path}/python/xVARep/embs.pkl'):
            with open(f'{self.path}/python/xVARep/embs.pkl', "rb") as pklFile:
                self.embeddings = pickle.load(pklFile)
        else:
            self.embeddings = {}

        # TODO, make sure the cache is invalidated if the .wav file is changed, detect that somehow
        self.cached_audio_feats = {}




    def load_state_dict (self, ckpt_path, sd):
        self.ckpt_path = ckpt_path
        self.model.load_state_dict(sd["state_dict"])


    def forward (self, data):
        data = torch.tensor(data).to(self.device).float()
        emb = self.model(data)
        emb = emb / emb.norm(dim=1)[:, None]
        del data
        return emb




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

        DO_MP = True
        if DO_MP and includeAllVoices:
            files_to_extract = {}

            for api, audio_path in enumerate(sampleWAVs):

                if voiceIds[api] in self.cached_audio_feats.keys():
                    audio_feats = self.cached_audio_feats[voiceIds[api]]
                    audio_feats_batch[voiceIds[api]] = audio_feats
                else:
                    files_to_extract[voiceIds[api]] = audio_path


            voiceIds_to_extract = list(files_to_extract.keys())
            paths_to_extract = [files_to_extract[voiceIds] for voiceIds in voiceIds_to_extract]


            if len(paths_to_extract):
                extracted_feats = mp_preprocess_extract_features(self.logger, self.norm_stats, paths_to_extract)

                for fi, audio_feats in enumerate(extracted_feats):
                    voiceId = voiceIds_to_extract[fi]
                    if voiceId not in self.cached_audio_feats.keys():
                        self.cached_audio_feats[voiceId] = audio_feats
                    audio_feats_batch[voiceId] = audio_feats

        else:
            for api, audio_path in enumerate(sampleWAVs):
                if voiceIds[api] in self.cached_audio_feats.keys():
                    audio_feats = self.cached_audio_feats[voiceIds[api]]
                else:
                    audio_feats = extract_features(audio_path)
                    audio_feats = [(val-self.norm_stats["mean"][vi]) / self.norm_stats["std"][vi] for vi,val in enumerate(audio_feats)]
                    self.cached_audio_feats[voiceIds[api]] = audio_feats
                audio_feats_batch[voiceIds[api]] = audio_feats

        audio_feats_batch = [audio_feats_batch[key] for key in voiceIds]




        # Compute embedding using the model
        embs = self.forward(audio_feats_batch)
        embs = list(embs.cpu().detach().numpy())
        for ei, emb in enumerate(embs):
            embeddings[voiceIds[ei]] = {"emb": emb, "name": voiceNames[ei], "gender": voiceGenders[ei], "gameId": gameIds[ei]}
            self.embeddings[voiceIds[ei]] = {"emb": emb, "name": voiceNames[ei], "gender": voiceGenders[ei], "gameId": gameIds[ei]}



        if includeAllVoices:
            for voiceId in list(self.embeddings.keys()):
                if voiceId not in embeddings.keys():
                    embeddings[voiceId] = {}
                    embeddings[voiceId]["emb"] = self.embeddings[voiceId]["emb"]
                    embeddings[voiceId]["name"] = self.embeddings[voiceId]["name"]
                    embeddings[voiceId]["gender"] = self.embeddings[voiceId]["gender"]
                    embeddings[voiceId]["gameId"] = self.embeddings[voiceId]["gameId"]
                    voiceIds.append(voiceId)
                    voiceNames.append(self.embeddings[voiceId]["name"])
                    voiceGenders.append(self.embeddings[voiceId]["gender"])
                    gameIds.append(self.embeddings[voiceId]["gameId"])

        if not onlyInstalled:
            for voice in non_installed_voices:
                embeddings[voice["voiceId"]] = {}
                embeddings[voice["voiceId"]]["emb"] = self.embeddings[voice["voiceId"]]["emb"]
                embeddings[voice["voiceId"]]["name"] = voice["voiceName"]
                embeddings[voice["voiceId"]]["gender"] = voice["gender"]
                embeddings[voice["voiceId"]]["gameId"] = voice["gameId"]
                voiceIds.append(voice["voiceId"])
                voiceNames.append(voice["voiceName"])
                voiceGenders.append(voice["gender"])
                gameIds.append(voice["gameId"])


        with open(f'{self.path}/python/xVARep/embs.pkl', "wb") as pklFile:
            pickle.dump(self.embeddings, pklFile)

        return voiceIds, voiceNames, voiceGenders, gameIds, embeddings



    def reduce_data_dimension (self, mappings, includeAllVoices, onlyInstalled, algorithm):

        voiceIds, voiceNames, voiceGenders, gameIds, embeddings = self.compile_emb_bank(mappings, includeAllVoices, onlyInstalled)

        tsne_input_data = [embeddings[voiceId]["emb"] for voiceId in voiceIds]

        if algorithm=="tsne":
            reduced_data = TSNE(n_components=3, random_state=0, perplexity=30).fit_transform(np.array(tsne_input_data))
        else: # pca
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(tsne_input_data)


        string_formatted_dict = []
        for vi, voiceId in enumerate(voiceIds):
            formatted_string = f'{voiceId}={voiceNames[vi]}={voiceGenders[vi]}={gameIds[vi]}='
            formatted_string += ",".join([str(val*(200 if algorithm=="pca" else 1)) for val in reduced_data[vi]])
            string_formatted_dict.append(formatted_string)

        return "\n".join(string_formatted_dict)

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device


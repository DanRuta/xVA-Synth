import ffmpeg
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch

class Wav2Vec2(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(Wav2Vec2, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        torch.backends.cudnn.benchmark = True

        self.init_model("english_basic")
        self.isReady = True


    def init_model (self, symbols_alphabet):
        self.processor = Wav2Vec2Processor.from_pretrained(f'{"./resources/app" if self.PROD else "."}/python/wav2vec2', local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(f'{"./resources/app" if self.PROD else "."}/python/wav2vec2', local_files_only=True)
        self.model.eval()

    def load_state_dict (self, ckpt_path, ckpt, n_speakers):
        self.ckpt_path = ckpt_path


    def infer (self, PROD, audiopath):

        ffmpeg_path = f'{"./resources/app" if PROD else "."}/python/ffmpeg.exe'

        stream = ffmpeg.input(audiopath)
        ffmpeg_options = {"ar": "16000"}
        stream = ffmpeg.output(stream, audiopath.replace(".wav", "_16khz.wav"), **ffmpeg_options)
        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
        audio_input, sample_rate = sf.read(audiopath.replace(".wav", "_16khz.wav"))

        # Tokenize
        input_values = self.processor(audio_input, sample_rate=sample_rate, return_tensors="pt", padding="longest").input_values
        # Retrieve logits
        logits = self.model(input_values.to(self.device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0]).lower()
        self.logger.info(f'transcription: {transcription}')
        return transcription

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)

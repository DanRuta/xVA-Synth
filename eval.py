import io
import os
import re
import librosa
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper

import numpy as np
from python import helpers, modules, rnn_wrappers
from scipy import signal

class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
    '''Initializes the model for inference.

    Sets "mel_outputs", "linear_outputs", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
    '''
    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.get_variable(
        'embedding', [149, hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)          # [N, T_in, embed_depth=256]

      # Encoder
      prenet_outputs = modules.prenet(embedded_inputs, is_training, hp.prenet_depths)    # [N, T_in, prenet_depths[-1]=128]
      encoder_outputs = modules.encoder_cbhg(prenet_outputs, input_lengths, is_training, # [N, T_in, encoder_depth=256]
                                     hp.encoder_depth)

      # Attention
      attention_cell = AttentionWrapper(
        rnn_wrappers.DecoderPrenetWrapper(GRUCell(hp.attention_depth), is_training, hp.prenet_depths),
        BahdanauAttention(hp.attention_depth, encoder_outputs),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, attention_depth=256]

      # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
      concat_cell = rnn_wrappers.ConcatOutputAndAttentionWrapper(attention_cell)              # [N, T_in, 2*attention_depth=512]

      # Decoder (layers specified bottom to top):
      decoder_cell = MultiRNNCell([
          OutputProjectionWrapper(concat_cell, hp.decoder_depth),
          ResidualWrapper(GRUCell(hp.decoder_depth)),
          ResidualWrapper(GRUCell(hp.decoder_depth))
        ], state_is_tuple=True)                                                  # [N, T_in, decoder_depth=256]

      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      helper = helpers.TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

      (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell, helper, decoder_init_state),
        maximum_iterations=hp.max_iters)                                         # [N, T_out/r, M*r]

      # Reshape outputs to be one output per entry
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels])   # [N, T_out, M]

      # Add post-processing CBHG:
      post_outputs = modules.post_cbhg(mel_outputs, hp.num_mels, is_training,            # [N, T_out, postnet_depth=256]
                               hp.postnet_depth)
      linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)                # [N, T_out, F]

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = np.power(10.0, threshold_db * 0.05)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''

  x = (tf.clip_by_value(spectrogram, 0, 1) * -hparams.min_level_db) + hparams.min_level_db + hparams.ref_level_db
  S = tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.pow(S, hparams.power)
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      n_fft, hop_length, win_length = _stft_parameters()
      est = tf.contrib.signal.stft(y, win_length, hop_length, n_fft, pad_end=False)

      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)

def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length

def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def hparams_debug_string():
  global hparams
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    tf.reset_default_graph()

    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

    with tf.variable_scope('model') as scope:
      self.model = Tacotron(hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    seq = text
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = signal.lfilter([1], [1, -hparams.preemphasis], wav)
    wav = wav[:find_endpoint(wav)]
    out = io.BytesIO()

    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(out, wav.astype(np.int16), hparams.sample_rate)

    return out.getvalue()



def loadModel(model, outputs, cmudict):
  global hparams
  print("loadModel")
  hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',

    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=20000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    outputs_per_step=outputs,
    embed_depth=256,
    prenet_depths=[256, 128],
    encoder_depth=256,
    postnet_depth=256,
    attention_depth=256,
    decoder_depth=256,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    use_cmudict=cmudict,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Eval:
    max_iters=200,
    griffin_lim_iters=60,
    power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
  )
  # print(hparams_debug_string())
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  synth = Synthesizer()
  synth.load(model)
  return synth

def syntesize(model, sequence, outfile):
  print("syntesize")
  with open(outfile, "wb") as f:
    f.write(model.synthesize(sequence))
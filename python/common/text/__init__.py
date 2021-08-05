""" from https://github.com/keithito/tacotron """
import re
from python.common.text import cleaners
from python.common.text.symbols import get_symbols

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, symbol_alphabet, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(symbol_alphabet, _clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(symbol_alphabet, _clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(symbol_alphabet, m.group(2))
    text = m.group(3)

  return sequence


def sequence_to_text(symbol_alphabet, sequence):

  _id_to_symbol = {i: s for i, s in enumerate(get_symbols(symbol_alphabet))}

  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbol_alphabet, symbols):
  _symbol_to_id = {s: i for i, s in enumerate(get_symbols(symbol_alphabet))}
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(_symbol_to_id, s)]


def _arpabet_to_sequence(symbol_alphabet, text):
  return _symbols_to_sequence(symbol_alphabet, ['@' + s for s in text.split()])


def _should_keep_symbol(_symbol_to_id, s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

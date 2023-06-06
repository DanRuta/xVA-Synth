
import os
import re
import sys
import json
import codecs
import glob
from unidecode import unidecode
# from g2pc import G2pC

from h2p_parser.h2p import H2p
from num2words import num2words
import pykakasi
import epitran
# https://www.lexilogos.com/keyboard/pinyin_conversion.htm
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# I really need to find a better way to do this (handling many different possible entry points)
try:
    sys.path.append(".")
    from resources.app.python.xvapitch.text.ipa_to_xvaarpabet import ESpeak, ipa2xvaarpabet, PUNCTUATION, ALL_SYMBOLS, PIN_YIN_ENDS, pinyin_to_arpabet_mappings, text_pinyin_to_pinyin_symbs, manual_phone_replacements
    from resources.app.python.xvapitch.text.en_numbers import normalize_numbers as en_normalize_numbers
    from resources.app.python.xvapitch.text.ro_numbers import generateWords as ro_generateWords
except:
    try:
        from python.xvapitch.text.ipa_to_xvaarpabet import ESpeak, ipa2xvaarpabet, PUNCTUATION, ALL_SYMBOLS, PIN_YIN_ENDS, pinyin_to_arpabet_mappings, text_pinyin_to_pinyin_symbs, manual_phone_replacements
        from python.xvapitch.text.en_numbers import normalize_numbers as en_normalize_numbers
        from python.xvapitch.text.ro_numbers import generateWords as ro_generateWords
    except:
        try:
            from text.ipa_to_xvaarpabet import ESpeak, ipa2xvaarpabet, PUNCTUATION, ALL_SYMBOLS, PIN_YIN_ENDS, pinyin_to_arpabet_mappings, text_pinyin_to_pinyin_symbs, manual_phone_replacements
            from text.en_numbers import normalize_numbers as en_normalize_numbers
            from text.ro_numbers import generateWords as ro_generateWords
        except:
            from ipa_to_xvaarpabet import ESpeak, ipa2xvaarpabet, PUNCTUATION, ALL_SYMBOLS, PIN_YIN_ENDS, pinyin_to_arpabet_mappings, text_pinyin_to_pinyin_symbs, manual_phone_replacements
            from en_numbers import normalize_numbers as en_normalize_numbers
            from ro_numbers import generateWords as ro_generateWords



# Processing order:
# - text-to-text, clean up numbers
# - text-to-text, clean up abbreviations
# - text->phone, Custom dict replacements
# - text->phone, Heteronyms detection and replacement
# - text->phone, built-in dicts replacements (eg CMUdict)
# - text->text/phone, missed words ngram/POS splitting, and re-trying built-in dicts (eg CMUdict)
# - text->phone, g2p (eg espeak)
# - phone->[integer], convert phonemes to their index numbers, for use by the models

# class EspeakWrapper(object):
#     def __init__(self, base_dir, lang):
#         super(EspeakWrapper, self).__init__()


#         from phonemizer.backend import EspeakBackend
#         from phonemizer.backend.espeak.base import BaseEspeakBackend
#         # from phonemizer.backend.espeak import EspeakBackend
#         from phonemizer.separator import Separator

#         base_dir = f'C:/Program Files/'
#         espeak_dll_path = f'{base_dir}/eSpeak_NG/libespeak-ng.dll'
#         # espeak_dll_path = f'{base_dir}/libespeak-ng.dll'
#         # espeak_dll_path = f'{base_dir}/'
#         print(f'espeak_dll_path, {espeak_dll_path}')

#         BaseEspeakBackend.set_library(espeak_dll_path)
#         # EspeakBackend.set_library(espeak_dll_path)
#         self.backend = EspeakBackend(lang)
#         print(f'self.backend, {self.backend}')
#         self.separator = Separator(phone="|", syllable="", word="")
#         print(f'self.separator, {self.separator}')


#     def phonemize (self, word):
#         return self.backend.phonemize(word, self.separator)


class TextPreprocessor():
    def __init__(self, lang_code, lang_code2, base_dir, add_blank=True, logger=None, use_g2p=True, use_epitran=False):
        super(TextPreprocessor, self).__init__()

        self.use_g2p = use_g2p
        self.use_epitran = use_epitran
        self.logger = logger
        self.ALL_SYMBOLS = ALL_SYMBOLS
        self.lang_code = lang_code
        self.lang_code2 = lang_code2
        self.g2p_cache = {}
        self.g2p_cache_path = None
        self.add_blank = add_blank
        self.dicts = []
        self.dict_words = [] # Cache
        self.dict_is_custom = [] # Built-in, or custom; Give custom dict entries priority over other pre-processing steps

        self._punctuation = '!\'(),.:;? ' # Standard english pronunciation symbols

        self.punct_to_whitespace_reg = re.compile(f'[\.,!?]*')

        self.espeak = None
        self.epitran = None
        # self.custom_g2p_fn = None
        if lang_code2:
            # if self.use_epitran and self.use_g2p:
            if self.use_epitran:
                self.epitran = epitran.Epitran(self.lang_code2)

            elif self.use_g2p:
                self.espeak = ESpeak(base_dir, language=self.lang_code2, keep_puncs=True)

        self.h2p = None
        if lang_code=="en":
            self.h2p = H2p(preload=True)

        # Regular expression matching text enclosed in curly braces:
        self._curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


        self.num2words_fn = None
        num2words_supported_langs = ["en","ar","cz","de","dk","en_GB","en_IN","es","es_CO","es_VE","eu","fi","fr","fr_CH","fr_BE","fr_DZ","he","id","it","ja","kn","ko","lt","lv","no","pl","pt","pt_BR","sl","sr","ro","ru","sl","tr","th","vi","nl","uk"]
        if lang_code in num2words_supported_langs:
            self.num2words_fn = num2words



    def init_post(self):

        self.re_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in self.abbreviations]

    # Override - language specific
    def clean_numbers(self, text):
        return text

    # Override - language specific
    def clean_am_pm(self, text):
        return text

    def clean_abbreviations(self, text):
        for regex, replacement in self.re_abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def collapse_whitespace(self, text):
        _whitespace_re = re.compile(r'\s+')
        return re.sub(_whitespace_re, ' ', text)

    def load_dict (self, dict_path, isCustom=False):
        pron_dict = {}
        if dict_path.endswith(".txt"):
            pron_dict = self.read_txt_dict(dict_path, pron_dict)
        elif dict_path.endswith(".json"):
            pron_dict = self.read_json_dict(dict_path, pron_dict)

        pron_dict = self.post_process_dict(pron_dict)

        self.dict_is_custom.append(isCustom)
        self.dicts.append(pron_dict)
        self.dict_words.append(list(pron_dict.keys()))

    # Override
    def post_process_dict(self, pron_dict):
        return pron_dict


    def read_txt_dict (self, dict_path, pron_dict):
        with codecs.open(dict_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line.strip()):
                # if len(line.strip()) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
                    word = line.split(" ")[0].lower()
                    pron = " ".join(line.split(" ")[1:]).strip().upper()
                    # TODO? Check if the phonemes are valid?
                    # TODO? Handle variants(1)
                    pron_dict[word] = pron

        return pron_dict

    def read_json_dict (self, dict_path, pron_dict):
        with codecs.open(dict_path, encoding="utf-8") as f:
            json_data = json.load(f)

            for word in list(json_data["data"].keys()):
                if json_data["data"][word]["enabled"]==True:
                    # TODO? Check if the phonemes are valid?
                    # TODO? Handle variants(1)
                    pron_dict[word.lower()] = json_data["data"][word]["arpabet"].upper()

        return pron_dict


    def dict_replace (self, text, customDicts):

        for di, pron_dict in enumerate(self.dicts):

            if (customDicts and self.dict_is_custom[di]) or (not customDicts and not self.dict_is_custom[di]):

                dict_words = self.dict_words[di]
                text_graphites = re.sub("{([^}]*)}", "", text, flags=re.IGNORECASE)

                # Don't run the ARPAbet replacement for every single word, as it would be too slow. Instead, do it only for words that are actually present in the prompt
                words_in_prompt = (text_graphites+" ").replace("}","").replace("{","").replace(",","").replace("?","").replace("!","").replace(";","").replace(":","").replace("...",".").replace(". "," ").lower().split(" ")


                words_in_prompt = [word.strip() for word in words_in_prompt if len(word.strip()) and word.lower() in dict_words]

                if len(words_in_prompt):
                    # Pad out punctuation, to make sure they don't get used in the word look-ups
                    text = " "+text.replace(",", " ,").replace(".", " .").replace("!", " !").replace("?", " ?")+" "

                    for di, dict_word in enumerate(words_in_prompt):

                        dict_word_with_spaces = "{"+pron_dict[dict_word]+"}"
                        dict_word_replace = dict_word.strip().replace(".", "\.").replace("(", "\(").replace(")", "\)")


                        # Do it twice, because re will not re-use spaces, so if you have two neighbouring words to be replaced,
                        # and they share a space character, one of them won't get changed
                        for _ in range(2):
                            text = re.sub(r'(?<!\{)\b'+dict_word_replace+r'\b(?![\w\s\(\)]*[\}])', dict_word_with_spaces, text, flags=re.IGNORECASE)


                    # Undo the punctuation padding, to retain the original sentence structure
                    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                    text = re.sub("^\s+", " ", text) if text.startswith("  ") else re.sub("^\s*", "", text)
                    text = re.sub("\s+$", " ", text) if text.endswith("  ") else re.sub("\s*$", "", text)

        return text


    def detect_and_fill_heteronyms (self, text):
        if self.h2p is not None:
            text = self.h2p.replace_het(text)
        return text

    def clean_POS_and_subword_misses (self, text):
        # Eg plurals, possessives, contractions, hyphenated, compounds, stem, etc
        # TODO
        return text

    def load_g2p_cache (self, cache_path):
        # print(f'[DEBUG] Loading cache: {cache_path}')
        self.g2p_cache_path = cache_path
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf8") as f:
                lines = f.read().split("\n")
                for line in lines:
                    if "|" in line:
                        word = line.split("|")[0]
                        phones = "|".join(line.split("|")[1:])
                        self.g2p_cache[word.lower().strip()] = phones.strip()
        else:
            print(f'g2p cache file not found at: {cache_path}')

    def save_g2p_cache (self):
        if self.g2p_cache_path:
            cache_out = []
            cache_keys = sorted(list(self.g2p_cache.keys()))
            for key in cache_keys:
                cache_out.append(f'{key}|{self.g2p_cache[key]}')

            with open(self.g2p_cache_path, "w+", encoding="utf8") as f:
                f.write("\n".join(cache_out))

    # Override
    def fill_missing_via_g2p (self, text):

        # TODO, switch to from nltk.tokenize import word_tokenize

        orig_text = text
        # print(f'[g2p] orig_text, |{orig_text}|')
        text_parts = text.split("{")
        text_parts2 = [(part.split("}")[1] if "}" in part else part) for part in text_parts]
        # print(f'[g2p] text_parts, {text_parts}')
        # print(f'[g2p] text_parts2, {text_parts2}')

        phonemised = []
        for part in text_parts2:
            words = part.split(" ")

            part_phonemes = []

            for word in words:

                word = word.strip()
                if len(word):
                    # print(f'\n[g2p] word, {word}')

                    sub_parts = []
                    sub_part_phonemes = []

                    # ====== punctuation stuff start ========
                    # Get which punctuation symbols are contained in the text fragment
                    puncs_contained = []
                    for punc in PUNCTUATION:
                        if punc in word:
                            puncs_contained.append(punc)

                    # Split away the punctuation from text
                    sub_parts = [word]
                    # print(f'puncs_contained, {puncs_contained}')
                    if len(puncs_contained):

                        for punc in puncs_contained:

                            # init a new sub part list (list 2)
                            sub_parts2 = []

                            # for each sub part...
                            for sp in sub_parts:
                                sp = sp.strip()
                                # ...if it not already a punctuation symbol, try splitting it by the current punctuation symbol
                                if sp not in PUNCTUATION:
                                    sp_split = sp.split(punc)
                                    # if the split list length is 1, add to list 2
                                    if len(sp_split)==1:
                                        sub_parts2.append(sp_split[0])
                                    else:
                                        # if it's more than 1
                                        # print(f'sp_split, {sp_split}')
                                        for spspi, sps_part in enumerate(sp_split):
                                            # iterate through each item, and add to list, but also add the punct, apart from the last item
                                            sub_parts2.append(sps_part)
                                            if spspi<(len(sp_split)-1):
                                                sub_parts2.append(punc)

                                else:
                                    # otherwise add the punct to list 2
                                    sub_parts2.append(sp)

                            # set the sub parts list to list 2, for the next loop, or ready
                            sub_parts = sub_parts2

                    else:
                        sub_parts = [word]
                    # ====== punctuation stuff end ========
                    # print(f'sub_parts, {sub_parts}')

                    for sp in sub_parts:
                        if sp in PUNCTUATION:
                            sub_part_phonemes.append(sp)
                        else:
                            sp = sp.replace("\"", "").replace(")", "").replace("(", "").replace("]", "").replace("[", "").strip()

                            if len(sp):
                                # print(f'sp, {sp}')
                                if sp.lower() in self.g2p_cache.keys() and len(self.g2p_cache[sp.lower()].strip()):
                                    # print("in cache")
                                    g2p_out = ipa2xvaarpabet(self.g2p_cache[sp.lower()])
                                    # print(f'g2p_out, {g2p_out}')
                                    sub_part_phonemes.append(g2p_out)
                                else:
                                    if self.use_g2p or "custom_g2p_fn" in dir(self) or self.use_epitran:
                                        # print(f'self.custom_g2p_fn, {self.custom_g2p_fn}')
                                        if "custom_g2p_fn" in dir(self):
                                            g2p_out = self.custom_g2p_fn(sp)
                                        elif self.use_epitran:
                                            g2p_out = self.epitran.transliterate(sp)
                                        else:
                                            g2p_out = self.espeak.phonemize(sp).replace("|", " ")
                                        # print(f'g2p_out, {g2p_out}')
                                        self.g2p_cache[sp.lower()] = g2p_out
                                        self.save_g2p_cache()
                                        g2p_out = ipa2xvaarpabet(g2p_out)
                                        # print(f'g2p_out, {g2p_out}')
                                        sub_part_phonemes.append(g2p_out)
                                        # print(f'sp, {sp} ({len(self.g2p_cache.keys())})  {g2p_out}')


                    part_phonemes.append(" ".join(sub_part_phonemes))

            phonemised.append(" _ ".join(part_phonemes))



        # print("--")
        # print(f'text_parts ({len(text_parts)}), {text_parts}')
        # print(f'[g2p] phonemised ({len(phonemised)}), {phonemised}')


        text = []
        for ppi, phon_part in enumerate(phonemised):
            # print(f'phon_part, {phon_part}')
            prefix = ""

            if "}" in text_parts[ppi]:
                if ppi<len(phonemised)-1 and text_parts[ppi].split("}")[1].startswith(" "):
                    prefix = text_parts[ppi].split("}")[0]+" _ "
                else:
                    prefix = text_parts[ppi].split("}")[0]+" "

            text.append(f'{prefix} {phon_part}')


        # print(f'[g2p] text ({len(text)}), {text}')

        text_final = []
        for tpi, text_part in enumerate(text):
            if tpi!=0 or text_part.strip()!="" or not orig_text.startswith("{"):
                # print(not orig_text.startswith("{"), tpi, f'|{text_part.strip()}|')
                text_final.append(text_part)

            if (tpi or orig_text.startswith(" ")) and ((tpi<len(text_parts2)-1 and text_parts2[tpi+1].startswith(" ")) or text_parts2[tpi].endswith(" ")):
                # print("adding _")
                text_final.append("_")

        text = " ".join(text_final).replace("  ", " ").replace("  ", " ").replace(" _ _ ", " _ ").replace(" _ _ ", " _ ")
        return text


    # Convert IPA fragments not already replaced by dicts/rules via espeak and post-processing
    def ipa_to_xVAARPAbet (self, ipa_text):
        xVAARPAbet = ipa2xvaarpabet(ipa_text)
        return xVAARPAbet


    def clean_special_chars(self, text):
        return text.replace("*","")

    def text_to_phonemes (self, text):
        text = self.clean_special_chars(text)
        text = self.collapse_whitespace(text).replace(" }", "}").replace("{ ", "{")
        text = self.clean_am_pm(text)
        text = self.clean_numbers(text)
        # print(f'clean_numbers: |{text}|')
        text = self.clean_abbreviations(text)
        # print(f'clean_abbreviations: |{text}|')
        text = self.dict_replace(text, customDicts=True)
        # print(f'dict_replace(custom): |{text}|')
        text = self.detect_and_fill_heteronyms(text)
        # print(f'detect_and_fill_heteronyms: |{text}|')
        text = self.dict_replace(text, customDicts=False)
        # print(f'dict_replace(built-in):, |{text}|')
        text = self.clean_POS_and_subword_misses(text)
        # print(f'clean_POS_and_subword_misses: |{text}|')
        text = self.fill_missing_via_g2p(text)
        # print(f'fill_missing_via_g2p: |{text}|')
        return text

    # Main entry-point for pre-processing text completely into phonemes
    # This converts not the phonemes, but to the index numbers for the phonemes list, as required by the models
    def text_to_sequence (self, text):

        orig_text = text
        text = self.text_to_phonemes(text) # Get 100% phonemes from the text
        text = self.collapse_whitespace(text).strip() # Get rid of duplicate/padding spaces
        phonemes = text.split(" ")
        phonemes_final = []
        for pi,phone in enumerate(phonemes):
            if phone in manual_phone_replacements.keys():
                phonemes_final.append(manual_phone_replacements[phone])
            else:
                phonemes_final.append(phone)


        # print(f'phonemes, {phonemes}')

        # with open(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/debug.txt', "w+") as f:
        #     f.write(" ".join(phonemes))

        # sequence = [ALL_SYMBOLS.index(phone) for phone in phonemes]
        # blacklist = ["#"]
        try:
            sequence = []
            for phone in phonemes_final:
                if phone=="#": # The g2p something returns things like "# foreign french". Cut away the commented out stuff, when this happens
                    break
                if len(phone.strip()):
                    sequence.append(ALL_SYMBOLS.index(phone))

            # sequence = [ALL_SYMBOLS.index(phone) for phone in phonemes_final if len(phone) and phone.strip() not in blacklist]
        except:
            print(orig_text, phonemes_final)
            raise

        # Intersperse blank symbol if required
        if self.add_blank:
            sequence_ = []
            for si,symb in enumerate(sequence):
                sequence_.append(symb)
                if si<len(sequence)-1:
                    # sequence_.append(len(ALL_SYMBOLS)-1)
                    sequence_.append(len(ALL_SYMBOLS)-2)
            sequence = sequence_

        cleaned_text = "|".join([ALL_SYMBOLS[index] for index in sequence])

        return sequence, cleaned_text

    def cleaned_text_to_sequence (self, text):
        text = self.collapse_whitespace(text).strip() # Get rid of duplicate/padding spaces
        phonemes = text.split(" ")
        sequence = [ALL_SYMBOLS.index(phone) for phone in phonemes]
        return sequence

    def sequence_to_text (self, sequence): # Used in debugging
        text = []
        for ind in sequence[0]:
            text.append(ALL_SYMBOLS[ind])
        return text




class EnglishTextPreprocessor(TextPreprocessor):

    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(EnglishTextPreprocessor, self).__init__("en", "en-us", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "English"

        self.abbreviations = [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort'),
        ]

        self.init_post()

        # from en_numbers import normalize_numbers
        self.normalize_numbers = en_normalize_numbers



    def post_process_dict (self, pron_dict):

        # CMUdict doesn't contain the symbols on the left. Therefore, these must be mapped to symbols that the models have actually
        # been trained with. This is only the case for CMUdict, so for English-trained models
        ARPAbet_replacements_dict = {
            "YO": "IY0 UW0",
            "UH": "UH0",
            "AR": "R",
            "EY": "EY0",
            "A": "AA0",
            "AW": "AW0",
            "X": "K S",
            "CX": "K HH",
            "AO": "AO0",
            "PF": "P F",
            "AY": "AY0",
            "OE": "OW0 IY0",
            "IY": "IY0",
            "EH": "EH0",
            "OY": "OY0",
            "IH": "IH0",
            "H": "HH"
        }

        for word in pron_dict.keys():

            phonemes = pron_dict[word]

            for key in ARPAbet_replacements_dict.keys():
                phonemes = phonemes.replace(f' {key} ', f' {ARPAbet_replacements_dict[key]} ')
                # Do it twice, because re will not re-use spaces, so if you have two neighbouring phonemes to be replaced,
                # and they share a space character, one of them won't get changed
                phonemes = phonemes.replace(f' {key} ', f' {ARPAbet_replacements_dict[key]} ')

            pron_dict[word] = phonemes
        return pron_dict

    def clean_am_pm (self, text):
        words_out = []
        numerals = ["0","1","2","3","4","5","6","7","8","9"]
        spelled_out = ["teen","one", "two", "three", "four", "five", "six", "seven", "eight", "nine","ten","twenty","thirty","forty","fivty","o'clock"]

        words = text.split(" ")
        for word in words:
            if word[:2].lower().strip()=="am":
                finishes_with_spelled_out_numeral = False
                for spelled_out_n in spelled_out:
                    if len(words_out) and words_out[-1].endswith(spelled_out_n):
                        finishes_with_spelled_out_numeral = True
                        break

                if len(words_out) and words_out[-1][-1] in numerals or finishes_with_spelled_out_numeral:
                    word = "{EY0 IH0} {EH0 M}"+word[2:]
            words_out.append(word)

        return " ".join(words_out)


    def clean_numbers (self, text):
        # This (inflect code) also does things like currency, magnitudes, etc

        final_parts = []

        # print(f'text, {text}')
        parts = re.split("({([^}]*)})", text)
        skip_next = False
        for part in parts:
            if "{" in part:
                final_parts.append(part)
                skip_next = True
                # print(f'[clean_numbers] keeping: {part}')
            else:
                if skip_next:
                    skip_next = False
                else:
                    # print(f'[clean_numbers] doing: {part}')
                    final_parts.append(self.normalize_numbers(part))


        text = "".join(final_parts)

        # print(f'[clean_numbers] parts, {parts}')
        return text

        # return self.normalize_numbers(text)

    def text_to_sequence(self, text):
        text = unidecode(text) # transliterate non-english letters to English, if they can be ascii
        return super(EnglishTextPreprocessor, self).text_to_sequence(text)


class FrenchTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(FrenchTextPreprocessor, self).__init__("fr", "fr-fr", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "French"
        self.abbreviations = [
            ("M", "monsieur"),
            ("Mlle", "mademoiselle"),
            ("Mlles", "mesdemoiselles"),
            ("Mme", "Madame"),
            ("Mmes", "Mesdames"),
            ("N.B", "nota bene"),
            ("M", "monsieur"),
            ("p.c.q", "parce que"),
            ("Pr", "professeur"),
            ("qqch", "quelque chose"),
            ("rdv", "rendez-vous"),
            ("no", "numéro"),
            ("adr", "adresse"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("jr", "junior"),
            ("sgt", "sergent"),
            ("capt", "capitain"),
            ("col", "colonel"),
            ("av", "avenue"),
            ("av. J.-C", "avant Jésus-Christ"),
            ("apr. J.-C", "après Jésus-Christ"),
            ("boul", "boulevard"),
            ("c.-à-d", "c’est-à-dire"),
            ("etc", "et cetera"),
            ("ex", "exemple"),
            ("excl", "exclusivement"),
            ("boul", "boulevard"),
        ]
        self.normalize_numbers = self.num2words_fn
        self.init_post()

# https://github.com/virgil-av/numbers-to-words-romanian/blob/master/src/index.ts
class RomanianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(RomanianTextPreprocessor, self).__init__("ro", "ro", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Romanian"
        self.abbreviations = [
        ]
        self.normalize_numbers = ro_generateWords

        self.init_post()

class ItalianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(ItalianTextPreprocessor, self).__init__("it", "it", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Italian"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class DanishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(DanishTextPreprocessor, self).__init__("da", "da", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Danish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class GermanTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(GermanTextPreprocessor, self).__init__("de", "de", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "German"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class AmharicTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(AmharicTextPreprocessor, self).__init__("am", "amh-Ethi", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Amharic"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class ArabicTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(ArabicTextPreprocessor, self).__init__("ar", "ar", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Arabic"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class MongolianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(MongolianTextPreprocessor, self).__init__("mn", "mon-Cyrl", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Mongolian"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class DutchTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(DutchTextPreprocessor, self).__init__("nl", "nl", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Dutch"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class FinnishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(FinnishTextPreprocessor, self).__init__("fi", "fi", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Finnish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class GreekTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(GreekTextPreprocessor, self).__init__("el", "el", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Greek"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class HausaTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(HausaTextPreprocessor, self).__init__("ha", "hau-Latn", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Hausa"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class HindiTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(HindiTextPreprocessor, self).__init__("hi", "hi", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Hindi"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class HungarianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(HungarianTextPreprocessor, self).__init__("hu", "hu", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Hungarian"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class JapaneseTextPreprocessor(TextPreprocessor):
# Japanese: https://github.com/coqui-ai/TTS/blob/main/TTS/tts/utils/text/japanese/phonemizer.py
    # https://pypi.org/project/pykakasi/
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(JapaneseTextPreprocessor, self).__init__("jp", "ja", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Japanese"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

    def text_to_phonemes (self, line):
        kks = pykakasi.kakasi()
        line = kks.convert(line)
        line = " ".join([part["hira"] for part in line])
        # print(f'line, {line}')
        return super(JapaneseTextPreprocessor, self).text_to_phonemes(line)

class KoreanTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(KoreanTextPreprocessor, self).__init__("ko", "ko", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Korean"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class LatinTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(LatinTextPreprocessor, self).__init__("la", "la", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Latin"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class PolishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(PolishTextPreprocessor, self).__init__("pl", "pl", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Polish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class PortugueseTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(PortugueseTextPreprocessor, self).__init__("pt", "pt", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Portuguese"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class RussianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(RussianTextPreprocessor, self).__init__("ru", "ru", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Russian"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class SpanishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(SpanishTextPreprocessor, self).__init__("es", "es", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Spanish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class SwahiliTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(SwahiliTextPreprocessor, self).__init__("sw", "sw", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Swahili"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class SwedishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(SwedishTextPreprocessor, self).__init__("sv", "sv", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Swedish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

# from thai_segmenter import sentence_segment
class ThaiTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        # super(ThaiTextPreprocessor, self).__init__("th", "th", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)
        super(ThaiTextPreprocessor, self).__init__("th", "tha-Thai", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)
        # super(ThaiTextPreprocessor, self).__init__("th", "hau-Latn", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=True)

        self.lang_name = "Thai"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

    def text_to_phonemes (self, line):

        final_line = line
        # try:
        #     line = line.encode('utf8', errors='ignore').decode('utf8', errors='ignore')
        #     sentence_parts = sentence_segment(line)
        #     for part in list(sentence_parts):
        #         for sub_part in part.pos:
        #             final_line.append(sub_part[0])
        #         final_line.append(".")

        #     final_line = " ".join(final_line)
        # except:
        #     pass

        return super(ThaiTextPreprocessor, self).text_to_phonemes(final_line)

class TurkishTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(TurkishTextPreprocessor, self).__init__("tr", "tr", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Turkish"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class UkrainianTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(UkrainianTextPreprocessor, self).__init__("uk", "uk", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Ukrainian"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class VietnameseTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(VietnameseTextPreprocessor, self).__init__("vi", "vi", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Vietnamese"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

# https://polyglotclub.com/wiki/Language/Wolof/Pronunciation/Alphabet-and-Pronunciation#:~:text=Wolof%20Alphabet,-VowelsEdit&text=Single%20vowels%20are%20short%2C%20geminated,British)%20English%20%22sawed%22.
# https://huggingface.co/abdouaziiz/wav2vec2-xls-r-300m-wolof
class WolofTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(WolofTextPreprocessor, self).__init__("wo", "wo", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=False)

        self.lang_name = "Wolof"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

    # A very basic, lossy Wolof -> IPA converter. There were no g2p libraries supporting Wolof at the time of writing. It was this or nothing.
    def custom_g2p_fn(self, word):
        # print(f'custom_g2p_fn | IN: {word}')

        word = word.lower()

        # lossy
        word = word.replace("à", "a")
        word = word.replace("ó", "o")

        word = word.replace("aa", "aː")
        word = re.sub('a(?!:)', 'ɐ', word)
        word = word.replace("bb", "bː")
        word = word.replace("cc", "cːʰ")
        word = word.replace("dd", "dː")
        word = word.replace("ee", "ɛː")
        word = word.replace("ée", "eː")
        word = word.replace("ëe", "əː")
        word = re.sub('e(?!:)', 'ɛ', word)
        word = re.sub('ë(?!:)', 'ə', word)
        word = word.replace("gg", "gː")
        word = word.replace("ii", "iː")
        word = word.replace("jj", "ɟːʰ")
        word = re.sub('j(?!:)', 'ɟ', word)
        word = word.replace("kk", "kːʰ")
        word = word.replace("ll", "ɫː")
        word = word.replace("mb", "m̩b")
        word = word.replace("mm", "mː")
        word = word.replace("nc", "ɲc")
        word = word.replace("nd", "n̩d")
        word = word.replace("ng", "ŋ̩g")
        word = word.replace("nj", "ɲɟ")
        word = word.replace("nk", "ŋ̩k")
        word = word.replace("nn", "nː")
        word = word.replace("nq", "ɴq")
        word = word.replace("nt", "n̩t")
        word = word.replace("ññ", "ɲː")
        word = word.replace("ŋŋ", "ŋː")
        word = re.sub('ñ(?!:)', 'ɲ', word)
        word = word.replace("oo", "oː")
        word = word.replace("o", "ɔ")
        word = word.replace("pp", "pːʰ")
        word = word.replace("rr", "rː")
        word = word.replace("tt", "tːʰ")
        word = word.replace("uu", "uː")
        word = word.replace("ww", "wː")
        word = word.replace("yy", "jː")
        word = word.replace("y", "j")

        # lossy
        word = word.replace("é", "e")
        word = word.replace("ë", "e")
        word = word.replace("ñ", "n")
        word = word.replace("ŋ", "n")

        # print(f'custom_g2p_fn | OUT: {word}')
        return word

    # def save_g2p_cache(self):
    #     # TEMPORARY
    #     pass



class YorubaTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(YorubaTextPreprocessor, self).__init__("yo", "yor-Latn", base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Yoruba"
        self.abbreviations = [
        ]
        self.normalize_numbers = self.num2words_fn

        self.init_post()

class ChineseTextPreprocessor(TextPreprocessor):
    def __init__(self, base_dir, logger=None, use_g2p=True, use_epitran=False):
        super(ChineseTextPreprocessor, self).__init__("zh", None, base_dir, logger=logger, use_g2p=use_g2p, use_epitran=use_epitran)

        self.lang_name = "Chinese"
        self.abbreviations = [
        ]
        self.init_post()
        # self.g2p = None
        # if self.use_g2p:
        #     self.g2p = G2pC()
        from g2pc import G2pC
        self.g2p = G2pC()

        self.TEMP_unhandled = []


    def split_pinyin (self, pinyin):
        symbs_split = []

        pinyin = pinyin.lower()

        splitting_symbs = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "z", "c", "s", "r", "j", "q", "x"]

        for ss in splitting_symbs:
            # if phon.startswith(ss) and not phon.endswith("i"):
            if pinyin.startswith(ss):
                symbs_split.append(ss.upper())
                pinyin = pinyin[len(ss):]
                break
        symbs_split.append(pinyin.upper())

        return symbs_split


    def post_process_pinyin_symbs (self, symbs):
        post_processed = []

        # splitting_symbs = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "z", "c", "s", "r", "j", "q", "x"]

        for symb in symbs.split(" "):
            if len(symb)==0:
                continue

            symbs = self.split_pinyin(symb)
            for symb in symbs:
                post_processed.append(symb)


            # for ss in splitting_symbs:
            #     # if phon.startswith(ss) and not phon.endswith("i"):
            #     if symb.startswith(ss):
            #         post_processed.append(ss.upper())
            #         symb = symb[len(ss):]
            #         break
            # post_processed.append(symb.upper())

        return " ".join(post_processed)

    def fill_missing_via_g2p_zh (self, text):
        # TODO, switch to from nltk.tokenize import word_tokenize

        orig_text = text
        # print(f'[g2p] orig_text, |{orig_text}|')
        text_parts = text.split("{")
        text_parts2 = [(part.split("}")[1] if "}" in part else part) for part in text_parts]
        # print(f'[g2p] text_parts, {text_parts}')
        # print(f'[g2p] text_parts2, {text_parts2}')

        phonemised = []
        for part in text_parts2:
            # words = part.split(" ")
            words = [part]

            part_phonemes = []

            for word in words:

                word = word.strip()
                if len(word):
                    # print(f'[g2p] word, {word}')

                    sub_parts = []
                    sub_part_phonemes = []

                    # ====== punctuation stuff start ========
                    # Get which punctuation symbols are contained in the text fragment
                    puncs_contained = []
                    for punc in PUNCTUATION:
                        if punc in word:
                            puncs_contained.append(punc)

                    # Split away the punctuation from text
                    sub_parts = [word]
                    # print(f'puncs_contained, {puncs_contained}')
                    if len(puncs_contained):

                        for punc in puncs_contained:

                            # init a new sub part list (list 2)
                            sub_parts2 = []

                            # for each sub part...
                            for sp in sub_parts:
                                sp = sp.strip()
                                # ...if it not already a punctuation symbol, try splitting it by the current punctuation symbol
                                if sp not in PUNCTUATION:
                                    sp_split = sp.split(punc)
                                    # if the split list length is 1, add to list 2
                                    if len(sp_split)==1:
                                        sub_parts2.append(sp_split[0])
                                    else:
                                        # if it's more than 1
                                        # print(f'sp_split, {sp_split}')
                                        for spspi, sps_part in enumerate(sp_split):
                                            # iterate through each item, and add to list, but also add the punct, apart from the last item
                                            sub_parts2.append(sps_part)
                                            if spspi<(len(sp_split)-1):
                                                sub_parts2.append(punc)

                                else:
                                    # otherwise add the punct to list 2
                                    sub_parts2.append(sp)

                            # set the sub parts list to list 2, for the next loop, or ready
                            sub_parts = sub_parts2

                    else:
                        sub_parts = [word]
                    # ====== punctuation stuff end ========
                    # print(f'sub_parts, {sub_parts}')

                    for sp in sub_parts:
                        if sp in PUNCTUATION:
                            sub_part_phonemes.append(sp)
                        else:
                            sp = sp.replace("\"", "").replace(")", "").replace("(", "").replace("]", "").replace("[", "").strip()

                            if len(sp):
                                if sp.lower() in self.g2p_cache.keys() and len(self.g2p_cache[sp.lower()].trim()):
                                    g2p_out = self.g2p_cache[sp.lower()]
                                    g2p_out = self.post_process_pinyin_symbs(g2p_out)
                                    sub_part_phonemes.append(g2p_out)
                                else:
                                    # print(f'sp, {sp} ({len(self.g2p_cache.keys())})')
                                    # g2p_out = self.espeak.phonemize(sp).replace("|", " ")
                                    g2p_out = self.g2p(sp)
                                    g2p_out = " ".join([out_part[2] for out_part in g2p_out])
                                    self.g2p_cache[sp.lower()] = g2p_out
                                    self.save_g2p_cache()
                                    # g2p_out = ipa2xvaarpabet(g2p_out)
                                    g2p_out = self.post_process_pinyin_symbs(g2p_out)
                                    # print(f'g2p_out, {g2p_out}')
                                    sub_part_phonemes.append(g2p_out)


                    part_phonemes.append(" ".join(sub_part_phonemes))

            phonemised.append(" _ ".join(part_phonemes))



        # print("--")
        # print(f'text_parts ({len(text_parts)}), {text_parts}')
        # print(f'[g2p] phonemised ({len(phonemised)}), {phonemised}')


        text = []
        for ppi, phon_part in enumerate(phonemised):
            # print(f'phon_part, {phon_part}')
            prefix = ""

            if "}" in text_parts[ppi]:
                if ppi<len(phonemised)-1 and text_parts[ppi].split("}")[1].startswith(" "):
                    prefix = text_parts[ppi].split("}")[0]+" _ "
                else:
                    prefix = text_parts[ppi].split("}")[0]+" "

            text.append(f'{prefix} {phon_part}')


        # print(f'[g2p] text ({len(text)}), {text}')

        text_final = []
        for tpi, text_part in enumerate(text):
            if tpi!=0 or text_part.strip()!="" or not orig_text.startswith("{"):
                # print(not orig_text.startswith("{"), tpi, f'|{text_part.strip()}|')
                text_final.append(text_part)

            if (tpi or orig_text.startswith(" ")) and ((tpi<len(text_parts2)-1 and text_parts2[tpi+1].startswith(" ")) or text_parts2[tpi].endswith(" ")):
                # print("adding _")
                text_final.append("_")

        text = " ".join(text_final).replace("  ", " ").replace("  ", " ").replace(" _ _ ", " _ ").replace(" _ _ ", " _ ")
        return text



    def preprocess_pinyin (self, text):



        # self.logger.info(f'preprocess_pinyin word_tokenize: {word_tokenize(text)}')

        tokens = word_tokenize(text)

        final_out = []

        is_inside_inline_arpabet = False # Used for determining whether to handle token as grapheme of inlined phonemes (or already preproccessed phonemes)
        # has_included_inlint_arpabet_start = False # Used to determine if to insert the inline phoneme delimiter start {

        for token in tokens:

            if token.startswith("{"):
                is_inside_inline_arpabet = True
                # if len(token.replace("{", "")):
                #     final_out.append(token.replace("{", ""))
                final_out.append(token)
            if token.endswith("}"):
                is_inside_inline_arpabet = False
                final_out.append(token)

            if is_inside_inline_arpabet: # The token is an already processed phoneme, from inline or previously processed phonemes. Include without changes
                final_out.append(token)
                continue


            text = text_pinyin_to_pinyin_symbs(token)


            text_final = []
            text = text.upper().split(" ")

            # self.logger.info(f'preprocess_pinyin text: {text}')

            for part in text:

                # self.logger.info(f'preprocess_pinyin part: {part}')


                final_parts = []
                # split_symbs = []
                do_again = True

                # print(f'part, {part}')
                while do_again:
                    # Check to see if the part is a pynyin that starts with one of the consonants that can be split away
                    split_symbs = self.split_pinyin(part)


                    # print(f'split_symbs, {split_symbs}')
                    do_again = False


                    if len(split_symbs)>1:
                        # A split happened. Add the first split-pinyin into the list...
                        final_parts.append(split_symbs[0])

                        # ... then check if the second half of the split starts with one of the "ending" pinyin phonemes
                        second_half = split_symbs[1]

                        for phone in PIN_YIN_ENDS:
                            if second_half.startswith(phone):
                                final_parts.append(phone)
                                second_half = second_half[len(phone):]
                                if len(second_half):
                                    do_again = True
                                break

                        # Check to see if the leftover starts with one of the pinyin to arpabet mappings
                        for phone_key in pinyin_to_arpabet_mappings.keys():
                            if second_half.startswith(phone_key):
                                final_parts.append(pinyin_to_arpabet_mappings[phone_key])
                                second_half = second_half[len(pinyin_to_arpabet_mappings[phone_key]):]
                                if len(second_half):
                                    do_again = True
                                break

                        part = second_half

                    else:
                        # If the part wasn't split up, then check if it starts with a "split" pinyin symbol, but not with the splitting consonants
                        for phone in PIN_YIN_ENDS:
                            if part.startswith(phone):
                                # Starts with an "ending" phoneme, so add to the list and remove from the part
                                final_parts.append(phone)
                                part = part[len(phone):]
                                if len(part):
                                    # Repeat the whole thing, if there's still any left-over stuff
                                    do_again = True
                                break

                        # Check to see if the leftover starts with one of the pinyin to arpabet mappings
                        for phone_key in pinyin_to_arpabet_mappings.keys():
                            if part.startswith(phone_key):
                                # Starts with a replacement phone, so add to the list and remove from the part
                                final_parts.append(pinyin_to_arpabet_mappings[phone_key])
                                part = part[len(pinyin_to_arpabet_mappings[phone_key]):]
                                if len(part):
                                    # Repeat the whole thing, if there's still any left-over stuff
                                    do_again = True
                                break

                # print(f'part, {part}')
                if len(part):
                    final_parts.append(part)


                # print(f'final_parts, {final_parts}')
                # self.logger.info(f'preprocess_pinyin final_parts: {final_parts}')
                all_split_are_pinyin = True

                final_parts_post = []

                for split in final_parts:

                    if split in pinyin_to_arpabet_mappings.keys():
                        # self.logger.info(f'preprocess_pinyin changing split from: {split} to {pinyin_to_arpabet_mappings[split]}')
                        split = pinyin_to_arpabet_mappings[split]
                    # if split=="J":
                    #     split = "JH"

                    if split in ALL_SYMBOLS:
                        final_parts_post.append(split)
                    else:
                        if split+"5" in ALL_SYMBOLS:
                            final_parts_post.append(split+"5")
                        else:
                            all_split_are_pinyin = False

                # self.logger.info(f'preprocess_pinyin final_parts_post: {final_parts_post}')

                if all_split_are_pinyin:
                    # text_final.append("{"+" ".join(final_parts)+"}")
                    text_final.append("{"+" ".join(final_parts_post)+"}")
                else:
                    text_final.append(part)

            # print(f'text_final, {text_final}')
            final_out.append(" ".join(text_final))

        # self.logger.info(f'preprocess_pinyin final_out: {final_out}')
        text = " ".join(final_out)
        # self.logger.info(f'preprocess_pinyin return text: {text}')
        return text


    def text_to_phonemes (self, text):
        # print(f'text_to_phonemes, {text}')
        text = self.collapse_whitespace(text).replace(" }", "}").replace("{ ", "{")

        text = self.preprocess_pinyin(text)

        # text = self.clean_numbers(text)
        # print(f'clean_numbers: |{text}|')
        # text = self.clean_abbreviations(text)
        # print(f'clean_abbreviations: |{text}|')
        # text = self.dict_replace(text, customDicts=True)
        # print(f'dict_replace(custom): |{text}|')
        # text = self.detect_and_fill_heteronyms(text)
        # print(f'detect_and_fill_heteronyms: |{text}|')
        # text = self.dict_replace(text, customDicts=False)
        # print(f'dict_replace(built-in):, |{text}|')
        # text = self.clean_POS_and_subword_misses(text)
        # self.logger.info(f'clean_POS_and_subword_misses: |{text}|')
        text = self.fill_missing_via_g2p_zh(text)
        # self.logger.info(f'1 text: {text}')
        # text = self.en_processor.text_to_phonemes(text)
        # self.logger.info(f'2 text: {text}')
        # print(f'fill_missing_via_g2p: |{text}|')
        return text

    def text_to_sequence (self, text):

        orig_text = text

        text = self.collapse_whitespace(text) # Get rid of duplicate/padding spaces
        text = text.replace("！", "!").replace("？", "?").replace("，", ",").replace("。", ",").replace("…", "...").replace("）", "").replace("（", "")\
            .replace("、", ",").replace("“", ",").replace("”", ",").replace("：", ":")

        text = self.text_to_phonemes(text) # Get 100% phonemes from the text

        # if self.logger is not None:
        #     self.logger.info(f'1 text: {text}')
        # text = self.en_processor.text_to_phonemes(text)
        # self.logger.info(f'2 text: {text}')

        phonemes = self.collapse_whitespace(text).strip().split(" ")
        # self.logger.info(f'1 phonemes: {phonemes}')

        sequence = []
        for pi,phone in enumerate(phonemes):
            phone = phone.replace(":","").strip()
            if len(phone):
                try:
                    sequence.append(ALL_SYMBOLS.index(phone))
                except:
                    if phone in pinyin_to_arpabet_mappings.keys():
                        sequence.append(ALL_SYMBOLS.index(pinyin_to_arpabet_mappings[phone]))
                    else:
                        if phone not in ["5"]:
                            self.TEMP_unhandled.append(f'{orig_text}: {phone}')
                            # with open(f'F:/Speech/xVA-Synth/python/xvapitch/text/DEBUG.txt', "w+") as f:
                            #     f.write("\n".join(self.TEMP_unhandled))

                # Add a space character between each symbol
                # if pi is not len(phonemes)-1:
                #     sequence.append(ALL_SYMBOLS.index("_"))

        # Intersperse blank symbol if required
        if self.add_blank:
            sequence_ = []
            for si,symb in enumerate(sequence):
                sequence_.append(symb)
                if si<len(sequence)-1:
                    sequence_.append(len(ALL_SYMBOLS)-1)
            sequence = sequence_

        cleaned_text = "|".join([ALL_SYMBOLS[index] for index in sequence])

        return sequence, cleaned_text





def get_text_preprocessor(code, base_dir, logger=None, override_useAnyG2P=None):

    tp_codes = {
        "am": {
            "name": "Amharic",
            "tp": AmharicTextPreprocessor,
            "dicts": [],
            "custom_dicts": [],
            "use_g2p": False,
            "use_epitran": True,
            "g2p_cache": [f'{base_dir}/g2p_cache/epitran/epitran_cache_am.txt']
        },

        "ar": {
            "name": "Arabic",
            "tp": ArabicTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/arabic.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_ar.txt']
        },

        "da": {
            "name": "Danish",
            "tp": DanishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/danish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_da.txt']
        },

        "de": {
            "name": "German",
            "tp": GermanTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/german.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_de.txt']
        },

        "el": {
            "name": "Greek",
            "tp": GreekTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/greek.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_el.txt']
        },

        "en": {
            "name": "English",
            "tp": EnglishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/cmudict.txt'],
            "custom_dicts": glob.glob(f'{base_dir}/../../../arpabet/*.json'),
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_en.txt']
        },

        "es": {
            "name": "Spanish",
            "tp": SpanishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/spanish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_es.txt']
        },

        "fi": {
            "name": "Finnish",
            "tp": FinnishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/finnish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_fi.txt']
        },

        "fr": {
            "name": "French",
            "tp": FrenchTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/french.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_fr.txt']
        },

        "ha": {
            "name": "Hausa",
            "tp": HausaTextPreprocessor,
            # "dicts": [f'{base_dir}/dicts/hausa.txt'],
            "dicts": [],
            "custom_dicts": [],
            "use_g2p": False,
            "use_epitran": True,
            "g2p_cache": [f'{base_dir}/g2p_cache/epitran/epitran_cache_ha.txt']
        },

        "hi": {
            "name": "Hindi",
            "tp": HindiTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/hindi.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_hi.txt']
        },

        "hu": {
            "name": "Hungarian",
            "tp": HungarianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/hungarian.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_hu.txt']
        },

        "it": {
            "name": "Italian",
            "tp": ItalianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/italian.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_it.txt']
        },

        "jp": {
            "name": "Japanese",
            "tp": JapaneseTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/japanese.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_jp.txt']
        },

        "ko": {
            "name": "Korean",
            "tp": KoreanTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/korean.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_ko.txt']
        },

        "la": {
            "name": "Latin",
            "tp": LatinTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/latin.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_la.txt']
        },

        "mn": {
            "name": "Mongolian",
            "tp": MongolianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/mongolian.txt'],
            "custom_dicts": [],
            "use_epitran": True,
            "g2p_cache": [f'{base_dir}/g2p_cache/epitran/epitran_cache_mn.txt']
        },

        "nl": {
            "name": "Dutch",
            "tp": DutchTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/dutch.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_nl.txt']
        },

        "pl": {
            "name": "Polish",
            "tp": PolishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/polish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_pl.txt']
        },

        "pt": {
            "name": "Portuguese",
            "tp": PortugueseTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/portuguese_br.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_pt.txt']
        },

        "ro": {
            "name": "Romanian",
            "tp": RomanianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/romanian.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_ro.txt']
        },

        "ru": {
            "name": "Russian",
            "tp": RussianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/russian.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_ru.txt']
        },

        "sv": {
            "name": "Swedish",
            "tp": SwedishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/swedish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_sv.txt']
        },

        "sw": {
            "name": "Swahili",
            "tp": SwahiliTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/swahili.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_sw.txt']
        },

        "th": {
            "name": "Thai",
            "tp": ThaiTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/thai.txt'],
            "custom_dicts": [],
            # "use_g2p": F
            # "use_g2p": False,
            # "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_th.txt']
            "g2p_cache": [f'{base_dir}/g2p_cache/epitran/epitran_cache_th.txt']
        },

        "tr": {
            "name": "Turkish",
            "tp": TurkishTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/turkish.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_tr.txt']
        },

        "uk": {
            "name": "Ukrainian",
            "tp": UkrainianTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/ukrainian.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_uk.txt']
        },

        "vi": {
            "name": "Vietnamese",
            "tp": VietnameseTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/vietnamese.txt'],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/espeak/espeak_cache_vi.txt']
        },

        "wo": {
            "name": "Wolof",
            "tp": WolofTextPreprocessor,
            # "dicts": [f'{base_dir}/dicts/wolof.txt'],
            "dicts": [],
            "custom_dicts": [],
            "g2p_cache": [f'{base_dir}/g2p_cache/g2p_cache_wo.txt']
        },

        "yo": {
            "name": "Yoruba",
            "tp": YorubaTextPreprocessor,
            "dicts": [f'{base_dir}/dicts/yoruba.txt'],
            "custom_dicts": [],
            "use_epitran": True,
            "g2p_cache": [f'{base_dir}/g2p_cache/epitran/epitran_cache_yo.txt']
        },

        "zh": {
            "name": "Chinese",
            "tp": ChineseTextPreprocessor,
            "dicts": [],
            "custom_dicts": [],
            # "use_g2p": False,
            "use_g2p": True,
            "g2p_cache": [f'{base_dir}/g2p_cache/g2pc_cache_zh.txt']
        },
    }

    use_g2p = tp_codes[code]["use_g2p"] if "use_g2p" in tp_codes[code].keys() else True
    # print(f'override_useAnyG2P, {override_useAnyG2P}')
    if override_useAnyG2P is False:
        use_g2p = override_useAnyG2P
        tp_codes[code]["use_epitran"] = override_useAnyG2P
        tp_codes[code]["use_g2p"] = override_useAnyG2P

    # print(f'tp_codes[code]["use_epitran"], {tp_codes[code]["use_epitran"]}')

    tp = tp_codes[code]["tp"](base_dir, logger=logger, use_g2p=use_g2p, use_epitran=tp_codes[code]["use_epitran"] if "use_epitran" in tp_codes[code].keys() else None)

    for builtin_dict in tp_codes[code]["dicts"]:
        tp.load_dict(builtin_dict)
    for custom_dict in tp_codes[code]["custom_dicts"]:
        tp.load_dict(custom_dict, isCustom=True)

    if len(tp_codes[code]["g2p_cache"]):
        tp.load_g2p_cache(tp_codes[code]["g2p_cache"][0])

    return tp



if __name__ == '__main__':

    import os
    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])

    # tp = RomanianTextPreprocessor(base_dir)
    # tp = ItalianTextPreprocessor(base_dir)
    # tp = GermanTextPreprocessor(base_dir)
    # tp = FrenchTextPreprocessor(base_dir)
            # tp = ArabicTextPreprocessor(base_dir)

    tp = get_text_preprocessor("jp", base_dir)

    # line = "Un test la 10 cuvinte"
    # line = "ein Testsatz mit 10 Wörtern"
    # line = "une phrase test de 10 mots"
    # line = "جملة اختبارية من 10 كلمات"
    # line = "かな漢字"
    # line = "10語の日本語文"
    # line = "aa a a "
    # line = "aa a baal rebb ceeb sàcc "
    line = "これしきで戦闘不能か…ひ弱なものだな。"
    # line = "これしきで せんとうふのう か…ひ じゃく なものだな。"
    line = "これ式で戦闘不能か費はなものだな."
    line = "これ しき で せんとうふのう か ひ はなものだな."


    # tp.espeak
    # print(f'tp.espeak, {tp.espeak}')
    # print(f'tp.espeak, {tp.espeak.supported_languages(base_dir)}')
    # # {'af': 'afrikaans-mbrola-1', 'am': 'Amharic', 'an': 'Aragonese', 'ar': 'Arabic', 'as': 'Assamese', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bpy': 'Bishnupriya_Manipuri', 'bs': 'Bosnian', 'ca': 'Catalan', 'chr-US-Qaaa-x-west': 'Cherokee_', 'cmn': 'Chinese_(Mandarin,_latin_as_English)', 'cmn-latn-pinyin': 'Chinese_(Mandarin,_latin_as_Pinyin)', 'cs': 'Czech', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'german-mbrola-8', 'el': 'greek-mbrola-1', 'en': 'en-swedish', 'en-029': 'English_(Caribbean)', 'en-gb': 'English_(Great_Britain)', 'en-gb-scotland': 'English_(Scotland)', 'en-gb-x-gbclan': 'English_(Lancaster)', 'en-gb-x-gbcwmd': 'English_(West_Midlands)', 'en-gb-x-rp': 'English_(Received_Pronunciation)', 'en-uk': 'english-mb-en1', 'en-us': 'us-mbrola-3', 'en-us-nyc': 'English_(America,_New_York_City)', 'eo': 'Esperanto', 'es': 'Spanish_(Spain)', 'es-419': 'Spanish_(Latin_America)', 'es-es': 'spanish-mbrola-2', 'es-mx': 'mexican-mbrola-2', 'es-vz': 'venezuala-mbrola-1', 'et': 'estonian-mbrola-1', 'eu': 'Basque', 'fa': 'persian-mb-ir1', 'fa-latn': 'Persian_(Pinglish)', 'fi': 'Finnish', 'fr': 'french-mbrola-7', 'fr-be': 'french-mbrola-5', 'fr-ca': 'fr-canadian-mbrola-2', 'fr-ch': 'French_(Switzerland)', 'fr-fr': 'french-mbrola-6', 'ga': 'Gaelic_(Irish)', 'gd': 'Gaelic_(Scottish)', 'gn': 'Guarani', 'grc': 'german-mbrola-6', 'gu': 'Gujarati', 'hak': 'Hakka_Chinese', 'haw': 'Hawaiian', 'he': 'hebrew-mbrola-2', 'hi': 'Hindi', 'hr': 'croatian-mbrola-1', 'ht': 'Haitian_Creole', 'hu': 'hungarian-mbrola-1', 'hy': 'Armenian_(East_Armenia)', 'hyw': 'Armenian_(West_Armenia)', 'ia': 'Interlingua', 'id': 'indonesian-mbrola-1', 'io': 'Ido', 'is': 'icelandic-mbrola-1', 'it': 'italian-mbrola-2', 'ja': 'Japanese', 'jbo': 'Lojban', 'ka': 'Georgian', 'kk': 'Kazakh', 'kl': 'Greenlandic', 'kn': 'Kannada', 'ko': 'Korean', 'kok': 'Konkani', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'latin-mbrola-1', 'lb': 'Luxembourgish', 'lfn': 'Lingua_Franca_Nova', 'lt': 'lithuanian-mbrola-2', 'ltg': 'Latgalian', 'lv': 'Latvian', 'mi': 'maori-mbrola-1', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Myanmar_(Burmese)', 'nb': 'Norwegian_Bokmål', 'nci': 'Nahuatl_(Classical)', 'ne': 'Nepali', 'nl': 'dutch-mbrola-3', 'nog': 'Nogai', 'om': 'Oromo', 'or': 'Oriya', 'pa': 'Punjabi', 'pap': 'Papiamento', 'piqd': 'Klingon', 'pl': 'polish-mbrola-1', 'pt': 'Portuguese_(Portugal)', 'pt-br': 'brazil-mbrola-4', 'pt-pt': 'portugal-mbrola-1', 'py': 'Pyash', 'qdb': 'Lang_Belta', 'qu': 'Quechua', 'quc': "K'iche'", 'qya': 'Quenya', 'ro': 'romanian-mbrola-1', 'ru': 'Russian', 'ru-lv': 'Russian_(Latvia)', 'sd': 'Sindhi', 'shn': 'Shan_(Tai_Yai)', 'si': 'Sinhala', 'sjn': 'Sindarin', 'sk': 'Slovak', 'sl': 'Slovenian', 'smj': 'Lule_Saami', 'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'swedish-mbrola-2', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'telugu-mbrola-1', 'th': 'Thai', 'tk': 'Turkmen', 'tn': 'Setswana', 'tr': 'turkish-mbrola-1', 'tt': 'Tatar', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese_(Northern)', 'vi-vn-x-central': 'Vietnamese_(Central)', 'vi-vn-x-south': 'Vietnamese_(Southern)', 'yue': 'Chinese_(Cantonese,_latin_as_Jyutping)', 'zh': 'chinese-mb-cn1'}
    # fdfgd()


    # kks = pykakasi.kakasi()
    # line = kks.convert(line)
    # line = " ".join([part["hira"] for part in line])

    print(f'line, {line}')

    print(f'Line: |{line}|')
    phonemes = tp.text_to_phonemes(line)
    print(f'xVAARPAbet: |{phonemes}|')
    ssd()



if __name__ == '__main__':
    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    tp = get_text_preprocessor("en", base_dir)

    with open("F:/Speech/custom-arpabets/elderscrolls-missing-post.txt") as f:
        words = f.read().split("\n")

    metadata_out = ["game_id|voice_id|text,out_path"]
    txt_out = []
    for word in words:
        if len(word.strip())>2:

            phones = tp.text_to_phonemes(word)
            print(f'word, {word}')
            print(f'phones, {phones}')

            metadata_out.append(f'skyrim|sk_femaleeventoned|This is what '+"{" + phones +"}"+f' sounds like.|./{word}.wav')
            txt_out.append(f'{word}|{phones}')

    with open(f'./g2p_batch.csv', "w+") as f:
        f.write("\n".join(metadata_out))
    with open(f'./txt_out.csv', "w+") as f:
        f.write("\n".join(txt_out))


    fddfg()

if __name__ == '__main__':
    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    # tp = get_text_preprocessor("th", base_dir)
    # tp = get_text_preprocessor("mn", base_dir)
    tp = get_text_preprocessor("wo", base_dir)

    # print(tp.text_to_phonemes("นี่คือประโยคภาษาไทยที่พูดโดย xVASynth ประมาณนี้ค่ะ"))
    # print(tp.text_to_phonemes("Энэ бол {EH1 G S V EY0 EY0 IH0 S IH0 N TH}-ийн ярьдаг монгол хэл дээрх өгүүлбэр юм. "))
    print(tp.text_to_phonemes(" Kii est ab baat ci wolof, janga par xvasynth "))

    fddfg()

if __name__ == '__main__':

    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    tp = get_text_preprocessor("ha", base_dir)

    print(tp.text_to_phonemes("Wannan jimla ce a cikin hausa, xVASynth ta yi magana "))


    fddfg()


# if __name__ == '__main__':
if False:


    print("Mass pre-caching g2p")

    def get_datasets (root_f):
        data_folders = os.listdir(root_f)
        data_folders = [f'{root_f}/{dataset_folder}' for dataset_folder in sorted(data_folders) if not dataset_folder.startswith("_") and "." not in dataset_folder]
        return data_folders

    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    # all_data_folders = get_datasets(f'D:/xVASpeech/DATASETS')+get_datasets(f'D:/xVASpeech/GAME_DATA')
    all_data_folders = get_datasets(f'D:/xVASpeech/GAME_DATA')

    for dfi,dataset_folder in enumerate(all_data_folders):
        lang = dataset_folder.split("/")[-1].split("_")[0]
        if "de_f4" in dataset_folder:
            continue
        # if lang not in ["zh"]:
        #     continue
        # if lang in ["am", "sw"]:
        #     continue # Skip currently running training
        tp = get_text_preprocessor(lang, base_dir)

        with open(f'{dataset_folder}/metadata.csv') as f:
            lines = f.read().split("\n")

        for li,line in enumerate(lines):

            print(f'\r{dfi+1}/{len(all_data_folders)} | {li+1}/{len(lines)} | {dataset_folder}                                    ', end="", flush=True)

            if "|" in line:
                text = line.split("|")[1]
                if len(text):
                    tp.text_to_phonemes(text)


    print("")
    fsdf()








    # kks = pykakasi.kakasi()

    # pron_dict = {}
    # # with open(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/dicts/japanese.txt') as f:
    # with open(f'F:/Speech/xVA-Synth/python/xvapitch/text/dicts/japanese.txt') as f:
    #     lines = f.read().split("\n")
    #     for li,line in enumerate(lines):

    #         print(f'\r{li+1}/{len(lines)}', end="", flush=True)

    #         if len(line.strip()):
    #             word = line.split(" ")[0]
    #             phon = " ".join(line.split(" ")[1:])

    #             word = kks.convert(word)
    #             word = "".join([part["hira"] for part in word])
    #             # word = word.replace(" ", "").replace(" ", "")

    #             pron_dict[word] = phon

    # csv_out = []
    # for key in pron_dict.keys():
    #     csv_out.append(f'{key} {pron_dict[key]}')

    # with open(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/dicts/japanese_h.txt', "w+") as f:
    #     f.write("\n".join(csv_out))





if False:

    tp = ChineseTextPreprocessor(base_dir)

    # tp.load_g2p_cache(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/g2p_cache/g2pc_cache_zh.txt')

    line = "你好。 这就是 xVASynth 声音的样子。"
    line = "遛弯儿都得躲远点。"

    # line = "Nǐ hǎo"
    # line = "Zhè shì yīgè jiào zhǎng de jùzi. Wǒ xīwàng tā shì zhèngquè de, yīnwèi wǒ zhèngzài shǐyòng gǔgē fānyì tā"

    # phones = tp.text_to_phonemes(line)
    # print(f'phones, |{phones}|')

    phones = tp.text_to_sequence(line)
    print(f'phones, |{phones[1]}|')








    print("start setup...")
    text = []
    # text.append("nords")
    # text.append("I read the book... It was a good book to read?{T EH S T}! Test dovahkiin word")
    # text.append(" I read the book... It was a good book to read?{T EH S T}! Test dovahkiin word")
    # text.append("{AY1 } read the book... It was a good book to read?{T EH S T}! Test 1 dovahkiin word")
    text.append(" {AY1 } read the book... It was a good book to read?{T EH S T}! Test 1 dovahkiin word ")
    # text.append("the scaffold hung with black; and the inhabitants of the neighborhood, having petitioned the sheriffs to remove the scene of execution to the old place,")
    text.append("oxenfurt")
    text.append("atomatoys")

    import os
    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])
    print(f'base_dir, {base_dir}')

    tp = EnglishTextPreprocessor(base_dir)
    tp.load_dict(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/dicts/cmudict.txt')
    tp.load_dict(f'F:/Speech/xVA-Synth/arpabet/xvadict-elder_scrolls.json', isCustom=True)

    # tp.load_g2p_cache(f'F:/Speech/xva-trainer/python/xvapitch/text_prep/g2p_cache/espeak/espeak_cache_en.txt')

    print("start inferring...")

    for line in text:
        print(f'Line: |{line}|')
        phonemes = tp.text_to_phonemes(line)
        print(f'xVAARPAbet: |{phonemes}|')


# TODO
# - Add the POS, and extra cleaning stuff

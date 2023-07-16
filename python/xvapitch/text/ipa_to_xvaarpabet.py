import os
import shutil
import logging
import abc
import sys
import re
import subprocess
import platform
from typing import Dict, List, Tuple
# import phonecode_tables

# try:
#     sys.path.append(".")
#     import resources.app.python.xvapitch.text.phonecode_tables as phonecode_tables
# except:
#     try:
#         import python.xvapitch.text.phonecode_tables as phonecode_tables
#     except:
#         try:
#             import xvapitch.text.phonecode_tables as phonecode_tables
#         except:
#             try:
#                 import text.phonecode_tables as phonecode_tables
#             except:
#                 import phonecode_tables as phonecode_tables
try:
    from . import phonecode_tables
except:
    try:
        sys.path.append(".")
        import resources.app.python.xvapitch.text.phonecode_tables as phonecode_tables
    except:
        try:
            import python.xvapitch.text.phonecode_tables as phonecode_tables
        except:
            try:
                import text.phonecode_tables as phonecode_tables
            except:
                import phonecode_tables as phonecode_tables



ARPABET_SYMBOLS = [
  'AA0', 'AA1', 'AA2', 'AA', 'AE0', 'AE1', 'AE2', 'AE', 'AH0', 'AH1', 'AH2', 'AH',
  'AO0', 'AO1', 'AO2', 'AO', 'AW0', 'AW1', 'AW2', 'AW', 'AY0', 'AY1', 'AY2', 'AY',
  'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'EH', 'ER0', 'ER1', 'ER2', 'ER',
  'EY0', 'EY1', 'EY2', 'EY', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IH', 'IY0', 'IY1',
  'IY2', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OW', 'OY0',
  'OY1', 'OY2', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UH',
  'UW0', 'UW1', 'UW2', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]
# These are actually in the spec, but they seem to not get used by CMUdict or FastPitch
extra_arpabet_symbols = [
    "AX", # commA
    "AXR", # lettER
    "IX", #  rosEs, rabbIt
    "UX", # dUde
    "DX", # buTTer
    "EL", # bottLE
    "EM", # rhythM
    "EN0", # buttON
    "EN1", # buttON
    "EN2", # buttON
    "EN", # buttON
    "NX", # wiNNer
    "Q", # <glottal stop>, eg uh-oh, english (cockney) bottle
    "WH", # why - but like, when people add H before the W
]
new_arpabet_symbols = [
    "RRR", # "rrr" - hard r
    "HR", # the hhhrrr sound common on arabic languages
    "OE", # ø, german möve, french eu (bleu)
    "RH", # "run" - soft r, but mostly h
    "TS", # T and S together -  romanian ț

    "RR", # "run" - soft r
    "UU", # ʏ, german über
    "OO", # ŏ, hard o
    "KH", # K H, but together
    "SJ", # swedish sj
    "HJ", # trying to say the "J" sound, but with a wide open mouth
    "BR", # b mixed in with rolling r, almost like "blowing a raspberry" sound

]
ARPABET_SYMBOLS = ARPABET_SYMBOLS + extra_arpabet_symbols + new_arpabet_symbols
PUNCTUATION = [".", ",", "!", "?", "-", ";", ":",   "—"]

PIN_YIN_ENDS = [
    "A1", "A2", "A3", "A4", "A5", "AI1", "AI2", "AI3", "AI4", "AI5", "AIR2", "AIR3", "AIR4", "AN1", "AN2", "AN3", "AN4", "AN5", "ANG1", "ANG2", "ANG3", "ANG4", "ANG5", "ANGR2", "ANGR3", "ANGR4", "ANR1", "ANR3", "ANR4", "AO1", "AO2", "AO3", "AO4", "AO5", "AOR1", "AOR2", "AOR3", "AOR4", "AOR5", "AR2", "AR3", "AR4", "AR5", "E1", "E2", "E3", "E4", "E5", "EI1", "EI2", "EI3", "EI4", "EI5", "EIR4", "EN1", "EN2", "EN3", "EN4", "EN5", "ENG1", "ENG2", "ENG3", "ENG4", "ENG5", "ENGR1", "ENGR4", "ENR1", "ENR2", "ENR3", "ENR4", "ENR5", "ER1", "ER2", "ER3", "ER4", "ER5", "I1", "I2", "I3", "I4", "I5", "IA1", "IA2", "IA3", "IA4", "IA5", "IAN1", "IAN2", "IAN3", "IAN4", "IAN5", "IANG1", "IANG2", "IANG3", "IANG4", "IANG5", "IANGR2", "IANR1", "IANR2", "IANR3", "IANR4", "IANR5", "IAO1", "IAO2", "IAO3", "IAO4", "IAO5", "IAOR1", "IAOR2", "IAOR3", "IAOR4", "IAR1", "IAR4", "IE1", "IE2", "IE3", "IE4", "IE5", "IN1", "IN2", "IN3", "IN4", "IN5", "ING1", "ING2", "ING3", "ING4", "ING5", "INGR2", "INGR4", "INR1", "INR4", "IONG1", "IONG2", "IONG3", "IONG4", "IONG5", "IR1", "IR3", "IR4", "IU1", "IU2", "IU3", "IU4", "IU5", "IUR1", "IUR2", "O1", "O2", "O3", "O4", "O5", "ONG1", "ONG2", "ONG3", "ONG4", "ONG5", "OR1", "OR2", "OU1", "OU2", "OU3", "OU4", "OU5", "OUR2", "OUR3", "OUR4", "OUR5", "U1", "U2", "U3", "U4", "U5", "UA1", "UA2", "UA3", "UA4", "UA5", "UAI1", "UAI2", "UAI3", "UAI4", "UAIR4", "UAIR5", "UAN1", "UAN2", "UAN3", "UAN4", "UAN5", "UANG1", "UANG2", "UANG3", "UANG4", "UANG5", "UANR1", "UANR2", "UANR3", "UANR4", "UAR1", "UAR2", "UAR4", "UE1", "UE2", "UE3", "UE4", "UE5", "UER2", "UER3", "UI1", "UI2", "UI3", "UI4", "UI5", "UIR1", "UIR2", "UIR3", "UIR4", "UN1", "UN2", "UN3", "UN4", "UN5", "UNR1", "UNR2", "UNR3", "UNR4", "UO1", "UO2", "UO3", "UO4", "UO5", "UOR1", "UOR2", "UOR3", "UOR5", "UR1", "UR2", "UR4", "UR5", "V2", "V3", "V4", "V5", "VE4", "VR3", "WA1", "WA2", "WA3", "WA4", "WA5", "WAI1", "WAI2", "WAI3", "WAI4", "WAN1", "WAN2", "WAN3", "WAN4", "WAN5", "WANG1", "WANG2", "WANG3", "WANG4", "WANG5", "WANGR2", "WANGR4", "WANR2", "WANR4", "WANR5", "WEI1", "WEI2", "WEI3", "WEI4", "WEI5", "WEIR1", "WEIR2", "WEIR3", "WEIR4", "WEIR5", "WEN1", "WEN2", "WEN3", "WEN4", "WEN5", "WENG1", "WENG2", "WENG3", "WENG4", "WENR2", "WO1", "WO2", "WO3", "WO4", "WO5", "WU1", "WU2", "WU3", "WU4", "WU5", "WUR3", "YA1", "YA2", "YA3", "YA4", "YA5", "YAN1", "YAN2", "YAN3", "YAN4", "YANG1", "YANG2", "YANG3", "YANG4", "YANG5", "YANGR4", "YANR3", "YAO1", "YAO2", "YAO3", "YAO4", "YAO5", "YE1", "YE2", "YE3", "YE4", "YE5", "YER4", "YI1", "YI2", "YI3", "YI4", "YI5", "YIN1", "YIN2", "YIN3", "YIN4", "YIN5", "YING1", "YING2", "YING3", "YING4", "YING5", "YINGR1", "YINGR2", "YINGR3", "YIR4", "YO1", "YO3", "YONG1", "YONG2", "YONG3", "YONG4", "YONG5", "YONGR3", "YOU1", "YOU2", "YOU3", "YOU4", "YOU5", "YOUR2", "YOUR3", "YOUR4", "YU1", "YU2", "YU3", "YU4", "YU5", "YUAN1", "YUAN2", "YUAN3", "YUAN4", "YUAN5", "YUANR2", "YUANR4", "YUE1", "YUE2", "YUE4", "YUE5", "YUER4", "YUN1", "YUN2", "YUN3", "YUN4",
]
PIN_YIN_STARTS = [ # NOT USED - already in ARPAbet, or manually mapped to equivalent (see below)
    "B", "C", "CH", "D", "F", "G",  "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "SH", "T", "X", "Z", "ZH"
# "H", "J", "Q", "X",
]

EXTRA = [ # Not currently used, but registered for compatibility, if I do start using them. Plus some extra slots
    "@BREATHE_IN",
    "@BREATHE_OUT",
    "@LAUGH",
    "@GIGGLE",
    "@SIGH",
    "@COUGH",
    "@AHEM",
    "@SNEEZE",
    "@WHISTLE",
    "@UGH",
    "@HMM",
    "@GASP",
    "@AAH",
    "@GRUNT",
    "@YAWN",
    "@SNIFF",

    "@_UNUSED_1",
    "@_UNUSED_2",
    "@_UNUSED_3",
    "@_UNUSED_4",
    "@_UNUSED_5",
]


# ALL_SYMBOLS = ARPABET_SYMBOLS + PUNCTUATION + PIN_YIN_STARTS + PIN_YIN_ENDS + ["_"]
ALL_SYMBOLS = ARPABET_SYMBOLS + PUNCTUATION + PIN_YIN_ENDS + EXTRA + ["<PAD>", "_"]

pinyin_to_arpabet_mappings = {
    "C": "TS",
    "E": "EH0",
    "H": "HH",
    "J": "ZH",
    "Q": "K",
    "X": "S",
}

def text_pinyin_to_pinyin_symbs (text):
    text = re.sub("āng", "ang1", text)
    text = re.sub("áng", "ang2", text)
    text = re.sub("ǎng", "ang3", text)
    text = re.sub("àng", "ang4", text)
    text = re.sub("ēng", "eng1", text)
    text = re.sub("éng", "eng2", text)
    text = re.sub("ěng", "eng3", text)
    text = re.sub("èng", "eng4", text)
    text = re.sub("īng", "ing1", text)
    text = re.sub("íng", "ing2", text)
    text = re.sub("ǐng", "ing3", text)
    text = re.sub("ìng", "ing4", text)
    text = re.sub("ōng", "ong1", text)
    text = re.sub("óng", "ong2", text)
    text = re.sub("ǒng", "ong3", text)
    text = re.sub("òng", "ong4", text)
    text = re.sub("ān", "an1", text)
    text = re.sub("án", "an2", text)
    text = re.sub("ǎn", "an3", text)
    text = re.sub("àn", "an4", text)
    text = re.sub("ēn", "en1", text)
    text = re.sub("én", "en2", text)
    text = re.sub("ěn", "en3", text)
    text = re.sub("èn", "en4", text)
    text = re.sub("īn", "in1", text)
    text = re.sub("ín", "in2", text)
    text = re.sub("ǐn", "in3", text)
    text = re.sub("ìn", "in4", text)
    text = re.sub("ūn", "un1", text)
    text = re.sub("ún", "un2", text)
    text = re.sub("ǔn", "un3", text)
    text = re.sub("ùn", "un4", text)
    text = re.sub("ér", "er2", text)
    text = re.sub("ěr", "er3", text)
    text = re.sub("èr", "er4", text)
    text = re.sub("āo", "aō", text)
    text = re.sub("áo", "aó", text)
    text = re.sub("ǎo", "aǒ", text)
    text = re.sub("ào", "aò", text)
    text = re.sub("ōu", "oū", text)
    text = re.sub("óu", "oú", text)
    text = re.sub("ǒu", "oǔ", text)
    text = re.sub("òu", "où", text)
    text = re.sub("āi", "aī", text)
    text = re.sub("ái", "aí", text)
    text = re.sub("ǎi", "aǐ", text)
    text = re.sub("ài", "aì", text)
    text = re.sub("ēi", "eī", text)
    text = re.sub("éi", "eí", text)
    text = re.sub("ěi", "eǐ", text)
    text = re.sub("èi", "eì", text)
    text = re.sub("ā", "a1", text)
    text = re.sub("á", "a2", text)
    text = re.sub("ǎ", "a3", text)
    text = re.sub("à", "a4", text)
    text = re.sub("ē", "e1", text)
    text = re.sub("é", "e2", text)
    text = re.sub("ě", "e3", text)
    text = re.sub("è", "e4", text)
    text = re.sub("ī", "i1", text)
    text = re.sub("í", "i2", text)
    text = re.sub("ǐ", "i3", text)
    text = re.sub("ì", "i4", text)
    text = re.sub("ō", "o1", text)
    text = re.sub("ó", "o2", text)
    text = re.sub("ǒ", "o3", text)
    text = re.sub("ò", "o4", text)
    text = re.sub("ū", "u1", text)
    text = re.sub("ú", "u2", text)
    text = re.sub("ǔ", "u3", text)
    text = re.sub("ù", "u4", text)
    text = re.sub("ǖ", "ü1", text)
    text = re.sub("ǘ", "ü2", text)
    text = re.sub("ǚ", "ü3", text)
    text = re.sub("ǜ", "ü4", text)

    text = re.sub("ng1a", "n1ga", text)
    text = re.sub("ng2a", "n2ga", text)
    text = re.sub("ng3a", "n3ga", text)
    text = re.sub("ng4a", "n4ga", text)
    text = re.sub("ng1e", "n1ge", text)
    text = re.sub("ng2e", "n2ge", text)
    text = re.sub("ng3e", "n3ge", text)
    text = re.sub("ng4e", "n4ge", text)
    text = re.sub("ng1o", "n1go", text)
    text = re.sub("ng2o", "n2go", text)
    text = re.sub("ng3o", "n3go", text)
    text = re.sub("ng4o", "n4go", text)
    text = re.sub("ng1u", "n1gu", text)
    text = re.sub("ng2u", "n2gu", text)
    text = re.sub("ng3u", "n3gu", text)
    text = re.sub("ng4u", "n4gu", text)

    text = re.sub("n1ang", "1nang", text)
    text = re.sub("n2ang", "2nang", text)
    text = re.sub("n3ang", "3nang", text)
    text = re.sub("n4ang", "4nang", text)
    text = re.sub("n1eng", "1neng", text)
    text = re.sub("n2eng", "2neng", text)
    text = re.sub("n3eng", "3neng", text)
    text = re.sub("n4eng", "4neng", text)
    text = re.sub("n1ing", "1ning", text)
    text = re.sub("n2ing", "2ning", text)
    text = re.sub("n3ing", "3ning", text)
    text = re.sub("n4ing", "4ning", text)
    text = re.sub("n1ong", "1nong", text)
    text = re.sub("n2ong", "2nong", text)
    text = re.sub("n3ong", "3nong", text)
    text = re.sub("n4ong", "4nong", text)

    text = re.sub("hen2an2", "he2nan2", text)
    text = re.sub("hun2an2", "hu2nan2", text)
    text = re.sub("zhun2an2", "zhu2nan2", text)
    text = re.sub("hun3an2", "hu3nan2", text)
    text = re.sub("lin3an2", "li3nan2", text)
    text = re.sub("zhin3an2", "zhi3nan2", text)
    text = re.sub("bun4an2", "bu4nan2", text)
    text = re.sub("chin4an2", "chi4nan2", text)
    text = re.sub("shin4an2", "shi4nan2", text)
    text = re.sub("man3an3", "ma3nan3", text)
    text = re.sub("ban1en4", "ba1nen4", text)
    text = re.sub("jin2an4", "ji2nan4", text)
    text = re.sub("yin2an4", "yi2nan4", text)
    text = re.sub("hen2an4", "he2nan4", text)
    text = re.sub("lin2an4", "li2nan4", text)
    text = re.sub("zen2an4", "ze2nan4", text)
    text = re.sub("kun3an4", "ku3nan4", text)
    text = re.sub("sin3an4", "si3nan4", text)
    text = re.sub("yun4an4", "yu4nan4", text)
    text = re.sub("qun4ian2", "qu4nian2", text)

    text = re.sub("ner4en3", "ne4ren3", text)
    text = re.sub("er4an2", "e4ran2", text)
    text = re.sub("ger4en2", "ge4ren2", text)
    text = re.sub("her2en2", "he2ren2", text)
    text = re.sub("zher2en2", "zhe2ren2", text)
    text = re.sub("zer2en2", "ze2ren2", text)
    text = re.sub("zer2en4", "ze2ren4", text)
    text = re.sub("der2en2", "de2ren2", text)
    text = re.sub("ker4en2", "ke4ren2", text)
    text = re.sub("ser4en2", "se4ren2", text)
    text = re.sub("ker4an2", "ke4ran2", text)
    return text

def _espeak_exe(base_dir, args: List, sync=False) -> List[str]:
    """Run espeak with the given arguments."""
    # espeak_lib = f'F:/Speech/espeak/eSpeak/command_line/espeak.exe'
    # espeak_lib = f'C:/Program Files (x86)/eSpeak/command_line/espeak.exe'
    # espeak_lib = f'F:/Speech/espeak/eSpeak_NG/espeak-ng.exe'
    base_dir = base_dir.replace("\\", "/")
    if platform.system() == 'Linux':
        espeak_lib = 'espeak-ng'
    else:
        espeak_lib = f'{base_dir}/eSpeak_NG/espeak-ng.exe'
    cmd = [
        espeak_lib,
        # f'--path="F:/Speech/espeak/eSpeak_NG"',
        f'--path="{base_dir}/eSpeak_NG"',
        "-q",
        "-b",
        "1",  # UTF8 text encoding
    ]
    cmd.extend(args)
    # logging.debug("espeakng: executing %s", repr(cmd))
    print("espeakng: executing %s", repr(" ".join(cmd)))

    os.makedirs("/usr/share/espeak-ng-data", exist_ok=True)
    if not os.path.exists("/usr/share/espeak-ng-data/phontab"):
        shutil.copytree(f'{base_dir}/eSpeak_NG/espeak-ng-data', "/usr/share/espeak-ng-data", dirs_exist_ok=True)
    # print(" ".join(cmd))
    # print(f'F:/Speech/espeak/eSpeak_NG/espeak-ng.exe --path="F:/Speech/espeak/eSpeak_NG" -q -b 1 -v ro --ipa=1 "bună ziua. Ce mai faceți?"')
    # print("---")


    try:
        with subprocess.Popen(
            cmd,
            # cmd,
            # f'F:/Speech/espeak/eSpeak_NG/espeak-ng.exe --path="F:/Speech/espeak/eSpeak_NG" -q -b 1 -v ro --ipa=1 "bună ziua. Ce mai faceți?"',
            # F:/Speech/espeak/eSpeak_NG/espeak-ng.exe --path="F:/Speech/espeak/eSpeak_NG" -q -b 1 -v ro --ipa=1 "bună ziua. Ce mai faceți?"
            # F:/Speech/espeak/eSpeak_NG/espeak-ng.exe --path="F:/Speech/espeak/eSpeak_NG" -q -b 1 -v ro --ipa=1 "bună ziua. Ce mai faceți?"

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            res = iter(p.stdout.readline, b"")
            if not sync:
                p.stdout.close()
                if p.stderr:
                    p.stderr.close()
                if p.stdin:
                    p.stdin.close()
                return res
            res2 = []
            for line in res:
                res2.append(line)
            p.stdout.close()
            if p.stderr:
                p.stderr.close()
            if p.stdin:
                p.stdin.close()
            p.wait()
        return res2
    except:
        print(f'espeak subprocess error. base_dir={base_dir}')
        raise

class BasePhonemizer(abc.ABC):
    """Base phonemizer class
    Phonemization follows the following steps:
        1. Preprocessing:
            - remove empty lines
            - remove punctuation
            - keep track of punctuation marks
        2. Phonemization:
            - convert text to phonemes
        3. Postprocessing:
            - join phonemes
            - restore punctuation marks
    Args:
        language (str):
            Language used by the phonemizer.
        punctuations (List[str]):
            List of punctuation marks to be preserved.
        keep_puncs (bool):
            Whether to preserve punctuation marks or not.
    """

    # def __init__(self, language, punctuations=Punctuation.default_puncs(), keep_puncs=False):
    def __init__(self, language, punctuations=None, keep_puncs=False):

        # ensure the backend is installed on the system
        # if not self.is_available():
        #     raise RuntimeError("{} not installed on your system".format(self.name()))  # pragma: nocover

        # ensure the backend support the requested language
        self._language = self._init_language(language)

        # setup punctuation processing
        self._keep_puncs = keep_puncs
        # self._punctuator = Punctuation(punctuations)

    def _init_language(self, language):
        """Language initialization
        This method may be overloaded in child classes (see Segments backend)
        """
        # if not self.is_supported_language(language):
        #     raise RuntimeError(f'language "{language}" is not supported by the ' f"{self.name()} backend")
        return language

    @property
    def language(self):
        """The language code configured to be used for phonemization"""
        return self._language

    # @staticmethod
    # @abc.abstractmethod
    # def name():
    #     """The name of the backend"""
    #     ...

    # @classmethod
    # @abc.abstractmethod
    # def is_available(cls):
    #     """Returns True if the backend is installed, False otherwise"""
    #     ...

    @classmethod
    @abc.abstractmethod
    def version(cls):
        """Return the backend version as a tuple (major, minor, patch)"""
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages():
        """Return a dict of language codes -> name supported by the backend"""
        ...

    def is_supported_language(self, language):
        """Returns True if `language` is supported by the backend"""
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text, separator):
        """The main phonemization method"""

    def _phonemize_preprocess(self, text) -> Tuple[List[str], List]:
        """Preprocess the text before phonemization
        1. remove spaces
        2. remove punctuation
        Override this if you need a different behaviour
        """
        text = text.strip()
        # if self._keep_puncs:
            # a tuple (text, punctuation marks)
        #     return self._punctuator.strip_to_restore(text)
        # return [self._punctuator.strip(text)], []
        return [text], []

    def _phonemize_postprocess(self, phonemized, punctuations) -> str:
        """Postprocess the raw phonemized output
        Override this if you need a different behaviour
        """
        # if self._keep_puncs:
        #     return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: str, separator="|") -> str:
        """Returns the `text` phonemized for the given language
        Args:
            text (str):
                Text to be phonemized.
            separator (str):
                string separator used between phonemes. Default to '_'.
        Returns:
            (str): Phonemized text
        """
        text, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in text:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        phonemized = self._phonemize_postprocess(phonemized, punctuations)
        # print("text", text)
        return phonemized

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")


class ESpeak(BasePhonemizer):
    """ESpeak wrapper calling `espeak` or `espeak-ng` from the command-line the perform G2P
    Args:
        language (str):
            Valid language code for the used backend.
        backend (str):
            Name of the backend library to use. `espeak` or `espeak-ng`. If None, set automatically
            prefering `espeak-ng` over `espeak`. Defaults to None.
        punctuations (str):
            Characters to be treated as punctuation. Defaults to Punctuation.default_puncs().
        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to True.
    Example:
        >>> from TTS.tts.utils.text.phonemizers import ESpeak
        >>> phonemizer = ESpeak("tr")
        >>> phonemizer.phonemize("Bu Türkçe, bir örnektir.", separator="|")
        'b|ʊ t|ˈø|r|k|tʃ|ɛ, b|ɪ|r œ|r|n|ˈɛ|c|t|ɪ|r.'
    """

    # _ESPEAK_LIB = _DEF_ESPEAK_LIB

    # def __init__(self, language: str, backend=None, punctuations=Punctuation.default_puncs(), keep_puncs=True):
    def __init__(self, base_dir, language: str, backend=None, punctuations=None, keep_puncs=True):

        self.base_dir = base_dir

        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        if backend is not None:
            self.backend = backend

    def phonemize_espeak(self, text: str, separator: str = "|", tie=False) -> str:
        """Convert input text to phonemes.
        Args:
            text (str):
                Text to be converted to phonemes.
            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """
        # set arguments
        args = ["-v", f"{self._language}"]
        # espeak and espeak-ng parses `ipa` differently
        if tie:
            args.append("--ipa=1")
        else:
            args.append("--ipa=1")
        if tie:
            args.append("--tie=%s" % tie)

        args.append('"' + text + '"')
        # compute phonemes
        phonemes = ""


        for line in _espeak_exe(self.base_dir, args, sync=True):
            print("[phonemize_espeak] line", line)
            logging.debug("line: %s", repr(line))
            # phonemes += line.decode("utf8").strip()[self.num_skip_chars :]  # skip initial redundant characters
            phonemes += line.decode("utf8").strip()
        phonemes = re.sub(r"(\([a-z][a-z]\))", "", phonemes)
        return phonemes.replace("_", separator)

    def _phonemize(self, text, separator=None):
        return self.phonemize_espeak(text, separator, tie=False)

    @staticmethod
    def supported_languages(base_dir) -> Dict:
        """Get a dictionary of supported languages.
        Returns:
            Dict: Dictionary of language codes.
        """
        # if _DEF_ESPEAK_LIB is None:
        #     return {}
        args = ["--voices"]
        langs = {}
        count = 0
        for line in _espeak_exe(base_dir, args, sync=True):
            print("[supported_languages] line", line)
            line = line.decode("utf8").strip()
            if count > 0:
                cols = line.split()
                lang_code = cols[1]
                lang_name = cols[3]
                langs[lang_code] = lang_name
            logging.debug("line: %s", repr(line))
            count += 1
        return langs

    def version(self) -> str:
        """Get the version of the used backend.
        Returns:
            str: Version of the used backend.
        """
        args = ["--version"]
        for line in _espeak_exe(self.base_dir, args, sync=True):
            version = line.decode("utf8").strip().split()[2]
            logging.debug("line: %s", repr(line))
            return version



# =========================
# https://github.com/jhasegaw/phonecodes/blob/master/src/phonecode_tables.py
def translate_string(s, d):
    '''(tl,ttf)=translate_string(s,d):
    Translate the string, s, using symbols from dict, d, as:
    1. Min # untranslatable symbols, then 2. Min # symbols.
    tl = list of translated or untranslated symbols.
    ttf[n] = True if tl[n] was translated, else ttf[n]=False.
    '''
    N = len(s)
    symcost = 1    # path cost per translated symbol
    oovcost = 10   # path cost per untranslatable symbol
    maxsym = max(len(k) for k in d.keys())  # max input symbol length
    # (pathcost to s[(n-m):n], n-m, translation[s[(n-m):m]], True/False)
    lattice = [ (0,0,'',True) ]
    for n in range(1,N+1):
        # Initialize on the assumption that s[n-1] is untranslatable
        lattice.append((oovcost+lattice[n-1][0],n-1,s[(n-1):n],False))
        # Search for translatable sequences s[(n-m):n], and keep the best
        for m in range(1,min(n+1,maxsym+1)):
            if s[(n-m):n] in d and symcost+lattice[n-m][0] < lattice[n][0]:
                lattice[n] = (symcost+lattice[n-m][0],n-m,d[s[(n-m):n]],True)
    # Back-trace
    tl = []
    translated = []
    n = N
    while n > 0:
        tl.append(lattice[n][2])
        translated.append(lattice[n][3])
        n = lattice[n][1]
    return((tl[::-1], translated[::-1]))
def attach_tones_to_vowels(il, tones, vowels, searchstep, catdir):
    '''Return a copy of il, with each tone attached to nearest vowel if any.
    searchstep=1 means search for next vowel, searchstep=-1 means prev vowel.
    catdir>=0 means concatenate after vowel, catdir<0 means cat before vowel.
    Tones are not combined, except those also included in the vowels set.
    '''
    ol = il.copy()
    v = 0 if searchstep>0 else len(ol)-1
    t = -1
    while 0<=v and v<len(ol):
        if (ol[v] in vowels or (len(ol[v])>1 and ol[v][0] in vowels)) and t>=0:
            ol[v]= ol[v]+ol[t] if catdir>=0 else ol[t]+ol[v]
            ol = ol[0:t] + ol[(t+1):]  # Remove the tone
            t = -1 # Done with that tone
        if v<len(ol) and ol[v] in tones:
            t = v
        v += searchstep
    return(ol)
def phonecode_ipa2arpabet (ipa_string):
    (il,ttf)=translate_string(ipa_string,phonecode_tables._ipa2arpabet)
    aprabet_string = attach_tones_to_vowels(il,'012',phonecode_tables._arpabet_vowels,1,1)
    # return il
    return "".join(aprabet_string)



manual_replace_dict = {
    # English
    "L": "L", # Not sure why this gets missed
    "ɐ": "AA",
    "ɾ": "R",
    "o": "OW0",
    "ᵻ": "EH0",
    "ɜ": "UH1",
    "r": "RRR", # Or "R" if not using custom ARPAbet symbols
    "H": "HH", # For consistency
    "x": "HH",
    "ɬ": "HH",
    # Arabic
    "a": "AE",
    "χ": "HR", # Or "H" if not using custom ARPAbet symbols
    "e": "EH0",
    "ɣ": "HH",
    "ħ": "HH",
    "q": "K",
    "ʕ": "RH",
    # German
    "ø": "OE", # ø, german möve, french eu (bleu)
    "œ": "ER",
    "œ̃": "ER",
    "ç": "HH",
    "y": "Y UW1",
    "ɒ": "AO",
    "A": "AA",
    # Greek
    "ʎ": "L IY",
    "ɲ": "N IY0",
    "c": "K HH",
    # Spanish
    "β": "V",
    "ɟ": "G",
    "ʝ": "IY",
    # French
    "ʁ": "RH",
    # Hindi
    "ʈ": "T",
    "ʋ": "V",
    "ɳ": "N",
    "ɖ": "D",
    "ʂ": "SH",
    # Dutch
    "ɵ": "UH",
    # Polish
    "ɕ": "SH",
    "ʑ": "ZH",
    # Russian
    "ɭ": "L",
    # Swedish
    "ʉ": "UW",
    # Turkish
    "ɫ": "L",
    "ɯ": "UW0",
    # Japanese
    "ä": "AA0",
    "Dʑ": "JH",
    "ũ": "N",
    # Thai
    "ɓ": "B",
    "ɔ̌": "AA0",
    "ɗ": "D",
    "ì": "IY0",
    "à": "AA0",
    "ǐ": "IY0",
    # Dutch
    "s̺": "S",
    "ɑ̈": "AA0",
    "ɒ̽": "AH0",
    "s̬": "S",
    "tˢ": "TS",
    "ɔ̽": "OO",
    # Mongolian
    "ɪ̆": "IH0",
    "ə̆": "AX0",
    "ʊ̈": "UH0",
    "ɬ": "HJ",
    # Yoruba
    "ã́": "AE",
    "á": "AE",
    "ã́": "AE",
    "ā": "AE",
    "í": "IY0",
    "ĩ́": "IY0",
    "ī": "IY0",
    "ò": "OW0",
    "ó": "OW0",
    "ọ́": "OW0",
    "ù": "UW0",
    "ú": "UW0",
    "ē": "EH0",
    "ɪ́": "IH0",
    "ʊ́": "UH0",
    "ṹ": "UH0",
    "ū": "UH0",
    "ɛ́": "EH0",
    "ŋ́": "NG",
    "ɔ́": "AO0",
}
manual_ignore_replace_dict = {
    "ː": "",
    "(eN)": "", # Arabic
    "(ar)": "", # Arabic
    "ˤ": "", # Arabic
    "?": "", # German
    "(De)": "", # German
    "(PL)": "", # German
    "(eL)": "", # Greek
    "(Fr)": "", # French
    "(HHIY)": "", # Hindi
    "(IYT)": "", # Italian
    "(eS)": "", # Italian
    "ʰ": "", # Latin
    "(NL)": "", # Dutch
    "(PT-PT)": "", # Portuguese
    "(rUW)": "", # Russian
    "ʲ": "", # Russian
    "(UWK)": "", # Ukrainian
    "(Ko)": "", # Korean
    "˥˩": "", # Thai
    "˦˥": "", # Thai
    "˧": "", # Thai
    "˩˦": "", # Thai
    "˩": "", # Thai
    "˨˩": "", # Thai
    "˨": "", # Thai
    "ꜜ": "", # Yoruba
    "͡": "", # Hausa
}

manual_phone_replacements = {
    "AX0": "AX"
}


# =========================

# Assume a space separated string of IPA symbols
# This is a lossy conversion from complex IPA, to a user-friendly ARPAbet format, with a few extra custom symbols,
# to better handle non-english phonemes not present in the original ARPAbet
def ipa2xvaarpabet (ipa_text):

    # print(f'ipa_text, {ipa_text}')

    ipa_text = ipa_text.replace("ː", " ")
    pc_phones = phonecode_ipa2arpabet(ipa_text)
    # print(f'pc_phones 1, {pc_phones}')

    phones_final = []

    for key in manual_ignore_replace_dict.keys():
        pc_phones = pc_phones.replace(key, manual_ignore_replace_dict[key])
        pc_phones = pc_phones.replace("  ", "").replace("  ", "").strip()

    # print(f'pc_phones 2, {pc_phones}')

    for phone in pc_phones.split(" "):
        phone = phone.strip()
        # print(f'(top) phone, {phone}')
        if len(phone):
            if phone in ARPABET_SYMBOLS:
                # print(f'phone in ARPABET_SYMBOLS, {phone}')
                phones_final.append(phone)
                phone = ""
            else:
                for outer_i in range(5):
                    phone = phone.strip()
                    # print(f'phone, {phone}')
                    # Check to see if multiple ARPAbet phones have been joined together, by checking the start letters against
                    # phones from aprabet, and remove them from the phone string, and checking again, etc

                    # Check up to 5 times, in case there are multiple phones and/or the order is non-alphabetic
                    for i in range(5):
                        if len(phone):
                            for arpabet_phone in ARPABET_SYMBOLS:
                                if len(phone) and phone.startswith(arpabet_phone):
                                    # print(f'phone arpabet_phone, {phone} ({len(phone)}) {arpabet_phone} ({len(arpabet_phone)}) -> {phone[len(arpabet_phone):]}')
                                    phones_final.append(arpabet_phone)
                                    phone = phone[len(arpabet_phone):]
                                    if phone in ARPABET_SYMBOLS:
                                        phones_final.append(phone)
                                        phone = ""


                    # If there's anything left over, check to see if I've handled this through the manual key/value replacements
                    # decided through manual inspection of unhandled phones
                    if len(phone):

                        # Replace numbered stress markers
                        if outer_i>2:
                            phone = phone.replace("0","").replace("1","").replace("2","").replace("3","")

                        # print(f'phone 2, {phone}')

                        if phone in manual_replace_dict.keys():
                            # print(f'manual_replace_dict[phone], {phone} -> {manual_replace_dict[phone]}')
                            phones_final.append(manual_replace_dict[phone])
                            phone = ""
                        else:

                            # Check up to 3 times, in case there are multiple phones and/or the order is non-alphabetic
                            for i in range(3):

                                for manual_phone_key in manual_replace_dict.keys():

                                    if len(phone) and phone.startswith(manual_phone_key):
                                        # print(f'manual_replace_dict[manual_phone_key], {phone} => {manual_phone_key} => {manual_replace_dict[manual_phone_key]}   -> {phone[len(manual_phone_key):]}')
                                        phones_final.append(manual_replace_dict[manual_phone_key])
                                        # phone = phone[len(manual_replace_dict[manual_phone_key]):]
                                        phone = phone[len(manual_phone_key):]
                                        if phone in manual_replace_dict.keys():
                                            phones_final.append(manual_replace_dict[phone])
                                            phone = ""


    # print(f'phones_final, {phones_final}')

    phones_final_post = []
    for phone in phones_final:
        if phone in manual_phone_replacements.keys():
            phones_final_post.append(manual_phone_replacements[phone])
        else:
            phones_final_post.append(phone)

    # print(f'phones_final, {phones_final}')
    # print(f'phones_final_post, {phones_final_post}')
    return " ".join(phones_final_post)



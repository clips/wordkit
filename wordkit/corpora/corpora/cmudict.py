"""Tools for working with CMUDICT."""
import re

import pandas as pd

from wordkit.corpora.base import reader

CMU_2IPA = {
    "AO": "ɔ",
    "AO0": "ɔ",
    "AO1": "ɔ",
    "AO2": "ɔ",
    "AA": "ɑ",
    "AA0": "ɑ",
    "AA1": "ɑ",
    "AA2": "ɑ",
    "IY": "i",
    "IY0": "i",
    "IY1": "i",
    "IY2": "i",
    "UW": "u",
    "UW0": "u",
    "UW1": "u",
    "UW2": "u",
    "EH": "e",
    "EH0": "e",
    "EH1": "e",
    "EH2": "e",
    "IH": "ɪ",
    "IH0": "ɪ",
    "IH1": "ɪ",
    "IH2": "ɪ",
    "UH": "ʊ",
    "UH0": "ʊ",
    "UH1": "ʊ",
    "UH2": "ʊ",
    "AH": "ʌ",
    "AH0": "ə",
    "AH1": "ʌ",
    "AH2": "ʌ",
    "AE": "æ",
    "AE0": "æ",
    "AE1": "æ",
    "AE2": "æ",
    "AX": "ə",
    "AX0": "ə",
    "AX1": "ə",
    "AX2": "ə",
    "EY": "eɪ",
    "EY0": "eɪ",
    "EY1": "eɪ",
    "EY2": "eɪ",
    "AY": "aɪ",
    "AY0": "aɪ",
    "AY1": "aɪ",
    "AY2": "aɪ",
    "OW": "oʊ",
    "OW0": "oʊ",
    "OW1": "oʊ",
    "OW2": "oʊ",
    "AW": "aʊ",
    "AW0": "aʊ",
    "AW1": "aʊ",
    "AW2": "aʊ",
    "OY": "ɔɪ",
    "OY0": "ɔɪ",
    "OY1": "ɔɪ",
    "OY2": "ɔɪ",
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "ɡ",
    "CH": "tʃ",
    "JH": "dʒ",
    "F": "f",
    "V": "v",
    "TH": "θ",
    "DH": "ð",
    "S": "s",
    "Z": "z",
    "SH": "ʃ",
    "ZH": "ʒ",
    "HH": "h",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "L": "l",
    "R": "r",
    "ER": "ɜr",
    "ER0": "ɜr",
    "ER1": "ɜr",
    "ER2": "ɜr",
    "AXR": "ər",
    "AXR0": "ər",
    "AXR1": "ər",
    "AXR2": "ər",
    "W": "w",
    "Y": "j",
}


def cmu_to_ipa(phonemes):
    """Convert CMU phonemes to IPA unicode format."""
    return tuple([CMU_2IPA[p] for p in phonemes])


def _open(path, **kwargs):
    """Open a file for reading."""
    df = []
    for line in open(path):
        line = line.split("#")[0]
        word, *rest = line.strip().split()
        word = brackets.sub("", word)
        df.append({"orthography": word, "phonology": rest})

    return pd.DataFrame(df)


brackets = re.compile(r"\(\d\)")


def cmu(path, fields=("orthography", "phonology"), language=None):
    """Extract structured information from CMUDICT."""
    return reader(
        path,
        fields,
        {"orthography": "orthography", "phonology": "phonology"},
        language="eng",
        opener=_open,
        preprocessors={"phonology": cmu_to_ipa},
    )


# Transpiled from here: https://github.com/virgil-av/numbers-to-words-romanian

import re
import math

TEN = 10
ONE_HUNDRED = 100
ONE_THOUSAND = 1000
ONE_MILLION = 1000000
ONE_BILLION = 1000000000
ONE_TRILLION = 1000000000000

LESS_THAN_TWENTY = [
    'zero',
    'unu',
    'doi',
    'trei',
    'patru',
    'cinci',
    'șase',
    'șapte',
    'opt',
    'nouă',
    'zece',
    'unsprezece',
    'doisprezece',
    'treisprezece',
    'paisprezece',
    'cincisprezece',
    'șaisprezece',
    'șaptesprezece',
    'optsprezece',
    'nouăsprezece',
]
TENTHS_LESS_THAN_HUNDRED = [
    'zero',
    'zece',
    'douăzeci',
    'treizeci',
    'patruzeci',
    'cincizeci',
    'șaizeci',
    'șaptezeci',
    'optzeci',
    'nouăzeci',
]

def parseDecimals (nr):
    decimals = int(str(int(nr*100)/100).split(".")[1])
    word = ""
    if decimals>0:
        word += " virgulă"

        if decimals < 10:
            word += "zero "
        word += generateWords(decimals, [])
    return word

def match (nr, numberUnitsSingular, numberUnitsPlural):
    string = ""

    if nr==1:
        string = numberUnitsSingular

    elif nr==2:
        string = "două " + numberUnitsPlural

    elif nr<20 or (nr > 100 and nr % 100 < 20):
        string = generateWords(nr, []) + " " + numberUnitsPlural

    else:
        words = generateWords(nr, [])
        if nr % 10 == 2:
            words = re.sub("doi$", "două",  words)
        string = words + " de " + numberUnitsPlural

    return string


def generateWords (nr, words = []):
    remainder = 0
    word = ""

    if math.isnan(nr):
        return "nan"

    if nr > ONE_TRILLION - 1:
        return nr

    if nr==0:
        if len(words):
            words = " ".join(words)
            words = re.sub(",$", "",  words)
            words = re.sub("\s{2,}", " ",  words)
            return words
        else:
            return "zero"

    if nr<0:
        words.append("minus")
        nr = abs(nr)


    if nr<20:
        remainder = 0
        word = LESS_THAN_TWENTY[int(nr)]
        word += parseDecimals(nr)

    elif nr < ONE_HUNDRED:
        remainder = int(nr % TEN)
        word = TENTHS_LESS_THAN_HUNDRED[int(nr / TEN)]
        if remainder:
            word += " și "

    elif nr < ONE_THOUSAND:
        remainder = nr % ONE_HUNDRED
        hundreds = int(nr / ONE_HUNDRED)

        if hundreds==1:
            word = "o sută"
        elif hundreds==2:
            word = "două sute"
        else:
            word = generateWords(hundreds, []) + " sute"

    elif nr<ONE_MILLION:
        remainder = nr % ONE_THOUSAND
        thousands = int(nr / ONE_THOUSAND)
        word = match(thousands, "o mie", "mii")

    elif nr<ONE_BILLION:
        remainder = nr % ONE_MILLION
        millions = int(nr / ONE_MILLION)
        word = match(millions, "un milion", "milioane")

    elif nr<ONE_TRILLION:
        remainder = nr % ONE_BILLION
        billions = int(nr/ONE_BILLION)
        word = match(billions, "un miliard", "miliarde")

    words.append(word)
    return generateWords(remainder, words)


if __name__ == '__main__':
    # Test

    print(generateWords(100))
    print(generateWords(125))
    print(generateWords(118931))
    print(generateWords(1259631))
    print(generateWords(101230465))
    print(generateWords(5101230465))
    print(generateWords(999999999999))


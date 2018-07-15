"use strict"

// const unidecode = require("unidecode")
const valid_symbols = [
  "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
  "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
  "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY",
  "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1",
  "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0",
  "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2", "UW",
  "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]
const _arpabet = valid_symbols.map(s => `@${s}`)
const symbols = "_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ".split("").concat(_arpabet)
const _symbol_to_id = {}
symbols.forEach((s, si) => _symbol_to_id[s] = si)
const _id_to_symbol = {} // refactor note, this is not needed, it behaves just like the original array
symbols.forEach((s, si) => _id_to_symbol[si] = s)

// const convert_to_ascii = text => unidecode(text)

// === Numbers start
const digitsTeens = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
const tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
const numGroups = ["", "thousand", "million"]

const numToString = num => {
    num = parseInt(num)

    const parts = []
    let chunkCount = 0

    while (num > 0) {
        if (num % 1000 != 0) {
            const chunk = numChunkToString(parseInt(num%1000)) + " " + numGroups[chunkCount]
            parts.unshift(chunk)
        }
        num = parseInt(num/1000)
        chunkCount++
    }

    return parts.join(" ").trim()
}

const numChunkToString = num => {

    const parts = []

    if (num>=100) {
        parts.push(digitsTeens[parseInt(num/100)])
        parts.push("hundred")
        num %= 100
    }

    if (num>=10 && num<=19) {
        parts.push(digitsTeens[num])
    } else if (num>=20) {
        parts.push(tens[parseInt(num/10)])
        num %= 10
    }

    if (num >= 1 && num <= 9) {
        parts.push(digitsTeens[num])
    }

    return parts.join(" ")
}

const isNumber = n => {
    return parseInt(n).toString().length == n.length
}

const normalize_numbers = text => {
    return text.split(" ").map(w => {

        if (!Number.isNaN(parseInt(w))) {
            w = w.replace(",", "")

            const ordinal = w.match(/[0-9]+(st|nd|rd|th)$/i)
            if (ordinal) {

                w = parseInt(w)

                if (w>20) {
                    switch (ordinal[1]) {
                        case "st":
                            w = numToString(parseInt(w/10)*10) + " first"
                            break
                        case "nd":
                            w = numToString(parseInt(w/10)*10) + " second"
                            break
                        case "rd":
                            w = numToString(parseInt(w/10)*10) + " third"
                            break
                        default:
                            w = numToString(w) + "th"
                            break
                    }
                } else if (w<10) {
                    switch (w) {
                        case 1:
                            w = "first"
                            break
                        case 2:
                            w = "second"
                            break
                        case 3:
                            w = "third"
                            break
                        default:
                            w = numToString(w) + "th"
                            break
                    }
                } else {
                    w = numToString(w) + "th"
                }

                return w
            }
        }

        if (!Number.isNaN(parseInt(w))) {

            w = w.replace(",", "")

            const sEnd = w.endsWith("s")
            w = w.replace(/s$/, "")

            if (isNumber(w)) {
                // Plain number
                w = numToString(w)

            } else if (w.endsWith("%")) {
                // Percent
                w = numToString(w) + " percent"
            } else if (w.includes(":") && !Number.isNaN(parseInt(w.split(":")[0])) && !Number.isNaN(parseInt(w.split(":")[1]))) {

                const mins = w.split(":")[1]
                w = w.split(":")[0]

                if (w >= 13) {
                    w -= 12
                }

                switch (mins) {
                    case "00":
                        w = `${w} o"clock`
                        break
                    case "15":
                        w = `quarter past ${numToString(w)}`
                        break
                    case "30":
                        w = `half past ${numToString(w)}`
                        break
                    default:
                        w = `${numToString(w)} ${numToString(mins)}`
                }
            }

            if (sEnd) {
                w += "s"
            }
        }

        if (w.startsWith("$")) {
            w = numToString(w.slice(1, w.length)) + " dollars"
        }
        if (w.startsWith("£")) {
            w = numToString(w.slice(1, w.length)) + " pounds"
        }
        if (w.startsWith("€")) {
            w = numToString(w.slice(1, w.length)) + " euros"
        }

        return w
    }).join(" ")
}
// === Numbers end

const expand_abbreviations = text => {
    return text.trim()
        .split(" ")
        .map(w => {
            return w
                .replace(/\smrs\s/ig, "misses")
                .replace(/\smr\s/ig, "mister")
                .replace(/\sdr\s/ig, "doctor")
                .replace(/\sst\s/ig, "saint")
                .replace(/\sco\s/ig, "company")
                .replace(/\sjr\s/ig, "junior")
                .replace(/\smaj\s/ig, "major")
                .replace(/\sgen\s/ig, "general")
                .replace(/\sdrs\s/ig, "doctors")
                .replace(/\srev\s/ig, "reverend")
                .replace(/\slt\s/ig, "lieutenant")
                .replace(/\shon\s/ig, "honorable")
                .replace(/\ssgt\s/ig, "sergeant")
                .replace(/\scapt\s/ig, "captain")
                .replace(/\sesq\s/ig, "esquire")
                .replace(/\sltd\s/ig, "limited")
                .replace(/\scol\s/ig, "colonel")
                .replace(/\sft\s/ig, "fort")
                .replace(/\s\s\s+/g, " ")
        }).join(" ")
}

const english_cleaners = text => {
    // return expand_abbreviations(normalize_numbers(convert_to_ascii(text).toLowerCase())).replace(/\s+/, " ")
    return expand_abbreviations(normalize_numbers(text.toLowerCase())).replace(/\s+/, " ")
}

const _should_keep_symbol = s => {
    return _symbol_to_id.hasOwnProperty(s) && s !== "_" && s !== "~"
}
const _symbols_to_sequence = symbols => {
    symbols = Array.isArray(symbols) ? symbols : symbols.split("")
    return symbols.filter(s => _should_keep_symbol(s)).map(s => _symbol_to_id[s])
}
const _arpabet_to_sequence = text => {
    return _symbols_to_sequence(text.split().map(s => `@${s}`))
}

const curlyRE = /(.*?)\{(.+?)\}(.*)/

const text_to_sequence = (text) => {

    const sequence = []

    while (text.length) {
        const match = text.match(curlyRE)

        if (!match) {
            const symbols = _symbols_to_sequence(english_cleaners(text))
            symbols.forEach(s => sequence.push(s))
            break
        }

        const symbols2 = _symbols_to_sequence(english_cleaners(match[1]))
        symbols2.forEach(s => sequence.push(s))
        const symbols3 = _arpabet_to_sequence(english_cleaners(match[2]))
        symbols3.forEach(s => sequence.push(s))
        text = match[3]
    }
    sequence.push(_symbol_to_id["~"])
    return sequence
}

exports.text_to_sequence = text_to_sequence
exports.english_cleaners = english_cleaners
"use strict"


// This doesn't seem to work anymore. TODO, make it work again
function getCaretPosition() {
    var sel = document.selection, range, rect;
    var x = 0, y = 0;
    if (sel) {
        if (sel.type != "Control") {
            range = sel.createRange();
            range.collapse(true);
            x = range.boundingLeft;
            y = range.boundingTop;
        }
    } else if (window.getSelection) {
        sel = window.getSelection();
        if (sel.rangeCount) {
            range = sel.getRangeAt(0).cloneRange();
            if (range.getClientRects) {
                range.collapse(true);
                if (range.getClientRects().length>0){
                    rect = range.getClientRects()[0];
                    x = rect.left;
                    y = rect.top;
                }
            }
            // Fall back to inserting a temporary element
            if (x == 0 && y == 0) {
                var span = document.createElement("span");
                if (span.getClientRects) {
                    // Ensure span has dimensions and position by
                    // adding a zero-width space character
                    span.appendChild( document.createTextNode("\u200b") );
                    range.insertNode(span);
                    rect = span.getClientRects()[0];
                    x = rect.left;
                    y = rect.top;
                    var spanParent = span.parentNode;
                    spanParent.removeChild(span);

                    // Glue any broken text nodes back together
                    spanParent.normalize();
                }
            }
        }
    }
    return { x: x, y: y };
}


window.ARPABET_SYMBOLS_v2 = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1","AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",     "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
    "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]

let ARPABET_SYMBOLS = [
  'AA0', 'AA1', 'AA2', 'AA', 'AE0', 'AE1', 'AE2', 'AE', 'AH0', 'AH1', 'AH2', 'AH',
  'AO0', 'AO1', 'AO2', 'AO', 'AW0', 'AW1', 'AW2', 'AW', 'AY0', 'AY1', 'AY2', 'AY',
  'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'EH', 'ER0', 'ER1', 'ER2', 'ER',
  'EY0', 'EY1', 'EY2', 'EY', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IH', 'IY0', 'IY1',
  'IY2', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OW', 'OY0',
  'OY1', 'OY2', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UH',
  'UW0', 'UW1', 'UW2', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

let extra_arpabet_symbols = [
    "AX",
    "AXR",
    "IX",
    "UX",
    "DX",
    "EL",
    "EM",
    "EN0",
    "EN1",
    "EN2",
    "EN",
    "NX",
    "Q",
    "WH",
]
let new_arpabet_symbols = [
    "RRR",
    "HR",
    "OE",
    "RH",
    "TS",

    "RR",
    "UU",
    "OO",
    "KH",
    "SJ",
    "HJ",
    "BR",
]
window.ARPABET_SYMBOLS_v3 = ARPABET_SYMBOLS.concat(extra_arpabet_symbols).concat(new_arpabet_symbols).sort((a,b)=>a<b?-1:1)

const preprocess = (caretStart) => {

    const defaultLang = "en"
    let text = dialogueInput.value

    const finishedParts = []
    // const parts = text.split(/\\lang\{[a-z]{0,2}\}\{/gi)
    const parts = text.split(/\\lang\[[a-z]{0,2}\]\[/gi)
    // const matchIterator = text.matchAll(/\\lang\{[a-z]{0,2}\}\{/gi)
    const matchIterator = text.matchAll(/\\lang\[[a-z]{0,2}\]\[/gi)
    finishedParts.push([defaultLang, parts[0]])

    const langStack = []

    // console.log("parts", parts)
    let textCounter = parts[0].length
    let caretInARPAbet = false

    parts.forEach((part,pi) => {
        if (pi) {
            const match = matchIterator.next().value[0]
            const langCode = match.split("lang[")[1].split("]")[0]
            langStack.push(langCode)
            // console.log("add langStack", langStack)

            textCounter += match.length

            let unescaped_part = ""

            part.split("]").forEach(sub_part => {
                // console.log("sub_part", sub_part, textCounter, textCounter+sub_part.length)

                if (caretStart > textCounter && caretStart <textCounter+sub_part.length) {
                    caretInARPAbet = true

                                                        // If backspace-ing a "{", and the next non-space character is "}", then delete that also

                                                        // Add ctrl+<space> to open the auto-complete

                }

                // if (sub_part.includes("[")) {
                //     unescaped_part += sub_part+"]"
                //     return
                // }

                sub_part = unescaped_part+sub_part

                unescaped_part = ""

                if (part.includes("]")) {
                    finishedParts.push([langStack.pop()||defaultLang, sub_part])
                } else {
                    finishedParts.push([langStack[langStack.length-1], sub_part])
                }
                // finishedParts.push([langStack.pop()||defaultLang, sub_part])
            })
        }
    })
    return caretInARPAbet
}

const hideAutocomplete = () => {
    textEditorTooltip.style.display = "none"
    textEditorTooltip.innerHTML = ""
    autocomplete_callback = undefined
}
let textWrittenSinceAutocompleteWasShown = ""
const filterOrHideAutocomplete = () => {
    if (textEditorTooltip.style.display=="flex") {
        highlightedAutocompleteIndex = 0
        let childrenShown = 0
        Array.from(textEditorTooltip.children).forEach(child => {
            if (child.classList.contains("autocomplete_option_active")) {
                child.classList.toggle("autocomplete_option_active")
            }

            if (child.innerText.toLowerCase().startsWith(textWrittenSinceAutocompleteWasShown) || textWrittenSinceAutocompleteWasShown.length==0) {
                child.style.display = "flex"
                childrenShown += 1
            } else {
                child.style.display = "none"
            }
        })
        if (childrenShown==0) {
            hideAutocomplete()
            return
        }
        setHighlightedAutocomplete(0, true)
    }
}
let autocomplete_callback = undefined
const showAutocomplete = (options, callback) => {
    const position = getCaretPosition(dialogueInput)

    // The getCaretPosition function doesn't work anymore. At least center it
    position.x = window.visualViewport.width/2

    textEditorTooltip.style.left = position.x + "px"
    textEditorTooltip.style.top = position.y + "px"

    highlightedAutocompleteIndex = 0
    textWrittenSinceAutocompleteWasShown = ""
    autocomplete_callback = callback

    options.forEach(option => {
        const optElem = createElem("div.autocomplete_option", option[0])
        optElem.dataset.autocomplete_return = option.length>1 ? option[1] : option[0]
        optElem.addEventListener("click", () => {
            callback(optElem.dataset.autocomplete_return)
            hideAutocomplete()
            refreshText()
        })
        textEditorTooltip.appendChild(optElem)
    })
    textEditorTooltip.style.display = "flex"
    setHighlightedAutocomplete(0, true)
    return
}

let highlightedAutocompleteIndex = 0
const setHighlightedAutocomplete = (delta, override=false) => {

    let NEW_highlightedAutocompleteIndex = Math.min(textEditorTooltip.children.length-1, Math.max(0, highlightedAutocompleteIndex+delta))
    if (override || NEW_highlightedAutocompleteIndex != highlightedAutocompleteIndex) {
        if (!override) {
            textEditorTooltip.children[highlightedAutocompleteIndex].classList.toggle("autocomplete_option_active")
        }
        textEditorTooltip.children[override ? delta : NEW_highlightedAutocompleteIndex].classList.toggle("autocomplete_option_active")
        highlightedAutocompleteIndex = NEW_highlightedAutocompleteIndex
        // textEditorTooltip.children[highlightedAutocompleteIndex].scrollIntoView()
    }
}


dialogueInput.addEventListener("keydown", event => {
    if (event.key=="Tab" || event.key=="Enter") {
        event.preventDefault()

        if (autocomplete_callback!==undefined) {
            autocomplete_callback(textEditorTooltip.children[highlightedAutocompleteIndex].dataset.autocomplete_return)
            hideAutocomplete()
            refreshText()
            return
        }

        let cursorIndex = dialogueInput.selectionStart
        if (cursorIndex) {
            // Move into the next [] bracket if already wrote down language
            let textPreCursor = dialogueInput.value.slice(0, cursorIndex)
            let textPostCursor = dialogueInput.value.slice(cursorIndex, dialogueInput.value.length)

            if (textPreCursor.slice(textPreCursor.length-8, 7)=="\\lang[") {
                dialogueInput.setSelectionRange(cursorIndex+2,cursorIndex+2)

            // Move out of [] if at the end
            } else if (textEditorTooltip.style.display=="none" && (textPostCursor.startsWith("]") || textPostCursor.startsWith("}"))) {
                console.log("moving one")
                dialogueInput.setSelectionRange(cursorIndex+1,cursorIndex+1)
            }

        }

    }
})

const splitWords = (sequence, addSpace) => {
    const words = []

    // const sequenceProcessed = sequence
    const sequenceProcessed = [] // Do further processing to also split on { } symbols, not just spaces
    sequence.forEach(word => {
        if (word.includes("{")) {
            word.split("{").forEach((w, wi) => {
                sequenceProcessed.push(wi ? ["{"+w, addSpace] : [w, false])
            })
        } else if (word.includes("}")) {
            word.split("}").forEach((w, wi) => {
                sequenceProcessed.push(wi ? [w, addSpace] : [w+"}", false])
            })
        } else {
            sequenceProcessed.push([word, addSpace])
        }
    })

    sequenceProcessed.forEach(([word, addSpace]) => {

        if (word.startsWith("\\lang[")) {
            words.push(word.split("][")[0]+"][")
            word = word.split("][")[1]
        }

        ["}","]","[","{"].forEach(char => {
            if (word.startsWith(char)) {
                words.push(char)
                word = word.slice(1,word.length)
            }
        })

        const endExtras = [];

        ["}","]","[","{"].forEach(char => {
            if (word.endsWith(char)) {
                endExtras.push(char)
                word = word.slice(0,word.length-1)
            }
        })

        words.push(word)
        endExtras.reverse().forEach(extra => words.push(extra))

        // if (word.startsWith("{")) {
        //     split_words.push("{")
        //     word = word.slice(1,word.length)
        // }
        // if (word.endsWith("}")) {
        //     split_words.push("{")
        //     word = word.slice(1,word.length)
        // }

        if (addSpace) {
            words.push(" ")
        }
    })

    return words
}

window.refreshText = () => {
    let all_text = dialogueInput.value
    textEditorElem.innerHTML = ""

    let openedCurlys = 0
    let openedLangs = 0

    let split_words = splitWords(all_text.split(" "), true)
    split_words = splitWords(split_words)
    split_words = splitWords(split_words)
    split_words = splitWords(split_words)

    // console.log("split_words", split_words)
    // dfgd()

    let caretCounter = 0
    let caretInARPAbet = false

    let firstOpenCurly = undefined
    let lastOpenCurly = undefined

    split_words.forEach(word => {

        if (caretCounter<=dialogueInput.selectionStart && (caretCounter+word.length)>dialogueInput.selectionStart) {
            // console.log(`caret (${dialogueInput.selectionStart}) in counter (${caretCounter}): `, word, openedCurlys, openedLangs)
            caretInARPAbet = openedCurlys > 0
        }
        caretCounter += word.length
        const spanElem = createElem("span.manyWhitespace", word)

        if (word.startsWith("\\lang[")) {
            openedLangs += 1
        }
        if (word.startsWith("{")) {
            openedCurlys += 1
            if (!caretInARPAbet) {
                firstOpenCurly = spanElem
            }
        }

        if (openedCurlys) {
            spanElem.style.fontWeight = "bold"
            spanElem.style.fontStyle = "italic"
        }
        if (openedLangs) {
            spanElem.style.background = "rgba(50, 150, 250, 0.2)"
        }

        ///====
        if (word.includes("part-highlighted")) {
            spanElem.style.textDecoration = "underline dotted red"
        } else  if (word.includes("highlighted")) {
            spanElem.style.textDecoration = "underline solid red"
        }
        ///====

        if (word.endsWith("]")) {
            openedLangs -= 1
        }
        if (word.endsWith("}")) {
            openedCurlys -= 1
            if (caretInARPAbet && lastOpenCurly===undefined) {
                lastOpenCurly = spanElem
            }
        }

        textEditorElem.appendChild(spanElem)

    })

    preprocess(dialogueInput.selectionStart)
    return [caretInARPAbet, firstOpenCurly, lastOpenCurly]
}

const languagesList = Object.keys(window.supportedLanguages)

const insertText = (inputTextArea, textToInsert, caretOffset=0) => {
    let cursorIndex = inputTextArea.selectionStart
    inputTextArea.value = inputTextArea.value.slice(0, cursorIndex) + textToInsert + inputTextArea.value.slice(cursorIndex, inputTextArea.value.length)
    caretOffset += textToInsert.length
    inputTextArea.setSelectionRange(cursorIndex+caretOffset,cursorIndex+caretOffset)
    refreshText()
}

dialogueInput.addEventListener("keydown", event => {

    generateVoiceButton.disabled = !dialogueInput.value.length

    if (event.key=="Enter") {
        event.stopPropagation()
        event.preventDefault()
        return
    }

    if (textEditorTooltip.style.display=="flex" && (event.key=="ArrowDown" || event.key=="ArrowUp" || (!window.shiftKeyIsPressed && event.key=="ArrowLeft") || (!window.shiftKeyIsPressed && event.key=="ArrowRight"))) {
    // if (textEditorTooltip.style.display=="flex" && (event.key=="ArrowDown" || event.key=="ArrowUp")) {
        event.stopPropagation()
        event.preventDefault()
        return
    }
    if (event.key=="}") {
        if (dialogueInput.value.slice(dialogueInput.selectionStart, dialogueInput.value.length-1).startsWith("}")) {
            dialogueInput.setSelectionRange(dialogueInput.selectionStart+1, dialogueInput.selectionStart+1)
            event.stopPropagation()
            event.preventDefault()
            return
        }
    }
    if (event.key=="]") {
        if (dialogueInput.value.slice(dialogueInput.selectionStart, dialogueInput.value.length-1).startsWith("]")) {
            dialogueInput.setSelectionRange(dialogueInput.selectionStart+1, dialogueInput.selectionStart+1)
            event.stopPropagation()
            event.preventDefault()
            return
        }
    }
})

let is_doing_gp2 = false
window.get_g2p = (text_to_g2p) => {
    return new Promise(resolve => {
        doFetch("http://localhost:8008/getG2P", {method: "Post", body: JSON.stringify({base_lang: base_lang_select.value, text: text_to_g2p})})
        .then(r=>r.text()).then(res => {
            is_doing_gp2 = false
            resolve(res)
        })
    })
}

const handleTextUpdate = (event) => {
    generateVoiceButton.disabled = !dialogueInput.value.length

    window.shiftKeyIsPressed = event.shiftKey

    if (textEditorTooltip.style.display=="flex" && (event.type=="click" || (!window.shiftKeyIsPressed && event.key=="ArrowDown") || (!window.shiftKeyIsPressed && event.key=="ArrowRight"))) {
        event.stopPropagation()
        event.preventDefault()
        setHighlightedAutocomplete(1)
        return
    }
    if (textEditorTooltip.style.display=="flex" && (event.type=="click" || (!window.shiftKeyIsPressed && event.key=="ArrowLeft") || (!window.shiftKeyIsPressed && event.key=="ArrowUp"))) {
        event.stopPropagation()
        event.preventDefault()
        setHighlightedAutocomplete(-1)
        return
    }


    if (event.type!="click" && (event.key=="Shift" || event.key=="Control")) {
        event.stopPropagation()
        event.preventDefault()
        return
    }

    const [caretInARPAbet, firstOpenCurly, lastOpenCurly] = refreshText()
    if (caretInARPAbet) {
        firstOpenCurly && (firstOpenCurly.style.color = "red")
        lastOpenCurly && (lastOpenCurly.style.color = "red")
    }


    textEditorElem.scrollTop = dialogueInput.scrollTop


    if (dialogueInput.selectionStart!=dialogueInput.selectionEnd && !is_doing_gp2) {

        hideAutocomplete()

        showAutocomplete([["&lt;Convert to phonemes&gt;"]], () => {

            const text_to_g2p = dialogueInput.value.slice(dialogueInput.selectionStart, dialogueInput.selectionEnd)
            is_doing_gp2 = true

            get_g2p(text_to_g2p).then(phonemes => {
                const initialStart = dialogueInput.selectionStart
                dialogueInput.value = dialogueInput.value.slice(0, dialogueInput.selectionStart) + dialogueInput.value.slice(dialogueInput.selectionEnd, dialogueInput.value.length)

                dialogueInput.selectionStart = initialStart

                insertText(dialogueInput, phonemes, 0)
            })
        })

    } else
    // } else {
        if (event.type!="click" && event.key.length==1 && event.key.match(/[a-z]/i)) {
            textWrittenSinceAutocompleteWasShown += event.key.toLowerCase()
            filterOrHideAutocomplete()
        } else if (event.type!="click" && event.key=="Backspace") {
            if (textWrittenSinceAutocompleteWasShown.length==0) {
                hideAutocomplete()
            } else {
                textWrittenSinceAutocompleteWasShown = textWrittenSinceAutocompleteWasShown.slice(0,textWrittenSinceAutocompleteWasShown.length-1)
                filterOrHideAutocomplete()
            }
        } else {
            hideAutocomplete()
        }

        const ctrlSpace = event.ctrlKey && event.code=="Space"

        if (event.type!="click" && (event.key=="{" || event.key=="") || ctrlSpace) {
            if (!ctrlSpace && event.key!="") {
                insertText(dialogueInput, "}", -1)
            }

            if (caretInARPAbet) {
                let symbols = window.ARPABET_SYMBOLS_v3
                if (window.currentModel&&window.currentModel.modelType=="FastPitch1.1") {
                    symbols = window.ARPABET_SYMBOLS_v2
                } else if (window.currentModel&&window.currentModel.modelType=="FastPitch") {
                    symbols = ["&lt;ARPAbet only available for v2+ models&gt;"]
                }
                showAutocomplete(symbols.map(v=>{return [v]}), option => {
                    if (symbols.length>1) {
                        insertText(dialogueInput, option.slice(textWrittenSinceAutocompleteWasShown.length, option.length)+" ", 0)
                    }
                })
            }
            if (event.key!="") {
                handleTextUpdate({type: "keydown", key: ""})
            }
        }

        if (event.type!="click" && event.key=="\\") {
            // showAutocomplete([["\\lang[language][text]", "\\lang[][]"], ["\\sil[milliseconds]", "\\sil[]"]], (option) => {
            showAutocomplete([["\\lang[language][text]", "\\lang[][]"]], (option) => {

                if (option.includes("lang")) {
                    insertText(dialogueInput, option.slice(1, option.length), -3)
                } else {
                    insertText(dialogueInput, option.slice(1, option.length), -1)
                }

                setTimeout(() => {
                    showAutocomplete(languagesList.map(v=>{return [v]}), option => {
                        insertText(dialogueInput, option.slice(textWrittenSinceAutocompleteWasShown.length, option.length), 2)
                    })
                }, 100)
            })
        }
    // }
// })
}
dialogueInput.addEventListener("click", event => handleTextUpdate(event))
dialogueInput.addEventListener("keyup", event => handleTextUpdate(event))
refreshText()
setTimeout(window.refreshText, 500)

window.addEventListener("click", event => {
    if (event.target && event.target!=textEditorTooltip && event.target.className && event.target.className.includes && !event.target.className.includes("autocomplete_option")) {
        hideAutocomplete()
    }
})
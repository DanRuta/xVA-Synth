"use strict"

window.arpabetMenuState = {
    currentDict: undefined,
    dictionaries: {},
    paginationIndex: 0,
    totalPages: 0,
    clickedRecord: undefined
}

window.ARPAbetSymbols = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "B", "CH", "D", "DH", "EH", "EL", "EM", "EN", "ER", "EY", "F", "G", "HH", "IH", "IX", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "UX", "V", "W", "WH", "Y", "Z", "ZH"]

window.refreshDictionariesList = () => {
    return new Promise(resolve => {
        window.arpabetMenuState.dictionaries = {}
        arpabet_dicts_list.innerHTML = ""

        const jsonFiles = fs.readdirSync(`${window.path}/arpabet`)
        jsonFiles.forEach(fname => {

            const dictId = fname.replace(".json", "")

            const jsonData = JSON.parse(fs.readFileSync(`${window.path}/arpabet/${fname}`, "utf8"))

            const dictButton = createElem("button", jsonData.title)
            dictButton.title = jsonData.description
            dictButton.style.background = window.currentGame ? `#${window.currentGame.themeColourPrimary}` : "#ccc"
            arpabet_dicts_list.appendChild(dictButton)

            window.arpabetMenuState.dictionaries[dictId] = jsonData//.data

            dictButton.addEventListener("click", ()=>handleDictClick(dictId))

        })
    })
}

window.handleDictClick = (dictId) => {

    if (window.arpabetMenuState.currentDict==dictId) {
        return
    }
    arpabet_enableall_button.disabled = false
    arpabet_disableall_button.disabled = false
    window.arpabetMenuState.currentDict = dictId
    window.arpabetMenuState.paginationIndex = 0
    window.arpabetMenuState.totalPages = 0

    arpabet_word_search_input.value = ""
    window.arpabetRunSearch()
    window.refreshDictWordList()
}

window.refreshDictWordList = () => {

    const dictId = window.arpabetMenuState.currentDict
    arpabetWordsListContainer.innerHTML = ""

    const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[dictId].filteredData)
    let startIndex = window.arpabetMenuState.paginationIndex*window.userSettings.arpabet_paginationSize
    const endIndex = Math.min(startIndex+window.userSettings.arpabet_paginationSize, wordKeys.length)

    window.arpabetMenuState.totalPages = Math.ceil(wordKeys.length/window.userSettings.arpabet_paginationSize)
    arpabet_pagination_numbers.innerHTML = window.i18n.PAGINATION_X_OF_Y.replace("_1", window.arpabetMenuState.paginationIndex+1).replace("_2", window.arpabetMenuState.totalPages)

    for (let i=startIndex; i<endIndex; i++) {
        const data = window.arpabetMenuState.dictionaries[dictId].filteredData[wordKeys[i]]
        const word = wordKeys[i]

        const rowElem = createElem("div.arpabetRow")
        const ckbx = createElem("input.arpabetRowItem", {type: "checkbox"})
        ckbx.checked = data.enabled
        ckbx.style.marginTop = 0
        ckbx.addEventListener("click", () => {
            window.arpabetMenuState.dictionaries[dictId].data[wordKeys[i]].enabled = ckbx.checked
            window.saveARPAbetDict(dictId)
        })

        const wordElem = createElem("div.arpabetRowItem", word)
        wordElem.title = word

        const arpabetElem = createElem("div.arpabetRowItem", data.arpabet)
        arpabetElem.title = data.arpabet


        rowElem.appendChild(createElem("div.arpabetRowItem", ckbx))
        rowElem.appendChild(wordElem)
        rowElem.appendChild(arpabetElem)

        rowElem.addEventListener("click", () => {
            window.arpabetMenuState.clickedRecord = {elem: rowElem, word}
            arpabet_word_input.value = word
            arpabet_arpabet_input.value = data.arpabet
        })

        arpabetWordsListContainer.appendChild(rowElem)
    }
}

window.saveARPAbetDict = (dictId) => {

    const dataOut = {
        title: window.arpabetMenuState.dictionaries[dictId].title,
        description: window.arpabetMenuState.dictionaries[dictId].description,
        version: window.arpabetMenuState.dictionaries[dictId].version,
        author: window.arpabetMenuState.dictionaries[dictId].author,
        nexusLink: window.arpabetMenuState.dictionaries[dictId].nexusLink,
        data: window.arpabetMenuState.dictionaries[dictId].data
    }

    doFetch(`http://localhost:8008/updateARPABet`, {
        method: "Post",
        body: JSON.stringify({})
    })//.then(r => r.text()).then(r => {console.log(r)})


    fs.writeFileSync(`${window.path}/arpabet/${dictId}.json`, JSON.stringify(dataOut, null, 4))
}

arpabet_save.addEventListener("click", () => {
    const word = arpabet_word_input.value.trim().toLowerCase()
    const arpabet = arpabet_arpabet_input.value.trim().toUpperCase().replace(/\s{2,}/g, " ")

    if (!word.length || !arpabet.length) {
        return window.errorModal(window.i18n.ARPABET_ERROR_EMPTY_INPUT)
    }

    const badSymbols = arpabet.split(" ").filter(symb => !window.ARPAbetSymbols.includes(symb))
    if (badSymbols.length) {
        return window.errorModal(window.i18n.ARPABET_ERROR_BAD_SYMBOLS.replace("_1", badSymbols.join(", ")))
    }

    const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)

    // Delete the old record
    if (window.arpabetMenuState.clickedRecord && window.arpabetMenuState.clickedRecord.word != word) {
        delete window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[window.arpabetMenuState.clickedRecord.word]
    }

    window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word] = {enabled: true, arpabet: arpabet}
    window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data

    window.refreshDictWordList()
    window.saveARPAbetDict(window.arpabetMenuState.currentDict)
})

arpabetModal.addEventListener("click", (event) => {
    if (window.arpabetMenuState.clickedRecord && event.target.className!="arpabetRow"&& event.target.className!="arpabetRowItem" && ![arpabet_word_input, arpabet_arpabet_input, arpabet_save, arpabet_prev_btn, arpabet_next_btn].includes(event.target)) {
        window.arpabetMenuState.clickedRecord = undefined
        arpabet_word_input.value = ""
        arpabet_arpabet_input.value = ""
    }
})
arpabet_prev_btn.addEventListener("click", () => {
    window.arpabetMenuState.paginationIndex = Math.max(0, window.arpabetMenuState.paginationIndex-1)
    window.refreshDictWordList()
})
arpabet_next_btn.addEventListener("click", () => {
    window.arpabetMenuState.paginationIndex = Math.min(window.arpabetMenuState.totalPages-1, window.arpabetMenuState.paginationIndex+1)
    window.refreshDictWordList()
})

window.arpabetRunSearch = () => {
    window.arpabetMenuState.paginationIndex = 0
    window.arpabetMenuState.totalPages = 0

    const query = arpabet_word_search_input.value.trim().toLowerCase()

    if (!query.length) {
        window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data
    } else {
        const filteredKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data).filter(key => key.includes(query))

        window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData = {}
        filteredKeys.forEach(key => {
            window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].filteredData[key] = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[key]
        })
    }

    window.refreshDictWordList()
}
let arpabetSearchInterval
arpabet_word_search_input.addEventListener("keyup", () => {
    if (arpabetSearchInterval!=null) {
        clearTimeout(arpabetSearchInterval)
    }
    arpabetSearchInterval = setTimeout(window.arpabetRunSearch, 500)
})

arpabet_enableall_button.addEventListener("click", () => {

    const dictName = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].title
    window.confirmModal(window.i18n.ARPABET_CONFIRM_ENABLE_ALL.replace("_1", dictName)).then(response => {
        if (response) {
            const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)
            wordKeys.forEach(word => {
                window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word].enabled = true
            })

            window.saveARPAbetDict(window.arpabetMenuState.currentDict)
            window.arpabetRunSearch()
        }
    })
})

arpabet_disableall_button.addEventListener("click", () => {

    const dictName = window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].title
    window.confirmModal(window.i18n.ARPABET_CONFIRM_DISABLE_ALL.replace("_1", dictName)).then(response => {
        if (response) {
            const wordKeys = Object.keys(window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data)
            wordKeys.forEach(word => {
                window.arpabetMenuState.dictionaries[window.arpabetMenuState.currentDict].data[word].enabled = false
            })

            window.saveARPAbetDict(window.arpabetMenuState.currentDict)
            window.arpabetRunSearch()
        }
    })
})




fs.watch(`${window.path}/arpabet`, {recursive: false, persistent: true}, (eventType, filename) => {refreshDictionariesList()})
refreshDictionariesList()
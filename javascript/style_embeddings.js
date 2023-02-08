"use strict"

window.allStyleEmbs = {}
window.styleEmbsMenuState = {
    embeddingsDir: `${window.path}/embeddings`,
    hasChangedEmb: false, // For clearing the use of the editor state when re-generating a line with a different embedding
    selectedEmb: undefined,
    activatedEmbeddings: {}
}


window.loadStyleEmbsFromDisk = () => {
    window.allStyleEmbs = {}

    // Read the activated embeddings file
    window.styleEmbsMenuState.activatedEmbeddings = {}
    if (fs.existsSync(`./embeddings.txt`)) {
        const embeddingsEnabled = fs.readFileSync(`./embeddings.txt`, "utf8").split("\n")
        embeddingsEnabled.forEach(emb => {
            window.styleEmbsMenuState.activatedEmbeddings[emb.replace("*","")] = emb.includes("*")
        })
    }


    // Read all the embedding files
    const embFiles = fs.readdirSync(window.styleEmbsMenuState.embeddingsDir)
    embFiles.forEach(jsonFName => {
        const jsonData = JSON.parse(fs.readFileSync(`${window.styleEmbsMenuState.embeddingsDir}/${jsonFName}`))
        jsonData.fileName = `${window.styleEmbsMenuState.embeddingsDir}/${jsonFName}`
        if (!Object.keys(window.styleEmbsMenuState.activatedEmbeddings).includes(jsonData.emb_id)) {
            window.styleEmbsMenuState.activatedEmbeddings[jsonData.emb_id] = true
        }
        jsonData.enabled = window.styleEmbsMenuState.activatedEmbeddings[jsonData.emb_id]
        window.allStyleEmbs[jsonData.voiceId] = window.allStyleEmbs[jsonData.voiceId] || []

        window.allStyleEmbs[jsonData.voiceId].push(jsonData)
    })
    window.saveEnabledStyleEmbs()
}
window.saveEnabledStyleEmbs = () => {
    fs.writeFileSync(`./embeddings.txt`, Object.keys(window.styleEmbsMenuState.activatedEmbeddings).map(key => {
        return `${window.styleEmbsMenuState.activatedEmbeddings[key]?"*":""}${key}`
    }).join("\n"), "utf8")
}
window.loadStyleEmbsFromDisk()


window.resetStyleEmbFields = () => {
    styleEmbAuthorInput.value = ""
    styleEmbGameIdInput.value = ""
    styleEmbVoiceIdInput.value = ""
    styleEmbNameInput.value = ""
    styleEmbDescriptionInput.value = ""
    styleEmbIdInput.value = ""
    wavFilepathForEmbComputeInput.value = ""
    styleEmbValuesInput.value = ""

    styleEmbGameIdInput.value = window.currentGame.gameId
    if (window.currentModel) {
        styleEmbVoiceIdInput.value = window.currentModel.voiceId
    }
}

window.refreshStyleEmbsTable = () => {
    styleembsRecordsContainer.innerHTML = ""
    window.styleEmbsMenuState.selectedEmb = undefined
    styleEmbDelete.disabled = true
    window.resetStyleEmbFields()


    Object.keys(window.allStyleEmbs).sort().forEach(key => {
        window.allStyleEmbs[key].forEach((emb,ei) => {
            const record = createElem("div")
            const enabledCkbx = createElem("input", {type: "checkbox"})
            enabledCkbx.checked = emb.enabled
            record.appendChild(createElem("div", enabledCkbx))

            const embName = createElem("div", emb["embeddingName"])
            embName.title = emb["embeddingName"]
            record.appendChild(embName)

            const embGameID = createElem("div", emb["gameId"])
            embGameID.title = emb["gameId"]
            record.appendChild(embGameID)

            const embVoiceID = createElem("div", emb["voiceId"])
            embVoiceID.title = emb["voiceId"]
            record.appendChild(embVoiceID)

            const embDescription = createElem("div", emb["description"]||"")
            embDescription.title = emb["description"]||""
            record.appendChild(embDescription)

            const embID = createElem("div", emb["emb_id"])
            embID.title = emb["emb_id"]
            record.appendChild(embID)

            const embVersion = createElem("div", emb["version"]||"1.0")
            embVersion.title = emb["version"]||"1.0"
            record.appendChild(embVersion)

            enabledCkbx.addEventListener("click", () => {
                window.allStyleEmbs[key][ei].enabled = !window.allStyleEmbs[key][ei].enabled
                window.styleEmbsMenuState.activatedEmbeddings[emb["emb_id"]] = window.allStyleEmbs[key][ei].enabled
                window.saveEnabledStyleEmbs()

                if (window.currentModel) {
                    window.loadStyleEmbsForVoice(window.currentModel)
                }
            })


            record.addEventListener("click", (e) => {
                if (e.target==enabledCkbx || e.target.nodeName=="BUTTON") {
                    return
                }
                // Clear visual selection of the old selected item, if there was already an item selected before
                if (window.styleEmbsMenuState.selectedEmb) {
                    window.styleEmbsMenuState.selectedEmb[0].style.background = "none"
                    Array.from(window.styleEmbsMenuState.selectedEmb[0].children).forEach(child => child.style.color = "white")
                }

                window.styleEmbsMenuState.selectedEmb = [record, emb]

                // Visually show that this row is selected
                window.styleEmbsMenuState.selectedEmb[0].style.background = "white"
                Array.from(window.styleEmbsMenuState.selectedEmb[0].children).forEach(child => child.style.color = "black")
                styleEmbDelete.disabled = false


                // Populate the edit fields
                styleEmbAuthorInput.value = emb.author||""
                styleEmbGameIdInput.value = emb.gameId||""
                styleEmbVoiceIdInput.value = emb.voiceId||""
                styleEmbNameInput.value = emb.embeddingName||""
                styleEmbDescriptionInput.value = emb.description||""
                styleEmbIdInput.value = emb.emb_id||""
                wavFilepathForEmbComputeInput.value = ""
                styleEmbValuesInput.value = emb.emb||""
            })

            styleembsRecordsContainer.appendChild(record)
        })
    })
}
styleembs_main.addEventListener("click", (e) => {
    if (e.target == styleembs_main) {
        window.refreshStyleEmbsTable()
    }
})

styleEmbSave.addEventListener("click", () => {

    const missingFieldsValues = []

    if (!styleEmbAuthorInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.AUTHOR)
    }
    if (!styleEmbGameIdInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.GAME_ID)
    }
    if (!styleEmbVoiceIdInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.VOICE_ID)
    }
    if (!styleEmbNameInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.EMB_NAME)
    }
    if (!styleEmbIdInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.EMB_ID)
    }
    if (!styleEmbValuesInput.value.trim().length) {
        missingFieldsValues.push(window.i18n.STYLE_EMB_VALUES)
    }

    if (missingFieldsValues.length) {
        window.errorModal(window.i18n.ERROR_MISSING_FIELDS.replace("_1", missingFieldsValues.join(", ")))
    } else {
        let outputFilename
        if (window.styleEmbsMenuState.selectedEmb) {
            outputFilename = window.styleEmbsMenuState.selectedEmb[1].fileName
        } else {
            outputFilename = `${window.styleEmbsMenuState.embeddingsDir}/${styleEmbVoiceIdInput.value.trim().toLowerCase()}.${styleEmbGameIdInput.value.trim().toLowerCase()}.${styleEmbIdInput.value.trim().toLowerCase()}.${styleEmbAuthorInput.value.trim().toLowerCase()}.json`
        }

        const jsonData = {
            "author": styleEmbAuthorInput.value.trim(),
            "version": "1.0", // Should I make this editable in the UI?
            "gameId": styleEmbGameIdInput.value.trim().toLowerCase(),
            "voiceId": styleEmbVoiceIdInput.value.trim().toLowerCase(),
            "description": styleEmbDescriptionInput.value.trim()||"",
            "embeddingName": styleEmbNameInput.value.trim(),
            "emb": styleEmbValuesInput.value.trim().split(",").map(v=>parseFloat(v)),
            "emb_id": styleEmbIdInput.value.trim()
        }

        fs.writeFileSync(outputFilename, JSON.stringify(jsonData, null, 4), "utf8")
        window.loadStyleEmbsFromDisk()
        window.refreshStyleEmbsTable()
    }
})


styleEmbDelete.addEventListener("click", () => {
    window.confirmModal(window.i18n.CONFIRM_DELETE_STYLE_EMB).then(response => {
        if (response) {
            fs.unlinkSync(window.styleEmbsMenuState.selectedEmb[1].fileName)
            window.loadStyleEmbsFromDisk()
            window.refreshStyleEmbsTable()
        }
    })
})


// Return the default embedding, plus any other ones
window.loadStyleEmbsForVoice = (currentModel) => {

    const embeddings = {}

    // Add the default option from the model json
    embeddings["default"] = [window.i18n.DEFAULT, currentModel.games[0].base_speaker_emb] // TODO, specialize to specific game?

    // Load any other style embeddings available
    if (Object.keys(window.allStyleEmbs).includes(currentModel.voiceId)) {
        window.allStyleEmbs[currentModel.voiceId].forEach(loadedStyleEmb => {
            if (loadedStyleEmb.enabled) {
                embeddings[loadedStyleEmb.emb_id] = [loadedStyleEmb.embeddingName, loadedStyleEmb.emb]
            }
        })
    }


    // Add every option to the embeddings selection dropdown
    style_emb_select.innerHTML = ""
    Object.keys(embeddings).forEach(key => {
        const opt = createElem("option", embeddings[key][0])
        opt.value = embeddings[key][1].join(",")
        style_emb_select.appendChild(opt)
    })
}
style_emb_select.addEventListener("change", () => window.styleEmbsMenuState.hasChangedEmb)



window.styleEmbsModalOpenCallback = () => {
    styleEmbGameIdInput.value = window.currentGame.gameId
    if (window.currentModel) {
        styleEmbVoiceIdInput.value = window.currentModel.voiceId
    }
    window.refreshStyleEmbsTable()
}
window.dragDropWavForEmbComputeFilepathInput = (eType, event) => {
    if (["dragenter", "dragover"].includes(eType)) {
        wavFileDragDropArea.style.background = "#5b5b5b"
        wavFileDragDropArea.style.color = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        wavFileDragDropArea.style.background = "rgba(0,0,0,0)"
        wavFileDragDropArea.style.color = "white"
    }

    event.preventDefault()
    event.stopPropagation()

    const dataLines = []

    if (eType=="drop") {
        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)

        if (files[0].path.endsWith(".wav")) {
            wavFilepathForEmbComputeInput.value = String(files[0].path).replaceAll(/\\/g, "/")
        } else {
            window.errorModal(window.i18n.ERROR_FILE_MUST_BE_WAV)
        }
    }
}

wavFileDragDropArea.addEventListener("dragenter", event => window.dragDropWavForEmbComputeFilepathInput("dragenter", event), false)
wavFileDragDropArea.addEventListener("dragleave", event => window.dragDropWavForEmbComputeFilepathInput("dragleave", event), false)
wavFileDragDropArea.addEventListener("dragover", event => window.dragDropWavForEmbComputeFilepathInput("dragover", event), false)
wavFileDragDropArea.addEventListener("drop", event => window.dragDropWavForEmbComputeFilepathInput("drop", event), false)



getStyleEmbeddingBtn.addEventListener("click", () => {
    if (!wavFilepathForEmbComputeInput.value.trim().length) {
        window.errorModal(window.i18n.ERROR_NEED_WAV_FILE)
    } else {
        doFetch(`http://localhost:8008/getWavV3StyleEmb`, {
            method: "Post",
            body: JSON.stringify({wav_path: wavFilepathForEmbComputeInput.value})
        }).then(r=>r.text()).then(v => {
            styleEmbValuesInput.value = v
        })
    }
})
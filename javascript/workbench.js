"use strict"
const er = require('@electron/remote')
const dialog = er.dialog


window.voiceWorkbenchState = {
    isInit: false,
    isStarted: false,
    currentAudioFilePath: undefined,
    newAudioFilePath: undefined,
    currentEmb: undefined,
    refAEmb: undefined,
    refBEmb: undefined,
}

window.initVoiceWorkbench = () => {
    if (!window.voiceWorkbenchState.isInit) {
        window.voiceWorkbenchState.isInit = true
        window.refreshExistingCraftedVoices()
        window.initDropdowns()
        voiceWorkbenchLanguageDropdown.value = "en"
    }
    window.refreshExistingCraftedVoices()
}

window.refreshExistingCraftedVoices = () => {
    voiceWorkbenchVoicesList.innerHTML = ""
    Object.keys(window.games).sort((a,b)=>a>b?1:-1).forEach(gameId => {
        const themeColour = window.gameAssets[gameId].themeColourPrimary
        window.games[gameId].models.forEach(model => {
            if (model.embOverABaseModel) {
                const button = createElem("div.voiceType", model.voiceName)
                button.style.background = `#${themeColour}`
                button.addEventListener("click", () => window.voiceWorkbenchLoadOrResetCraftedVoice(model))
                voiceWorkbenchVoicesList.appendChild(button)
            }
        })
    })
}

window.voiceWorkbenchLoadOrResetCraftedVoice = (model) => {

    voiceWorkbenchModelDropdown.value = model ? model.embOverABaseModel : "<base>/base_v1.0"
    voiceWorkbenchVoiceNameInput.value = model ? model.voiceName : ""
    voiceWorkbenchVoiceIDInput.value = model ? model.variants[0].voiceId : ""
    voiceWorkbenchGenderDropdown.value = model ? model.variants[0].gender : "male"
    voiceWorkbenchAuthorInput.value = model ? (model.variants[0].author || "Anonymous") : ""
    voiceWorkbenchLanguageDropdown.value = model ? model.variants[0].lang : "en"
    voiceWorkbenchGamesDropdown.value = model ? model.gameId : "other"
    voiceWorkbenchCurrentEmbeddingInput.value = model ? model.variants[0].base_speaker_emb : ""
    window.voiceWorkbenchState.currentEmb = model ? model.variants[0].base_speaker_emb : undefined

    window.voiceWorkbenchState.currentAudioFilePath = undefined
    window.voiceWorkbenchState.newAudioFilePath = undefined
    window.voiceWorkbenchState.refAEmb = undefined
    window.voiceWorkbenchState.refBEmb = undefined

    voiceWorkbenchRefAInput.value = ""
    voiceWorkbenchRefBInput.value = ""

    if (model) {
        voiceWorkbenchDeleteButton.disabled = false
        voiceWorkbenchStartButton.click()
    }
}

window.voiceWorkbenchGenerateVoice = async () => {

    if (!voiceWorkbenchCurrentEmbeddingInput.value.length) {
        return window.errorModal(window.i18n.ENTER_VOICE_CRAFTING_STARTING_EMB)
    }

    // Load the model if it hasn't been loaded already
    let voiceId = voiceWorkbenchModelDropdown.value.split("/").at(-1)
    if (!window.currentModel || window.currentModel.voiceId!=voiceId) {
        let modelPath
        if (voiceId.includes("Base xVAPitch Model")) {
            modelPath = `${window.path}/python/xvapitch/base_v1.0.pt`
        } else {
            const gameId = voiceWorkbenchModelDropdown.value.split("/").at(0)
            modelPath = window.userSettings[`modelspath_${gameId}`]+"/"+voiceId
        }
        await window.voiceWorkbenchChangeModel(modelPath, voiceId)
    }

    const base_lang = voiceWorkbenchLanguageDropdown.value//voiceId.includes("Base xVAPitch Model") ? "en" : window.currentModel.lang
    let newEmb = undefined

    // const currentEmbedding = voiceWorkbenchCurrentEmbeddingInput.value.split(",").map(v=>parseFloat(v))
    const currentEmbedding = window.voiceWorkbenchState.currentEmb
    // const currentDelta = voiceWorkbenchCurrentDeltaInput.value.split(",").map(v=>parseFloat(v))
    const newEmbedding = window.getVoiceWorkbenchNewEmbedding()

    const tempFileNum = `${Math.random().toString().split(".")[1]}`
    const currentTempFileLocation = `${path}/output/temp-${tempFileNum}_current.wav`
    const newTempFileLocation = `${path}/output/temp-${tempFileNum}_new.wav`

    // Do the current embedding first
    const synthRequests = []
    synthRequests.push(doSynth(JSON.stringify({
            sequence: voiceWorkbenchInputTextArea.value.trim(),
            useCleanup: true, // TODO, user setting?
            base_lang, base_emb: currentEmbedding.join(","), outfile: currentTempFileLocation
        })))
    let doingNewAudioFile = false
    if (voiceWorkbenchCurrentDeltaInput.value.length) {
        doingNewAudioFile = true
        synthRequests.push(doSynth(JSON.stringify({
            sequence: voiceWorkbenchInputTextArea.value.trim(),
            useCleanup: true, // TODO, user setting?
            base_lang, base_emb: newEmbedding.join(","), outfile: newTempFileLocation
        })))
    }

    // toggleSpinnerButtons()
    spinnerModal(`${window.i18n.SYNTHESIZING}`)
    Promise.all(synthRequests).then(res => {
        closeModal(undefined, [workbenchContainer])
        window.voiceWorkbenchState.currentAudioFilePath = currentTempFileLocation
        voiceWorkbenchAudioCurrentPlayPauseBtn.disabled = false
        voiceWorkbenchAudioCurrentSaveBtn.disabled = false

        if (doingNewAudioFile) {
            window.voiceWorkbenchState.newAudioFilePath = newTempFileLocation
            voiceWorkbenchAudioNewPlayBtn.disabled = false
            voiceWorkbenchAudioNewSaveBtn.disabled = false
        }
    })
}
const doSynth = (body) => {
    return new Promise(resolve => {
        doFetch("http://localhost:8008/synthesizeSimple", {
            method: "Post",
            body
        }).then(r=>r.text()).then(resolve)
    })
}

window.getVoiceWorkbenchNewEmbedding = () => {
    const currentDelta = voiceWorkbenchCurrentDeltaInput.value.split(",").map(v=>parseFloat(v))
    const newEmb = window.voiceWorkbenchState.currentEmb.map((v,vi) => {
        return v + currentDelta[vi]//*strength
    })
    return newEmb
}

window.voiceWorkbenchChangeModel = (modelPath, voiceId) => {
    window.currentModel = {
        outputs: undefined,
        model: modelPath.replace(".pt", ""),
        modelType: "xVAPitch",
        base_lang: voiceWorkbenchLanguageDropdown.value,
        isBaseModel: true,
        voiceId: voiceId
    }
    generateVoiceButton.dataset.modelQuery = JSON.stringify(window.currentModel)
    return window.loadModel()
}
voiceWorkbenchGenerateSampleButton.addEventListener("click", window.voiceWorkbenchGenerateVoice)


window.initDropdowns = () => {
    // Games dropdown
    Object.keys(window.games).sort((a,b)=>a>b?1:-1).forEach(gameId => {
        if (gameId!="other") {
            const gameName = window.games[gameId].gameTheme.gameName
            const option = createElem("option", gameName)
            option.value = gameId
            voiceWorkbenchGamesDropdown.appendChild(option)
        }
    })

    // Models dropdown
    Object.keys(window.games).forEach(gameId => {
        const gameName = window.games[gameId].gameTheme.gameName
        window.games[gameId].models.forEach(modelMeta => {
            const voiceName = modelMeta.voiceName
            const voiceId = modelMeta.variants[0].voiceId

            // Variants are not supported by v3 models, so pick the first one only. Also, filter out crafted voices
            if (modelMeta.variants[0].modelType=="xVAPitch" && !modelMeta.embOverABaseModel) {

                const option = createElem("option", `[${gameName}] ${voiceName}`)
                option.value = `${gameId}/${voiceId}`
                voiceWorkbenchModelDropdown.appendChild(option)
            }
        })
    })
}

voiceWorkbenchStartButton.addEventListener("click", () => {
    window.voiceWorkbenchState.isStarted = true

    voiceWorkbenchLoadedContent.style.display = "flex"
    voiceWorkbenchLoadedContent2.style.display = "flex"
    voiceWorkbenchStartButton.style.display = "none"


    // Load the base model's embedding as a starting point, if it's not the built-in base model
    let voiceId = voiceWorkbenchModelDropdown.value.split("/").at(-1)
    if (voiceId.includes("Base xVAPitch Model")) {
    } else {
        const baseModelData = window.games[voiceWorkbenchModelDropdown.value.split("/")[0]].models.filter(model => {
            return model.variants[0].voiceId == voiceWorkbenchModelDropdown.value.split("/").at(-1)
        })[0]
        voiceWorkbenchCurrentEmbeddingInput.value = baseModelData.variants[0].base_speaker_emb.join(",")
        window.voiceWorkbenchState.currentEmb = baseModelData.variants[0].base_speaker_emb
    }
})

window.setupVoiceWorkbenchDropArea = (container, inputField, callback=undefined) => {
    const dropFn = (eType, event) => {
        if (["dragenter", "dragover"].includes(eType)) {
            container.style.background = "#5b5b5b"
            container.style.color = "white"
        }
        if (["dragleave", "drop"].includes(eType)) {
            container.style.background = "rgba(0,0,0,0)"
            container.style.color = "white"
        }

        event.preventDefault()
        event.stopPropagation()

        const dataLines = []

        if (eType=="drop") {
            const dataTransfer = event.dataTransfer
            const files = Array.from(dataTransfer.files)

            if (files[0].path.endsWith(".wav")) {
                const filePath = String(files[0].path).replaceAll(/\\/g, "/")
                console.log("filePath", filePath)
                window.getSpeakerEmbeddingFromFilePath(filePath).then(embedding => {
                    inputField.value = embedding
                    if (callback) {
                        callback(filePath)
                    }
                })
            } else {
                window.errorModal(window.i18n.ERROR_FILE_MUST_BE_WAV)
            }
        }
    }

    container.addEventListener("dragenter", event => dropFn("dragenter", event), false)
    container.addEventListener("dragleave", event => dropFn("dragleave", event), false)
    container.addEventListener("dragover", event => dropFn("dragover", event), false)
    container.addEventListener("drop", event => dropFn("drop", event), false)
}

window.setupVoiceWorkbenchDropArea(voiceWorkbenchCurrentEmbeddingDropzone, voiceWorkbenchCurrentEmbeddingInput, () => {
    window.voiceWorkbenchState.currentEmb = voiceWorkbenchCurrentEmbeddingInput.value.split(",").map(v=>parseFloat(v))
})
voiceWorkbenchCurrentEmbeddingInput.addEventListener("change", ()=>{
    window.voiceWorkbenchState.currentEmb = voiceWorkbenchCurrentEmbeddingInput.value.split(",").map(v=>parseFloat(v))
})
window.setupVoiceWorkbenchDropArea(voiceWorkbenchRefADropzone, voiceWorkbenchRefAInput, (filePath) => {
    voiceWorkbenchRefAFilePath.innerHTML = window.i18n.FROM_FILE_IS_FILEPATH.replace("_1", filePath)
    voiceWorkshopApplyDeltaButton.disabled = false
    window.voiceWorkbenchState.refAEmb = voiceWorkbenchRefAInput.value.split(",").map(v=>parseFloat(v))
    window.voiceWorkbenchUpdateDelta()
})
window.setupVoiceWorkbenchDropArea(voiceWorkbenchRefBDropzone, voiceWorkbenchRefBInput, (filePath) => {
    voiceWorkbenchRefBFilePath.innerHTML = window.i18n.FROM_FILE_IS_FILEPATH.replace("_1", filePath)
    window.voiceWorkbenchState.refBEmb = voiceWorkbenchRefBInput.value.split(",").map(v=>parseFloat(v))
    window.voiceWorkbenchUpdateDelta()
})

voiceWorkbenchInputTextArea.addEventListener("keyup", () => {
    voiceWorkbenchGenerateSampleButton.disabled = voiceWorkbenchInputTextArea.value.trim().length==0
})
voiceWorkbenchAudioCurrentPlayPauseBtn.addEventListener("click", () => {
    const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
        src: window.voiceWorkbenchState.currentAudioFilePath
    }))
    audioPreview.setSinkId(window.userSettings.base_speaker)
})
voiceWorkbenchAudioCurrentSaveBtn.addEventListener("click", async () => {
    const userChosenPath = await dialog.showSaveDialog({ defaultPath: window.voiceWorkbenchState.currentAudioFilePath })
    if (userChosenPath && userChosenPath.filePath) {
        const outFilePath = userChosenPath.filePath.split(".").at(-1)=="wav" ? userChosenPath.filePath : userChosenPath.filePath+".wav"
        fs.copyFileSync(window.voiceWorkbenchState.currentAudioFilePath, outFilePath)
    }
})
voiceWorkbenchAudioNewPlayBtn.addEventListener("click", () => {
    const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
        src: window.voiceWorkbenchState.newAudioFilePath
    }))
    audioPreview.setSinkId(window.userSettings.base_speaker)
})
voiceWorkbenchAudioNewSaveBtn.addEventListener("click", async () => {
    const userChosenPath = await dialog.showSaveDialog({ defaultPath: window.voiceWorkbenchState.newAudioFilePath })
    if (userChosenPath && userChosenPath.filePath) {
        const outFilePath = userChosenPath.filePath.split(".").at(-1)=="wav" ? userChosenPath.filePath : userChosenPath.filePath+".wav"
        fs.copyFileSync(window.voiceWorkbenchState.newAudioFilePath, outFilePath)
    }
})

window.voiceWorkbenchUpdateDelta = () => {
    // Don't do anything if reference file A isn't given
    if (!window.voiceWorkbenchState.refAEmb) {
        return
    }

    const strengthValue = parseFloat(voiceWorkbenchStrengthInput.value)

    let delta

    // When only Ref A is used, the delta is from <current> towards the first reference file A
    if (window.voiceWorkbenchState.refBEmb == undefined) {
        delta = window.voiceWorkbenchState.currentEmb.map((v,vi) => {
            return (window.voiceWorkbenchState.refAEmb[vi] - v) * strengthValue
        })
    } else {
        // When Ref B is also used, the delta is from ref A to ref B
        delta = window.voiceWorkbenchState.refAEmb.map((v,vi) => {
            return (window.voiceWorkbenchState.refBEmb[vi] - v) * strengthValue
        })
    }

    voiceWorkbenchCurrentDeltaInput.value = delta.join(",")
}

voiceWorkbenchStrengthSlider.addEventListener("change", () => {
    voiceWorkbenchStrengthInput.value = voiceWorkbenchStrengthSlider.value
    window.voiceWorkbenchUpdateDelta()
})
voiceWorkbenchStrengthInput.addEventListener("change", () => {
    voiceWorkbenchStrengthSlider.value = voiceWorkbenchStrengthInput.value
    window.voiceWorkbenchUpdateDelta()
})
voiceWorkshopApplyDeltaButton.addEventListener("click", () => {
    if (voiceWorkbenchCurrentDeltaInput.value.length) {
        const newEmb = window.getVoiceWorkbenchNewEmbedding()
        window.voiceWorkbenchState.currentEmb = newEmb
        voiceWorkbenchCurrentEmbeddingInput.value = newEmb.join(",")
        voiceWorkbenchCurrentDeltaInput.value = ""
        voiceWorkshopApplyDeltaButton.disabled = true
        voiceWorkbenchRefAInput.value = ""
        window.voiceWorkbenchState.refAEmb = undefined
        voiceWorkbenchRefBInput.value = ""
        window.voiceWorkbenchState.refBEmb = undefined
    }
})

/*
Drop file A over the reference audio file A area, to get its embedding
    When only the reference A file is used, the current delta is this embedding multiplied by the strength

Drop file B over the B area, to get a second embedding
    When both this and A are active, the current delta is the direction from A to B, multiplied by the strength
        direction meaning B minus A, instead of <current> minus A
*/

voiceWorkbenchSaveButton.addEventListener("click", () => {

    const voiceName = voiceWorkbenchVoiceNameInput.value
    const voiceId = voiceWorkbenchVoiceIDInput.value
    const gender = voiceWorkbenchGenderDropdown.value
    const author = voiceWorkbenchAuthorInput.value || "Anonymous"
    const lang = voiceWorkbenchLanguageDropdown.value


    if (!voiceName.trim().length) {
        return window.errorModal(window.i18n.ENTER_VOICE_NAME)
    }
    if (!voiceId.trim().length) {
        return window.errorModal(window.i18n.ENTER_VOICE_ID)
    }

    const modelJson = {
        "version": "3.0",
        "modelVersion": "3.0",
        "modelType": "xVAPitch",
        "author": author,
        "lang": lang,
        "embOverABaseModel": voiceWorkbenchModelDropdown.value,
        "games": [
            {
                "gameId": voiceWorkbenchGamesDropdown.value,
                "voiceId": voiceId,
                "variant": "Default",
                "voiceName": voiceName,
                "base_speaker_emb": window.voiceWorkbenchState.currentEmb,
                "gender": gender
            }
        ]
    }
    const gameModelsPath = `${window.userSettings[`modelspath_${voiceWorkbenchGamesDropdown.value}`]}`

    const jsonDestination = `${gameModelsPath}/${voiceId}.json`
    fs.writeFileSync(jsonDestination, JSON.stringify(modelJson, null, 4))

    doSynth(JSON.stringify({
        sequence: " This is what my voice sounds like. ",
        useCleanup: true, // TODO, user setting?
        base_lang: lang,
        base_emb: window.voiceWorkbenchState.currentEmb.join(","), outfile: jsonDestination.replace(".json", ".wav")
    })).then(() => {
        voiceWorkbenchDeleteButton.disabled = false
        window.currentModel = undefined
        generateVoiceButton.dataset.modelQuery = null
        window.infoModal(window.i18n.VOICE_CREATED_AT.replace("_1", jsonDestination))

        // Clean up the temp file from the clean-up post-processing, if it exists
        if (fs.existsSync(jsonDestination.replace(".json", "_preCleanup.wav"))) {
            fs.unlinkSync(jsonDestination.replace(".json", "_preCleanup.wav"))
        }

        window.loadAllModels().then(() => {
            window.refreshExistingCraftedVoices()

            // Refresh the main page voice models if the same game is loaded as the target game models directory as saved into
            if (window.currentGame.gameId==voiceWorkbenchGamesDropdown.value) {
                window.changeGame(window.currentGame)
                window.refreshExistingCraftedVoices()
            }
        })
    })
})

voiceWorkbenchGamesDropdown.addEventListener("change", () => {
    const gameModelsPath = `${window.userSettings[`modelspath_${voiceWorkbenchGamesDropdown.value}`]}`
    const voiceId = voiceWorkbenchVoiceIDInput.value
    const jsonLocation = `${gameModelsPath}/${voiceId}.json`
    voiceWorkbenchDeleteButton.disabled = !fs.existsSync(jsonLocation)
})
voiceWorkbenchVoiceIDInput.addEventListener("change", () => {
    const gameModelsPath = `${window.userSettings[`modelspath_${voiceWorkbenchGamesDropdown.value}`]}`
    const voiceId = voiceWorkbenchVoiceIDInput.value
    const jsonLocation = `${gameModelsPath}/${voiceId}.json`
    voiceWorkbenchDeleteButton.disabled = !fs.existsSync(jsonLocation)
})
voiceWorkbenchDeleteButton.addEventListener("click", () => {
    const gameModelsPath = `${window.userSettings[`modelspath_${voiceWorkbenchGamesDropdown.value}`]}`
    const voiceId = voiceWorkbenchVoiceIDInput.value
    const jsonLocation = `${gameModelsPath}/${voiceId}.json`
    window.confirmModal(window.i18n.CONFIRM_DELETE_CRAFTED_VOICE.replace("_1", voiceWorkbenchVoiceNameInput.value).replace("_2", jsonLocation)).then(resp => {
        if (resp) {
            if (fs.existsSync(jsonLocation.replace(".json", ".wav"))) {
                fs.unlinkSync(jsonLocation.replace(".json", ".wav"))
            }
            fs.unlinkSync(jsonLocation)
        }
        window.infoModal(window.i18n.SUCCESSFULLY_DELETED_CRAFTED_VOICE)
        window.loadAllModels().then(() => {

            // Refresh the main page voice models if the same game is loaded as the target game models directory deleted from
            if (window.currentGame.gameId==voiceWorkbenchGamesDropdown.value) {

                window.changeGame(window.currentGame)
                window.refreshExistingCraftedVoices()
            }
            voiceWorkbenchCancelButton.click()
        })
    })
})



voiceWorkbenchCancelButton.addEventListener("click", () => {
    window.voiceWorkbenchState.isStarted = false

    voiceWorkbenchLoadedContent.style.display = "none"
    voiceWorkbenchLoadedContent2.style.display = "none"
    voiceWorkbenchStartButton.style.display = "flex"

    window.voiceWorkbenchLoadOrResetCraftedVoice()
})
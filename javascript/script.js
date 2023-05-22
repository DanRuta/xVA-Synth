"use strict"
window.appVersion = "v3.0.0"

window.PRODUCTION = module.filename.includes("resources")
const path = window.PRODUCTION ? "./resources/app" : "."
window.path = path

const fs = require("fs")
const zipdir = require('zip-dir')
const {shell, ipcRenderer, clipboard} = require("electron")
const doFetch = require("node-fetch")
const {xVAAppLogger} = require("./javascript/appLogger.js")
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
process.on(`uncaughtException`, (data, origin) => {window.appLogger.log(`uncaughtException: ${data}`);window.appLogger.log(`uncaughtException: ${origin}`)})
window.onerror = (err, url, lineNum) => {window.appLogger.log(`onerror: ${err.stack}`)}
require("./javascript/i18n.js")
require("./javascript/util.js")
require("./javascript/nexus.js")
require("./javascript/dragdrop_model_install.js")
require("./javascript/embeddings.js")
require("./javascript/totd.js")
require("./javascript/arpabet.js")
require("./javascript/style_embeddings.js")
const {Editor} = require("./javascript/editor.js")
require("./javascript/textarea.js")
const {saveUserSettings, deleteFolderRecursive} = require("./javascript/settingsMenu.js")
const xVASpeech = require("./javascript/speech2speech.js")
require("./javascript/batch.js")
require("./javascript/outputFiles.js")
require("./javascript/workbench.js")
const er = require('@electron/remote')
window.electronBrowserWindow = er.getCurrentWindow()
const child = require("child_process").execFile
const spawn = require("child_process").spawn

// Newly introduced in v3. I will slowly start moving global context variables into this, and update code throughout to reference this
// instead of old variables such as window.games, window.currentModel, etc.
window.appState = {}


// Start the server
if (window.PRODUCTION) {
    window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
}

const {PluginsManager} = require("./javascript/plugins_manager.js")
window.pluginsContext = {}
window.pluginsManager = new PluginsManager(window.path, window.appLogger, window.appVersion)
window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["pre"], event="pre start")


let themeColour
let secondaryThemeColour
const oldCError = console.error
console.error = (...rest) => {
    window.appLogger.log(`console.error: ${rest}`)
    oldCError(rest)
}

window.addEventListener("error", function (e) {window.appLogger.log(`error: ${e.error.stack}`)})
window.addEventListener('unhandledrejection', function (e) {window.appLogger.log(`unhandledrejection: ${e.stack}`)})


setTimeout(() => {
    window.electron = require("electron")
}, 1000)


window.games = {}
window.models = {}
window.sequenceEditor = new Editor()
window.currentModel = undefined
window.currentModelButton = undefined
window.watchedModelsDirs = []

window.appLogger.log(`Settings: ${JSON.stringify(window.userSettings)}`)

// Set up folders
try {fs.mkdirSync(`${path}/models`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/output`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/assets`)} catch (e) {/*Do nothing*/}

// Clean up temp files
const clearOldTempFiles = () => {
    fs.readdir(`${__dirname.replace("/javascript", "")}/output`, (err, files) => {
        if (err) {
            window.appLogger.log(`Error cleaning up temp files: ${err}`)
        }
        if (files && files.length) {
            files.filter(f => f.startsWith("temp-")).forEach(file => {
                fs.unlink(`${__dirname.replace("/javascript", "")}/output/${file}`, err => err&&console.log(err))
            })
        }
    })
}
clearOldTempFiles()

let fileRenameCounter = 0
let fileChangeCounter = 0
window.isGenerating = false




window.registerModel = (modelsPath, gameFolder, model, {gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i}) => {
    // Add game, if the game hasn't been added yet
    let audioPreviewPath
    try {
        audioPreviewPath = `${modelsPath}/${model.games.find(({gameId}) => gameId==gameFolder).voiceId}`
    } catch (e) {}
    if (!window.games.hasOwnProperty(gameId)) {

        const gameAsset = fs.readdirSync(`${path}/assets`).find(f => f==gameId+".json")
        if (gameAsset && gameAsset.length && gameAsset[0]) {
            const gameTheme = JSON.parse(fs.readFileSync(`${path}/assets/${gameAsset}`))

            window.games[gameId] = {
                models: [],
                gameTheme,
                gameAsset
            }

        } else {
            window.appLogger.log(`Something not right with loading model: ${voiceId} . The asset file for its game (${gameId}) could not be found here: ${path}/assets. You need a ${gameId}.json file. Loading a generic theme for this voice's game/category.`)

            const dummyGameTheme = {
                "gameName": gameId,
                "assetFile": "other.jpg",
                "themeColourPrimary": "aaaaaa",
                "themeColourSecondary": null,
                "gameCode": "x",
                "nexusGamePageIDs": []
            }
            const dummyGameAsset = "other.json"
            window.games[gameId] = {
                models: [],
                dummyGameTheme,
                dummyGameAsset
            }
            audioPreviewPath = `${modelsPath}/${model.games[0].voiceId}`
        }
    }

    // Catch duplicates, for when/if a model is registered for multiple games, but there's already the same model in that game, from another version
    // const existingDuplicates = []
    // window.games[gameId].models.forEach((item,i) => {
    //     if (item.voiceId==voiceId) {
    //         existingDuplicates.push([item, i])
    //     }
    // })

    // Check if a variant has already been added for this voice name, for this game
    let foundVariantIndex = undefined
    window.games[gameId].models.forEach((item,i) => {
        if (foundVariantIndex!=undefined) return

        if (item.voiceName.toLowerCase().trim()==voiceName.toLowerCase().trim()) {
            foundVariantIndex = i
        }
    })

    // Add the initial model metadata, if no existing variant has been added (will happen most of the time)
    if (!foundVariantIndex) {
        const modelData = {
            gameId,
            modelsPath,
            voiceName,
            embOverABaseModel: model.embOverABaseModel,
            variants: []
        }
        window.games[gameId].models.push(modelData)
        foundVariantIndex = window.games[gameId].models.length-1
    }

    const variantData = {
        author: model.author,
        version: model.version,
        modelVersion: model.modelVersion,
        modelType: model.modelType,
        base_speaker_emb: model.modelType=="xVAPitch" ? model.games[0].base_speaker_emb : undefined,
        voiceId,
        audioPreviewPath,
        hifi: undefined,
        num_speakers: model.emb_size,
        emb_i,
        variantName: variant ? variant.replace("Default :", "Default:").replace("Default:", "").trim() : "Default",
        voiceDescription,
        lang: model.lang,
        gender,
        modelType: modelType||model.modelType,
        model,
    }
    const potentialHiFiPath = `${modelsPath}/${voiceId}.hg.pt`
    if (fs.existsSync(potentialHiFiPath)) {
        variantData.hifi = potentialHiFiPath
    }

    const isDefaultVariant = !variant || variant.toLowerCase().startsWith("default")

    if (isDefaultVariant) {
        // Place first in the list, if it's default
        window.games[gameId].models[foundVariantIndex].audioPreviewPath = audioPreviewPath
        window.games[gameId].models[foundVariantIndex].variants.splice(0,0,variantData)
    } else {
        window.games[gameId].models[foundVariantIndex].variants.push(variantData)
    }


    // // Using the detected duplicates, use only the latest version
    // if (existingDuplicates.length) {
    //     if (existingDuplicates[0][0].modelVersion<model.modelVersion) {
    //         window.games[gameId].models.splice(existingDuplicates[0][1], 1)
    //         window.games[gameId].models.push(modelData)
    //     }
    // } else {
    //     window.games[gameId].models.push(modelData)
    // }
}

window.loadAllModels = (forceUpdate=false) => {
    return new Promise(resolve => {

        if (!forceUpdate && window.nexusState.installQueue.length) {
            return
        }

        let gameFolder
        let modelPathsKeys = Object.keys(window.userSettings).filter(key => key.includes("modelspath_"))
        window.games = {}

        // Do the current game first, and stop blocking the render process
        if (window.currentGame) {
            const currentGameFolder = window.userSettings[`modelspath_${window.currentGame.gameId}`]
            gameFolder = currentGameFolder
            try {
                const files = fs.readdirSync(modelsPath).filter(f => f.endsWith(".json"))
                files.forEach(fileName => {
                    try {
                        if (!models.hasOwnProperty(`${gameFolder}/${fileName}`)) {
                            models[`${gameFolder}/${fileName}`] = null
                        }
                        const model = JSON.parse(fs.readFileSync(`${modelsPath}/${fileName}`, "utf8"))
                        model.games.forEach(({gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i}) => {
                            window.registerModel(currentGameFolder, gameFolder, model, {gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i})
                        })

                    } catch (e) {
                        console.log(e)
                        // window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME.replace("_1", gameFolder)} `+fileName)
                        // window.appLogger.log(e)
                        // window.appLogger.log(e.stack)
                    }
                })
            } catch (e) {
                window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME}: `+ gameFolder)
                window.appLogger.log(e)
            }
            resolve() // Continue the rest but asynchronously
        }


        modelPathsKeys.forEach(modelsPathKey => {
            const modelsPath = window.userSettings[modelsPathKey]
            try {
                const files = fs.readdirSync(modelsPath).filter(f => f.endsWith(".json"))

                if (!files.length) {
                    return
                }

                files.forEach(fileName => {

                    gameFolder = modelsPathKey.split("_")[1]

                    try {
                        if (!models.hasOwnProperty(`${gameFolder}/${fileName}`)) {
                            models[`${gameFolder}/${fileName}`] = null
                        }

                        const model = JSON.parse(fs.readFileSync(`${modelsPath}/${fileName}`, "utf8"))
                        model.games.forEach(({gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i}) => {
                            window.registerModel(modelsPath, gameFolder, model, {gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i})
                        })
                    } catch (e) {
                        console.log(e)
                        setTimeout(() => {
                            window.errorModal(`${fileName}<br><br>${e.stack}`)
                        }, 1000)
                        window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME.replace("_1", gameFolder)} `+fileName)
                        window.appLogger.log(e)
                        window.appLogger.log(e.stack)
                    }
                })
            } catch (e) {
                window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME}: `+ gameFolder)
                window.appLogger.log(e)
            }
        })
        window.updateGameList(false)
        resolve()
    })
}


// Change variant
let oldVariantSelection = undefined // For reverting, if versioning checks fail
variant_select.addEventListener("change", () => {

    const model = window.games[window.currentGame.gameId].models.find(model => model.voiceName== window.currentModel.voiceName)
    const variant = model.variants.find(variant => variant.variantName==variant_select.value)

    const appVersionOk = window.checkVersionRequirements(variant.version, appVersion)
    if (!appVersionOk) {
        window.errorModal(`${window.i18n.MODEL_REQUIRES_VERSION} v${variant.version}<br><br>${window.i18n.THIS_APP_VERSION}: ${window.appVersion}`)
        variant_select.value = oldVariantSelection
        return
    }

    generateVoiceButton.dataset.modelQuery = JSON.stringify({
        outputs: parseInt(model.outputs),
        model: model.embOverABaseModel ? window.userSettings[`modelspath_${model.embOverABaseModel.split("/")[0]}`]+`/${model.embOverABaseModel.split("/")[1]}` : `${model.modelsPath}/${variant.voiceId}`,
        modelType: variant.modelType,
        version: variant.version,
        model_speakers: model.num_speakers,
        base_lang: model.lang || "en"
    })
    oldVariantSelection = variant_select.value

    titleInfoVoiceID.innerHTML = variant.voiceId
    titleInfoGender.innerHTML = variant.gender || "?"
    titleInfoAppVersion.innerHTML = variant.version || "?"
    titleInfoModelVersion.innerHTML = variant.modelVersion || "?"
    titleInfoModelType.innerHTML = variant.modelType || "?"
    titleInfoLanguage.innerHTML = variant.lang || window.currentModel.games[0].lang || "en"
    titleInfoAuthor.innerHTML = variant.author || "?"

    generateVoiceButton.click()
})



// Change game
window.changeGame = (meta) => {

    titleInfo.style.display = "none"
    window.currentGame = meta
    themeColour = meta.themeColourPrimary
    secondaryThemeColour = meta.themeColourSecondary
    let titleID = meta.gameCode

    generateVoiceButton.disabled = true
    generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
    selectedGameDisplay.innerHTML = meta.gameName

    // Change the app title
    titleName.innerHTML = window.i18n.SELECT_VOICE_TYPE
    if (window.games[window.currentGame.gameId] == undefined) {
        titleName.innerHTML = `${window.i18n.NO_MODELS_IN}: ${window.userSettings[`modelspath_${window.currentGame.gameId}`]}`
    }

    const gameFolder = meta.gameId
    const gameName = meta.gameName

    setting_models_path_container.style.display = "flex"
    setting_out_path_container.style.display = "flex"
    setting_models_path_label.innerHTML = `<i style="display:inline">${gameName}</i><span>${window.i18n.SETTINGS_MODELS_PATH}</span>`
    setting_models_path_input.value = window.userSettings[`modelspath_${gameFolder}`]
    setting_out_path_label.innerHTML = `<i style="display:inline">${gameName}</i> ${window.i18n.SETTINGS_OUTPUT_PATH}`
    setting_out_path_input.value = window.userSettings[`outpath_${gameFolder}`]

    window.setTheme(window.currentGame)
    try {
        window.displayAllModels()
    } catch (e) {console.log(e)}

    try {fs.mkdirSync(`${path}/output/${meta.gameId}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", JSON.stringify(meta))

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""

    const buttons = []
    const totalNumVoices = (window.games[meta.gameId] ? window.games[meta.gameId].models : []).reduce((p,c)=>p+c.variants.length, 0)
    voiceSearchInput.placeholder = window.i18n.SEARCH_N_VOICES.replace("_", window.games[meta.gameId] ? totalNumVoices : "0")
    voiceSearchInput.value = ""

    if (!window.games[meta.gameId]) {
        return
    }

    (window.games[meta.gameId] ? window.games[meta.gameId].models : []).forEach(({modelsPath, audioPreviewPath, gameId, variants, voiceName, embOverABaseModel}) => {

        const {voiceId, voiceDescription, hifi, model} = variants[0]
        const modelVersion = variants[0].version

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
        if (embOverABaseModel) {
            button.style.fontStyle = "italic"
        }
        if (window.userSettings.do_model_version_highlight && parseFloat(modelVersion)<window.userSettings.model_version_highlight) {
            button.style.border =  `2px solid #${themeColour}`
            button.style.padding =  "0"
            button.style.background = "none"
        }
        button.dataset.modelId = voiceId
        if (secondaryThemeColour) {
            button.style.color = `#${secondaryThemeColour}`
            button.style.textShadow = `none`
        }

        // Quick voice set preview, if there is a preview file
        button.addEventListener("contextmenu", () => {
            window.appLogger.log(`${audioPreviewPath}.wav`)
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: `${audioPreviewPath}.wav`
            }))
            audioPreview.style.height = "25px"
            audioPreview.setSinkId(window.userSettings.base_speaker)
        })

        if (embOverABaseModel) {
            const gameOfBaseModel = embOverABaseModel.split("/")[0]
            if (gameOfBaseModel=="<base>") {
                // For included base v3 models
                modelsPath = `${window.path}/python/xvapitch/${embOverABaseModel.split("/")[1]}`
            } else {
                // For any other model
                const gameModelsPath = `${window.userSettings[`outpath_${gameOfBaseModel}`]}`
                modelsPath = `${gameModelsPath}/${embOverABaseModel.split("/")[1]}`
            }
        }

        button.addEventListener("click", event => window.selectVoice(event, variants, hifi, gameId, voiceId, model, button, audioPreviewPath, modelsPath, meta, embOverABaseModel))
        buttons.push(button)
    })

    buttons.sort((a,b) => a.innerHTML.toLowerCase()<b.innerHTML.toLowerCase()?-1:1)
        .forEach(button => voiceTypeContainer.appendChild(button))

}


window.selectVoice = (event, variants, hifi, gameId, voiceId, model, button, audioPreviewPath, modelsPath, meta, embOverABaseModel) => {
    // Just for easier packaging of the voice models for publishing - yes, lazy
    if (event.ctrlKey && event.shiftKey) {
        window.packageVoice(event.altKey, variants, {modelsPath, gameId})
    }

    variant_select.innerHTML = ""
    oldVariantSelection = undefined
    if (variants.length==1) {
        variantElements.style.display = "none"
    } else {
        variantElements.style.display = "flex"
        variants.forEach(variant => {
            const option = createElem("option", {value: variant.variantName})
            option.innerHTML = variant.variantName
            variant_select.appendChild(option)
            if (!oldVariantSelection) {
                oldVariantSelection = variant.variantName
            }
        })
    }


    if (hifi) {
        // Remove the bespoke hifi option if there was one already there
        Array.from(vocoder_select.children).forEach(opt => {
            if (opt.innerHTML=="Bespoke HiFi GAN") {
                vocoder_select.removeChild(opt)
            }
        })
        bespoke_hifi_bolt.style.opacity = 1
        const option = createElem("option", "Bespoke HiFi GAN")
        option.value = `${gameId}/${voiceId}.hg.pt`
        vocoder_select.appendChild(option)
    } else {
        bespoke_hifi_bolt.style.opacity = 0
        // Set the vocoder select to quick-and-dirty if bespoke hifi-gan was selected
        if (vocoder_select.value.includes(".hg.")) {
            vocoder_select.value = "qnd"
            window.changeVocoder("qnd")
        }
        // Remove the bespoke hifi option if there was one already there
        Array.from(vocoder_select.children).forEach(opt => {
            if (opt.innerHTML=="Bespoke HiFi GAN") {
                vocoder_select.removeChild(opt)
            }
        })
    }

    window.currentModel = model
    window.currentModel.voiceId = voiceId
    window.currentModel.voiceName = button.innerHTML
    window.currentModel.hifi = hifi
    window.currentModel.audioPreviewPath = audioPreviewPath
    window.currentModelButton = button


    generateVoiceButton.dataset.modelQuery = null

    // The model is already loaded. Don't re-load it.
    if (generateVoiceButton.dataset.modelIDLoaded == voiceId) {
        generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
        generateVoiceButton.dataset.modelQuery = "null"

    } else {
        generateVoiceButton.innerHTML = window.i18n.LOAD_MODEL
        generateVoiceButton.dataset.modelQuery = JSON.stringify({
            outputs: parseInt(model.outputs),
            model: model.embOverABaseModel ? window.userSettings[`modelspath_${model.embOverABaseModel.split("/")[0]}`]+`/${model.embOverABaseModel.split("/")[1]}` : `${modelsPath}/${model.voiceId}`,
            modelType: model.modelType,
            version: model.version,
            model_speakers: model.emb_size,
            cmudict: model.cmudict,
            base_lang: model.lang || "en"
        })
        generateVoiceButton.dataset.modelIDToLoad = voiceId
    }
    generateVoiceButton.disabled = false

    titleName.innerHTML = button.innerHTML
    titleInfo.style.display = "flex"
    titleInfoName.innerHTML = window.currentModel.voiceName
    titleInfoVoiceID.innerHTML = voiceId
    titleInfoGender.innerHTML = window.currentModel.games[0].gender || "?"
    titleInfoAppVersion.innerHTML = window.currentModel.version || "?"
    titleInfoModelVersion.innerHTML = window.currentModel.modelVersion || "?"
    titleInfoModelType.innerHTML = window.currentModel.modelType || "?"
    titleInfoLanguage.innerHTML = window.currentModel.lang || window.currentModel.games[0].lang || "en"
    titleInfoAuthor.innerHTML = window.currentModel.author || "?"
    titleInfoLicense.innerHTML = window.currentModel.license || window.i18n.UNKNOWN

    title.dataset.modelId = voiceId
    keepSampleButton.style.display = "none"
    samplePlayPause.style.display = "none"

    // Voice samples
    voiceSamples.innerHTML = ""

    window.initMainPagePagination(`${window.userSettings[`outpath_${meta.gameId}`]}/${button.dataset.modelId}`)
    window.refreshRecordsList()
}

titleInfo.addEventListener("click", () => titleDetails.style.display = titleDetails.style.display=="none" ? "block" : "none")
window.addEventListener("click", event => {
    if (event.target!=titleInfo && event.target!=titleDetails && event.target.parentNode && event.target.parentNode!=titleDetails && event.target.parentNode.parentNode!=titleDetails) {
        titleDetails.style.display = "none"
    }
})
titleDetails.style.display = "none"


window.loadModel = () => {
    return new Promise(resolve => {
        if (window.batch_state.state) {
            window.errorModal(window.i18n.BATCH_ERR_IN_PROGRESS)
            return
        }

        const body = JSON.parse(generateVoiceButton.dataset.modelQuery)

        const appVersionOk = window.checkVersionRequirements(body.version, appVersion)
        if (!appVersionOk) {
            window.errorModal(`${window.i18n.MODEL_REQUIRES_VERSION} v${body.version}<br><br>${window.i18n.THIS_APP_VERSION}: ${window.appVersion}`)
            return
        }


        window.appLogger.log(`${window.i18n.LOADING_VOICE}: ${JSON.parse(generateVoiceButton.dataset.modelQuery).model}`)
        window.batch_state.lastModel = JSON.parse(generateVoiceButton.dataset.modelQuery).model.split("/").reverse()[0]

        body["pluginsContext"] = JSON.stringify(window.pluginsContext)

        spinnerModal(`${window.i18n.LOADING_VOICE}`)
        doFetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: JSON.stringify(body)
        }).then(r=>r.text()).then(res => {

            window.currentModel.loaded = true
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad

            // Set the editor pitch/energy dropdowns to pitch, and freeze them, if energy is not supported by the model
            window.appState.currentModelEmbeddings = {}
            if (window.currentModel.modelType.toLowerCase()=="xvapitch" && !window.currentModel.isBaseModel) {
                vocoder_options_container.style.display = "none"
                base_lang_select.disabled = false
                style_emb_select.disabled = false
                window.loadStyleEmbsForVoice(window.currentModel)
                mic_SVG.children[0].style.fill = "white"
                base_lang_select.value = window.currentModel.lang
            } else {
                vocoder_options_container.style.display = "inline-block"
                base_lang_select.disabled = true
                style_emb_select.disabled = true
                mic_SVG.children[0].style.fill = "grey"
            }
            if (window.currentModel.modelType.toLowerCase()=="fastpitch") {
                seq_edit_view_select.value = "pitch"
                seq_edit_edit_select.value = "pitch"
                seq_edit_view_select.disabled = true
                seq_edit_edit_select.disabled = true
            } else {
                seq_edit_view_select.value = "pitch_energy"
                seq_edit_view_select.disabled = false
                seq_edit_edit_select.disabled = false
            }

            if (window.userSettings.defaultToHiFi && window.currentModel.hifi) {
                vocoder_select.value = Array.from(vocoder_select.children).find(opt => opt.innerHTML=="Bespoke HiFi GAN").value
                window.changeVocoder(vocoder_select.value).then(() => dialogueInput.focus())
            } else if (window.userSettings.vocoder.includes(".hg.pt")) {
                window.changeVocoder("qnd").then(() => dialogueInput.focus())
            } else {
                closeModal(null, [workbenchContainer]).then(() => dialogueInput.focus())
            }
            resolve()
        }).catch(e => {
            console.log(e)
            if (e.code =="ENOENT") {
                closeModal(null, [modalContainer, workbenchContainer]).then(() => {
                    window.errorModal(window.i18n.ERR_SERVER)
                    resolve()
                })
            }
        })
    })
}

window.synthesizeSample = () => {

    const game = window.currentGame.gameId

    if (window.isGenerating) {
        return
    }
    if (!window.speech2speechState.s2s_running) {
        clearOldTempFiles()
    }

    let sequence = dialogueInput.value.replace("…", "...").replace("’", "'")
    if (window.userSettings.spacePadding) { // Pad start and end of the input sequence with spaces
        sequence = " "+sequence.trim()+" "
    }

    if (sequence.length==0) {
        return
    }
    window.isGenerating = true

    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["generate-voice"]["pre"], event="pre generate-voice")

    if (window.wavesurfer) {
        try {
            window.wavesurfer.stop()
        } catch (e) {
            console.log(e)
        }
        wavesurferContainer.style.opacity = 0
    }
    toggleSpinnerButtons()

    const voiceType = title.dataset.modelId
    const outputFileName = dialogueInput.value.slice(0, 260).replace(/\n/g, " ").replace(/[\/\\:\*?<>"|]*/g, "").replace(/^[\.\s]+/, "")

    try {fs.unlinkSync(localStorage.getItem("tempFileLocation"))} catch (e) {/*Do nothing*/}

    // For some reason, the samplePlay audio element does not update the source when the file name is the same
    const tempFileNum = `${Math.random().toString().split(".")[1]}`
    let tempFileLocation = `${path}/output/temp-${tempFileNum}.wav`
    let pitch = []
    let duration = []
    let energy = []
    let emAngry = []
    let emHappy = []
    let emSad = []
    let emSurprise = []
    let editorStyles = {}
    let isFreshRegen = true
    let old_sequence = undefined

    if (editorContainer.innerHTML && editorContainer.innerHTML.length && generateVoiceButton.dataset.modelIDLoaded==window.sequenceEditor.currentVoice) {
        if (window.sequenceEditor.audioInput || window.sequenceEditor.sequence && sequence!=window.sequenceEditor.inputSequence) {
            old_sequence = window.sequenceEditor.inputSequence
        }
    }
    // Don't use the old_sequence if running speech-to-speech
    if (window.speech2speechState.s2s_running) {
        old_sequence = undefined
        window.speech2speechState.s2s_running = false
    }

    // Check if editing an existing line (otherwise it's a fresh new line)
    const languageHasChanged = window.sequenceEditor.base_lang && window.sequenceEditor.base_lang != base_lang_select.value
    const promptHasChanged = !window.sequenceEditor.historyState.length || window.sequenceEditor.historyState.at(-1)!=sequence
    if (!promptHasChanged && !languageHasChanged && !window.arpabetMenuState.hasChangedARPAbet && !window.styleEmbsMenuState.hasChangedEmb &&
        (speech2speechState.s2s_autogenerate || (editorContainer.innerHTML && editorContainer.innerHTML.length && (window.userSettings.keepEditorOnVoiceChange || generateVoiceButton.dataset.modelIDLoaded==window.sequenceEditor.currentVoice)))) {

        speech2speechState.s2s_autogenerate = false
        pitch = window.sequenceEditor.pitchNew.map(v=> v==undefined?0:v)
        duration = window.sequenceEditor.dursNew.map(v => v*pace_slid.value).map(v=> v==undefined?0:v)
        energy = window.sequenceEditor.energyNew ? window.sequenceEditor.energyNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []
        if (window.currentModel.modelType=="xVAPitch") {
            emAngry = window.sequenceEditor.emAngryNew ? window.sequenceEditor.emAngryNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []
            emHappy = window.sequenceEditor.emHappyNew ? window.sequenceEditor.emHappyNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []
            emSad = window.sequenceEditor.emSadNew ? window.sequenceEditor.emSadNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []
            emSurprise = window.sequenceEditor.emSurpriseNew ? window.sequenceEditor.emSurpriseNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []

            window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
                editorStyles[styleKey] = {
                    embedding: window.appState.currentModelEmbeddings[styleKey][1],
                    sliders: window.sequenceEditor.styleValuesNew[styleKey].map(v => v==undefined?0:v).filter(v => !isNaN(v))// : []
                }
            })
        }
        isFreshRegen = false
    }

    window.arpabetMenuState.hasChangedARPAbet = false
    window.styleEmbsMenuState.hasChangedEmb = false
    window.sequenceEditor.currentVoice = generateVoiceButton.dataset.modelIDLoaded

    const speaker_i = window.currentModel.games[0].emb_i
    const pace = (window.userSettings.keepPaceOnNew && isFreshRegen)?pace_slid.value:1

    window.appLogger.log(`${window.i18n.SYNTHESIZING}: ${sequence}`)

    doFetch(`http://localhost:8008/synthesize`, {
        method: "Post",
        body: JSON.stringify({
            sequence, pitch, duration, energy, emAngry, emHappy, emSad, emSurprise, editorStyles, speaker_i, pace,
            base_lang: base_lang_select.value,
            base_emb: style_emb_select.value||"",
            modelType: window.currentModel.modelType,
            old_sequence, // For partial re-generation
            device: window.userSettings.installation=="cpu"?"cpu":(window.userSettings.useGPU?"cuda:0":"cpu"),
            // device: window.userSettings.useGPU?"gpu":"cpu",   // Switch to this once DirectML is installed
            useSR: useSRCkbx.checked,
            useCleanup: useCleanupCkbx.checked,
            outfile: tempFileLocation,
            pluginsContext: JSON.stringify(window.pluginsContext),
            vocoder: window.currentModel.modelType=="xVAPitch" ? "n/a" : window.userSettings.vocoder,
            waveglowPath: vocoder_select.value=="256_waveglow" ? window.userSettings.waveglow_path : window.userSettings.bigwaveglow_path
        })
    }).then(r=>r.text()).then(res => {
        window.isGenerating = false

        if (res=="ENOENT" || res.startsWith("ERR:")) {
            console.log(res)
            if (res.startsWith("ERR:")) {
                if (res.includes("ARPABET_NOT_IN_LIST")) {
                    const symbolNotInList = res.split(":").reverse()[0]
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${window.i18n.ERR_ARPABET_NOT_EXIST.replace("_1", symbolNotInList)}`)
                } else {
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${res.replace("ERR:","").replaceAll(/\n/g, "<br>")}`)
                }
            } else {
                window.appLogger.log(res)
                window.errorModal(`${window.i18n.BATCH_MODEL_NOT_FOUND}.${vocoder_select.value.includes("waveglow")?" "+window.i18n.BATCH_DOWNLOAD_WAVEGLOW:""}`)
            }
            toggleSpinnerButtons()
            return
        }

        dialogueInput.focus()
        window.sequenceEditor.historyState.push(sequence)

        if (window.userSettings.clear_text_after_synth) {
            dialogueInput.value = ""
        }

        res = res.split("\n")
        let pitchData = res[0]
        let durationsData = res[1]
        let energyData = res[2]
        let em_angryData = res[3]
        let em_happyData = res[4]
        let em_sadData = res[5]
        let em_surpriseData = res[6]
        const editorStyles = res[7]&&res[7].length ? JSON.parse(res[7]) : undefined
        let cleanedSequence = res[8].split("|").map(c=>c.replaceAll("{", "").replaceAll("}", "").replace(/\s/g, "_"))
        const start_index = res[9]
        const end_index = res[10]
        pitchData = pitchData.split(",").map(v => parseFloat(v))

        // For use in adjusting editor range
        const maxPitchVal = pitchData.reduce((p,c)=>Math.max(p, Math.abs(c)), 0)
        if (maxPitchVal>window.sequenceEditor.default_pitchSliderRange) {
            window.sequenceEditor.pitchSliderRange = maxPitchVal
        } else {
            window.sequenceEditor.pitchSliderRange = window.sequenceEditor.default_pitchSliderRange
        }

        em_angryData = em_angryData.length ? em_angryData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v)) : []
        em_happyData = em_happyData.length ? em_happyData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v)) : []
        em_sadData = em_sadData.length ? em_sadData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v)) : []
        em_surpriseData = em_surpriseData.length ? em_surpriseData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v)) : []

        if (energyData.length) {
            energyData = energyData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v))

            // For use in adjusting editor range
            const maxEnergyVal = energyData.reduce((p,c)=>Math.max(p, c), 0)
            const minEnergyVal = energyData.reduce((p,c)=>Math.min(p, c), 100)

            if (minEnergyVal<window.sequenceEditor.default_MIN_ENERGY) {
                window.sequenceEditor.MIN_ENERGY = minEnergyVal
            } else {
                window.sequenceEditor.MIN_ENERGY = window.sequenceEditor.default_MIN_ENERGY
            }
            if (maxEnergyVal>window.sequenceEditor.default_MAX_ENERGY) {
                window.sequenceEditor.MAX_ENERGY = maxEnergyVal
            } else {
                window.sequenceEditor.MAX_ENERGY = window.sequenceEditor.default_MAX_ENERGY
            }

        } else {
            energyData = []
        }
        durationsData = durationsData.split(",").map(v => isFreshRegen ? parseFloat(v) : parseFloat(v)/pace_slid.value)

        const doTheRest = () => {

            window.sequenceEditor.base_lang = base_lang_select.value
            window.sequenceEditor.inputSequence = sequence
            window.sequenceEditor.sequence = cleanedSequence

            if (pitch.length==0 || isFreshRegen) {
                window.sequenceEditor.resetPitch = pitchData
                window.sequenceEditor.resetDurs = durationsData
                window.sequenceEditor.resetEnergy = energyData
                window.sequenceEditor.resetEmAngry = em_angryData
                window.sequenceEditor.resetEmHappy = em_happyData
                window.sequenceEditor.resetEmSad = em_sadData
                window.sequenceEditor.resetEmSurprise = em_surpriseData
            }

            window.sequenceEditor.letters = cleanedSequence
            window.sequenceEditor.pitchNew = pitchData.map(p=>p)
            window.sequenceEditor.dursNew = durationsData.map(v=>v)
            if (window.currentModel.modelType=="xVAPitch") {
                window.sequenceEditor.energyNew = energyData.map(v=>v)
                window.sequenceEditor.emAngryNew = em_angryData.map(v=>v)
                window.sequenceEditor.emHappyNew = em_happyData.map(v=>v)
                window.sequenceEditor.emSadNew = em_sadData.map(v=>v)
                window.sequenceEditor.emSurpriseNew = em_surpriseData.map(v=>v)
                window.sequenceEditor.loadStylesData(editorStyles)
            }
            window.sequenceEditor.init()
            window.sequenceEditor.update(window.currentModel.modelType)

            window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
            window.sequenceEditor.autoInferTimer = null
            window.sequenceEditor.hasChanged = false


            toggleSpinnerButtons()
            if (keepSampleButton.dataset.newFileLocation && keepSampleButton.dataset.newFileLocation.startsWith("BATCH_EDIT")) {
                console.log("_debug_")
            } else {
                if (window.userSettings[`outpath_${game}`]) {
                    keepSampleButton.dataset.newFileLocation = `${window.userSettings[`outpath_${game}`]}/${voiceType}/${outputFileName}.wav`
                } else {
                    keepSampleButton.dataset.newFileLocation = `${__dirname.replace(/\\/g,"/")}/output/${voiceType}/${outputFileName}.wav`
                }
            }
            keepSampleButton.disabled = false
            window.tempFileLocation = tempFileLocation


            // Wavesurfer
            if (!window.wavesurfer) {
                window.initWaveSurfer(`${__dirname.replace("/javascript", "")}/output/${tempFileLocation.split("/").reverse()[0]}`)
            } else {
                window.wavesurfer.load(`${__dirname.replace("/javascript", "")}/output/${tempFileLocation.split("/").reverse()[0]}`)
            }

            window.wavesurfer.on("ready",  () => {

                wavesurferContainer.style.opacity = 1

                if (window.userSettings.autoPlayGen) {

                    if (window.userSettings.playChangedAudio) {
                        const playbackStartEnd = window.sequenceEditor.getChangedTimeStamps(start_index, end_index, window.wavesurfer.getDuration())
                        if (playbackStartEnd) {
                            wavesurfer.play(playbackStartEnd[0], playbackStartEnd[1])
                        } else {
                            wavesurfer.play()
                        }
                    } else {
                        wavesurfer.play()
                    }
                    window.sequenceEditor.adjustedLetters = new Set()
                    samplePlayPause.innerHTML = window.i18n.PAUSE
                }
            })

            // Persistance across sessions
            localStorage.setItem("tempFileLocation", tempFileLocation)
        }


        if (window.userSettings.audio.ffmpeg) {
            const options = {
                hz: window.userSettings.audio.hz,
                padStart: window.userSettings.audio.padStart,
                padEnd: window.userSettings.audio.padEnd,
                bit_depth: window.userSettings.audio.bitdepth,
                amplitude: window.userSettings.audio.amplitude,
                pitchMult: window.userSettings.audio.pitchMult,
                tempo: window.userSettings.audio.tempo,
                deessing: window.userSettings.audio.deessing,
                nr: window.userSettings.audio.nr,
                nf: window.userSettings.audio.nf,
                useNR: window.userSettings.audio.useNR,
                useSR: useSRCkbx.checked,
                useCleanup: useCleanupCkbx.checked,
            }

            const extraInfo = {
                game: window.currentGame.gameId,
                voiceId: window.currentModel.voiceId,
                voiceName: window.currentModel.voiceName,
                inputSequence: sequence,
                letters: cleanedSequence,
                pitch: pitchData.map(p=>p),
                energy: energyData.map(p=>p),
                em_angry: em_angryData.map(p=>p),
                em_happy: em_happyData.map(p=>p),
                em_sad: em_sadData.map(p=>p),
                em_surprise: em_surpriseData.map(p=>p),
                durations: durationsData.map(v=>v)
            }

            doFetch(`http://localhost:8008/outputAudio`, {
                method: "Post",
                body: JSON.stringify({
                    input_path: tempFileLocation,
                    output_path: tempFileLocation.replace(".wav", `_ffmpeg.${window.userSettings.audio.format}`),
                    pluginsContext: JSON.stringify(window.pluginsContext),
                    extraInfo: JSON.stringify(extraInfo),
                    isBatchMode: false,
                    options: JSON.stringify(options)
                })
            }).then(r=>r.text()).then(res => {
                if (res.length && res!="-") {
                    console.log("res", res)
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${res}`).then(() => toggleSpinnerButtons())
                } else {
                    tempFileLocation = tempFileLocation.replace(".wav", `_ffmpeg.${window.userSettings.audio.format}`)
                    doTheRest()
                }
            }).catch(res => {
                console.log(res)
                window.appLogger.log(`outputAudio error: ${res}`)
                // closeModal().then(() => {
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${res}`)
                // })
            })
        } else {
            doTheRest()
        }


    }).catch(res => {
        window.isGenerating = false
        console.log(res)
        window.appLogger.log(res)
        window.errorModal(window.i18n.SOMETHING_WENT_WRONG)
        toggleSpinnerButtons()
    })
}

generateVoiceButton.addEventListener("click", () => {
    try {fs.mkdirSync(window.userSettings[`outpath_${game}`])} catch (e) {/*Do nothing*/}
    try {fs.mkdirSync(`${window.userSettings[`outpath_${game}`]}/${voiceId}`)} catch (e) {/*Do nothing*/}

    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {
        window.loadModel()
    } else {
        window.synthesizeSample()
    }
})




window.saveFile = (from, to, skipUIRecord=false) => {
    to = to.split("%20").join(" ")
    to = to.replace(".wav", `.${window.userSettings.audio.format}`)

    // Make the containing folder if it does not already exist
    let containerFolderPath = to.split("/")
    containerFolderPath = containerFolderPath.slice(0,containerFolderPath.length-1).join("/")

    try {fs.mkdirSync(containerFolderPath)} catch (e) {/*Do nothing*/}

    // For plugins
    const pluginData = {
        game: window.currentGame.gameId,
        voiceId: window.currentModel.voiceId,
        voiceName: window.currentModel.voiceName,
        inputSequence: window.sequenceEditor.inputSequence,
        letters: window.sequenceEditor.letters,
        pitch: window.sequenceEditor.pitchNew,
        durations: window.sequenceEditor.dursNew,
        vocoder: vocoder_select.value,
        from, to
    }
    const options = {
        hz: window.userSettings.audio.hz,
        padStart: window.userSettings.audio.padStart,
        padEnd: window.userSettings.audio.padEnd,
        bit_depth: window.userSettings.audio.bitdepth,
        amplitude: window.userSettings.audio.amplitude,
        pitchMult: window.userSettings.audio.pitchMult,
        tempo: window.userSettings.audio.tempo,
        deessing: window.userSettings.audio.deessing,
        nr: window.userSettings.audio.nr,
        nf: window.userSettings.audio.nf,
        useNR: window.userSettings.audio.useNR,
        useSR: useSRCkbx.checked
    }
    pluginData.audioOptions = options
    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["pre"], event="pre keep-sample", pluginData)

    const jsonDataOut = {
        inputSequence: dialogueInput.value.trim(),
        pacing: parseFloat(pace_slid.value),
        letters: window.sequenceEditor.letters,
        currentVoice: window.sequenceEditor.currentVoice,
        resetEnergy: window.sequenceEditor.resetEnergy,
        resetPitch: window.sequenceEditor.resetPitch,
        resetDurs: window.sequenceEditor.resetDurs,
        resetEmAngry: window.sequenceEditor.resetEmAngry,
        resetEmHappy: window.sequenceEditor.resetEmHappy,
        resetEmSad: window.sequenceEditor.resetEmSad,
        resetEmSurprise: window.sequenceEditor.resetEmSurprise,
        styleValuesReset: window.sequenceEditor.styleValuesReset,
        ampFlatCounter: window.sequenceEditor.ampFlatCounter,
        inputSequence: window.sequenceEditor.inputSequence,
        sequence: window.sequenceEditor.sequence,
        pitchNew: window.sequenceEditor.pitchNew,
        energyNew: window.sequenceEditor.energyNew,
        dursNew: window.sequenceEditor.dursNew,
        emAngryNew: window.sequenceEditor.emAngryNew,
        emHappyNew: window.sequenceEditor.emHappyNew,
        emSadNew: window.sequenceEditor.emSadNew,
        emSurpriseNew: window.sequenceEditor.emSurpriseNew,
        styleValuesNew: window.sequenceEditor.styleValuesNew,
    }

    let outputFileName = to.split("/").reverse()[0].split(".").reverse().slice(1, 1000)
    const toExt = to.split(".").reverse()[0]

    if (window.userSettings.filenameNumericalSeq) {
        outputFileName = outputFileName[0]+"."+outputFileName.slice(1,1000).reverse().join(".")
    } else {
        outputFileName = outputFileName.reverse().join(".")
    }
    to = `${to.split("/").reverse().slice(1,10000).reverse().join("/")}/${outputFileName}`


    const allFiles = fs.readdirSync(`${path}/output`).filter(fname => fname.includes(from.split("/").reverse()[0].split(".")[0]))
    const toFolder = to.split("/").reverse().slice(1, 1000).reverse().join("/")


    allFiles.forEach(fname => {
        const ext = fname.split(".").reverse()[0]
        fs.copyFile(`${path}/output/${fname}`, `${toFolder}/${outputFileName}.${ext}`, err => {
            if (err) {
                console.log(err)
                window.appLogger.log(`Error in saveFile->outputAudio[no ffmpeg]: ${err}`)
                if (!fs.existsSync(from)) {
                    window.appLogger.log(`${window.i18n.TEMP_FILE_NOT_EXIST}: ${from}`)
                }
                if (!fs.existsSync(toFolder)) {
                    window.appLogger.log(`${window.i18n.OUT_DIR_NOT_EXIST}: ${toFolder}`)
                }
            } else {
                if (window.userSettings.outputJSON && window.sequenceEditor.letters.length) {
                    fs.writeFileSync(`${to}.${toExt}.json`, JSON.stringify(jsonDataOut, null, 4))
                }
                if (!skipUIRecord) {
                    window.initMainPagePagination(`${window.userSettings[`outpath_${window.currentGame.gameId}`]}/${window.currentModel.voiceId}`)
                    window.refreshRecordsList()
                }
                window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
            }
        })
    })
}

window.keepSampleFunction = shiftClick => {
    if (keepSampleButton.dataset.newFileLocation) {

        const skipUIRecord = keepSampleButton.dataset.newFileLocation.includes("BATCH_EDIT")
        let fromLocation = window.tempFileLocation
        let toLocation = keepSampleButton.dataset.newFileLocation.replace("BATCH_EDIT", "")

        if (!skipUIRecord) {
            toLocation = toLocation.split("/")
            toLocation[toLocation.length-1] = toLocation[toLocation.length-1].replace(/[\/\\:\*?<>"|]*/g, "")
            toLocation[toLocation.length-1] = toLocation[toLocation.length-1].replace(/\.wav$/, "").slice(0, window.userSettings.max_filename_chars).replace(/\.$/, "")
        }


        // Numerical file name counter
        if (!skipUIRecord && window.userSettings.filenameNumericalSeq) {
            let existingFiles = []
            try {
                existingFiles = fs.readdirSync(toLocation.slice(0, toLocation.length-1).join("/")).filter(fname => !fname.endsWith(".json"))
            } catch (e) {
                console.log(e)
            }
            existingFiles = existingFiles.filter(fname => fname.includes(toLocation[toLocation.length-1]))
            existingFiles = existingFiles.map(fname => {
                const parts = fname.split(".")
                parts.reverse()
                if (parts.length>2 && parts.reverse()[0].length) {
                    if (parseInt(parts[0]) != NaN) {
                        return parseInt(parts[0])
                    }
                }
                return null
            })
            existingFiles = existingFiles.filter(val => !!val)
            if (existingFiles.length==0) {
                existingFiles.push(0)
            }

            if (existingFiles.length) {
                existingFiles = existingFiles.sort((a,b) => {a<b?-1:1})
                toLocation[toLocation.length-1] = `${toLocation[toLocation.length-1]}.${String(existingFiles[existingFiles.length-1]+1).padStart(4, "0")}`
            }
        }


        if (!skipUIRecord) {
            toLocation[toLocation.length-1] += ".wav"
            toLocation = toLocation.join("/")
        }


        const outFolder = toLocation.split("/").reverse().slice(2, 100).reverse().join("/")
        if (!fs.existsSync(outFolder)) {
            return void window.errorModal(`${window.i18n.OUT_DIR_NOT_EXIST}:<br><br><i>${outFolder}</i><br><br>${window.i18n.YOU_CAN_CHANGE_IN_SETTINGS}`)
        }

        // File name conflict
        const alreadyExists = fs.existsSync(toLocation)
        if (alreadyExists || shiftClick) {

            const promptText = alreadyExists ? window.i18n.FILE_EXISTS_ADJUST : window.i18n.ENTER_FILE_NAME

            createModal("prompt", {
                prompt: promptText,
                value: toLocation.split("/").reverse()[0].replace(".wav", `.${window.userSettings.audio.format}`)
            }).then(newFileName => {

                let toLocationOut = toLocation.split("/").reverse()
                toLocationOut[0] = newFileName.replace(`.${window.userSettings.audio.format}`, "") + `.${window.userSettings.audio.format}`
                let outDir = toLocationOut
                outDir.shift()

                newFileName = (newFileName.replace(`.${window.userSettings.audio.format}`, "") + `.${window.userSettings.audio.format}`).replace(/[\/\\:\*?<>"|]*/g, "")
                toLocationOut.reverse()
                toLocationOut.push(newFileName)

                if (fs.existsSync(outDir.slice(0, outDir.length-1).join("/"))) {
                    const existingFiles = fs.readdirSync(outDir.slice(0, outDir.length-1).join("/"))
                    const existingFileConflict = existingFiles.filter(name => name==newFileName)


                    const finalOutLocation = toLocationOut.join("/")

                    if (existingFileConflict.length) {
                        // Remove the entry from the output files' preview
                        Array.from(voiceSamples.querySelectorAll("div.sample")).forEach(sampleElem => {
                            const source = sampleElem.querySelector("source")
                            let sourceSrc = source.src.split("%20").join(" ").replace("file:///", "")
                            sourceSrc = sourceSrc.split("/").reverse()
                            const finalFileName = finalOutLocation.split("/").reverse()

                            if (sourceSrc[0] == finalFileName[0]) {
                                sampleElem.parentNode.removeChild(sampleElem)
                            }
                        })

                        // Remove the old file and write the new one in
                        fs.unlink(finalOutLocation, err => {
                            if (err) {
                                console.log(err)
                                window.appLogger.log(`Error in keepSample: ${err}`)
                            }
                            window.saveFile(fromLocation, finalOutLocation, skipUIRecord)
                        })
                        return
                    } else {
                        window.saveFile(fromLocation, toLocationOut.join("/"), skipUIRecord)
                        return
                    }
                }
                window.saveFile(fromLocation, toLocationOut.join("/"), skipUIRecord)
            })

        } else {
            window.saveFile(fromLocation, toLocation, skipUIRecord)
        }
    }
}
keepSampleButton.addEventListener("click", event => keepSampleFunction(event.shiftKey))



// Weird recursive intermittent promises to repeatedly check if the server is up yet - but it works!
window.serverIsUp = false
const serverStartingMessage = `${window.i18n.LOADING}...<br>${window.i18n.MAY_TAKE_A_MINUTE}<br><br>${window.i18n.STARTING_PYTHON}...`
window.doWeirdServerStartupCheck = () => {
    const check = () => {
        return new Promise(topResolve => {
            if (window.serverIsUp) {
                topResolve()
            } else {
                (new Promise((resolve, reject) => {
                    // Gather the model paths to send to the server
                    const modelsPaths = {}
                    Object.keys(window.userSettings).filter(key => key.includes("modelspath_")).forEach(key => {
                        modelsPaths[key.split("_")[1]] = window.userSettings[key]
                    })

                    doFetch(`http://localhost:8008/checkReady`, {
                        method: "Post",
                        body: JSON.stringify({
                            device: (window.userSettings.useGPU&&window.userSettings.installation=="gpu")?"gpu":"cpu",
                            modelsPaths: JSON.stringify(modelsPaths)
                        })
                    }).then(r => r.text()).then(r => {
                        closeModal([document.querySelector("#activeModal"), modalContainer], [totdContainer, EULAContainer], true).then(() => {
                            window.pluginsManager.updateUI()
                            if (!window.pluginsManager.hasRunPostStartPlugins) {
                                window.pluginsManager.hasRunPostStartPlugins = true
                                window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["post"], event="post start")
                                window.electronBrowserWindow.setProgressBar(-1)
                                window.checkForWorkshopInstallations()
                            }
                        })
                        window.serverIsUp = true
                        if (window.userSettings.installation=="cpu") {

                            if (useGPUCbx.checked) {
                                doFetch(`http://localhost:8008/setDevice`, {
                                    method: "Post",
                                    body: JSON.stringify({device: "cpu"})
                                })
                            }
                            useGPUCbx.checked = false
                            useGPUCbx.disabled = true
                            window.userSettings.useGPU = false
                            saveUserSettings()
                        }

                        resolve()
                    }).catch((err) => {
                        reject()
                    })
                })).catch(() => {
                    setTimeout(async () => {
                        await check()
                        topResolve()
                    }, 100)
                })
            }
        })
    }

    check()
}
window.doWeirdServerStartupCheck()

modalContainer.addEventListener("click", event => {
    try {
        if (event.target==modalContainer && activeModal.dataset.type!="spinner") {
            closeModal()
        }
    } catch (e) {}
})


// Cached UI stuff
// =========
dialogueInput.addEventListener("keyup", (event) => {
    localStorage.setItem("dialogueInput", " "+dialogueInput.value.trim()+" ")
    window.sequenceEditor.hasChanged = true
})

const dialogueInputCache = localStorage.getItem("dialogueInput")

if (dialogueInputCache) {
    dialogueInput.value = dialogueInputCache
}

// =========








vocoder_select.value = window.userSettings.vocoder.includes(".hg.") ? "qnd" : window.userSettings.vocoder
window.changeVocoder = vocoder => {
    return new Promise(resolve => {
        spinnerModal(window.i18n.CHANGING_MODELS)
        doFetch(`http://localhost:8008/setVocoder`, {
            method: "Post",
            body: JSON.stringify({
                vocoder,
                modelPath: vocoder=="256_waveglow" ? window.userSettings.waveglow_path : window.userSettings.bigwaveglow_path
            })
        }).then(r=>r.text()).then((res) => {
            closeModal().then(() => {
                setTimeout(() => {
                    if (res=="ENOENT") {
                        vocoder_select.value = window.userSettings.vocoder
                        window.errorModal(`${window.i18n.BATCH_MODEL_NOT_FOUND}.${vocoder.includes("waveglow")?" "+window.i18n.BATCH_DOWNLOAD_WAVEGLOW:""}`)
                        resolve()
                    } else {
                        window.batch_state.lastVocoder = vocoder
                        window.userSettings.vocoder = vocoder
                        saveUserSettings()
                        resolve()
                    }
                }, 300)
            })
        })
    })
}
vocoder_select.addEventListener("change", () => window.changeVocoder(vocoder_select.value))

useSRCkbx.addEventListener("click", () => {
    let userHasSeenThisAlready = localStorage.getItem("useSRHintSeen")
    if (useSRCkbx.checked && !userHasSeenThisAlready) {
        window.confirmModal(window.i18n.USE_SR_HINT).then(resp => {
            if (resp) {
                localStorage.setItem("useSRHintSeen", "true")
            }
        })
    }

})

dialogueInput.addEventListener("contextmenu", event => {
    event.preventDefault()
    ipcRenderer.send('show-context-menu')
})
ipcRenderer.on('context-menu-command', (e, command) => {
    if (command=="context-copy") {
        if (dialogueInput.selectionStart != dialogueInput.selectionEnd) {
            clipboard.writeText(dialogueInput.value.slice(dialogueInput.selectionStart, dialogueInput.selectionEnd))
        }
    } else if (command=="context-paste") {
        if (clipboard.readText().length) {
            let newString = dialogueInput.value.slice(0, dialogueInput.selectionStart) + clipboard.readText() + dialogueInput.value.slice(dialogueInput.selectionEnd, dialogueInput.value.length)
            dialogueInput.value = newString
        }
    }
})


window.setupModal(workbenchIcon, workbenchContainer, () => window.initVoiceWorkbench())


// Info
// ====
window.setupModal(infoIcon, infoContainer)


// Patreon
// =======
window.setupModal(patreonIcon, patreonContainer, () => {
    const data = fs.readFileSync(`${path}/patreon.txt`, "utf8") + ", minermanb"
    creditsList.innerHTML = data
})


// Updates
// =======
app_version.innerHTML = window.appVersion
updatesVersions.innerHTML = `${window.i18n.THIS_APP_VERSION}: ${window.appVersion}`

const checkForUpdates = () => {
    doFetch("http://danruta.co.uk/xvasynth_updates.txt").then(r=>r.json()).then(data => {
        fs.writeFileSync(`${path}/updates.json`, JSON.stringify(data), "utf8")
        checkUpdates.innerHTML = window.i18n.CHECK_FOR_UPDATES
        window.showUpdates()
    }).catch(() => {
        checkUpdates.innerHTML = window.i18n.CANT_REACH_SERVER
    })
}
window.showUpdates = () => {
    window.updatesLog = fs.readFileSync(`${path}/updates.json`, "utf8")
    window.updatesLog = JSON.parse(window.updatesLog)
    const sortedLogVersions = Object.keys(window.updatesLog).map( a => a.split('.').map( n => +n+100000 ).join('.') ).sort()
        .map( a => a.split('.').map( n => +n-100000 ).join('.') )

    const appVersion = window.appVersion.replace("v", "")
    const appIsUpToDate = sortedLogVersions.indexOf(appVersion)==(sortedLogVersions.length-1) || sortedLogVersions.indexOf(appVersion)==-1

    if (!appIsUpToDate) {
        update_nothing.style.display = "none"
        update_something.style.display = "block"
        updatesVersions.innerHTML = `${window.i18n.THIS_APP_VERSION}: ${appVersion}. ${window.i18n.AVAILABLE}: ${sortedLogVersions[sortedLogVersions.length-1]}`
    } else {
        updatesVersions.innerHTML = `${window.i18n.THIS_APP_VERSION}: ${appVersion}. ${window.i18n.UPTODATE}`
    }

    updatesLogList.innerHTML = ""
    sortedLogVersions.reverse().forEach(version => {
        const versionLabel = createElem("h2", version)
        const logItem = createElem("div", versionLabel)
        window.updatesLog[version].split("\n").forEach(line => {
            logItem.appendChild(createElem("div", line))
        })
        updatesLogList.appendChild(logItem)
    })
}
checkForUpdates()
window.setupModal(updatesIcon, updatesContainer)

checkUpdates.addEventListener("click", () => {
    checkUpdates.innerHTML = window.i18n.CHECKING_FOR_UPDATES
    checkForUpdates()
})
window.showUpdates()


// Batch generation
// ========
window.setupModal(batchIcon, batchGenerationContainer)

// Settings
// ========
window.setupModal(settingsCog, settingsContainer)

// Change Game
// ===========
window.setupModal(changeGameButton, gameSelectionContainer)
changeGameButton.addEventListener("click", () => searchGameInput.focus())

window.gameAssets = {}

window.updateGameList = (doLoadAllModels=true) => {
    gameSelectionListContainer.innerHTML = ""
    const fileNames = fs.readdirSync(`${window.path}/assets`)

    let totalVoices = 0
    let totalGames = new Set()

    const itemsToSort = []
    // const gameIDs = doLoadAllModels ? fileNames.filter(fn=>fn.endsWith(".json")) : Object.keys(window.games).map(gID => gID+".json")
    const gameIDs = fileNames.filter(fn=>fn.endsWith(".json"))

    gameIDs.forEach(gameId => {

        const metadata = fs.existsSync(`${window.path}/assets/${gameId}`) ? JSON.parse(fs.readFileSync(`${window.path}/assets/${gameId}`)) : window.games[gameId.replace(".json", "")].dummyGameTheme
        gameId = gameId.replace(".json", "")
        metadata.gameId = gameId
        const assetFile = metadata.assetFile

        const gameSelection = createElem("div.gameSelection")
        gameSelection.style.background = `url("assets/${assetFile}")`

        const gameName = metadata.gameName
        const gameSelectionContent = createElem("div.gameSelectionContent")


        let numVoices = 0
        const modelsPath = window.userSettings[`modelspath_${gameId}`]
        if (fs.existsSync(modelsPath)) {
            const files = fs.readdirSync(modelsPath)
            numVoices = files.filter(fn => fn.includes(".json")).length
            totalVoices += numVoices
        }
        if (numVoices==0) {
            gameSelectionContent.style.background = "rgba(150,150,150,0.7)"
        } else {
            gameSelectionContent.classList.add("gameSelectionContentToHover")
            totalGames.add(gameId)
        }

        gameSelectionContent.appendChild(createElem("div", `${numVoices} ${(numVoices>1||numVoices==0)?window.i18n.VOICE_PLURAL:window.i18n.VOICE}`))
        gameSelectionContent.appendChild(createElem("div", gameName))

        gameSelection.appendChild(gameSelectionContent)

        window.gameAssets[gameId] = metadata
        gameSelectionContent.addEventListener("click", () => {
            voiceSearchInput.focus()
            searchGameInput.value = ""
            changeGame(metadata)
            closeModal(gameSelectionContainer)
            Array.from(gameSelectionListContainer.children).forEach(elem => elem.style.display="flex")
        })

        itemsToSort.push([numVoices, gameSelection])

        const modelsDir = window.userSettings[`modelspath_${gameId}`]
        if (!window.watchedModelsDirs.includes(modelsDir)) {
            window.watchedModelsDirs.push(modelsDir)

            try {
                fs.watch(modelsDir, {recursive: false, persistent: true}, (eventType, filename) => {
                    if (window.userSettings.autoReloadVoices) {
                        if (doLoadAllModels) {
                            loadAllModels().then(() => changeGame(metadata))
                        }
                    }
                })
            } catch (e) {
                // console.log(e)
            }
        }
    })


    itemsToSort.sort((a,b) => a[0]<b[0]?1:-1).forEach(([numVoices, elem]) => {
        gameSelectionListContainer.appendChild(elem)
    })

    searchGameInput.addEventListener("keyup", (event) => {

        if (event.key=="Enter") {
            const voiceElems = Array.from(gameSelectionListContainer.children).filter(elem => elem.style.display=="flex")
            if (voiceElems.length==1) {
                voiceElems[0].children[0].click()
                searchGameInput.value = ""
            }
        }

        const voiceElems = Array.from(gameSelectionListContainer.children)
        if (searchGameInput.value.length) {
            voiceElems.forEach(elem => {
                if (elem.children[0].children[1].innerHTML.toLowerCase().includes(searchGameInput.value)) {
                    elem.style.display="flex"
                } else {
                    elem.style.display="none"
                }
            })

        } else {
            voiceElems.forEach(elem => elem.style.display="block")
        }
    })

    searchGameInput.placeholder = window.i18n.SEARCH_N_GAMES_WITH_N2_VOICES.replace("_1", Array.from(totalGames).length).replace("_2", totalVoices)

    if (doLoadAllModels) {
        loadAllModels().then(() => {
            // Load the last selected game
            const lastGame = localStorage.getItem("lastGame")

            if (lastGame) {
                changeGame(JSON.parse(lastGame))
            }
        })
    }
}
window.updateGameList()


// Embeddings
// ==========
window.setupModal(embeddingsIcon, embeddingsContainer, () => {
    setTimeout(() => {
        if (window.embeddingsState.isReady) {
            window.embeddings_updateSize()
        }
    }, 100)
    window.embeddings_updateSize()
    window.embeddingsState.isOpen = true
    if (!window.embeddingsState.ready) {
        setTimeout(() => {
            window.embeddingsState.ready = true
            window.initEmbeddingsScene()
            setTimeout(() => {
                window.computeEmbsAndDimReduction(true)
            }, 300)
        }, 100)
    }
}, () => {
    window.embeddingsState.isOpen = false
})

// Arpabet
// =======
window.setupModal(arpabetIcon, arpabetContainer, () => setTimeout(()=> !window.arpabetMenuState.hasInitialised && window.refreshDictionariesList(), 100))


// Plugins
// =======
window.setupModal(pluginsIcon, pluginsContainer)


window.setupModal(style_emb_manage_btn, styleEmbeddingsContainer, window.styleEmbsModalOpenCallback)


// Other
// =====
window.setupModal(reset_what_open_btn, resetContainer)
window.setupModal(i18n_batch_metadata_open_btn, batchMetadataCSVContainer)

voiceSearchInput.addEventListener("keyup", () => {

    if (event.key=="Enter") {
        const voiceElems = Array.from(voiceTypeContainer.children).filter(elem => elem.style.display=="block")
        if (voiceElems.length==1) {
            voiceElems[0].click()
            generateVoiceButton.click()
            voiceSearchInput.value = ""
        }
    }

    const voiceElems = Array.from(voiceTypeContainer.children)

    if (voiceSearchInput.value.length) {
        voiceElems.forEach(elem => {
            if (elem.innerHTML.toLowerCase().includes(voiceSearchInput.value)) {
                elem.style.display="block"
            } else {
                elem.style.display="none"
            }
        })

    } else {
        voiceElems.forEach(elem => elem.style.display="block")
    }
})

// Splash/EULA
splashNextButton1.addEventListener("click", () => {
    splash_screen1.style.display = "none"
    splash_screen2.style.display = "flex"
})
EULA_closeButon.addEventListener("click", () => {
    if (EULA_accept_ckbx.checked) {
        closeModal(EULAContainer)
        window.userSettings.EULA_accepted_2023 = true
        saveUserSettings()

        if (!window.totd_state.startupChecked) {
            window.showTipIfEnabledAndNewDay().then(() => {
                if (!serverIsUp) {
                    spinnerModal(serverStartingMessage)
                }
            })
        }

        if (!serverIsUp && totdContainer.style.display=="none") {
            window.spinnerModal(serverStartingMessage)
        }
    }
})
if (!Object.keys(window.userSettings).includes("EULA_accepted_2023") || !window.userSettings.EULA_accepted_2023) {
    EULAContainer.style.opacity = 0
    EULAContainer.style.display = "flex"
    requestAnimationFrame(() => requestAnimationFrame(() => EULAContainer.style.opacity = 1))
    requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
} else {
    window.showTipIfEnabledAndNewDay().then(() => {
        // If not, or the user closed the window quickly, show the server is starting message if still booting up
        if (!window.serverIsUp) {
            spinnerModal(serverStartingMessage)
        }
    })
}


// Links
document.querySelectorAll('a[href^="http"]').forEach(a => a.addEventListener("click", e => {
    event.preventDefault()
    shell.openExternal(a.href)
}))

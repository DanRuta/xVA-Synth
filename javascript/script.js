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
window.onerror = (err, url, lineNum) => window.appLogger.log(`onerror: ${err.stack}`)
require("./javascript/i18n.js")
require("./javascript/util.js")
require("./javascript/nexus.js")
require("./javascript/embeddings.js")
require("./javascript/totd.js")
require("./javascript/arpabet.js")
require("./javascript/style_embeddings.js")
const {Editor} = require("./javascript/editor.js")
const {saveUserSettings, deleteFolderRecursive} = require("./javascript/settingsMenu.js")
const xVASpeech = require("./javascript/speech2speech.js")
require("./javascript/batch.js")
window.electronBrowserWindow = require("electron").remote.getCurrentWindow()
const child = require("child_process").execFile
const spawn = require("child_process").spawn


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


// Audio player
window.initWaveSurfer = (src) => {
    if (window.wavesurfer) {
        window.wavesurfer.stop()
        wavesurferContainer.innerHTML = ""
    } else {
        window.wavesurfer = WaveSurfer.create({
            container: '#wavesurferContainer',
            backend: 'MediaElement',
            waveColor: `#${window.currentGame.themeColourPrimary}`,
            height: 100,
            progressColor: 'white',
            responsive: true,
        })
    }
    try {
        window.wavesurfer.setSinkId(window.userSettings.base_speaker)
    } catch (e) {
        console.log("Can't set sinkId")
    }
    if (src) {
        window.wavesurfer.load(src)
    }
    window.wavesurfer.on("finish", () => {
        samplePlayPause.innerHTML = window.i18n.PLAY
    })
    window.wavesurfer.on("seek", event => {
        if (event!=0) {
            window.wavesurfer.play()
            samplePlayPause.innerHTML = window.i18n.PAUSE
        }
    })
}


window.registerModel = (modelsPath, gameFolder, model, {gameId, voiceId, voiceName, voiceDescription, gender, variant, modelType, emb_i}) => {
    // Add game, if the game hasn't been added yet
    if (!window.games.hasOwnProperty(gameId)) {

        const gameAsset = fs.readdirSync(`${path}/assets`).find(f => f==gameId+".json")
        const gameTheme = JSON.parse(fs.readFileSync(`${path}/assets/${gameAsset}`))

        window.games[gameId] = {
            models: [],
            gameTheme,
            gameAsset
        }
    }

    const audioPreviewPath = `${modelsPath}/${model.games.find(({gameId}) => gameId==gameFolder).voiceId}`

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
                        window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME.replace("_1", gameFolder)} `+fileName)
                        window.appLogger.log(e)
                        window.appLogger.log(e.stack)
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
        resolve()
    })
}
setting_models_path_input.addEventListener("change", () => {
    const gameFolder = window.currentGame.gameId

    setting_models_path_input.value = setting_models_path_input.value.replace(/\/\//g, "/").replace(/\\/g,"/")
    window.userSettings[`modelspath_${gameFolder}`] = setting_models_path_input.value
    saveUserSettings()
    loadAllModels().then(() => {
        changeGame(window.currentGame)
    })

    if (!window.watchedModelsDirs.includes(setting_models_path_input.value)) {
        window.watchedModelsDirs.push(setting_models_path_input.value)
        fs.watch(setting_models_path_input.value, {recursive: false, persistent: true}, (eventType, filename) => {
            changeGame(window.currentGame)
        })
    }
    window.updateGameList()

    // Gather the model paths to send to the server
    const modelsPaths = {}
    Object.keys(window.userSettings).filter(key => key.includes("modelspath_")).forEach(key => {
        modelsPaths[key.split("_")[1]] = window.userSettings[key]
    })
    doFetch(`http://localhost:8008/setAvailableVoices`, {
        method: "Post",
        body: JSON.stringify({
            modelsPaths: JSON.stringify(modelsPaths)
        })
    })
})

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
        model: `${model.modelsPath}/${variant.voiceId}`,
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
    } else if (titleID) {
        document.title = `${titleID}VA Synth`
        dragBar.innerHTML = `${titleID}VA Synth`
    } else {
        document.title = `xVA Synth`
        dragBar.innerHTML = `xVA Synth`
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

    (window.games[meta.gameId] ? window.games[meta.gameId].models : []).forEach(({modelsPath, audioPreviewPath, gameId, variants, voiceName}) => {

        const {voiceId, voiceDescription, hifi, model} = variants[0]
        const modelVersion = variants[0].version

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
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
            audioPreview.setSinkId(window.userSettings.base_speaker)
        })

        button.addEventListener("click", event => {

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

                const modelGameFolder = audioPreviewPath.split("/")[0]

                generateVoiceButton.dataset.modelQuery = JSON.stringify({
                    outputs: parseInt(model.outputs),
                    model: `${modelsPath}/${voiceId}`,
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

            title.dataset.modelId = voiceId
            keepSampleButton.style.display = "none"
            samplePlayPause.style.display = "none"

            // Voice samples
            voiceSamples.innerHTML = ""
            refreshRecordsList(`${window.userSettings[`outpath_${meta.gameId}`]}/${button.dataset.modelId}`)
        })
        buttons.push(button)
    })

    buttons.sort((a,b) => a.innerHTML.toLowerCase()<b.innerHTML.toLowerCase()?-1:1)
        .forEach(button => voiceTypeContainer.appendChild(button))

}

window.samplePlayPauseHandler = event => {
    if (window.wavesurfer) {
        if (event.ctrlKey) {
            if (window.wavesurfer.sink_id!=window.userSettings.alt_speaker) {
                window.wavesurfer.setSinkId(window.userSettings.alt_speaker)
            }
        } else {
            if (window.wavesurfer.sink_id!=window.userSettings.base_speaker) {
                window.wavesurfer.setSinkId(window.userSettings.base_speaker)
            }
        }
    }

    if (window.wavesurfer.isPlaying()) {
        samplePlayPause.innerHTML = window.i18n.PLAY
        window.wavesurfer.playPause()
    } else {
        samplePlayPause.innerHTML = window.i18n.PAUSE
        window.wavesurfer.playPause()
    }
}
samplePlayPause.addEventListener("click", window.samplePlayPauseHandler)

window.makeSample = (src, newSample) => {
    const fileName = src.split("/").reverse()[0].split("%20").join(" ")
    const fileFormat = fileName.split(".").reverse()[0]
    const fileNameElem = createElem("div", fileName)
    const promptText = createElem("div.samplePromptText")

    if (fs.existsSync(src+".json")) {
        try {
            const lineMeta = fs.readFileSync(src+".json", "utf8")
            promptText.innerHTML = JSON.parse(lineMeta).inputSequence
            if (promptText.innerHTML.length > 130) {
                promptText.innerHTML = promptText.innerHTML.slice(0, 130)+"..."
            }
        } catch (e) {
            // console.log(e)
        }
    }
    const sample = createElem("div.sample", createElem("div", fileNameElem, promptText))
    const audioControls = createElem("div.sampleAudioControls")
    const audio = createElem("audio", {controls: true}, createElem("source", {
        src: src,
        type: `audio/${fileFormat}`
    }))
    audio.addEventListener("play", () => {
        if (window.ctrlKeyIsPressed) {
            audio.setSinkId(window.userSettings.alt_speaker)
        } else {
            audio.setSinkId(window.userSettings.base_speaker)
        }
    })
    audio.setSinkId(window.userSettings.base_speaker)
    const openFileLocationButton = createElem("div", {title: window.i18n.OPEN_CONTAINING_FOLDER})
    openFileLocationButton.innerHTML = `<svg class="openFolderSVG" id="svg" version="1.1" xmlns="http:\/\/www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="400" height="350" viewBox="0, 0, 400,350"><g id="svgg"><path id="path0" d="M39.960 53.003 C 36.442 53.516,35.992 53.635,30.800 55.422 C 15.784 60.591,3.913 74.835,0.636 91.617 C -0.372 96.776,-0.146 305.978,0.872 310.000 C 5.229 327.228,16.605 339.940,32.351 345.172 C 40.175 347.773,32.175 347.630,163.000 347.498 L 281.800 347.378 285.600 346.495 C 304.672 342.065,321.061 332.312,330.218 319.944 C 330.648 319.362,332.162 317.472,333.581 315.744 C 335.001 314.015,336.299 312.420,336.467 312.200 C 336.634 311.980,337.543 310.879,338.486 309.753 C 340.489 307.360,342.127 305.341,343.800 303.201 C 344.460 302.356,346.890 299.375,349.200 296.575 C 351.510 293.776,353.940 290.806,354.600 289.975 C 355.260 289.144,356.561 287.505,357.492 286.332 C 358.422 285.160,359.952 283.267,360.892 282.126 C 362.517 280.153,371.130 269.561,375.632 264.000 C 376.789 262.570,380.427 258.097,383.715 254.059 C 393.790 241.689,396.099 237.993,398.474 230.445 C 403.970 212.972,394.149 194.684,376.212 188.991 C 369.142 186.747,368.803 186.724,344.733 186.779 C 330.095 186.812,322.380 186.691,322.216 186.425 C 322.078 186.203,321.971 178.951,321.977 170.310 C 321.995 146.255,321.401 141.613,317.200 133.000 C 314.009 126.457,307.690 118.680,303.142 115.694 C 302.560 115.313,301.300 114.438,300.342 113.752 C 295.986 110.631,288.986 107.881,282.402 106.704 C 280.540 106.371,262.906 106.176,220.400 106.019 L 161.000 105.800 160.763 98.800 C 159.961 75.055,143.463 56.235,120.600 52.984 C 115.148 52.208,45.292 52.225,39.960 53.003 M120.348 80.330 C 130.472 83.988,133.993 90.369,133.998 105.071 C 134.003 120.968,137.334 127.726,147.110 131.675 L 149.400 132.600 213.800 132.807 C 272.726 132.996,278.392 133.071,280.453 133.690 C 286.872 135.615,292.306 141.010,294.261 147.400 C 294.928 149.578,294.996 151.483,294.998 168.000 L 295.000 186.200 292.800 186.449 C 291.590 186.585,254.330 186.725,210.000 186.759 C 163.866 186.795,128.374 186.977,127.000 187.186 C 115.800 188.887,104.936 192.929,96.705 198.458 C 95.442 199.306,94.302 200.000,94.171 200.000 C 93.815 200.000,89.287 203.526,87.000 205.583 C 84.269 208.039,80.083 212.649,76.488 217.159 C 72.902 221.657,72.598 222.031,70.800 224.169 C 70.030 225.084,68.770 226.620,68.000 227.582 C 67.230 228.544,66.054 229.977,65.387 230.766 C 64.720 231.554,62.727 234.000,60.957 236.200 C 59.188 238.400,56.346 241.910,54.642 244.000 C 52.938 246.090,50.163 249.510,48.476 251.600 C 44.000 257.146,36.689 266.126,36.212 266.665 C 35.985 266.921,34.900 268.252,33.800 269.623 C 32.700 270.994,30.947 273.125,29.904 274.358 C 28.861 275.591,28.006 276.735,28.004 276.900 C 28.002 277.065,27.728 277.200,27.395 277.200 C 26.428 277.200,26.700 96.271,27.670 93.553 C 30.020 86.972,35.122 81.823,40.800 80.300 C 44.238 79.378,47.793 79.296,81.800 79.351 L 117.800 79.410 120.348 80.330 M369.400 214.800 C 374.239 217.220,374.273 222.468,369.489 228.785 C 367.767 231.059,364.761 234.844,364.394 235.200 C 364.281 235.310,362.373 237.650,360.154 240.400 C 357.936 243.150,354.248 247.707,351.960 250.526 C 347.732 255.736,346.053 257.821,343.202 261.400 C 341.505 263.530,340.849 264.336,334.600 271.965 C 332.400 274.651,330.204 277.390,329.720 278.053 C 329.236 278.716,328.246 279.945,327.520 280.785 C 326.794 281.624,325.300 283.429,324.200 284.794 C 323.100 286.160,321.726 287.845,321.147 288.538 C 320.568 289.232,318.858 291.345,317.347 293.233 C 308.372 304.449,306.512 306.609,303.703 309.081 C 299.300 312.956,290.855 317.633,286.000 318.886 C 277.958 320.960,287.753 320.819,159.845 320.699 C 33.557 320.581,42.330 320.726,38.536 318.694 C 34.021 316.276,35.345 310.414,42.386 301.647 C 44.044 299.583,45.940 297.210,46.600 296.374 C 47.260 295.538,48.340 294.169,49.000 293.332 C 49.660 292.495,51.550 290.171,53.200 288.167 C 54.850 286.164,57.100 283.395,58.200 282.015 C 59.300 280.635,60.920 278.632,61.800 277.564 C 62.680 276.496,64.210 274.617,65.200 273.389 C 66.190 272.162,67.188 270.942,67.418 270.678 C 67.649 270.415,71.591 265.520,76.179 259.800 C 80.767 254.080,84.634 249.310,84.773 249.200 C 84.913 249.090,87.117 246.390,89.673 243.200 C 92.228 240.010,95.621 235.780,97.213 233.800 C 106.328 222.459,116.884 215.713,128.200 213.998 C 129.300 213.832,183.570 213.719,248.800 213.748 L 367.400 213.800 369.400 214.800 " stroke="none" fill="#050505" fill-rule="evenodd"></path><path id="path1" d="M0.000 46.800 C 0.000 72.540,0.072 93.600,0.159 93.600 C 0.246 93.600,0.516 92.460,0.759 91.066 C 3.484 75.417,16.060 60.496,30.800 55.422 C 35.953 53.648,36.338 53.550,40.317 52.981 C 46.066 52.159,114.817 52.161,120.600 52.984 C 143.463 56.235,159.961 75.055,160.763 98.800 L 161.000 105.800 220.400 106.019 C 262.906 106.176,280.540 106.371,282.402 106.704 C 288.986 107.881,295.986 110.631,300.342 113.752 C 301.300 114.438,302.560 115.313,303.142 115.694 C 307.690 118.680,314.009 126.457,317.200 133.000 C 321.401 141.613,321.995 146.255,321.977 170.310 C 321.971 178.951,322.078 186.203,322.216 186.425 C 322.380 186.691,330.095 186.812,344.733 186.779 C 368.803 186.724,369.142 186.747,376.212 188.991 C 381.954 190.814,388.211 194.832,391.662 198.914 C 395.916 203.945,397.373 206.765,399.354 213.800 C 399.842 215.533,399.922 201.399,399.958 107.900 L 400.000 0.000 200.000 0.000 L 0.000 0.000 0.000 46.800 M44.000 79.609 C 35.903 81.030,30.492 85.651,27.670 93.553 C 26.700 96.271,26.428 277.200,27.395 277.200 C 27.728 277.200,28.002 277.065,28.004 276.900 C 28.006 276.735,28.861 275.591,29.904 274.358 C 30.947 273.125,32.700 270.994,33.800 269.623 C 34.900 268.252,35.985 266.921,36.212 266.665 C 36.689 266.126,44.000 257.146,48.476 251.600 C 50.163 249.510,52.938 246.090,54.642 244.000 C 56.346 241.910,59.188 238.400,60.957 236.200 C 62.727 234.000,64.720 231.554,65.387 230.766 C 66.054 229.977,67.230 228.544,68.000 227.582 C 68.770 226.620,70.030 225.084,70.800 224.169 C 72.598 222.031,72.902 221.657,76.488 217.159 C 80.083 212.649,84.269 208.039,87.000 205.583 C 89.287 203.526,93.815 200.000,94.171 200.000 C 94.302 200.000,95.442 199.306,96.705 198.458 C 104.936 192.929,115.800 188.887,127.000 187.186 C 128.374 186.977,163.866 186.795,210.000 186.759 C 254.330 186.725,291.590 186.585,292.800 186.449 L 295.000 186.200 294.998 168.000 C 294.996 151.483,294.928 149.578,294.261 147.400 C 292.306 141.010,286.872 135.615,280.453 133.690 C 278.392 133.071,272.726 132.996,213.800 132.807 L 149.400 132.600 147.110 131.675 C 137.334 127.726,134.003 120.968,133.998 105.071 C 133.993 90.369,130.472 83.988,120.348 80.330 L 117.800 79.410 81.800 79.351 C 62.000 79.319,44.990 79.435,44.000 79.609 M128.200 213.998 C 116.884 215.713,106.328 222.459,97.213 233.800 C 95.621 235.780,92.228 240.010,89.673 243.200 C 87.117 246.390,84.913 249.090,84.773 249.200 C 84.634 249.310,80.767 254.080,76.179 259.800 C 71.591 265.520,67.649 270.415,67.418 270.678 C 67.188 270.942,66.190 272.162,65.200 273.389 C 64.210 274.617,62.680 276.496,61.800 277.564 C 60.920 278.632,59.300 280.635,58.200 282.015 C 57.100 283.395,54.850 286.164,53.200 288.167 C 51.550 290.171,49.660 292.495,49.000 293.332 C 48.340 294.169,47.260 295.538,46.600 296.374 C 45.940 297.210,44.044 299.583,42.386 301.647 C 35.345 310.414,34.021 316.276,38.536 318.694 C 42.330 320.726,33.557 320.581,159.845 320.699 C 287.753 320.819,277.958 320.960,286.000 318.886 C 290.855 317.633,299.300 312.956,303.703 309.081 C 306.512 306.609,308.372 304.449,317.347 293.233 C 318.858 291.345,320.568 289.232,321.147 288.538 C 321.726 287.845,323.100 286.160,324.200 284.794 C 325.300 283.429,326.794 281.624,327.520 280.785 C 328.246 279.945,329.236 278.716,329.720 278.053 C 330.204 277.390,332.400 274.651,334.600 271.965 C 340.849 264.336,341.505 263.530,343.202 261.400 C 346.053 257.821,347.732 255.736,351.960 250.526 C 354.248 247.707,357.936 243.150,360.154 240.400 C 362.373 237.650,364.281 235.310,364.394 235.200 C 364.761 234.844,367.767 231.059,369.489 228.785 C 374.273 222.468,374.239 217.220,369.400 214.800 L 367.400 213.800 248.800 213.748 C 183.570 213.719,129.300 213.832,128.200 213.998 M399.600 225.751 C 399.600 231.796,394.623 240.665,383.715 254.059 C 380.427 258.097,376.789 262.570,375.632 264.000 C 371.130 269.561,362.517 280.153,360.892 282.126 C 359.952 283.267,358.422 285.160,357.492 286.332 C 356.561 287.505,355.260 289.144,354.600 289.975 C 353.940 290.806,351.510 293.776,349.200 296.575 C 346.890 299.375,344.460 302.356,343.800 303.201 C 342.127 305.341,340.489 307.360,338.486 309.753 C 337.543 310.879,336.634 311.980,336.467 312.200 C 336.299 312.420,335.001 314.015,333.581 315.744 C 332.162 317.472,330.648 319.362,330.218 319.944 C 321.061 332.312,304.672 342.065,285.600 346.495 L 281.800 347.378 163.000 347.498 C 32.175 347.630,40.175 347.773,32.351 345.172 C 16.471 339.895,3.810 325.502,0.820 309.326 C 0.591 308.085,0.312 306.979,0.202 306.868 C 0.091 306.757,-0.000 327.667,-0.000 353.333 L 0.000 400.000 200.000 400.000 L 400.000 400.000 400.000 312.400 C 400.000 264.220,399.910 224.800,399.800 224.800 C 399.690 224.800,399.600 225.228,399.600 225.751 " stroke="none" fill="#fbfbfb" fill-rule="evenodd"></path></g></svg>`
    openFileLocationButton.addEventListener("click", () => {
        shell.showItemInFolder(src)
    })

    if (fs.existsSync(`${src}.json`)) {
        const editButton = createElem("div", {title: window.i18n.ADJUST_SAMPLE_IN_EDITOR})
        editButton.innerHTML = `<svg class="renameSVG" version="1.0" xmlns="http:\/\/www.w3.org/2000/svg" width="344.000000pt" height="344.000000pt" viewBox="0 0 344.000000 344.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,344.000000) scale(0.100000,-0.100000)" fill="#555555" stroke="none"><path d="M1489 2353 l-936 -938 -197 -623 c-109 -343 -195 -626 -192 -629 2 -3 284 84 626 193 l621 198 937 938 c889 891 937 940 934 971 -11 108 -86 289 -167 403 -157 219 -395 371 -655 418 l-34 6 -937 -937z m1103 671 c135 -45 253 -135 337 -257 41 -61 96 -178 112 -241 l12 -48 -129 -129 -129 -129 -287 287 -288 288 127 127 c79 79 135 128 148 128 11 0 55 -12 97 -26z m-1798 -1783 c174 -79 354 -248 436 -409 59 -116 72 -104 -213 -196 l-248 -80 -104 104 c-58 58 -105 109 -105 115 0 23 154 495 162 495 5 0 37 -13 72 -29z"/></g></svg>`
        editButton.addEventListener("click", () => {
            let editData = fs.readFileSync(`${src}.json`, "utf8")
            editData = JSON.parse(editData)

            generateVoiceButton.dataset.modelIDLoaded = editData.pitchEditor ? editData.pitchEditor.currentVoice : editData.currentVoice

            window.sequenceEditor.inputSequence = editData.inputSequence
            window.sequenceEditor.pacing = editData.pacing
            window.sequenceEditor.letters = editData.pitchEditor ? editData.pitchEditor.letters : editData.letters
            window.sequenceEditor.currentVoice = editData.pitchEditor ? editData.pitchEditor.currentVoice : editData.currentVoice
            window.sequenceEditor.resetEnergy = (editData.pitchEditor && editData.pitchEditor.resetEnergy) ? editData.pitchEditor.resetEnergy : editData.resetEnergy
            window.sequenceEditor.resetPitch = editData.pitchEditor ? editData.pitchEditor.resetPitch : editData.resetPitch
            window.sequenceEditor.resetDurs = editData.pitchEditor ? editData.pitchEditor.resetDurs : editData.resetDurs
            window.sequenceEditor.letterFocus = []
            window.sequenceEditor.ampFlatCounter = 0
            window.sequenceEditor.hasChanged = false
            window.sequenceEditor.sequence = editData.pitchEditor ? editData.pitchEditor.sequence : editData.sequence
            window.sequenceEditor.energyNew = (editData.pitchEditor && editData.pitchEditor.energyNew) ? editData.pitchEditor.energyNew : editData.energyNew
            window.sequenceEditor.pitchNew = editData.pitchEditor ? editData.pitchEditor.pitchNew : editData.pitchNew
            window.sequenceEditor.dursNew = editData.pitchEditor ? editData.pitchEditor.dursNew : editData.dursNew


            window.sequenceEditor.init()
            window.sequenceEditor.update()
            window.sequenceEditor.autoInferTimer = null

            dialogueInput.value = editData.inputSequence
            paceNumbInput.value = editData.pacing
            pace_slid.value = editData.pacing

            window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
            window.sequenceEditor.update()

            if (!window.wavesurfer) {
                window.initWaveSurfer(src)
            } else {
                window.wavesurfer.load(src)
            }

            samplePlayPause.style.display = "block"
        })
        audioControls.appendChild(editButton)
    }

    const renameButton = createElem("div", {title: window.i18n.RENAME_THE_FILE})
    renameButton.innerHTML = `<svg class="renameSVG" version="1.0" xmlns="http://www.w3.org/2000/svg" width="166.000000pt" height="336.000000pt" viewBox="0 0 166.000000 336.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,336.000000) scale(0.100000,-0.100000)" fill="#000000" stroke="none"> <path d="M165 3175 c-30 -31 -35 -42 -35 -84 0 -34 6 -56 21 -75 42 -53 58 -56 324 -56 l245 0 0 -1290 0 -1290 -245 0 c-266 0 -282 -3 -324 -56 -15 -19 -21 -41 -21 -75 0 -42 5 -53 35 -84 l36 -35 281 0 280 0 41 40 c30 30 42 38 48 28 5 -7 9 -16 9 -21 0 -4 15 -16 33 -27 30 -19 51 -20 319 -20 l287 0 36 35 c30 31 35 42 35 84 0 34 -6 56 -21 75 -42 53 -58 56 -324 56 l-245 0 0 1290 0 1290 245 0 c266 0 282 3 324 56 15 19 21 41 21 75 0 42 -5 53 -35 84 l-36 35 -287 0 c-268 0 -289 -1 -319 -20 -18 -11 -33 -23 -33 -27 0 -5 -4 -14 -9 -21 -6 -10 -18 -2 -48 28 l-41 40 -280 0 -281 0 -36 -35z"/></g></svg>`

    renameButton.addEventListener("click", () => {
        createModal("prompt", {
            prompt: window.i18n.ENTER_NEW_FILENAME_UNCHANGED_CANCEL,
            value: sample.querySelector("div").innerHTML
        }).then(newFileName => {
            if (newFileName!=fileName) {
                const oldPath = src.split("/").reverse()
                const newPath = src.split("/").reverse()
                oldPath[0] = sample.querySelector("div").innerHTML
                newPath[0] = newFileName

                const oldPathComposed = oldPath.reverse().join("/")
                const newPathComposed = newPath.reverse().join("/")
                fs.renameSync(oldPathComposed, newPathComposed)

                if (fs.existsSync(`${oldPathComposed}.json`)) {
                    fs.renameSync(oldPathComposed+".json", newPathComposed+".json")
                }
                if (fs.existsSync(`${oldPathComposed.replace(/\.wav$/, "")}.lip`)) {
                    fs.renameSync(oldPathComposed.replace(/\.wav$/, "")+".lip", newPathComposed.replace(/\.wav$/, "")+".lip")
                }
                if (fs.existsSync(`${oldPathComposed.replace(/\.wav$/, "")}.fuz`)) {
                    fs.renameSync(oldPathComposed.replace(/\.wav$/, "")+".fuz", newPathComposed.replace(/\.wav$/, "")+".fuz")
                }

                oldPath.reverse()
                oldPath.splice(0,1)
                refreshRecordsList(oldPath.reverse().join("/"))
            }
        })
    })

    const editInProgramButton = createElem("div", {title: window.i18n.EDIT_IN_EXTERNAL_PROGRAM})
    editInProgramButton.innerHTML = `<svg class="renameSVG" version="1.0" width="175.000000pt" height="240.000000pt" viewBox="0 0 175.000000 240.000000"  preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,240.000000) scale(0.100000,-0.100000)" fill="#000000" stroke="none"><path d="M615 2265 l-129 -125 -68 0 c-95 0 -98 -4 -98 -150 0 -146 3 -150 98 -150 l68 0 129 -125 c128 -123 165 -145 179 -109 8 20 8 748 0 768 -14 36 -51 14 -179 -109z"/> <path d="M1016 2344 c-22 -21 -20 -30 10 -51 66 -45 126 -109 151 -162 22 -47 27 -69 27 -141 0 -72 -5 -94 -27 -141 -25 -53 -85 -117 -151 -162 -30 -20 -33 -39 -11 -57 22 -18 64 3 132 64 192 173 164 491 -54 636 -54 35 -56 35 -77 14z"/> <path d="M926 2235 c-8 -22 1 -37 46 -70 73 -53 104 -149 78 -241 -13 -44 -50 -92 -108 -136 -26 -21 -27 -31 -6 -52 37 -38 150 68 179 167 27 91 13 181 -41 259 -49 70 -133 112 -148 73z"/> <path d="M834 2115 c-9 -23 2 -42 33 -57 53 -25 56 -108 4 -134 -35 -18 -44 -30 -36 -53 8 -25 34 -27 76 -6 92 48 92 202 0 250 -38 19 -70 19 -77 0z"/> <path d="M1381 1853 c-33 -47 -182 -253 -264 -364 -100 -137 -187 -262 -187 -270 0 -8 140 -204 177 -249 5 -6 41 41 109 141 30 45 60 86 65 93 48 54 197 276 226 336 33 68 37 83 37 160 1 71 -3 93 -23 130 -53 101 -82 106 -140 23z"/> <path d="M211 1861 c-56 -60 -68 -184 -27 -283 15 -38 106 -168 260 -371 130 -173 236 -320 236 -328 0 -8 -9 -25 -20 -39 -11 -14 -20 -29 -20 -33 0 -5 -10 -23 -23 -40 -12 -18 -27 -41 -33 -52 -13 -24 -65 -114 -80 -138 -10 -17 -13 -16 -60 7 -98 49 -209 43 -305 -17 -83 -51 -129 -141 -129 -251 0 -161 115 -283 275 -294 101 -6 173 22 243 96 56 58 79 97 133 227 46 112 101 203 164 274 l53 60 42 -45 c27 -29 69 -103 124 -217 86 -176 133 -250 197 -306 157 -136 405 -73 478 123 37 101 21 202 -46 290 -91 118 -275 147 -402 63 -30 -20 -42 -23 -49 -14 -5 7 -48 82 -96 167 -47 85 -123 202 -168 260 -45 58 -111 143 -146 190 -85 110 -251 326 -321 416 -31 40 -65 84 -76 100 -11 15 -35 46 -54 68 -19 23 -45 58 -59 79 -30 45 -54 47 -91 8z m653 -943 c20 -28 20 -33 0 -52 -42 -43 -109 10 -69 54 24 26 50 25 69 -2z m653 -434 c49 -20 87 -85 87 -149 -2 -135 -144 -209 -257 -134 -124 82 -89 265 58 299 33 8 64 4 112 -16z m-1126 -20 c47 -24 73 -71 77 -139 3 -50 0 -65 -20 -94 -34 -50 -71 -73 -125 -78 -99 -9 -173 53 -181 152 -11 135 126 223 249 159z"/></g></svg>`
    editInProgramButton.addEventListener("click", () => {

        if (window.userSettings.externalAudioEditor && window.userSettings.externalAudioEditor.length) {
            const fileName = audio.children[0].src.split("file:///")[1].split("%20").join(" ")
            const sp = spawn(window.userSettings.externalAudioEditor, [fileName], {'detached': true}, (err, data) => {
                if (err) {
                    console.log(err)
                    console.log(err.message)
                    window.errorModal(err.message)
                }
            })

            sp.on("error", err => {
                if (err.message.includes("ENOENT")) {
                    window.errorModal(`${window.i18n.FOLLOWING_PATH_NOT_VALID}:<br><br> ${window.userSettings.externalAudioEditor}`)
                } else {
                    window.errorModal(err.message)
                }
            })

        } else {
            window.errorModal(window.i18n.SPECIFY_EDIT_TOOL)
        }
    })


    const deleteFileButton = createElem("div", {title: window.i18n.DELETE_FILE})
    deleteFileButton.innerHTML = "&#10060;"
    deleteFileButton.addEventListener("click", () => {
        confirmModal(`${window.i18n.SURE_DELETE}<br><br><i>${fileName}</i>`).then(confirmation => {
            if (confirmation) {
                window.appLogger.log(`${newSample?window.i18n.DELETING_NEW_FILE:window.i18n.DELETING}: ${src}`)
                if (fs.existsSync(src)) {
                    fs.unlinkSync(src)
                }
                sample.remove()
                if (fs.existsSync(`${src}.json`)) {
                    fs.unlinkSync(`${src}.json`)
                }
            }
        })
    })
    audioControls.appendChild(renameButton)
    audioControls.appendChild(audio)
    audioControls.appendChild(editInProgramButton)
    audioControls.appendChild(openFileLocationButton)
    audioControls.appendChild(deleteFileButton)
    sample.appendChild(audioControls)
    return sample
}


generateVoiceButton.addEventListener("click", () => {

    const game = window.currentGame.gameId

    try {fs.mkdirSync(window.userSettings[`outpath_${game}`])} catch (e) {/*Do nothing*/}
    try {fs.mkdirSync(`${window.userSettings[`outpath_${game}`]}/${voiceId}`)} catch (e) {/*Do nothing*/}


    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {

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
            if (window.currentModel.modelType.toLowerCase()=="xvapitch") {
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
                closeModal().then(() => dialogueInput.focus())
            }
        }).catch(e => {
            console.log(e)
            if (e.code =="ENOENT") {
                closeModal(null, modalContainer).then(() => {
                    window.errorModal(window.i18n.ERR_SERVER)
                })
            }
        })
    } else {

        if (window.isGenerating) {
            return
        }
        clearOldTempFiles()

        dialogueInput.value = " "+dialogueInput.value.trim()+" "
        let sequence = dialogueInput.value.replace("…", "...").replace("’", "'")
        if (sequence.length==0) {
            return
        }
        window.isGenerating = true

        window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["generate-voice"]["pre"], event="pre generate-voice")

        if (window.wavesurfer) {
            window.wavesurfer.stop()
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
        if (!window.arpabetMenuState.hasChangedARPAbet && !window.styleEmbsMenuState.hasChangedEmb &&
            (speech2speechState.s2s_autogenerate || (editorContainer.innerHTML && editorContainer.innerHTML.length && (window.userSettings.keepEditorOnVoiceChange || generateVoiceButton.dataset.modelIDLoaded==window.sequenceEditor.currentVoice)))) {

            speech2speechState.s2s_autogenerate = false
            pitch = window.sequenceEditor.pitchNew.map(v=> v==undefined?0:v)
            duration = window.sequenceEditor.dursNew.map(v => v*pace_slid.value).map(v=> v==undefined?0:v)
            energy = window.sequenceEditor.energyNew ? window.sequenceEditor.energyNew.map(v => v==undefined?0:v).filter(v => !isNaN(v)) : []
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
                sequence, pitch, duration, energy, speaker_i, pace,
                base_lang: base_lang_select.value,
                base_emb: style_emb_select.value||"",
                modelType: window.currentModel.modelType,
                old_sequence, // For partial re-generation
                useSR: useSRCkbx.checked,
                outfile: tempFileLocation,
                pluginsContext: JSON.stringify(window.pluginsContext),
                vocoder: window.userSettings.vocoder,
                waveglowPath: vocoder_select.value=="256_waveglow" ? window.userSettings.waveglow_path : window.userSettings.bigwaveglow_path
            })
        }).then(r=>r.text()).then(res => {

            if (res=="ENOENT" || res.startsWith("ERR:")) {
                if (res.startsWith("ERR:")) {
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${res.replace("ERR:","").replaceAll(/\n/g, "<br>")}`)
                } else {
                    window.errorModal(`${window.i18n.BATCH_MODEL_NOT_FOUND}.${vocoder_select.value.includes("waveglow")?" "+window.i18n.BATCH_DOWNLOAD_WAVEGLOW:""}`)
                }
                toggleSpinnerButtons()
                return
            }

            dialogueInput.focus()

            if (window.userSettings.clear_text_after_synth) {
                dialogueInput.value = ""
            }

            window.isGenerating = false
            res = res.split("\n")
            let pitchData = res[0]
            let durationsData = res[1]
            let energyData = res[2]
            let cleanedSequence = res[3].split("|").map(c=>c.replaceAll("{", "").replaceAll("}", "").replace(/\s/g, "_"))
            const start_index = res[4]
            const end_index = res[5]
            pitchData = pitchData.split(",").map(v => parseFloat(v))

            // For use in adjusting editor range
            const maxPitchVal = pitchData.reduce((p,c)=>Math.max(p, Math.abs(c)), 0)
            if (maxPitchVal>window.sequenceEditor.default_pitchSliderRange) {
                window.sequenceEditor.pitchSliderRange = maxPitchVal
            } else {
                window.sequenceEditor.pitchSliderRange = window.sequenceEditor.default_pitchSliderRange
            }

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

                window.sequenceEditor.inputSequence = sequence
                window.sequenceEditor.sequence = cleanedSequence

                if (pitch.length==0 || isFreshRegen) {
                    window.sequenceEditor.resetPitch = pitchData
                    window.sequenceEditor.resetDurs = durationsData
                    window.sequenceEditor.resetEnergy = energyData
                }

                window.sequenceEditor.letters = cleanedSequence
                window.sequenceEditor.pitchNew = pitchData.map(p=>p)
                window.sequenceEditor.dursNew = durationsData.map(v=>v)
                window.sequenceEditor.energyNew = energyData.map(v=>v)
                window.sequenceEditor.init()
                window.sequenceEditor.update(window.currentModel.modelType)

                window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
                window.sequenceEditor.autoInferTimer = null
                window.sequenceEditor.hasChanged = false


                toggleSpinnerButtons()
                if (keepSampleButton.dataset.newFileLocation && keepSampleButton.dataset.newFileLocation.startsWith("BATCH_EDIT")) {
                    console.log("_debug_")
                } else {
                    keepSampleButton.dataset.newFileLocation = `${window.userSettings[`outpath_${game}`]}/${voiceType}/${outputFileName}.wav`
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


            if (window.userSettings.audio.ffmpeg && setting_audio_ffmpeg_preview.checked) {
                const options = {
                    hz: window.userSettings.audio.hz,
                    padStart: window.userSettings.audio.padStart,
                    padEnd: window.userSettings.audio.padEnd,
                    bit_depth: window.userSettings.audio.bitdepth,
                    amplitude: window.userSettings.audio.amplitude,
                    pitchMult: window.userSettings.audio.pitchMult,
                    tempo: window.userSettings.audio.tempo,
                    nr: window.userSettings.audio.nr,
                    nf: window.userSettings.audio.nf,
                    useNR: window.userSettings.audio.useNR,
                    useSR: useSRCkbx.checked
                }

                const extraInfo = {
                    game: window.currentGame.gameId,
                    voiceId: window.currentModel.voiceId,
                    voiceName: window.currentModel.voiceName,
                    inputSequence: sequence,
                    letters: cleanedSequence,
                    pitch: pitchData.map(p=>p),
                    energy: energyData.map(p=>p),
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
})


window.refreshRecordsList = (directory) => {

    if (!fs.existsSync(directory)) {
        return
    }

    const records = []
    const reverse = window.userSettings.voiceRecordsOrderByOrder=="ascending"
    const sortBy = window.userSettings.voiceRecordsOrderBy//"name"

    const files = fs.readdirSync(directory)
    files.forEach(file => {
        if (file.endsWith(".json") || file.endsWith(".lip") || file.endsWith(".fuz")) {
            return
        }
        const record = {}

        if (!file.toLowerCase().includes(voiceSamplesSearch.value.toLowerCase().trim())) {
            return
        }

        if (voiceSamplesSearchPrompt.value.length) {
            if (fs.existsSync(`${directory}/${file}.json`)) {
                try {
                    const lineMeta = fs.readFileSync(`${directory}/${file}.json`, "utf8")

                    if (!JSON.parse(lineMeta).inputSequence.toLowerCase().includes(voiceSamplesSearchPrompt.value.toLowerCase().trim())) {
                        return
                    }
                } catch (e) {
                    // console.log(e)
                }
            }
        }

        record.fileName = file
        record.lastChanged = fs.statSync(`${directory}/${file}`).mtime
        record.jsonPath = `${directory}/${file}`
        records.push(record)
    })

    voiceSamples.innerHTML = ""
    records.sort((a,b) => {
        if (sortBy=="name") {
            return a.fileName.toLowerCase()<b.fileName.toLowerCase() ? (reverse?-1:1) : (reverse?1:-1)
        } else if (sortBy=="time") {
            return a.lastChanged<b.lastChanged ? (reverse?-1:1) : (reverse?1:-1)
        } else {
            console.warn("sort by type not recognised", sortBy)
        }
    }).forEach(record => {
        voiceSamples.appendChild(window.makeSample(record.jsonPath))
    })
}

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
        ampFlatCounter: window.sequenceEditor.ampFlatCounter,
        inputSequence: window.sequenceEditor.inputSequence,
        sequence: window.sequenceEditor.sequence,
        pitchNew: window.sequenceEditor.pitchNew,
        energyNew: window.sequenceEditor.energyNew,
        dursNew: window.sequenceEditor.dursNew,
    }

    let outputFileName = to.split("/").reverse()[0].split(".").reverse().slice(1, 1000)
    const toExt = to.split(".").reverse()[0]

    if (window.userSettings.filenameNumericalSeq) {
        outputFileName = outputFileName[0]+"."+outputFileName.slice(1,1000).reverse().join(".")
    } else {
        outputFileName = outputFileName.reverse().join(".")
    }
    to = `${to.split("/").reverse().slice(1,10000).reverse().join("/")}/${outputFileName}`


    if (!setting_audio_ffmpeg_preview.checked && window.userSettings.audio.ffmpeg) {
        spinnerModal(window.i18n.SAVING_AUDIO_FILE)

        window.appLogger.log(`${window.i18n.ABOUT_TO_SAVE_FROM_N1_TO_N2_WITH_OPTIONS}: ${JSON.stringify(options)}`.replace("_1", from).replace("_2", to))

        doFetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body: JSON.stringify({
                input_path: from,
                output_path: to,
                pluginsContext: JSON.stringify(window.pluginsContext),
                isBatchMode: false,
                extraInfo: JSON.stringify(pluginData),
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            closeModal(undefined, undefined, true).then(() => {
                if (res.length && res!="-") {
                    console.log("res", res)
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${window.i18n.INPUT}: ${from}<br>${window.i18n.OUTPUT}: ${to}<br><br>${res}`)
                } else {
                    if (window.userSettings.outputJSON) {
                        fs.writeFileSync(`${to}.${toExt}.json`, JSON.stringify(jsonDataOut, null, 4))
                    }
                    if (!skipUIRecord) {
                        refreshRecordsList(containerFolderPath)
                    }
                    dialogueInput.focus()
                    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
                }
            })
        }).catch(res => {
            window.appLogger.log(`Error in saveFile->outputAudio[ffmpeg]: ${res}`)
            closeModal(undefined, undefined, true).then(() => {
                window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${window.i18n.INPUT}: ${from}<br>${window.i18n.OUTPUT}: ${to}<br><br>${res}`)
            })
        })
    } else {

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
                    if (window.userSettings.outputJSON) {
                        fs.writeFileSync(`${to}.${toExt}.json`, JSON.stringify(jsonDataOut, null, 4))
                    }
                    if (!skipUIRecord) {
                        refreshRecordsList(containerFolderPath)
                    }
                    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
                }
            })
        })
    }
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

if (Object.keys(window.userSettings).includes("voiceRecordsOrderBy")) {
    const labels = {
        "name": window.i18n.NAME,
        "time": window.i18n.TIME
    }
    voiceRecordsOrderByButton.innerHTML = labels[window.userSettings.voiceRecordsOrderBy]
} else {
    window.userSettings.voiceRecordsOrderBy = "name"
    saveUserSettings()
}
if (Object.keys(window.userSettings).includes("voiceRecordsOrderByOrder")) {
    const labels = {
        "ascending": window.i18n.ASCENDING,
        "descending": window.i18n.DESCENDING
    }
    voiceRecordsOrderByOrderButton.innerHTML = labels[window.userSettings.voiceRecordsOrderByOrder]
} else {
    window.userSettings.voiceRecordsOrderByOrder = "ascending"
    saveUserSettings()
}
// =========


// Delete all output files for a voice
voiceRecordsDeleteAllButton.addEventListener("click", () => {
    if (window.currentModel) {
        const outDir = window.userSettings[`outpath_${window.currentGame.gameId}`]+`/${currentModel.voiceId}`

        const files = fs.readdirSync(outDir)
        if (files.length) {
            window.confirmModal(window.i18n.DELETE_ALL_FILES_CONFIRM.replace("_1", files.length).replace("_2", outDir)).then(resp => {
                if (resp) {
                    window.deleteFolderRecursive(outDir, true)
                    refreshRecordsList(outDir)
                }
            })
        } else {
            window.errorModal(window.i18n.DELETE_ALL_FILES_ERR_NO_FILES.replace("_1", outDir))
        }
    }
})

voiceRecordsOrderByButton.addEventListener("click", () => {
    window.userSettings.voiceRecordsOrderBy = window.userSettings.voiceRecordsOrderBy=="name" ? "time" : "name"
    saveUserSettings()
    const labels = {
        "name": window.i18n.NAME,
        "time": window.i18n.TIME
    }
    voiceRecordsOrderByButton.innerHTML = labels[window.userSettings.voiceRecordsOrderBy]
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame.gameId}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})
voiceRecordsOrderByOrderButton.addEventListener("click", () => {
    window.userSettings.voiceRecordsOrderByOrder = window.userSettings.voiceRecordsOrderByOrder=="ascending" ? "descending" : "ascending"
    saveUserSettings()
    const labels = {
        "ascending": window.i18n.ASCENDING,
        "descending": window.i18n.DESCENDING
    }
    voiceRecordsOrderByOrderButton.innerHTML = labels[window.userSettings.voiceRecordsOrderByOrder]
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame.gameId}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})
voiceSamplesSearch.addEventListener("keyup", () => {
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame.gameId}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})
voiceSamplesSearchPrompt.addEventListener("keyup", () => {
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame.gameId}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})








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


// Info
// ====
window.setupModal(infoIcon, infoContainer)


// Patreon
// =======
window.setupModal(patreonIcon, patreonContainer, () => {
    const data = fs.readFileSync(`${path}/patreon.txt`, "utf8")
    const names = new Set()
    data.split("\r\n").forEach(name => names.add(name))
    names.add("minermanb")

    let content = ``
    creditsList.innerHTML = ""
    names.forEach(name => content += `<br>${name}`)
    creditsList.innerHTML = content
})
// Steam does not like patreon
// patreonButton.addEventListener("click", () => {
    // shell.openExternal("https://patreon.com")
// })
doFetch("http://danruta.co.uk/patreon.txt").then(r=>r.text()).then(data => fs.writeFileSync(`${path}/patreon.txt`, data, "utf8")).catch(e => {})


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

window.updateGameList = () => {
    gameSelectionListContainer.innerHTML = ""
    const fileNames = fs.readdirSync(`${window.path}/assets`)

    let totalVoices = 0
    let totalGames = new Set()

    const itemsToSort = []

    fileNames.filter(fn=>fn.endsWith(".json")).forEach(gameId => {

        const metadata = JSON.parse(fs.readFileSync(`${window.path}/assets/${gameId}`))
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
                        loadAllModels().then(() => changeGame(metadata))
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

    loadAllModels().then(() => {
        // Load the last selected game
        const lastGame = localStorage.getItem("lastGame")

        if (lastGame) {
            changeGame(JSON.parse(lastGame))
        }
    })
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

// EULA
EULA_closeButon.addEventListener("click", () => {
    if (EULA_accept_ckbx.checked) {
        closeModal(EULAContainer)
        window.userSettings.EULA_accepted = true
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
if (!Object.keys(window.userSettings).includes("EULA_accepted") || !window.userSettings.EULA_accepted) {
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


// TEMP - pre-alpha builds
setInterval(() => {
    dragBar.innerHTML = "xVASynth PRE-ALPHA v3 PREVIEW"
    dragBar.style.backgroundColor = "red"
}, 1000)
"use strict"

window.appVersion = "v2.0.0"

window.PRODUCTION = process.mainModule.filename.includes("resources")
const path = window.PRODUCTION ? "./resources/app" : "."
window.path = path

const fs = require("fs")
const zipdir = require('zip-dir')
const {shell, ipcRenderer} = require("electron")
const fetch = require("node-fetch")
const {xVAAppLogger} = require("./javascript/appLogger.js")
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
process.on(`uncaughtException`, data => window.appLogger.log(data))
window.onerror = (err, url, lineNum) => window.appLogger.log(err)
require("./javascript/i18n.js")
require("./javascript/util.js")
require("./javascript/nexus.js")
require("./javascript/embeddings.js")
const {Editor} = require("./javascript/editor.js")
const {saveUserSettings, deleteFolderRecursive} = require("./javascript/settingsMenu.js")
const xVASpeech = require("./javascript/xVASpeech.js")
const {startBatch} = require("./javascript/batch.js")
window.electronBrowserWindow = require("electron").remote.getCurrentWindow()
const child = require("child_process").execFile
const spawn = require("child_process").spawn


// Start the server
if (window.PRODUCTION) {
    window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
}

const {PluginsManager} = require("./javascript/plugins_manager.js")
window.pluginsManager = new PluginsManager(window.path, window.appLogger, window.appVersion)
window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["pre"], event="pre start")

let themeColour
let secondaryThemeColour
const oldCError = console.error
console.error = (data) => {
    window.appLogger.log(data)
    oldCError(arguments)
}

window.addEventListener("error", function (e) {window.appLogger.log(e.error.stack)})
window.addEventListener('unhandledrejection', function (e) {window.appLogger.log(e.reason.stack)})

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
fs.readdir(`${__dirname.replace("/javascript", "")}/output`, (err, files) => {
    if (err) {
        window.appLogger.log(err)
    }
    if (files && files.length) {
        files.filter(f => f.startsWith("temp-")).forEach(file => {
            fs.unlink(`${__dirname.replace("/javascript", "")}/output/${file}`, err => err&&console.log(err))
        })
    }
})

let fileRenameCounter = 0
let fileChangeCounter = 0
let isGenerating = false

window.loadAllModels = () => {
    return new Promise(resolve => {

        let gameFolder
        let modelPathsKeys = Object.keys(window.userSettings).filter(key => key.includes("modelspath_"))
        window.games = {}

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
                        model.games.forEach(({gameId, voiceId, voiceName, voiceDescription, gender}) => {

                            if (!window.games.hasOwnProperty(gameId)) {

                                const gameAsset = fs.readdirSync(`${path}/assets`).find(f => f.startsWith(gameId))
                                const option = document.createElement("option")
                                option.value = gameAsset
                                option.innerHTML = gameAsset.split("-").reverse()[0].split(".")[0]
                                window.games[gameId] = {
                                    models: [],
                                    gameAsset
                                }
                            }

                            const audioPreviewPath = `${modelsPath}/${model.games.find(({gameId}) => gameId==gameFolder).voiceId}`
                            const existingDuplicates = []
                            window.games[gameId].models.forEach((item,i) => {
                                if (item.voiceId==voiceId) {
                                    existingDuplicates.push([item, i])
                                }
                            })

                            const modelData = {model, modelsPath, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, gender, modelVersion: model.modelVersion, hifi: undefined, xvaspeech: undefined}
                            const potentialHiFiPath = `${modelsPath}/${voiceId}.hg.pt`
                            if (fs.existsSync(potentialHiFiPath)) {
                                modelData.hifi = potentialHiFiPath
                            }
                            const potentialxVASpeechPath = `${modelsPath}/${voiceId}.xvaspeech.pt`
                            if (fs.existsSync(potentialxVASpeechPath)) {
                                modelData.xvaspeech = potentialxVASpeechPath
                            }

                            if (existingDuplicates.length) {
                                if (existingDuplicates[0][0].modelVersion<model.modelVersion) {
                                    window.games[gameId].models.splice(existingDuplicates[0][1], 1)
                                    window.games[gameId].models.push(modelData)
                                }
                            } else {
                                window.games[gameId].models.push(modelData)
                            }
                        })
                    } catch (e) {
                        console.log(e)
                        window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME_WITH_FILENAME.replace("_1", gameFolder)} `+fileName)
                        window.appLogger.log(e)
                        window.appLogger.log(e.stack)
                    }
                })
            } catch (e) {
                // console.log(e)
                window.appLogger.log(`${window.i18n.ERR_LOADING_MODELS_FOR_GAME}: `+ gameFolder)
                window.appLogger.log(e)
            }

            resolve()
        })
    })
}
setting_models_path_input.addEventListener("change", () => {
    const gameFolder = window.currentGame[0]

    setting_models_path_input.value = setting_models_path_input.value.replace(/\/\//g, "/").replace(/\\/g,"/")
    window.userSettings[`modelspath_${gameFolder}`] = setting_models_path_input.value
    saveUserSettings()
    loadAllModels().then(() => {
        changeGame(window.currentGame.join("-"))
    })

    if (!window.watchedModelsDirs.includes(setting_models_path_input.value)) {
        window.watchedModelsDirs.push(setting_models_path_input.value)
        fs.watch(setting_models_path_input.value, {recursive: false, persistent: true}, (eventType, filename) => {
            changeGame(window.currentGame.join("-"))
        })
    }
    window.updateGameList()
})

// Change game
window.changeGame = (meta) => {

    meta = meta.split("-")
    window.currentGame = meta
    themeColour = meta[1]
    let titleID
    if (meta.length==5) {
        secondaryThemeColour = meta[2]
        titleID = meta[3]
    } else {
        secondaryThemeColour = undefined
        titleID = meta[2]
    }
    generateVoiceButton.disabled = true
    generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
    selectedGameDisplay.innerHTML = meta.length==5 ? meta[4].split(".")[0] : meta[3].split(".")[0]

    // Change the app title
    title.innerHTML = window.i18n.SELECT_VOICE_TYPE
    if (window.games[window.currentGame[0]] == undefined) {
        title.innerHTML = `${window.i18n.NO_MODELS_IN}: ${window.userSettings[`modelspath_${window.currentGame[0]}`]}`
        console.log(title.innerHTML)
    } else if (titleID) {
        document.title = `${titleID}VA Synth`
        dragBar.innerHTML = `${titleID}VA Synth`
    } else {
        document.title = `xVA Synth`
        dragBar.innerHTML = `xVA Synth`
    }

    const gameFolder = meta[0]
    const gameName = meta[meta.length-1].split(".")[0]

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

    try {fs.mkdirSync(`${path}/output/${meta[0]}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", window.currentGame.join("-"))

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""

    // No models found
    if (!Object.keys(window.games).length) {
        title.innerHTML = window.i18n.NO_MODELS_FOUND
        return
    }

    const buttons = []

    voiceSearchInput.placeholder = window.i18n.SEARCH_N_VOICES.replace("_", window.games[meta[0]] ? window.games[meta[0]].models.length : "0")
    voiceSearchInput.value = ""

    if (!window.games[meta[0]]) {
        return
    }

    window.games[meta[0]].models.forEach(({model, modelsPath, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, hifi}) => {

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
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
                if (event.altKey) {
                    const files = fs.readdirSync(`./output`).filter(fname => fname.includes("temp-") && fname.includes(".wav"))
                    if (files.length) {
                        const options = {
                            hz: window.userSettings.audio.hz,
                            padStart: window.userSettings.audio.padStart,
                            padEnd: window.userSettings.audio.padEnd,
                            bit_depth: window.userSettings.audio.bitdepth,
                            amplitude: window.userSettings.audio.amplitude
                        }

                        fetch(`http://localhost:8008/outputAudio`, {
                            method: "Post",
                            body: JSON.stringify({
                                input_path: `./output/${files[0]}`,
                                isBatchMode: false,
                                output_path: `${modelsPath}/${voiceId}.wav`,
                                options: JSON.stringify(options)
                            })
                        }).then(r=>r.text()).then(console.log)
                    }

                } else {
                    fs.mkdirSync(`./build/${voiceId}`)
                    fs.mkdirSync(`./build/${voiceId}/resources`)
                    fs.mkdirSync(`./build/${voiceId}/resources/app`)
                    fs.mkdirSync(`./build/${voiceId}/resources/app/models`)
                    fs.mkdirSync(`./build/${voiceId}/resources/app/models/${gameId}`)
                    fs.copyFileSync(`${modelsPath}/${voiceId}.json`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.json`)
                    fs.copyFileSync(`${modelsPath}/${voiceId}.wav`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.wav`)
                    fs.copyFileSync(`${modelsPath}/${voiceId}.pt`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.pt`)
                    if (hifi) {
                        fs.copyFileSync(`${modelsPath}/${voiceId}.hg.pt`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.hg.pt`)
                    }
                    zipdir(`./build/${voiceId}`, {saveTo: `./build/${voiceId}.zip`}, (err, buffer) => deleteFolderRecursive(`./build/${voiceId}`))
                }
                return
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
                    changeVocoder("qnd")
                }
                // Remove the bespoke hifi option if there was one already there
                Array.from(vocoder_select.children).forEach(opt => {
                    if (opt.innerHTML=="Bespoke HiFi GAN") {
                        vocoder_select.removeChild(opt)
                    }
                })
            }

            const appVersionRequirement = model.version.toString().split(".").map(v=>parseInt(v))
            const appVersionInts = appVersion.replace("v", "").split(".").map(v=>parseInt(v))
            let appVersionOk = true
            if (appVersionRequirement[0] <= appVersionInts[0] ) {
                if (appVersionRequirement.length>1 && parseInt(appVersionRequirement[0]) == appVersionInts[0]) {
                    if (appVersionRequirement[1] <= appVersionInts[1] ) {
                        if (appVersionRequirement.length>2 && parseInt(appVersionRequirement[1]) == appVersionInts[1]) {
                            if (appVersionRequirement[2] <= appVersionInts[2] ) {
                            } else {
                                appVersionOk = false
                            }
                        }
                    } else {
                        appVersionOk = false
                    }
                }
            } else {
                appVersionOk = false
            }
            if (!appVersionOk) {
                window.errorModal(`${window.i18n.MODEL_REQUIRES_VERSION} v${model.version}<br><br>${window.i18n.THIS_APP_VERSION}: ${window.appVersion}`)
                return
            }

            window.currentModel = model
            window.currentModel.voiceId = voiceId
            window.currentModel.voiceName = button.innerHTML
            window.currentModel.hifi = hifi
            window.currentModel.audioPreviewPath = audioPreviewPath
            window.currentModelButton = button

            if (voiceDescription) {
                description.innerHTML = voiceDescription
                description.className = "withContent"
            } else {
                description.innerHTML = ""
                description.className = ""
            }

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
                    model_speakers: model.emb_size,
                    cmudict: model.cmudict
                })
                generateVoiceButton.dataset.modelIDToLoad = voiceId
            }
            generateVoiceButton.disabled = false

            title.innerHTML = button.innerHTML
            title.dataset.modelId = voiceId
            keepSampleButton.style.display = "none"
            samplePlay.style.display = "none"

            // Voice samples
            voiceSamples.innerHTML = ""
            refreshRecordsList(`${window.userSettings[`outpath_${meta[0]}`]}/${button.dataset.modelId}`)
        })
        buttons.push(button)
    })

    buttons.sort((a,b) => a.innerHTML<b.innerHTML?-1:1)
        .forEach(button => voiceTypeContainer.appendChild(button))

}


const makeSample = (src, newSample) => {
    const fileName = src.split("/").reverse()[0].split("%20").join(" ")
    const fileFormat = fileName.split(".").reverse()[0]
    const sample = createElem("div.sample", createElem("div", fileName))
    const audioControls = createElem("div")
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

            window.sequenceEditor.inputSequence = editData.inputSequence
            window.sequenceEditor.pacing = editData.pacing
            window.sequenceEditor.letters = editData.pitchEditor ? editData.pitchEditor.letters : editData.letters
            window.sequenceEditor.currentVoice = editData.pitchEditor ? editData.pitchEditor.currentVoice : editData.currentVoice
            window.sequenceEditor.resetPitch = editData.pitchEditor ? editData.pitchEditor.resetPitch : editData.resetPitch
            window.sequenceEditor.resetDurs = editData.pitchEditor ? editData.pitchEditor.resetDurs : editData.resetDurs
            window.sequenceEditor.letterFocus = []
            window.sequenceEditor.ampFlatCounter = 0
            window.sequenceEditor.hasChanged = false
            window.sequenceEditor.sequence = editData.pitchEditor ? editData.pitchEditor.sequence : editData.sequence
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

            if (samplePlay.style.display!="none") {
                samplePlay.removeChild(samplePlay.children[0])
                samplePlay.appendChild(createElem("audio", {controls: true}, createElem("source", {
                    src: src,
                    type: `audio/${fileFormat}`
                })))
                samplePlay.addEventListener("play", () => {
                    if (window.ctrlKeyIsPressed) {
                        samplePlay.setSinkId(window.userSettings.alt_speaker)
                    } else {
                        samplePlay.setSinkId(window.userSettings.base_speaker)
                    }
                })
                samplePlay.setSinkId(window.userSettings.base_speaker)
            }
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

    const game = window.currentGame[0]

    try {fs.mkdirSync(window.userSettings[`outpath_${game}`])} catch (e) {/*Do nothing*/}
    try {fs.mkdirSync(`${window.userSettings[`outpath_${game}`]}/${voiceId}`)} catch (e) {/*Do nothing*/}


    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {

        if (window.batch_state.state) {
            window.errorModal(window.i18n.BATCH_ERR_IN_PROGRESS)
            return
        }

        window.appLogger.log(`${window.i18n.LOADING_VOICE}: ${JSON.parse(generateVoiceButton.dataset.modelQuery).model}`)
        window.batch_state.lastModel = JSON.parse(generateVoiceButton.dataset.modelQuery).model.split("/").reverse()[0]

        spinnerModal(`${window.i18n.LOADING_VOICE}`)
        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad

            if (window.userSettings.defaultToHiFi && window.currentModel.hifi) {
                vocoder_select.value = Array.from(vocoder_select.children).find(opt => opt.innerHTML=="Bespoke HiFi GAN").value
                changeVocoder(vocoder_select.value).then(() => dialogueInput.focus())
            } else if (window.userSettings.vocoder.includes(".hg.pt")) {
                changeVocoder("qnd").then(() => dialogueInput.focus())
            } else {
                closeModal().then(() => dialogueInput.focus())
            }
        }).catch(e => {
            console.log(e)
            if (e.code =="ENOENT") {
                closeModal(null, modalContainer).then(() => {
                    createModal("error", window.i18n.ERR_SERVER)
                })
            }
        })
    } else {

        if (isGenerating) {
            return
        }

        let sequence = dialogueInput.value.trim().replace("…", "...")
        if (sequence.length==0) {
            return
        }
        isGenerating = true

        window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["generate-voice"]["pre"], event="pre generate-voice")
        sequence = dialogueInput.value.trim().replace("…", "...")

        const existingSample = samplePlay.querySelector("audio")
        if (existingSample) {
            existingSample.pause()
        }

        toggleSpinnerButtons()

        const voiceType = title.dataset.modelId
        const outputFileName = dialogueInput.value.slice(0, 260).replace(/\n/g, " ").replace(/[\/\\:\*?<>"|]*/g, "")

        try {fs.unlinkSync(localStorage.getItem("tempFileLocation"))} catch (e) {/*Do nothing*/}

        // For some reason, the samplePlay audio element does not update the source when the file name is the same
        const tempFileNum = `${Math.random().toString().split(".")[1]}`
        let tempFileLocation = `${path}/output/temp-${tempFileNum}.wav`
        let pitch = []
        let duration = []
        let isFreshRegen = true
        let old_sequence = undefined

        if (editorContainer.innerHTML && editorContainer.innerHTML.length && generateVoiceButton.dataset.modelIDLoaded==window.sequenceEditor.currentVoice) {
            if (window.sequenceEditor.audioInput || window.sequenceEditor.sequence && sequence!=window.sequenceEditor.inputSequence) {
                old_sequence = window.sequenceEditor.inputSequence
            }
        }

        if (editorContainer.innerHTML && editorContainer.innerHTML.length && (window.userSettings.keepEditorOnVoiceChange || generateVoiceButton.dataset.modelIDLoaded==window.sequenceEditor.currentVoice)) {
            pitch = window.sequenceEditor.pitchNew.map(v=> v==undefined?0:v)
            duration = window.sequenceEditor.dursNew.map(v => v*pace_slid.value).map(v=> v==undefined?0:v)
            isFreshRegen = false
        }
        window.sequenceEditor.currentVoice = generateVoiceButton.dataset.modelIDLoaded

        const speaker_i = window.currentModel.games[0].emb_i
        const pace = (window.userSettings.keepPaceOnNew && isFreshRegen)?pace_slid.value:1


        window.appLogger.log(`${window.i18n.SYNTHESIZING}: ${sequence}`)

        fetch(`http://localhost:8008/synthesize`, {
            method: "Post",
            body: JSON.stringify({
                sequence, pitch, duration, speaker_i, pace,
                old_sequence, // For partial re-generation
                outfile: tempFileLocation,
                vocoder: window.userSettings.vocoder,
                waveglowPath: vocoder_select.value=="256_waveglow" ? window.userSettings.waveglow_path : window.userSettings.bigwaveglow_path
            })
        }).then(r=>r.text()).then(res => {

            if (res=="ENOENT") {
                window.errorModal(`Model not found.${vocoder_select.value.includes("waveglow")?" Download WaveGlow files separately if you haven't, or check the path in the settings.":""}`)
                toggleSpinnerButtons()
                return
            }

            const doTheRest = () => {
                dialogueInput.focus()
                isGenerating = false
                res = res.split("\n")
                let pitchData = res[0]
                let durationsData = res[1]
                let cleanedSequence = res[2]
                pitchData = pitchData.split(",").map(v => parseFloat(v))
                durationsData = durationsData.split(",").map(v => isFreshRegen ? parseFloat(v) : parseFloat(v)/pace_slid.value)
                window.sequenceEditor.inputSequence = sequence
                window.sequenceEditor.sequence = cleanedSequence

                if (pitch.length==0 || isFreshRegen) {
                    window.sequenceEditor.resetPitch = pitchData
                    window.sequenceEditor.resetDurs = durationsData
                }


                window.sequenceEditor.letters = cleanedSequence.replace(/\s/g, "_").split("")
                window.sequenceEditor.pitchNew = pitchData.map(p=>p)
                window.sequenceEditor.dursNew = durationsData.map(v=>v)
                window.sequenceEditor.init()
                window.sequenceEditor.update()

                window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
                window.sequenceEditor.autoInferTimer = null
                window.sequenceEditor.hasChanged = false


                toggleSpinnerButtons()
                if (keepSampleButton.dataset.newFileLocation && keepSampleButton.dataset.newFileLocation.startsWith("BATCH_EDIT")) {
                } else {
                    keepSampleButton.dataset.newFileLocation = `${window.userSettings[`outpath_${game}`]}/${voiceType}/${outputFileName}.wav`
                }
                keepSampleButton.disabled = false
                samplePlay.dataset.tempFileLocation = tempFileLocation
                samplePlay.innerHTML = ""

                const audio = createElem("audio", {controls: true, style: {width:"150px"}}, createElem("source", {src: tempFileLocation, type: "audio/wav"}))
                audio.setSinkId(window.userSettings.base_speaker)
                audio.addEventListener("play", () => {
                    if (window.ctrlKeyIsPressed) {
                        audio.setSinkId(window.userSettings.alt_speaker)
                    } else {
                        audio.setSinkId(window.userSettings.base_speaker)
                    }
                })
                samplePlay.appendChild(audio)
                audio.load()
                if (window.userSettings.autoPlayGen) {
                    audio.play()
                }

                // Persistance across sessions
                localStorage.setItem("tempFileLocation", tempFileLocation)
            }


            if (window.userSettings.audio.ffmpeg && setting_audio_ffmpeg_preview.checked) {
                const options = {
                    hz: window.userSettings.audio.hz,
                    padStart: window.userSettings.audio.padStart,
                    padEnd: window.userSettings.audio.padEnd,
                    bit_depth: window.userSettings.audio.bitdepth,
                    amplitude: window.userSettings.audio.amplitude
                }

                fetch(`http://localhost:8008/outputAudio`, {
                    method: "Post",
                    body: JSON.stringify({
                        input_path: tempFileLocation,
                        output_path: tempFileLocation.replace(".wav", `_ffmpeg.${window.userSettings.audio.format}`),
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
                    window.appLogger.log(res)
                    closeModal().then(() => {
                        window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${res}`)
                    })
                })
            } else {
                doTheRest()
            }


        }).catch(res => {
            isGenerating = false
            console.log(res)
            window.errorModal(window.i18n.SOMETHING_WENT_WRONG)
            toggleSpinnerButtons()
        })
    }
})


const refreshRecordsList = (directory) => {

    if (!fs.existsSync(directory)) {
        return
    }

    const records = []
    const reverse = window.userSettings.voiceRecordsOrderByOrder=="ascending"
    const sortBy = window.userSettings.voiceRecordsOrderBy//"name"

    const files = fs.readdirSync(directory)
    files.forEach(file => {
        if (file.endsWith(".json")) {
            return
        }
        const record = {}

        if (!file.toLowerCase().includes(voiceSamplesSearch.value.toLowerCase().trim())) {
            return
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
        voiceSamples.appendChild(makeSample(record.jsonPath))
    })
}

const saveFile = (from, to, skipUIRecord=false) => {
    to = to.split("%20").join(" ")
    to = to.replace(".wav", `.${window.userSettings.audio.format}`)

    // Make the containing folder if it does not already exist
    let containerFolderPath = to.split("/")
    containerFolderPath = containerFolderPath.slice(0,containerFolderPath.length-1).join("/")

    try {fs.mkdirSync(containerFolderPath)} catch (e) {/*Do nothing*/}

    // For plugins
    const pluginData = {
        game: window.currentGame[0],
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
        amplitude: window.userSettings.audio.amplitude
    }
    pluginData.audioOptions = options
    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["pre"], event="pre keep-sample", pluginData)

    const jsonDataOut = {
        inputSequence: dialogueInput.value.trim(),
        pacing: parseFloat(pace_slid.value),
        letters: window.sequenceEditor.letters,
        currentVoice: window.sequenceEditor.currentVoice,
        resetPitch: window.sequenceEditor.resetPitch,
        resetDurs: window.sequenceEditor.resetDurs,
        ampFlatCounter: window.sequenceEditor.ampFlatCounter,
        inputSequence: window.sequenceEditor.inputSequence,
        sequence: window.sequenceEditor.sequence,
        pitchNew: window.sequenceEditor.pitchNew,
        dursNew: window.sequenceEditor.dursNew,
    }

    if (!setting_audio_ffmpeg_preview.checked && window.userSettings.audio.ffmpeg) {
        spinnerModal(window.i18n.SAVING_AUDIO_FILE)

        window.appLogger.log(`${window.i18n.ABOUT_TO_SAVE_FROM_N1_TO_N2_WITH_OPTIONS}: ${JSON.stringify(options)}`.replace("_1", from).replace("_2", to))

        const extraInfo = {
            game: window.currentGame[0],
            voiceId: window.currentModel.voiceId,
            voiceName: window.currentModel.voiceName
        }

        fetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body: JSON.stringify({
                input_path: from,
                output_path: to,
                isBatchMode: false,
                extraInfo: JSON.stringify(pluginData),
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            closeModal().then(() => {
                if (res.length && res!="-") {
                    console.log("res", res)
                    window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${window.i18n.INPUT}: ${from}<br>${window.i18n.OUTPUT}: ${to}<br><br>${res}`)
                } else {
                    if (window.userSettings.outputJSON) {
                        fs.writeFileSync(`${to}.json`, JSON.stringify(jsonDataOut, null, 4))
                    }
                    if (!skipUIRecord) {
                        refreshRecordsList(containerFolderPath)
                    }
                    dialogueInput.focus()
                    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
                }
            })
        }).catch(res => {
            window.appLogger.log(res)
            closeModal().then(() => {
                window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}<br><br>${window.i18n.INPUT}: ${from}<br>${window.i18n.OUTPUT}: ${to}<br><br>${res}`)
            })
        })
    } else {
        fs.copyFile(from, to, err => {
            if (err) {
                console.log(err)
                window.appLogger.log(err)
                if (!fs.existsSync(from)) {
                    window.appLogger.log(`${window.i18n.TEMP_FILE_NOT_EXIST}: ${from}`)
                }
                const outputFolder = to.split("/").reverse().slice(1,1000).reverse().join("/")
                if (!fs.existsSync(outputFolder)) {
                    window.appLogger.log(`${window.i18n.OUT_DIR_NOT_EXIST}: ${outputFolder}`)
                }
            } else {
                if (window.userSettings.outputJSON) {
                    fs.writeFileSync(`${to}.json`, JSON.stringify(jsonDataOut, null, 4))
                }
                if (!skipUIRecord) {
                    refreshRecordsList(containerFolderPath)
                }
                window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
            }
        })
    }
}

window.keepSampleFunction = shiftClick => {
    if (keepSampleButton.dataset.newFileLocation) {

        const skipUIRecord = keepSampleButton.dataset.newFileLocation.includes("BATCH_EDIT")
        let fromLocation = samplePlay.dataset.tempFileLocation
        let toLocation = keepSampleButton.dataset.newFileLocation.replace("BATCH_EDIT", "")

        if (!skipUIRecord) {
            toLocation = toLocation.split("/")
            toLocation[toLocation.length-1] = toLocation[toLocation.length-1].replace(/[\/\\:\*?<>"|]*/g, "")
            toLocation[toLocation.length-1] = toLocation[toLocation.length-1].replace(/\.wav$/, "").slice(0, 75).replace(/\.$/, "")
        }


        // Numerical file name counter
        if (window.userSettings.filenameNumericalSeq) {
            let existingFiles = []
            try {
                existingFiles = fs.readdirSync(toLocation.slice(0, toLocation.length-1).join("/")).filter(fname => !fname.endsWith(".json"))
            } catch (e) {
            }
            existingFiles = existingFiles.filter(fname => fname.includes(toLocation[toLocation.length-1]))
            existingFiles = existingFiles.map(fname => {
                const parts = fname.split(".")
                if (parts.length>2 && parts[parts.length-2].length) {
                    if (parseInt(parts[parts.length-2]) != NaN) {
                        return parseInt(parts[parts.length-2])
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
                                window.appLogger.log(err)
                            }
                            console.log(fromLocation, "finalOutLocation", finalOutLocation)
                            saveFile(fromLocation, finalOutLocation, skipUIRecord)
                        })
                        return
                    } else {
                        saveFile(fromLocation, toLocationOut.join("/"), skipUIRecord)
                        return
                    }
                }
                saveFile(fromLocation, toLocationOut.join("/"), skipUIRecord)
            })

        } else {
            saveFile(fromLocation, toLocation, skipUIRecord)
        }
    }
}
keepSampleButton.addEventListener("click", event => keepSampleFunction(event.shiftKey))



// Weird recursive intermittent promises to repeatedly check if the server is up yet - but it works!
window.doWeirdServerStartupCheck = (startUpMessage) => {
    let serverIsUp = false
    const check = () => {
        return new Promise(topResolve => {
            if (serverIsUp) {
                topResolve()
            } else {
                // console.log("checking");
                (new Promise((resolve, reject) => {
                    fetch(`http://localhost:8008/checkReady`, {
                        method: "Post",
                        body: JSON.stringify({device: (window.userSettings.useGPU&&window.userSettings.installation=="gpu")?"gpu":"cpu"})
                    }).then(r => r.text()).then(r => {
                        // console.log("r", r)
                        closeModal().then(() => {
                            window.pluginsManager.updateUI()
                            if (!window.pluginsManager.hasRunPostStartPlugins) {
                                window.pluginsManager.hasRunPostStartPlugins = true
                                window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["post"], event="post start")
                            }
                        })
                        serverIsUp = true
                        if (window.userSettings.installation=="cpu") {

                            if (useGPUCbx.checked) {
                                fetch(`http://localhost:8008/setDevice`, {
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
                    }).catch(() => reject())
                })).catch(() => {
                    setTimeout(async () => {
                        await check()
                        topResolve()
                    }, 100)
                })
            }
        })
    }
    spinnerModal(startUpMessage)
    check()
}
window.doWeirdServerStartupCheck(`${window.i18n.LOADING}...<br>${window.i18n.MAY_TAKE_A_MINUTE}<br><br>${window.i18n.STARTING_PYTHON}...`)


modalContainer.addEventListener("click", event => {
    try {
        if (event.target==modalContainer && activeModal.dataset.type!="spinner") {
            closeModal()
        }
    } catch (e) {}
})


// Cached UI stuff
// =========
dialogueInput.addEventListener("keyup", () => {
    localStorage.setItem("dialogueInput", dialogueInput.value)
    window.sequenceEditor.hasChanged = true
})

const dialogueInputCache = localStorage.getItem("dialogueInput")

if (dialogueInputCache) {
    dialogueInput.value = dialogueInputCache
}

if (Object.keys(window.userSettings).includes("voiceRecordsOrderBy")) {
    const labels = {
        "name": "Name",
        "time": "Time"
    }
    voiceRecordsOrderByButton.innerHTML = labels[window.userSettings.voiceRecordsOrderBy]
} else {
    window.userSettings.voiceRecordsOrderBy = "name"
    saveUserSettings()
}
if (Object.keys(window.userSettings).includes("voiceRecordsOrderByOrder")) {
    const labels = {
        "ascending": "Ascending",
        "descending": "Descending"
    }
    voiceRecordsOrderByOrderButton.innerHTML = labels[window.userSettings.voiceRecordsOrderByOrder]
} else {
    window.userSettings.voiceRecordsOrderByOrder = "ascending"
    saveUserSettings()
}
// =========


voiceRecordsOrderByButton.addEventListener("click", () => {
    window.userSettings.voiceRecordsOrderBy = window.userSettings.voiceRecordsOrderBy=="name" ? "time" : "name"
    saveUserSettings()
    const labels = {
        "name": "Name",
        "time": "Time"
    }
    voiceRecordsOrderByButton.innerHTML = labels[window.userSettings.voiceRecordsOrderBy]
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame[0]}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})
voiceRecordsOrderByOrderButton.addEventListener("click", () => {
    window.userSettings.voiceRecordsOrderByOrder = window.userSettings.voiceRecordsOrderByOrder=="ascending" ? "descending" : "ascending"
    saveUserSettings()
    const labels = {
        "ascending": "Ascending",
        "descending": "Descending"
    }
    voiceRecordsOrderByOrderButton.innerHTML = labels[window.userSettings.voiceRecordsOrderByOrder]
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame[0]}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})
voiceSamplesSearch.addEventListener("keyup", () => {
    if (window.currentModel) {
        const voiceRecordsList = window.userSettings[`outpath_${window.currentGame[0]}`]+`/${window.currentModel.voiceId}`
        refreshRecordsList(voiceRecordsList)
    }
})








vocoder_select.value = window.userSettings.vocoder.includes(".hg.") ? "qnd" : window.userSettings.vocoder
const changeVocoder = vocoder => {
    return new Promise(resolve => {
        spinnerModal(window.i18n.CHANGING_MODELS)
        fetch(`http://localhost:8008/setVocoder`, {
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
                        window.errorModal(`Model not found.${vocoder.includes("waveglow")?" Download WaveGlow files separately if you haven't, or check the path in the settings.":""}`)
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
vocoder_select.addEventListener("change", () => changeVocoder(vocoder_select.value))



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
patreonButton.addEventListener("click", () => {
    shell.openExternal("https://patreon.com")
})
fetch("http://danruta.co.uk/patreon.txt").then(r=>r.text()).then(data => fs.writeFileSync(`${path}/patreon.txt`, data, "utf8")).catch(e => {})


// Updates
// =======
app_version.innerHTML = window.appVersion
updatesVersions.innerHTML = `${window.i18n.THIS_APP_VERSION}: ${window.appVersion}`

const checkForUpdates = () => {
    fetch("http://danruta.co.uk/xvasynth_updates.txt").then(r=>r.json()).then(data => {
        fs.writeFileSync(`${path}/updates.json`, JSON.stringify(data), "utf8")
        checkUpdates.innerHTML = window.i18n.CHECK_FOR_UPDATES
        showUpdates()
    }).catch(() => {
        checkUpdates.innerHTML = window.i18n.CANT_REACH_SERVER
    })
}
const showUpdates = () => {
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
showUpdates()


// Batch generation
// ========
window.setupModal(batchIcon, batchGenerationContainer)

// Settings
// ========
window.setupModal(settingsCog, settingsContainer)

// Change Game
// ===========
window.setupModal(changeGameButton, gameSelectionContainer)

window.gameAssets = {}

window.updateGameList = () => {
    gameSelectionListContainer.innerHTML = ""
    const fileNames = fs.readdirSync(`${window.path}/assets`)

    let totalVoices = 0
    let totalGames = new Set()

    const itemsToSort = []

    fileNames.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png")) && (fn.split("-").length==4 || fn.split("-").length==5)).forEach(fileName => {
        const gameSelection = createElem("div.gameSelection")
        gameSelection.style.background = `url("assets/${fileName}")`

        const gameId = fileName.split("-")[0]
        const gameName = fileName.split("-").reverse()[0].split(".")[0]
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

        window.gameAssets[gameId] = fileName
        gameSelectionContent.addEventListener("click", () => {
            changeGame(fileName)
            closeModal(gameSelectionContainer)
        })

        itemsToSort.push([numVoices, gameSelection])

        const modelsDir = window.userSettings[`modelspath_${gameId}`]
        if (!window.watchedModelsDirs.includes(modelsDir)) {
            window.watchedModelsDirs.push(modelsDir)

            try {
                fs.watch(modelsDir, {recursive: false, persistent: true}, (eventType, filename) => {
                    if (window.userSettings.autoReloadVoices) {
                        window.appLogger.log(`${eventType}: ${filename}`)
                        loadAllModels().then(() => changeGame(fileName))
                    }
                })
            } catch (e) {}
        }
    })

    itemsToSort.sort((a,b) => a[0]<b[0]?1:-1).forEach(([numVoices, elem]) => {
        gameSelectionListContainer.appendChild(elem)
    })

    searchGameInput.addEventListener("keyup", () => {
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
            changeGame(lastGame)
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


// Plugins
// =======
window.setupModal(pluginsIcon, pluginsContainer)

// Speech-to-Speech
// ================
window.setupModal(s2s_selectVoiceBtn, s2sSelectContainer, () => window.populateS2SVoiceList())
window.setupModal(s2s_settingsRecNoiseBtn, s2sSelectContainer, () => window.populateS2SVoiceList())




// Other
// =====
voiceSearchInput.addEventListener("keyup", () => {
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

// ELUA
EULA_closeButon.addEventListener("click", () => {
    if (EULA_accept_ckbx.checked) {
        closeModal(EULAContainer)
        window.userSettings.EULA_accepted = true
        saveUserSettings()
    }
})
if (!Object.keys(window.userSettings).includes("EULA_accepted") || window.userSettings.EULA_accepted==false) {
    EULAContainer.style.opacity = 0
    EULAContainer.style.display = "flex"
    chrome.style.opacity = 1
    requestAnimationFrame(() => requestAnimationFrame(() => EULAContainer.style.opacity = 1))
}
// Links
document.querySelectorAll('a[href^="http"]').forEach(a => a.addEventListener("click", e => {
    event.preventDefault()
    shell.openExternal(a.href)
}))
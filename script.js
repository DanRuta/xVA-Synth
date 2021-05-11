"use strict"

window.appVersion = "v1.3.7"

const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "./resources/app" : "."
window.path = path

const fs = require("fs")
const zipdir = require('zip-dir')
const {shell, ipcRenderer} = require("electron")
const fetch = require("node-fetch")
const {text_to_sequence, english_cleaners} = require("./text.js")
const {xVAAppLogger} = require("./appLogger.js")
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
const {saveUserSettings, deleteFolderRecursive} = require("./settingsMenu.js")
const {startBatch} = require("./batch.js")
window.electronBrowserWindow = require("electron").remote.getCurrentWindow()
const child = require("child_process").execFile
const spawn = require("child_process").spawn

const {PluginsManager} = require("./plugins_manager.js")
window.pluginsManager = new PluginsManager(window.path, window.appLogger, window.appVersion)
window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["pre"], event="pre start")

let themeColour
const oldCError = console.error
console.error = (data) => {
    window.appLogger.log(data)
    oldCError(arguments)
}
process.on(`uncaughtException`, data => window.appLogger.log(data))
window.onerror = (err, url, lineNum) => window.appLogger.log(err)

window.addEventListener("error", function (e) {window.appLogger.log(e.error.stack)})
window.addEventListener('unhandledrejection', function (e) {window.appLogger.log(e.reason.stack)})

window.games = {}
window.models = {}
window.pitchEditor = {letters: [], currentVoice: null, resetPitch: null, resetDurs: null, letterFocus: [], ampFlatCounter: 0, hasChanged: false}
window.currentModel = undefined
window.currentModelButton = undefined
window.watchedModelsDirs = []

window.appLogger.log(`Settings: ${JSON.stringify(window.userSettings)}`)

// Set up folders
try {fs.mkdirSync(`${path}/models`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/output`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/assets`)} catch (e) {/*Do nothing*/}

// Clean up temp files
fs.readdir(`${__dirname}/output`, (err, files) => {
    if (err) {
        window.appLogger.log(err)
    }
    if (files && files.length) {
        files.filter(f => f.startsWith("temp-")).forEach(file => {
            fs.unlink(`${__dirname}/output/${file}`, err => err&&console.log(err))
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

                            const modelData = {model, modelsPath, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, gender, modelVersion: model.modelVersion, hifi: undefined}
                            const potentialHiFiPath = `${modelsPath}/${voiceId}.hg.pt`
                            if (fs.existsSync(potentialHiFiPath)) {
                                modelData.hifi = potentialHiFiPath
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
                        window.appLogger.log("ERROR loading models for game: "+ gameFolder  + " with fileName: "+fileName)
                        window.appLogger.log(e)
                        window.appLogger.log(e.stack)
                    }
                })
            } catch (e) {
                console.log(e)
                window.appLogger.log("ERROR loading models for game: "+ gameFolder)
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
})

// Change game
window.changeGame = (meta) => {

    meta = meta.split("-")
    window.currentGame = meta
    themeColour = meta[1]
    generateVoiceButton.disabled = true
    generateVoiceButton.innerHTML = "Generate Voice"
    selectedGameDisplay.innerHTML = meta[3].split(".")[0]


    // Change batch panel colours, if it is initialized
    try {
        Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${window.currentGame[1]}`)
    } catch (e) {}
    try {
        Array.from(pluginsRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${window.currentGame[1]}`)
    } catch (e) {}

    // Change the app title
    title.innerHTML = "Select Voice Type"
    if (window.games[window.currentGame[0]] == undefined) {
        title.innerHTML = `No models in: ${window.userSettings[`modelspath_${window.currentGame[0]}`]}`
        console.log(title.innerHTML)
    } else if (meta[2]) {
        document.title = `${meta[2]}VA Synth`
        dragBar.innerHTML = `${meta[2]}VA Synth`
    } else {
        document.title = `xVA Synth`
        dragBar.innerHTML = `xVA Synth`
    }

    const gameFolder = meta[0]
    const gameName = meta[meta.length-1].split(".")[0]

    setting_models_path_container.style.display = "flex"
    setting_out_path_container.style.display = "flex"
    setting_models_path_label.innerHTML = `<i style="display:inline">${gameName}</i><span>models path</span>`
    setting_models_path_input.value = window.userSettings[`modelspath_${gameFolder}`]
    setting_out_path_label.innerHTML = `<i style="display:inline">${gameName}</i> output path`
    setting_out_path_input.value = window.userSettings[`outpath_${gameFolder}`]

    if (meta) {
        const background = `linear-gradient(0deg, grey 0px, rgba(0,0,0,0)), url("assets/${meta.join("-")}")`
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.background = `#${themeColour}`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.background = `#${themeColour}`)
        Array.from(document.querySelectorAll(".spinner")).forEach(e => e.style.borderLeftColor = `#${themeColour}`)

        // Fade the background image transition
        rightBG1.style.background = background
        rightBG2.style.opacity = 0
        setTimeout(() => {
            rightBG2.style.background = rightBG1.style.background
            rightBG2.style.opacity = 1
        }, 1000)
    }

    cssHack.innerHTML = `::selection {
        background: #${themeColour};
    }
    ::-webkit-scrollbar-thumb {
        background-color: #${themeColour} !important;
    }
    .slider::-webkit-slider-thumb {
        background-color: #${themeColour} !important;
    }
    a {color: #${themeColour}};
    #batchRecordsHeader > div {background-color: #${themeColour} !important;}
    #pluginsRecordsHeader > div {background-color: #${themeColour} !important;}
    `

    try {fs.mkdirSync(`${path}/output/${meta[0]}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", window.currentGame.join("-"))

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""


    // No models found
    if (!Object.keys(window.games).length) {
        title.innerHTML = "No models found"
        return
    }

    const buttons = []

    voiceSearchInput.placeholder = `Search ${window.games[meta[0]] ? window.games[meta[0]].models.length : "0"} voices...`
    voiceSearchInput.value = ""

    if (!window.games[meta[0]]) {
        return
    }

    window.games[meta[0]].models.forEach(({model, modelsPath, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, hifi}) => {

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
        button.dataset.modelId = voiceId

        // Quick voice set preview, if there is a preview file
        button.addEventListener("contextmenu", () => {
            window.appLogger.log(`${audioPreviewPath}.wav`)
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: `${audioPreviewPath}.wav`
            }))
        })

        button.addEventListener("click", event => {

            // Just for easier packaging of the voice models for publishing - yes, lazy
            if (event.ctrlKey && event.shiftKey) {
                if (event.altKey) {
                    const files = fs.readdirSync(`./output`).filter(fname => fname.includes("temp-") && fname.includes(".wav"))
                    console.log("files", files)
                    if (files.length) {
                        const options = {
                            hz: window.userSettings.audio.hz,
                            padStart: window.userSettings.audio.padStart,
                            padEnd: window.userSettings.audio.padEnd,
                            bit_depth: window.userSettings.audio.bitdepth,
                            amplitude: window.userSettings.audio.amplitude
                        }

                        // console.log(`About to save file from ${from} to ${to} with options: ${JSON.stringify(options)}`)
                        // window.appLogger.log(`About to save file from ${from} to ${to} with options: ${JSON.stringify(options)}`)
                        fetch(`http://localhost:8008/outputAudio`, {
                            method: "Post",
                            body: JSON.stringify({
                                input_path: `./output/${files[0]}`,
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
                window.errorModal(`This model requires app version v${model.version}<br><br>This app version: ${window.appVersion}`)
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
                generateVoiceButton.innerHTML = "Generate Voice"
                generateVoiceButton.dataset.modelQuery = "null"

            } else {
                generateVoiceButton.innerHTML = "Load model"

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
            fs.readdir(`${window.userSettings[`outpath_${meta[0]}`]}/${button.dataset.modelId}`, (err, files) => {

                if (err) return

                files.forEach(file => {
                    if (file.endsWith(".json")) {
                        return
                    }
                    voiceSamples.appendChild(makeSample(`${window.userSettings[`outpath_${meta[0]}`]}/${button.dataset.modelId}/${file}`))
                })
            })
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
    const openFileLocationButton = createElem("div", {title: "Open containing folder"})
    openFileLocationButton.innerHTML = "&#10064;"
    openFileLocationButton.addEventListener("click", () => {
        console.log("open dir", src)
        shell.showItemInFolder(src)
    })

    if (fs.existsSync(`${src}.json`)) {
        const editButton = createElem("div", {title: "Adjust sample in the editor"})
        editButton.innerHTML = `<svg class="renameSVG" version="1.0" xmlns="http:\/\/www.w3.org/2000/svg" width="344.000000pt" height="344.000000pt" viewBox="0 0 344.000000 344.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,344.000000) scale(0.100000,-0.100000)" fill="#555555" stroke="none"><path d="M1489 2353 l-936 -938 -197 -623 c-109 -343 -195 -626 -192 -629 2 -3 284 84 626 193 l621 198 937 938 c889 891 937 940 934 971 -11 108 -86 289 -167 403 -157 219 -395 371 -655 418 l-34 6 -937 -937z m1103 671 c135 -45 253 -135 337 -257 41 -61 96 -178 112 -241 l12 -48 -129 -129 -129 -129 -287 287 -288 288 127 127 c79 79 135 128 148 128 11 0 55 -12 97 -26z m-1798 -1783 c174 -79 354 -248 436 -409 59 -116 72 -104 -213 -196 l-248 -80 -104 104 c-58 58 -105 109 -105 115 0 23 154 495 162 495 5 0 37 -13 72 -29z"/></g></svg>`
        editButton.addEventListener("click", () => {
            let editData = fs.readFileSync(`${src}.json`, "utf8")
            editData = JSON.parse(editData)
            window.pitchEditor = editData.pitchEditor
            dialogueInput.value = editData.inputSequence
            setPitchEditorValues(undefined, undefined, undefined, true)
            pace_slid.value = editData.pacing

            if (samplePlay.style.display!="none") {
                samplePlay.removeChild(samplePlay.children[0])
                samplePlay.appendChild(createElem("audio", {controls: true}, createElem("source", {
                    src: src,
                    type: `audio/${fileFormat}`
                })))
            }
        })
        audioControls.appendChild(editButton)
    }

    const renameButton = createElem("div", {title: "Rename the file"})
    renameButton.innerHTML = `<svg class="renameSVG" version="1.0" xmlns="http://www.w3.org/2000/svg" width="166.000000pt" height="336.000000pt" viewBox="0 0 166.000000 336.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,336.000000) scale(0.100000,-0.100000)" fill="#000000" stroke="none"> <path d="M165 3175 c-30 -31 -35 -42 -35 -84 0 -34 6 -56 21 -75 42 -53 58 -56 324 -56 l245 0 0 -1290 0 -1290 -245 0 c-266 0 -282 -3 -324 -56 -15 -19 -21 -41 -21 -75 0 -42 5 -53 35 -84 l36 -35 281 0 280 0 41 40 c30 30 42 38 48 28 5 -7 9 -16 9 -21 0 -4 15 -16 33 -27 30 -19 51 -20 319 -20 l287 0 36 35 c30 31 35 42 35 84 0 34 -6 56 -21 75 -42 53 -58 56 -324 56 l-245 0 0 1290 0 1290 245 0 c266 0 282 3 324 56 15 19 21 41 21 75 0 42 -5 53 -35 84 l-36 35 -287 0 c-268 0 -289 -1 -319 -20 -18 -11 -33 -23 -33 -27 0 -5 -4 -14 -9 -21 -6 -10 -18 -2 -48 28 l-41 40 -280 0 -281 0 -36 -35z"/></g></svg>`

    renameButton.addEventListener("click", () => {
        createModal("prompt", {
            prompt: "Enter new file name, or submit unchanged to cancel.",
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

                sample.querySelector("div").innerHTML = newFileName
                if (samplePlay.style.display!="none") {
                    samplePlay.removeChild(samplePlay.children[0])
                    samplePlay.appendChild(createElem("audio", {controls: true}, createElem("source", {
                        src: newPathComposed,
                        type: `audio/${fileFormat}`
                    })))
                }
            }
        })
    })

    const editInProgramButton = createElem("div", {title: "Edit in external program"})
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
                    window.errorModal(`The following program path is not valid:<br><br> ${window.userSettings.externalAudioEditor}`)
                } else {
                    window.errorModal(err.message)
                }
            })

        } else {
            window.errorModal("Specify your audio editing tool in the settings")
        }
    })


    const deleteFileButton = createElem("div", {title: "Delete file"})
    deleteFileButton.innerHTML = "&#10060;"
    deleteFileButton.addEventListener("click", () => {
        confirmModal(`Are you sure you'd like to delete this file?<br><br><i>${fileName}</i>`).then(confirmation => {
            if (confirmation) {
                window.appLogger.log(`Deleting${newSample?"new":" "} file: ${src}`)
                fs.unlinkSync(src)
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


window.toggleSpinnerButtons = () => {
    const spinnerVisible = window.getComputedStyle(spinner).display == "block"
    spinner.style.display = spinnerVisible ? "none" : "block"
    keepSampleButton.style.display = spinnerVisible ? "block" : "none"
    generateVoiceButton.style.display = spinnerVisible ? "block" : "none"
    samplePlay.style.display = spinnerVisible ? "flex" : "none"
}

generateVoiceButton.addEventListener("click", () => {

    const game = window.currentGame[0]

    try {fs.mkdirSync(window.userSettings[`outpath_${game}`])} catch (e) {/*Do nothing*/}
    try {fs.mkdirSync(`${window.userSettings[`outpath_${game}`]}/${voiceId}`)} catch (e) {/*Do nothing*/}


    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {

        if (window.batch_state.state) {
            window.errorModal("Batch synthesis is in progress. Loading a model in the main app now would break things.")
            return
        }

        window.appLogger.log(`Loading voice set: ${JSON.parse(generateVoiceButton.dataset.modelQuery).model}`)
        window.batch_state.lastModel = JSON.parse(generateVoiceButton.dataset.modelQuery).model.split("/").reverse()[0]

        spinnerModal("Loading voice set<br>(may take a minute... but not much more!)")
        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = "Generate Voice"
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad

            if (window.userSettings.defaultToHiFi && window.currentModel.hifi) {
                vocoder_select.value = Array.from(vocoder_select.children).find(opt => opt.innerHTML=="Bespoke HiFi GAN").value
                changeVocoder(vocoder_select.value)
            } else if (window.userSettings.vocoder.includes(".hg.pt")) {
                changeVocoder("qnd")
            }
        }).catch(e => {
            console.log(e)
            if (e.code =="ENOENT") {
                closeModal(null, modalContainer).then(() => {
                    createModal("error", "There was an issue connecting to the python server.<br><br>Try again in a few seconds. If the issue persists, make sure localhost port 8008 is free, or send the server.log file to me on GitHub or Nexus.")
                })
            }
        })
    } else {

        if (isGenerating) {
            return
        }
        isGenerating = true

        const sequence = dialogueInput.value.trim().replace("…", "...")
        if (sequence.length==0) {
            return
        }

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
        const tempFileLocation = `${path}/output/temp-${tempFileNum}.wav`
        let pitch = []
        let duration = []
        let isFreshRegen = true
        let old_sequence = undefined


        if (editor.innerHTML && editor.innerHTML.length && generateVoiceButton.dataset.modelIDLoaded==window.pitchEditor.currentVoice) {
            if (window.pitchEditor.sequence && sequence!=window.pitchEditor.inputSequence) {
                old_sequence = window.pitchEditor.inputSequence
            }
        }

        if (editor.innerHTML && editor.innerHTML.length && generateVoiceButton.dataset.modelIDLoaded==window.pitchEditor.currentVoice) {
            pitch = window.pitchEditor.pitchNew.map(v=> v==undefined?0:v)
            duration = window.pitchEditor.dursNew.map(v => v*pace_slid.value).map(v=> v==undefined?0:v)
            isFreshRegen = false
        }
        window.pitchEditor.currentVoice = generateVoiceButton.dataset.modelIDLoaded

        const speaker_i = window.currentModel.games[0].emb_i
        const pace = (window.userSettings.keepPaceOnNew && isFreshRegen)?pace_slid.value:1


        window.appLogger.log(`Synthesising audio: ${sequence}`)

        fetch(`http://localhost:8008/synthesize`, {
            method: "Post",
            body: JSON.stringify({
                sequence, pitch, duration, speaker_i, pace,
                old_sequence, // For partial re-generation
                outfile: tempFileLocation,
                vocoder: window.userSettings.vocoder
            })
        }).then(r=>r.text()).then(res => {
            isGenerating = false
            res = res.split("\n")
            let pitchData = res[0]
            let durationsData = res[1]
            let cleanedSequence = res[2]
            pitchData = pitchData.split(",").map(v => parseFloat(v))
            durationsData = durationsData.split(",").map(v => isFreshRegen ? parseFloat(v) : parseFloat(v)/pace_slid.value)
            window.pitchEditor.inputSequence = sequence
            window.pitchEditor.sequence = cleanedSequence

            if (pitch.length==0 || isFreshRegen) {
                window.pitchEditor.ampFlatCounter = 0
                window.pitchEditor.resetPitch = pitchData
                window.pitchEditor.resetDurs = durationsData
            }

            setPitchEditorValues(cleanedSequence.replace(/\s/g, "_").split(""), pitchData, durationsData, isFreshRegen, pace)

            toggleSpinnerButtons()
            keepSampleButton.dataset.newFileLocation = `${window.userSettings[`outpath_${game}`]}/${voiceType}/${outputFileName}.wav`
            keepSampleButton.disabled = false
            samplePlay.dataset.tempFileLocation = tempFileLocation
            samplePlay.innerHTML = ""

            const finalOutSrc = `./output/temp-${tempFileNum}.wav`.replace("..", ".")

            const audio = createElem("audio", {controls: true, style: {width:"150px"}},
                    createElem("source", {src: finalOutSrc, type: "audio/wav"}))
            samplePlay.appendChild(audio)
            audio.load()
            if (window.userSettings.autoPlayGen) {
                audio.play()
            }

            // Persistance across sessions
            localStorage.setItem("tempFileLocation", tempFileLocation)
        }).catch(res => {
            isGenerating = false
            console.log(res)
            window.errorModal(`Something went wrong`)
            toggleSpinnerButtons()
        })
    }
})

const saveFile = (from, to) => {
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
        inputSequence: window.pitchEditor.inputSequence,
        letters: window.pitchEditor.letters,
        pitch: window.pitchEditor.pitchNew,
        durations: window.pitchEditor.dursNew,
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

    if (window.userSettings.audio.ffmpeg) {
        spinnerModal("Saving the audio file...")

        console.log(`About to save file from ${from} to ${to} with options: ${JSON.stringify(options)}`)
        window.appLogger.log(`About to save file from ${from} to ${to} with options: ${JSON.stringify(options)}`)


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
                extraInfo: JSON.stringify(pluginData),
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            closeModal().then(() => {
                if (res.length) {
                    console.log("res", res)
                    window.errorModal(`Something went wrong<br><br>Input: ${from}<br>Output: ${to}<br><br>${res}`)
                } else {
                    if (window.userSettings.outputJSON) {
                        fs.writeFileSync(`${to}.json`, JSON.stringify({inputSequence: dialogueInput.value.trim(), pitchEditor: window.pitchEditor, pacing: parseFloat(pace_slid.value)}, null, 4))
                    }
                    voiceSamples.appendChild(makeSample(to, true))
                    window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
                }
            })
        }).catch(res => {
            window.appLogger.log(res)
            console.log("CATCH res", res)
            closeModal().then(() => {
                window.errorModal(`Something went wrong<br><br>Input: ${from}<br>Output: ${to}<br><br>${res}`)
            })
        })
    } else {
        fs.copyFile(from, to, err => {
            if (err) {
                console.log(err)
                window.appLogger.log(err)
            }
            if (window.userSettings.outputJSON) {
                fs.writeFileSync(`${to}.json`, JSON.stringify({inputSequence: dialogueInput.value.trim(), pitchEditor: window.pitchEditor, pacing: parseFloat(pace_slid.value)}, null, 4))
            }
            voiceSamples.appendChild(makeSample(to, true))
            window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["keep-sample"]["post"], event="post keep-sample", pluginData)
        })
    }
}

const keepSampleFunction = shiftClick => {
    if (keepSampleButton.dataset.newFileLocation) {

        let fromLocation = samplePlay.dataset.tempFileLocation
        let toLocation = keepSampleButton.dataset.newFileLocation

        toLocation = toLocation.split("/")
        toLocation[toLocation.length-1] = toLocation[toLocation.length-1].replace(/[\/\\:\*?<>"|]*/g, "")
        toLocation = toLocation.join("/")

        const outFolder = toLocation.split("/").reverse().slice(2, 100).reverse().join("/")
        if (!fs.existsSync(outFolder)) {
            return void window.errorModal(`The output directory does not exist:<br><br><i>${outFolder}</i><br><br>You can change this in the settings.`)
        }

        // File name conflict
        const alreadyExists = fs.existsSync(toLocation)
        if (alreadyExists || shiftClick) {

            const promptText = alreadyExists ? `File already exists. Adjust the file name here, or submit without changing to overwrite the old file.` : `Enter file name`

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
                            saveFile(fromLocation, finalOutLocation)
                        })
                        return
                    } else {
                        saveFile(fromLocation, toLocationOut.join("/"))
                        return
                    }
                }
                saveFile(fromLocation, toLocationOut.join("/"))
            })

        } else {
            saveFile(fromLocation, toLocation)
        }
    }
}
keepSampleButton.addEventListener("click", event => keepSampleFunction(event.shiftKey))



window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))
window.errorModal = message => new Promise(resolve => resolve(createModal("error", message)))
const createModal = (type, message) => {
    return new Promise(resolve => {
        modalContainer.innerHTML = ""
        const displayMessage = message.prompt ? message.prompt : message
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span", displayMessage))
        modal.dataset.type = type

        if (type=="confirm") {
            const yesButton = createElem("button", {style: {background: `#${themeColour}`}})
            yesButton.innerHTML = "Yes"
            const noButton = createElem("button", {style: {background: `#${themeColour}`}})
            noButton.innerHTML = "No"
            modal.appendChild(createElem("div", yesButton, noButton))

            yesButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(true)
                })
            })
            noButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="error") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = "Close"
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="prompt") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = "Submit"
            const inputElem = createElem("input", {type: "text", value: message.value})
            modal.appendChild(createElem("div", inputElem))
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(inputElem.value)
                })
            })
        } else {
            modal.appendChild(createElem("div.spinner", {style: {borderLeftColor: document.querySelector("button").style.background}}))
        }

        modalContainer.appendChild(modal)
        modalContainer.style.opacity = 0
        modalContainer.style.display = "flex"

        requestAnimationFrame(() => requestAnimationFrame(() => modalContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
    })
}
window.closeModal = (container=undefined, notThisOne=undefined) => {
    return new Promise(resolve => {
        const allContainers = [batchGenerationContainer, gameSelectionContainer, updatesContainer, infoContainer, settingsContainer, patreonContainer, container, pluginsContainer, modalContainer]
        const containers = container==undefined ? allContainers : [container]
        containers.forEach(cont => {
            if ((notThisOne!=undefined&&notThisOne!=cont) && (notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                cont.style.opacity = 0
            }
        })

        const someOpenContainer = allContainers.find(container => container!=undefined && container.style.opacity==1 && container.style.display!="none" && container!=modalContainer)
        if (!someOpenContainer || someOpenContainer==container) {
            chrome.style.opacity = 0.88
        }

        containers.forEach(cont => {
            if ((notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                cont.style.opacity = 0
            }
        })

        setTimeout(() => {
            containers.forEach(cont => {
                if ((notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                    cont.style.display = "none"
                }
            })
        }, 200)
        try {
            activeModal.remove()
        } catch (e) {}
        resolve()
    })
}

let startingSplashInterval
let loadingStage = 0
let hasRunPostStartPlugins = false
startingSplashInterval = setInterval(() => {
    if (fs.existsSync(`${path}/FASTPITCH_LOADING`)) {
        if (loadingStage==0) {
            spinnerModal("Loading...<br>May take a minute (but not much more)<br><br>Building FastPitch model...")
            loadingStage = 1
        }
    } else if (fs.existsSync(`${path}/WAVEGLOW_LOADING`)) {
        if (loadingStage==1) {
            activeModal.children[0].innerHTML = "Loading...<br>May take a minute (but not much more)<br><br>Loading WaveGlow model..."
            loadingStage = 2
        }
    } else if (fs.existsSync(`${path}/SERVER_STARTING`)) {
        if (loadingStage==2) {
            activeModal.children[0].innerHTML = "Loading...<br>May take a minute (but not much more)<br><br>Starting up the python backend..."
            loadingStage = 3
        }
    } else {
        closeModal().then(() => {
            clearInterval(startingSplashInterval)
            if (!hasRunPostStartPlugins) {
                hasRunPostStartPlugins = true
                window.pluginsManager.runPlugins(window.pluginsManager.pluginsModules["start"]["post"], event="post start")
            }
        })
    }
}, 100)




modalContainer.addEventListener("click", event => {
    if (event.target==modalContainer && activeModal.dataset.type!="spinner") {
        closeModal()
    }
})

dialogueInput.addEventListener("keyup", () => {
    localStorage.setItem("dialogueInput", dialogueInput.value)
    window.pitchEditor.hasChanged = true
})

const dialogueInputCache = localStorage.getItem("dialogueInput")

if (dialogueInputCache) {
    dialogueInput.value = dialogueInputCache
}

window.addEventListener("resize", e => {
    window.userSettings.customWindowSize = `${window.innerHeight},${window.innerWidth}`
    saveUserSettings()
})


const setLetterFocus = (l, multi) => {
    if (window.pitchEditor.letterFocus.length && !multi) {
        window.pitchEditor.letterFocus.forEach(li => letterElems[li].style.color = "black")
        window.pitchEditor.letterFocus = []
    }
    window.pitchEditor.letterFocus.push(l)
    window.pitchEditor.letterFocus = Array.from(new Set(window.pitchEditor.letterFocus.sort()))
    window.pitchEditor.letterFocus.forEach(li => letterElems[li].style.color = "red")

    if (window.pitchEditor.letterFocus.length==1) {
        letterLength.value = parseFloat(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
        letterLengthNumb.value = letterLength.value

        letterLength.disabled = false
        letterPitchNumb.disabled = false
        letterLengthNumb.disabled = false
    } else {
        letterPitchNumb.disabled = true
        letterPitchNumb.value = ""
        letterLengthNumb.disabled = true
        letterLengthNumb.value = ""
    }
}

let sliders = []
let letterElems = []
let autoinfer_timer = null
let has_been_changed = false
let css_hack_items = []
let elemsWidths = []
const infer = () => {
    has_been_changed = false
    if (!isGenerating) {
        generateVoiceButton.click()
    }
}
const set_letter_display = (elem, elem_i, length=null, value=null) => {
    if (length != null && elem) {
        const elem_length = length/2
        elem.style.width = `${parseInt(elem_length/2)}px`
        elem.children[1].style.height = `${elem_length}px`
        elem.children[1].style.marginTop = `${-parseInt(elem_length/2)+90}px`
        css_hack_items[elem_i].innerHTML = `#slider_${elem_i}::-webkit-slider-thumb {height: ${elem_length}px;}`
        elemsWidths[elem_i] = elem_length
        elem.style.paddingLeft = `${parseInt(elem_length/2)}px`
        editor.style.width = `${parseInt(elemsWidths.reduce((p,c)=>p+c,1)*1.25)}px`
    }

    if (value != null) {
        elem.children[1].value = value
    }
}
const setPitchEditorValues = (letters, pitchOrig, lengthsOrig, isFreshRegen, pace=1) => {

    Array.from(editor.children).forEach(child => editor.removeChild(child))

    letters = letters ? letters : window.pitchEditor.letters
    pitchOrig = pitchOrig ? pitchOrig : window.pitchEditor.pitchNew
    lengthsOrig = lengthsOrig ? lengthsOrig : window.pitchEditor.dursNew

    if (isFreshRegen) {
        window.pitchEditor.letterFocus = []
        pace_slid.value = pace
    }

    window.pitchEditor.letters = letters
    window.pitchEditor.pitchNew = pitchOrig.map(p=>p)
    window.pitchEditor.dursNew = lengthsOrig.map(v=>v)

    sliders = []
    letterElems = []
    autoinfer_timer = null
    has_been_changed = false
    css_hack_items = []
    elemsWidths = []

    letters.forEach((letter, l) => {

        const letterLabel = createElem("div.letterElem", letter)
        const letterDiv = createElem("div.letter", letterLabel)
        const slider = createElem(`input.slider#slider_${l}`, {
            type: "range",
            orient: "vertical",
            step: 0.01,
            min: -3,
            max:  3,
            value: pitchOrig[l]
        })
        sliders.push(slider)
        letterDiv.appendChild(slider)

        letterLabel.addEventListener("click", event => setLetterFocus(l, event.ctrlKey))
        let multiLetterPitchDelta = undefined
        let multiLetterStartPitchVals = []
        slider.addEventListener("mousedown", () => {
            if (window.pitchEditor.letterFocus.length <= 1 || (!event.ctrlKey && !window.pitchEditor.letterFocus.includes(l))) {
                setLetterFocus(l)
            }

            if (window.pitchEditor.letterFocus.length>1) {
                multiLetterPitchDelta = slider.value
                multiLetterStartPitchVals = sliders.map(slider => parseFloat(slider.value))
            }

            // Tooltip
            if (window.userSettings.sliderTooltip) {
                const sliderRect = slider.getClientRects()[0]
                editorTooltip.style.display = "flex"
                const tooltipRect = editorTooltip.getClientRects()[0]
                editorTooltip.style.left = `${parseInt(sliderRect.left)-parseInt(tooltipRect.width) - 15}px`
                editorTooltip.style.top = `${parseInt(sliderRect.top)+parseInt(sliderRect.height/2) - parseInt(tooltipRect.height*0.75)}px`
                editorTooltip.innerHTML = slider.value
            }
        })
        slider.addEventListener("mouseup", () => editorTooltip.style.display = "none")
        slider.addEventListener("input", () => {
            editorTooltip.innerHTML = slider.value

            if (window.pitchEditor.letterFocus.length>1) {
                window.pitchEditor.letterFocus.forEach(li => {
                    if (li!=l) {
                        sliders[li].value = multiLetterStartPitchVals[li]+(slider.value-multiLetterPitchDelta)
                    }
                    window.pitchEditor.pitchNew[li] = parseFloat(sliders[li].value)
                })
            } else if (window.pitchEditor.letterFocus.length==1) {
                letterPitchNumb.value = slider.value
            }
        })


        slider.addEventListener("change", () => {
            if (window.pitchEditor.letterFocus.length==1) {
                window.pitchEditor.pitchNew[l] = parseFloat(slider.value)
                letterPitchNumb.value = slider.value
            }
            has_been_changed = true
            if (autoplay_ckbx.checked) {
                generateVoiceButton.click()
            }
            editorTooltip.style.display = "none"
        })

        if (window.pitchEditor.letterFocus.includes(l)) {
            letterDiv.style.color = "red"
        }


        let length = window.pitchEditor.dursNew[l] * pace_slid.value * 10 + 50

        letterDiv.style.width = `${parseInt(length/2)}px`
        slider.style.height = `${length}px`

        slider.style.marginLeft = `${-100}px`
        letterDiv.style.paddingLeft = `${parseInt(length/2)}px`

        const css_hack_elem = createElem("style", `#slider_${l}::-webkit-slider-thumb {height: ${length}px;}`)
        css_hack_items.push(css_hack_elem)
        css_hack_pitch_editor.appendChild(css_hack_elem)
        elemsWidths.push(length)
        editor.style.width = `${parseInt(elemsWidths.reduce((p,c)=>p+c,1)*1.15)}px`

        editor.appendChild(letterDiv)
        letterElems.push(letterDiv)

        set_letter_display(letterDiv, l, length, pitchOrig[l])
    })
}

// Un-select letters when clicking anywhere else
right.addEventListener("click", event => {
    if (event.target.nodeName=="BUTTON" || event.target.nodeName=="INPUT" || event.target.nodeName=="SVG" || event.target.nodeName=="IMG" || event.target.nodeName=="path" ||
        ["letterElem", "infoContainer"].includes(event.target.className)) {
        return
    }

    window.pitchEditor.letterFocus = []
    letterElems.forEach((letterDiv, l) => {
        letterDiv.style.color = "black"
    })
})

letterPitchNumb.addEventListener("input", () => {
    const lpnValue = parseFloat(letterPitchNumb.value) || 0
    if (window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]!=lpnValue) {
        has_been_changed = true
    }
    window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]] = lpnValue
    sliders[window.pitchEditor.letterFocus[0]].value = letterPitchNumb.value
    if (autoplay_ckbx.checked) {
        generateVoiceButton.click()
    }
})
letterPitchNumb.addEventListener("change", () => {
    const lpnValue = parseFloat(letterPitchNumb.value) || 0
    if (window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]!=lpnValue) {
        has_been_changed = true
    }
    window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]] = lpnValue
    sliders[window.pitchEditor.letterFocus[0]].value = letterPitchNumb.value
    if (autoplay_ckbx.checked) {
        generateVoiceButton.click()
    }
})

resetLetter_btn.addEventListener("click", () => {
    if (window.pitchEditor.letterFocus.length==0) {
        return
    }

    window.pitchEditor.letterFocus.forEach(l => {
        if (window.pitchEditor.dursNew[l] != window.pitchEditor.resetDurs[l]) {
            has_been_changed = true
        }
        window.pitchEditor.dursNew[l] = window.pitchEditor.resetDurs[l]
        window.pitchEditor.pitchNew[l] = window.pitchEditor.resetPitch[l]
        set_letter_display(letterElems[l], l, window.pitchEditor.resetDurs[l]* pace_slid.value*10+50, window.pitchEditor.pitchNew[l])
    })

    if (window.pitchEditor.letterFocus.length==1) {
        letterLength.value = parseFloat(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterLengthNumb.value = parseFloat(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
})
const updateLetterLengthFromInput = () => {
    if (window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]] != letterLength.value) {
        has_been_changed = true
    }
    window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]] = parseFloat(letterLength.value)

    window.pitchEditor.letterFocus.forEach(l => {
        const letterElem = letterElems[l]
        const newWidth = window.pitchEditor.dursNew[l] * pace_slid.value //* 100
        set_letter_display(letterElem, l, newWidth * 10 + 50)
    })
}
let multiLetterLengthDelta = undefined
let multiLetterStartLengthVals = []
letterLength.addEventListener("mousedown", () => {
    if (window.pitchEditor.letterFocus.length>1) {
        multiLetterLengthDelta = letterLength.value
        multiLetterStartLengthVals = window.pitchEditor.dursNew.map(v=>v)
    }
})
letterLength.addEventListener("input", () => {
    if (window.pitchEditor.letterFocus.length>1) {
        window.pitchEditor.letterFocus.forEach(li => {
            window.pitchEditor.dursNew[li] = multiLetterStartLengthVals[li]+(parseFloat(letterLength.value)-multiLetterLengthDelta)
        })
        updateLetterLengthFromInput()
        return
    }

    letterLengthNumb.value = letterLength.value
    updateLetterLengthFromInput()

    // Tooltip
    if (window.userSettings.sliderTooltip) {
        const sliderRect = letterLength.getClientRects()[0]
        editorTooltip.style.display = "flex"
        const tooltipRect = editorTooltip.getClientRects()[0]

        editorTooltip.style.left = `${parseInt(sliderRect.left)+parseInt(sliderRect.width/2) - parseInt(tooltipRect.width*0.75)}px`
        editorTooltip.style.top = `${parseInt(sliderRect.top)-parseInt(tooltipRect.height) - 15}px`
        editorTooltip.innerHTML = letterLength.value
    }
})
letterLength.addEventListener("mouseup", () => {
    if (has_been_changed) {
        if (autoinfer_timer != null) {
            clearTimeout(autoinfer_timer)
            autoinfer_timer = null
        }
        if (autoplay_ckbx.checked) {
            autoinfer_timer = setTimeout(infer, 500)
        }
    }
    editorTooltip.style.display = "none"
})
letterLengthNumb.addEventListener("input", () => {
    letterLength.value = letterLengthNumb.value
    updateLetterLengthFromInput()
})
letterLengthNumb.addEventListener("change", () => {
    letterLength.value = letterLengthNumb.value
    updateLetterLengthFromInput()
})

// Reset button
reset_btn.addEventListener("click", () => {
    window.pitchEditor.dursNew = window.pitchEditor.resetDurs
    window.pitchEditor.pitchNew = window.pitchEditor.resetPitch.map(p=>p)
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, window.pitchEditor.resetDurs[l]*10+50, window.pitchEditor.pitchNew[l]))
    letterLength.value = parseFloat(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
    if (window.pitchEditor.letterFocus.length==1) {
        letterLengthNumb.value = parseFloat(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
    pace_slid.value = 1
})
amplify_btn.addEventListener("click", () => {
    window.pitchEditor.ampFlatCounter += 1
    window.pitchEditor.pitchNew = window.pitchEditor.resetPitch.map((p, pi) => {
        if (window.pitchEditor.letterFocus.length>1 && window.pitchEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        const newVal = p*(1+window.pitchEditor.ampFlatCounter*0.025)
        return newVal>0 ? Math.min(3, newVal) : Math.max(-3, newVal)
    })
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    if (window.pitchEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
})
flatten_btn.addEventListener("click", () => {
    window.pitchEditor.ampFlatCounter -= 1
    window.pitchEditor.pitchNew = window.pitchEditor.resetPitch.map((p,pi) => {
        if (window.pitchEditor.letterFocus.length>1 && window.pitchEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        return p*Math.max(0, 1+window.pitchEditor.ampFlatCounter*0.025)
    })
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    if (window.pitchEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
})
increase_btn.addEventListener("click", () => {
    window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map((p,pi) => {
        if (window.pitchEditor.letterFocus.length>1 && window.pitchEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        return p+0.1
    })
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    if (window.pitchEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
})
decrease_btn.addEventListener("click", () => {
    window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map((p,pi) => {
        if (window.pitchEditor.letterFocus.length>1 && window.pitchEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        return p-0.1
    })
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    if (window.pitchEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseFloat(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
    }
})
pace_slid.addEventListener("change", () => {
    editorTooltip.style.display = "none"
    if (autoplay_ckbx.checked) {
        generateVoiceButton.click()
    }
})

pace_slid.addEventListener("input", () => {
    const new_lengths = window.pitchEditor.dursNew.map((v,l) => v * pace_slid.value)
    window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, new_lengths[l]* 10 + 50, null))

    // Tooltip
    if (window.userSettings.sliderTooltip) {
        const sliderRect = pace_slid.getClientRects()[0]
        editorTooltip.style.display = "flex"
        const tooltipRect = editorTooltip.getClientRects()[0]

        editorTooltip.style.left = `${parseInt(sliderRect.left)+parseInt(sliderRect.width/2) - parseInt(tooltipRect.width*0.75)}px`
        editorTooltip.style.top = `${parseInt(sliderRect.top)-parseInt(tooltipRect.height) - 15}px`
        editorTooltip.innerHTML = pace_slid.value
    }
})
autoplay_ckbx.addEventListener("change", () => {
    window.userSettings.autoplay = autoplay_ckbx.checked
    saveUserSettings()
})

vocoder_select.value = window.userSettings.vocoder.includes(".hg.") ? "qnd" : window.userSettings.vocoder
const changeVocoder = vocoder => {
    window.userSettings.vocoder = vocoder
    window.batch_state.lastVocoder = vocoder
    spinnerModal("Changing models...")
    fetch(`http://localhost:8008/setVocoder`, {
        method: "Post",
        body: JSON.stringify({vocoder})
    }).then(() => {
        closeModal().then(() => {
            saveUserSettings()
        })
    })
}
vocoder_select.addEventListener("change", () => changeVocoder(vocoder_select.value))



// Keyboard actions
// ================
window.addEventListener("keydown", event => {

    // The Enter key to submit text input prompts in modals
    if (event.key=="Enter" && modalContainer.style.display!="none" && event.target.tagName=="INPUT") {
        activeModal.querySelector("button").click()
    }

    // Disable keyboard controls while in a text input
    if (event.target.tagName=="INPUT" && event.target.tagName!=dialogueInput) {
        return
    }

    if (event.target==dialogueInput || event.target==letterPitchNumb || event.target==letterLengthNumb) {
        // Enter: Generate sample
        if (event.key=="Enter") {
            generateVoiceButton.click()
            event.preventDefault()
        }
        return
    }

    const key = event.key.toLowerCase()

    // Escape: close modals
    if (key=="escape") {
        closeModal()
    }
    // Space: bring focus to the input textarea
    if (key==" ") {
        setTimeout(() => dialogueInput.focus(), 0)
    }
    // CTRL-S: Keep sample
    if (key=="s" && event.ctrlKey && !event.shiftKey) {
        keepSampleFunction(false)
    }
    // CTRL-SHIFT-S: Keep sample (but with rename prompt)
    if (key=="s" && event.ctrlKey && event.shiftKey) {
        keepSampleFunction(true)
    }
    // Y/N for prompt modals
    if (key=="y" || key=="n") {
        if (document.querySelector("#activeModal")) {
            const buttons = Array.from(document.querySelector("#activeModal").querySelectorAll("button"))
            const yesBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="yes")
            const noBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="no")
            if (key=="y") yesBtn.click()
            if (key=="n") noBtn.click()
        }
    }
    // Left/Right arrows: Move between letter focused (clears multi-select, picks the min(0,first-1) if left, max($L, last+1) if right)
    // SHIFT-Left/Right: multi-letter create selection range
    if ((key=="arrowleft" || key=="arrowright") && !event.ctrlKey) {
        event.preventDefault()
        if (window.pitchEditor.letterFocus.length==0) {
            setLetterFocus(0)
        } else if (window.pitchEditor.letterFocus.length==1) {
            const curr_l = window.pitchEditor.letterFocus[0]
            const newIndex = key=="arrowleft" ? Math.max(0, curr_l-1) : Math.min(curr_l+1, window.pitchEditor.letters.length-1)
            setLetterFocus(newIndex, event.shiftKey)
        } else {
            if (key=="arrowleft") {
                setLetterFocus(Math.max(0, Math.min(...window.pitchEditor.letterFocus)-1), event.shiftKey)
            } else {
                setLetterFocus(Math.min(Math.max(...window.pitchEditor.letterFocus)+1, window.pitchEditor.letters.length-1), event.shiftKey)
            }
        }
    }

    // Up/Down arrows: Move pitch up/down for the letter(s) selected
    if ((key=="arrowup" || key=="arrowdown") && !event.ctrlKey) {
        event.preventDefault()
        if (window.pitchEditor.letterFocus.length) {
            window.pitchEditor.letterFocus.forEach(li => {
                sliders[li].value = parseFloat(sliders[li].value) + (key=="arrowup" ? 0.1 : -0.1)
                window.pitchEditor.pitchNew[li] = parseFloat(sliders[li].value)
                has_been_changed = true
            })
            if (autoinfer_timer != null) {
                clearTimeout(autoinfer_timer)
                autoinfer_timer = null
            }
            if (autoplay_ckbx.checked) {
                autoinfer_timer = setTimeout(infer, 500)
            }

            if (window.pitchEditor.letterFocus.length==1) {
                letterPitchNumb.value = sliders[window.pitchEditor.letterFocus[0]].value
            }
        }
    }

    // CTRL+Left/Right arrows: change the sequence-wide pacing
    if ((key=="arrowleft" || key=="arrowright") && event.ctrlKey) {
        pace_slid.value = parseFloat(pace_slid.value) + (key=="arrowleft"? -0.01 : 0.01)
        const new_lengths = window.pitchEditor.dursNew.map((v,l) => v * pace_slid.value)
        window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, new_lengths[l]* 10 + 50, null))
        // if (autoinfer_timer != null) {
        //     clearTimeout(autoinfer_timer)
        //     autoinfer_timer = null
        // }
        // if (autoplay_ckbx.checked) {
        //     autoinfer_timer = setTimeout(infer, 500)
        // }
    }

    // CTRL+Up/Down arrows: increase/decrease buttons
    if (key=="arrowup" && event.ctrlKey && !event.shiftKey) {
        increase_btn.click()
    }
    if (key=="arrowdown" && event.ctrlKey && !event.shiftKey) {
        decrease_btn.click()
    }
    // CTRL+SHIFT+Up/Down arrows: amplify/flatten buttons
    if (key=="arrowup" && event.ctrlKey && event.shiftKey) {
        amplify_btn.click()
    }
    if (key=="arrowdown" && event.ctrlKey && event.shiftKey) {
        flatten_btn.click()
    }
})

window.setupModal = (openingButton, modalContainerElem, callback) => {
    openingButton.addEventListener("click", () => {
        if (callback) {
            callback()
        }
        closeModal(undefined, modalContainerElem).then(() => {
            modalContainerElem.style.opacity = 0
            modalContainerElem.style.display = "flex"
            // chrome.style.opacity = 0.88
            requestAnimationFrame(() => requestAnimationFrame(() => modalContainerElem.style.opacity = 1))
            requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
        })
    })
    modalContainerElem.addEventListener("click", event => {
        if (event.target==modalContainerElem) {
            window.closeModal(modalContainerElem)
        }
    })
}


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
fetch("http://danruta.co.uk/patreon.txt").then(r=>r.text()).then(data => fs.writeFileSync(`${path}/patreon.txt`, data, "utf8"))


// Updates
// =======
app_version.innerHTML = window.appVersion
updatesVersions.innerHTML = `This app version: ${window.appVersion}`

const checkForUpdates = () => {
    fetch("http://danruta.co.uk/xvasynth_updates.txt").then(r=>r.json()).then(data => {
        fs.writeFileSync(`${path}/updates.json`, JSON.stringify(data), "utf8")
        checkUpdates.innerHTML = "Check for updates now"
        showUpdates()
    }).catch(() => {
        checkUpdates.innerHTML = "Can't reach server"
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
        updatesVersions.innerHTML = `This app version: ${appVersion}. Available: ${sortedLogVersions[sortedLogVersions.length-1]}`
        console.log(`Update available: This: ${appVersion}, available: ${sortedLogVersions[sortedLogVersions.length-1]}`)
    } else {
        updatesVersions.innerHTML = `This app version: ${appVersion}. Up-to-date.`
        console.log("App is up-to-date")
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
    checkUpdates.innerHTML = "Checking for updates..."
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

fs.readdir(`${path}/assets`, (err, fileNames) => {

    let totalVoices = 0
    let totalGames = new Set()

    const itemsToSort = []

    fileNames.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png"))&&fn.split("-").length==4).forEach(fileName => {
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

        gameSelectionContent.appendChild(createElem("div", `${numVoices} voice${(numVoices>1||numVoices==0)?"s":""}`))
        gameSelectionContent.appendChild(createElem("div", gameName))

        gameSelection.appendChild(gameSelectionContent)

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

    searchGameInput.placeholder = `Search ${Array.from(totalGames).length} games with ${totalVoices} voices...`
})






// Plugins
// =======
window.setupModal(pluginsIcon, pluginsContainer)


// Other
// =====
if (fs.existsSync(`${path}/models/nvidia_waveglowpyt_fp32_20190427.pt`)) {
    loadAllModels().then(() => {
        // Load the last selected game
        const lastGame = localStorage.getItem("lastGame")

        if (lastGame) {
            changeGame(lastGame)
        }
    })
} else {
    console.log("No Waveglow")
    setTimeout(() => {
        window.errorModal("WaveGlow model not found. Download it also (separate download), and place the .pt file in the models folder.")
    }, 1500)
}

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
    event.preventDefault();
    shell.openExternal(a.href);
}))
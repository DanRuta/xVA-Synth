"use strict"

const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "./resources/app" : "."

const fs = require("fs")
const {shell, ipcRenderer} = require("electron")
const fetch = require("node-fetch")
const {text_to_sequence, english_cleaners} = require("./text.js")
const {xVAAppLogger} = require("./appLogger.js")
const {saveUserSettings} = require("./settingsMenu.js")

let themeColour
window.appVersion = "v1.2.0"
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
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

const loadAllModels = () => {
    return new Promise(resolve => {
        fs.readdir(`${path}/models`, (err, gameDirs) => {
            gameDirs.filter(name => !name.includes(".")).forEach(gameFolder => {

                const files = fs.readdirSync(`${path}/models/${gameFolder}`).filter(f => f.endsWith(".json"))

                if (!files.length) {
                    return
                }

                files.forEach(fileName => {

                    if (!models.hasOwnProperty(`${gameFolder}/${fileName}`)) {

                        models[`${gameFolder}/${fileName}`] = null

                        const model = JSON.parse(fs.readFileSync(`${path}/models/${gameFolder}/${fileName}`, "utf8"))
                        model.games.forEach(({gameId, voiceId, voiceName, voiceDescription, gender}) => {

                            if (!games.hasOwnProperty(gameId)) {

                                const gameAsset = fs.readdirSync(`${path}/assets`).find(f => f.startsWith(gameId))
                                const option = document.createElement("option")
                                option.value = gameAsset
                                option.innerHTML = gameAsset.split("-").reverse()[0].split(".")[0]
                                games[gameId] = {
                                    models: [],
                                    gameAsset
                                }

                                // Insert the dropdown option, in alphabetical order, except for Other
                                const existingOptions = Array.from(gameDropdown.childNodes)

                                if (existingOptions.length && option.innerHTML!="Other") {
                                    const afterElement = existingOptions.find(el => el.text>option.innerHTML || el.text=="Other")
                                    gameDropdown.insertBefore(option, afterElement)
                                } else {
                                    gameDropdown.appendChild(option)
                                }
                            }

                            const audioPreviewPath = `${gameFolder}/${model.games.find(({gameId}) => gameId==gameFolder).voiceId}`
                            const existingDuplicates = []
                            games[gameId].models.forEach((item,i) => {
                                if (item.voiceId==voiceId) {
                                    existingDuplicates.push([item, i])
                                }
                            })

                            const modelData = {model, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, gender, modelVersion: model.modelVersion, hifi: undefined}
                            const potentialHiFiPath = `models/${audioPreviewPath}.hg.pt`
                            if (fs.existsSync(potentialHiFiPath)) {
                                modelData.hifi = potentialHiFiPath
                            }

                            if (existingDuplicates.length) {
                                if (existingDuplicates[0][0].modelVersion<model.modelVersion) {
                                    games[gameId].models.splice(existingDuplicates[0][1], 1)
                                    games[gameId].models.push(modelData)
                                }
                            } else {
                                games[gameId].models.push(modelData)
                            }
                        })
                    }
                })
            })

            resolve()
        })
    })
}

// Change game
const changeGame = () => {

    const meta = gameDropdown.value.split("-")
    themeColour = meta[1]
    generateVoiceButton.disabled = true
    generateVoiceButton.innerHTML = "Generate Voice"

    // Change the app title
    if (meta[2]) {
        document.title = `${meta[2]}VA Synth`
        dragBar.innerHTML = `${meta[2]}VA Synth`
    } else {
        document.title = `xVA Synth`
        dragBar.innerHTML = `xVA Synth`
    }

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
    `

    try {fs.mkdirSync(`${path}/output/${meta[0]}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", gameDropdown.value)

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""
    title.innerHTML = "Select Voice Type"

    // No models found
    if (!Object.keys(games).length) {
        title.innerHTML = "No models found"
        return
    }

    const buttons = []

    games[meta[0]].models.forEach(({model, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription, hifi}) => {

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
        button.dataset.modelId = voiceId

        // Quick voice set preview, if there is a preview file
        button.addEventListener("contextmenu", () => {
            window.appLogger.log(`${path}/models/${audioPreviewPath}.wav`)
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: `./models/${audioPreviewPath}.wav`
            }))
        })

        button.addEventListener("click", () => {

            if (hifi) {
                bespoke_hifi_bolt.style.opacity = 1
                const option = createElem("option", "Bespoke HiFi GAN")
                option.value = `${gameId}/${voiceId}.hg.pt`
                vocoder_select.appendChild(option)
            } else {
                bespoke_hifi_bolt.style.opacity = 0
                // Remove the bespoke hifi option if there was one already there
                Array.from(vocoder_select.children).forEach(opt => {
                    if (opt.innerHTML=="Bespoke HiFi GAN") {
                        vocoder_select.removeChild(opt)
                    }
                })
                vocoder_select.value = "qnd"
                changeVocoder("qnd")
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
                const modelFileName = audioPreviewPath.split("/")[1].split(".wav")[0]

                generateVoiceButton.dataset.modelQuery = JSON.stringify({
                    outputs: parseInt(model.outputs),
                    model: `${path}/models/${modelGameFolder}/${modelFileName}`,
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
    const openFileLocationButton = createElem("div", "&#10064;")
    openFileLocationButton.addEventListener("click", () => {
        console.log("open dir", src)
        shell.showItemInFolder(src)
    })

    if (fs.existsSync(`${src}.json`)) {
        const editButton = createElem("div", `<svg class="renameSVG" version="1.0" xmlns="http:\/\/www.w3.org/2000/svg" width="344.000000pt" height="344.000000pt" viewBox="0 0 344.000000 344.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,344.000000) scale(0.100000,-0.100000)" fill="#555555" stroke="none"><path d="M1489 2353 l-936 -938 -197 -623 c-109 -343 -195 -626 -192 -629 2 -3 284 84 626 193 l621 198 937 938 c889 891 937 940 934 971 -11 108 -86 289 -167 403 -157 219 -395 371 -655 418 l-34 6 -937 -937z m1103 671 c135 -45 253 -135 337 -257 41 -61 96 -178 112 -241 l12 -48 -129 -129 -129 -129 -287 287 -288 288 127 127 c79 79 135 128 148 128 11 0 55 -12 97 -26z m-1798 -1783 c174 -79 354 -248 436 -409 59 -116 72 -104 -213 -196 l-248 -80 -104 104 c-58 58 -105 109 -105 115 0 23 154 495 162 495 5 0 37 -13 72 -29z"/></g></svg>`)
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

    const renameButton = createElem("div", `<svg class="renameSVG" version="1.0" xmlns="http://www.w3.org/2000/svg" width="166.000000pt" height="336.000000pt" viewBox="0 0 166.000000 336.000000" preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,336.000000) scale(0.100000,-0.100000)" fill="#000000" stroke="none"> <path d="M165 3175 c-30 -31 -35 -42 -35 -84 0 -34 6 -56 21 -75 42 -53 58 -56 324 -56 l245 0 0 -1290 0 -1290 -245 0 c-266 0 -282 -3 -324 -56 -15 -19 -21 -41 -21 -75 0 -42 5 -53 35 -84 l36 -35 281 0 280 0 41 40 c30 30 42 38 48 28 5 -7 9 -16 9 -21 0 -4 15 -16 33 -27 30 -19 51 -20 319 -20 l287 0 36 35 c30 31 35 42 35 84 0 34 -6 56 -21 75 -42 53 -58 56 -324 56 l-245 0 0 1290 0 1290 245 0 c266 0 282 3 324 56 15 19 21 41 21 75 0 42 -5 53 -35 84 l-36 35 -287 0 c-268 0 -289 -1 -319 -20 -18 -11 -33 -23 -33 -27 0 -5 -4 -14 -9 -21 -6 -10 -18 -2 -48 28 l-41 40 -280 0 -281 0 -36 -35z"/></g></svg>`)

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

    const deleteFileButton = createElem("div", "&#10060;")
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

    const game = gameDropdown.value.split("-")[0]

    try {fs.mkdirSync(window.userSettings[`outpath_${game}`])} catch (e) {/*Do nothing*/}
    try {fs.mkdirSync(`${window.userSettings[`outpath_${game}`]}/${voiceId}`)} catch (e) {/*Do nothing*/}


    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {

        window.appLogger.log(`Loading voice set: ${JSON.parse(generateVoiceButton.dataset.modelQuery).model}`)

        spinnerModal("Loading voice set<br>(may take a minute... but not much more!)")
        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            closeModal().then(() => {
                generateVoiceButton.dataset.modelQuery = null
                generateVoiceButton.innerHTML = "Generate Voice"
                generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad
            })
        }).catch(e => {
            console.log(e)
            if (e.code =="ENOENT") {
                closeModal().then(() => {
                    createModal("error", "There was an issue connecting to the python server.<br><br>Try again in a few seconds. If the issue persists, make sure localhost port 8008 is free, or send the server.log file to me on GitHub or Nexus.")
                })
            }
        })
    } else {

        if (isGenerating) {
            return
        }
        isGenerating = true

        const sequence = dialogueInput.value.trim()
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

        if (editor.innerHTML && editor.innerHTML.length && window.pitchEditor.sequence && sequence==window.pitchEditor.inputSequence
            && generateVoiceButton.dataset.modelIDLoaded==window.pitchEditor.currentVoice) {
            pitch = window.pitchEditor.pitchNew
            duration = window.pitchEditor.dursNew.map(v => v*pace_slid.value)
            isFreshRegen = false
        }
        window.pitchEditor.currentVoice = generateVoiceButton.dataset.modelIDLoaded

        const speaker_i = window.currentModel.games[0].emb_i

        window.appLogger.log(`Synthesising audio: ${sequence}`)
        fetch(`http://localhost:8008/synthesize`, {
            method: "Post",
            body: JSON.stringify({
                sequence, pitch, duration, speaker_i,
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
            durationsData = durationsData.split(",").map(v => parseFloat(v)/pace_slid.value)
            window.pitchEditor.inputSequence = sequence
            window.pitchEditor.sequence = cleanedSequence

            if (pitch.length==0 || isFreshRegen) {
                window.pitchEditor.ampFlatCounter = 0
                window.pitchEditor.resetPitch = pitchData
                window.pitchEditor.resetDurs = durationsData
            }

            setPitchEditorValues(cleanedSequence.replace(/\s/g, "_").split(""), pitchData, durationsData, isFreshRegen)

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

    if (window.userSettings.audio.ffmpeg) {
        spinnerModal("Saving the audio file...")
        const options = {
            hz: window.userSettings.audio.hz,
            padStart: window.userSettings.audio.padStart,
            padEnd: window.userSettings.audio.padEnd,
            bit_depth: window.userSettings.audio.bitdepth,
            amplitude: window.userSettings.audio.amplitude
        }

        window.appLogger.log(`About to save file from ${from} to ${to} with options: ${JSON.stringify(options)}`)
        fetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body: JSON.stringify({
                input_path: from,
                output_path: to,
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            closeModal().then(() => {
                if (res.length) {
                    console.log("res", res)
                    window.errorModal(`Something went wrong<br><br>Input: ${from}<br>Output: ${to}<br><br>${res}`)
                } else {
                    fs.writeFileSync(`${to}.json`, JSON.stringify({inputSequence: dialogueInput.value.trim(), pitchEditor: window.pitchEditor, pacing: parseFloat(pace_slid.value)}, null, 4))
                    voiceSamples.appendChild(makeSample(to, true))
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
            fs.writeFileSync(`${to}.json`, JSON.stringify({inputSequence: dialogueInput.value.trim(), pitchEditor: window.pitchEditor, pacing: parseFloat(pace_slid.value)}, null, 4))
            voiceSamples.appendChild(makeSample(to, true))
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

                if (fs.existsSync(outDir)) {
                    const existingFiles = fs.readdirSync(outDir.reverse().join("/"))
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


gameDropdown.addEventListener("change", changeGame)


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
                closeModal().then(() => {
                    resolve(true)
                })
            })
            noButton.addEventListener("click", () => {
                closeModal().then(() => {
                    resolve(false)
                })
            })
        } else if (type=="error") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = "Close"
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal().then(() => {
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
                closeModal().then(() => {
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
window.closeModal = (container=modalContainer) => {
    return new Promise(resolve => {
        updatesContainer.style.opacity = 0
        infoContainer.style.opacity = 0
        settingsContainer.style.opacity = 0
        container.style.opacity = 0
        chrome.style.opacity = 0.88
        setTimeout(() => {
            updatesContainer.style.display = "none"
            infoContainer.style.display = "none"
            settingsContainer.style.display = "none"
            container.style.display = "none"
            try {
                activeModal.remove()
            } catch (e) {}
            resolve()
        }, 300)
    })
}

let startingSplashInterval
let loadingStage = 0
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
        letterLength.value = parseInt(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
const setPitchEditorValues = (letters, pitchOrig, lengthsOrig, isFreshRegen) => {

    Array.from(editor.children).forEach(child => editor.removeChild(child))

    letters = letters ? letters : window.pitchEditor.letters
    pitchOrig = pitchOrig ? pitchOrig : window.pitchEditor.pitchNew
    lengthsOrig = lengthsOrig ? lengthsOrig : window.pitchEditor.dursNew

    if (isFreshRegen) {
        window.pitchEditor.letterFocus = []
        pace_slid.value = 1
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

        const letterLabel = createElem("div", letter)
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
            if (window.pitchEditor.letterFocus.length <= 1) {
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
        set_letter_display(letterElems[l], l, window.pitchEditor.resetDurs[l]*10+50, window.pitchEditor.pitchNew[l])
    })

    if (window.pitchEditor.letterFocus.length==1) {
        letterLength.value = parseInt(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterLengthNumb.value = parseInt(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
    letterLength.value = parseInt(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
    if (window.pitchEditor.letterFocus.length==1) {
        letterLengthNumb.value = parseInt(window.pitchEditor.dursNew[window.pitchEditor.letterFocus[0]])
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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
        letterPitchNumb.value = parseInt(window.pitchEditor.pitchNew[window.pitchEditor.letterFocus[0]]*1000)/1000
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

vocoder_select.value = window.userSettings.vocoder
const changeVocoder = vocoder => {
    window.userSettings.vocoder = vocoder
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

infoIcon.addEventListener("click", () => {
    closeModal().then(() => {
        infoContainer.style.opacity = 0
        infoContainer.style.display = "flex"
        chrome.style.opacity = 0.88
        requestAnimationFrame(() => requestAnimationFrame(() => infoContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
    })
})
infoContainer.addEventListener("click", event => {
    if (event.target==infoContainer) {
        closeModal(infoContainer)
    }
})

// Patreon
// =======
patreonIcon.addEventListener("click", () => {

    const data = fs.readFileSync(`${path}/patreon.txt`, "utf8")
    const names = new Set()
    data.split("\r\n").forEach(name => names.add(name))
    names.add("minermanb")

    let content = `You can support development on patreon at this link:<br>
        <span id="patreonButton" href="https://www.patreon.com/bePatron?u=48461563" data-patreon-widget-type="become-patron-button">
        <svg style="height: 1rem;width: 1rem;" viewBox="0 0 569 546" xmlns="http://www.w3.org/2000/svg"><g><circle cx="362.589996" fill="#ffffff" cy="204.589996" data-fill="1" id="Oval" r="204.589996"></circle><rect fill="#ffffff" data-fill="2" height="545.799988" id="Rectangle" width="100" x="0" y="0"></rect></g></svg>
        <span style="width:5px"></span>
        Become a Patron!</span>
        <br><hr><br>Special thanks:`
    names.forEach(name => content += `<br>${name}`)

    closeModal().then(() => {
        createModal("error", content)
        patreonButton.addEventListener("click", () => {
            shell.openExternal("https://patreon.com")
        })
    })
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
updatesIcon.addEventListener("click", () => {
    closeModal().then(() => {
        updatesContainer.style.opacity = 0
        updatesContainer.style.display = "flex"
        chrome.style.opacity = 0.88
        requestAnimationFrame(() => requestAnimationFrame(() => updatesContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
    })
})
updatesContainer.addEventListener("click", event => {
    if (event.target==updatesContainer) {
        closeModal(updatesContainer)
    }
})
checkUpdates.addEventListener("click", () => {
    checkUpdates.innerHTML = "Checking for updates..."
    checkForUpdates()
})
showUpdates()


// Settings
// ========
settingsCog.addEventListener("click", () => {
    closeModal().then(() => {
        settingsContainer.style.opacity = 0
        settingsContainer.style.display = "flex"
        chrome.style.opacity = 0.88
        requestAnimationFrame(() => requestAnimationFrame(() => settingsContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
    })
})
settingsContainer.addEventListener("click", event => {
    if (event.target==settingsContainer) {
        window.closeModal(settingsContainer)
    }
})



if (fs.existsSync(`${path}/models/nvidia_waveglowpyt_fp32_20190427.pt`)) {
    loadAllModels().then(() => {
        // Load the last selected game
        const lastGame = localStorage.getItem("lastGame")

        if (lastGame) {
            gameDropdown.value = lastGame
        }
        changeGame()
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
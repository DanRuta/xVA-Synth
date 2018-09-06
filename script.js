"use strict"

const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "./resources/app" : "."

const fs = require("fs")
const {shell, ipcRenderer} = require("electron")
const fetch = require("node-fetch")
const {text_to_sequence, english_cleaners} = require("./text.js")

let themeColour
window.games = {}
window.models = {}

const customWindowSize = localStorage.getItem("customWindowSize")
if (customWindowSize) {
    const [height, width] = customWindowSize.split(",").map(v => parseInt(v))
    ipcRenderer.send("resize", {height, width})
}

// Set up folders
try {fs.mkdirSync(`${path}/models`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/output`)} catch (e) {/*Do nothing*/}
try {fs.mkdirSync(`${path}/assets`)} catch (e) {/*Do nothing*/}

// Clean up temp files
fs.readdir("output", (err, files) => {
    files.filter(f => f.startsWith("temp-")).forEach(file => {
        fs.unlink(`output/${file}`, err => err&&console.log(err))
    })
})

let fileRenameCounter = 0
let fileChangeCounter = 0

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

                        const model = JSON.parse(fs.readFileSync(`models/${gameFolder}/${fileName}`, "utf8"))
                        model.games.forEach(({gameId, voiceId, voiceName, voiceDescription}) => {

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
                            games[gameId].models.push({model, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription})
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
    document.title = `${meta[2]}VA Synth`
    dragBar.innerHTML = `${meta[2]}VA Synth`

    if (meta) {
        const background = `linear-gradient(0deg, grey 0, rgba(0,0,0,0)), url("assets/${meta.join("-")}"), grey`
        right.style.background = background
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.background = `#${themeColour}`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.background = `#${themeColour}`)
        Array.from(document.querySelectorAll(".spinner")).forEach(e => e.style.borderLeftColor = `#${themeColour}`)
    }

    cssHack.innerHTML = `::selection {
        background: #${themeColour};
    }`

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

    games[meta[0]].models.forEach(({model, audioPreviewPath, gameId, voiceId, voiceName, voiceDescription}) => {

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`
        button.dataset.modelId = voiceId

        // Quick voice set preview, if there is a preview file
        button.addEventListener("contextmenu", () => {
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: `${path}/models/${audioPreviewPath}.wav`
            }))
        })

        button.addEventListener("click", () => {

            if (voiceDescription) {
                description.innerHTML = voiceDescription
                description.className = "withContent"
            } else {
                description.innerHTML = ""
                description.className = ""
            }

            try {fs.mkdirSync(`${path}/output/${meta[0]}/${voiceId}`)} catch (e) {/*Do nothing*/}

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
            fs.readdir(`${path}/output/${meta[0]}/${button.dataset.modelId}`, (err, files) => {

                if (err) return

                files.filter(f => f.endsWith(".wav")).forEach(file => {
                    voiceSamples.appendChild(makeSample(`./output/${meta[0]}/${button.dataset.modelId}/${file}`))
                })
            })
        })
        buttons.push(button)
    })

    buttons.sort((a,b) => a.innerHTML<b.innerHTML?-1:1)
        .forEach(button => voiceTypeContainer.appendChild(button))

}

const makeSample = src => {
    const sample = createElem("div.sample", createElem("div", src.split("/").reverse()[0].split(".wav")[0]))
    const audioControls = createElem("div")
    const audio = createElem("audio", {controls: true}, createElem("source", {
        src: src,
        type: "audio/wav"
    }))
    const openFileLocationButton = createElem("div", "&#10064;")
    openFileLocationButton.addEventListener("click", () => shell.showItemInFolder(`${__dirname}/${src}`))

    const deleteFileButton = createElem("div", "&#10060;")
    deleteFileButton.addEventListener("click", () => {
        confirmModal("Are you sure you'd like to delete this file?").then(confirmation => {
            if (confirmation) {
                fs.unlinkSync(src)
                sample.remove()
            }
        })
    })
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

    if (generateVoiceButton.dataset.modelQuery && generateVoiceButton.dataset.modelQuery!="null") {

        spinnerModal("Loading model<br>(may take a minute...)")
        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            closeModal()
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = "Generate Voice"
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad
        }).catch(e => {
            if (e.code =="ENOENT") {
                closeModal().then(() => {
                    createModal("error", "There was an issue connecting to the python server.<br><br>Try again in a few seconds. If the issue persists, make sure localhost port 8008 is free, or send the server.log file to me on GitHub or Nexus.")
                })
            }
        })
    } else {

        const existingSample = samplePlay.querySelector("audio")
        if (existingSample) {
            existingSample.pause()
        }

        toggleSpinnerButtons()

        const game = gameDropdown.value.split("-")[0]
        const voiceType = title.dataset.modelId

        const sequence = text_to_sequence(dialogueInput.value).join(",")
        const outputFileName = dialogueInput.value.slice(0, 260).replace("?", "")

        try {fs.unlinkSync(localStorage.getItem("tempFileLocation"))} catch (e) {/*Do nothing*/}

        // For some reason, the samplePlay audio element does not update the source when the file name is the same
        const tempFileLocation = `./output/temp-${Math.random().toString().split(".")[1]}.wav`

        fetch(`http://localhost:8008/synthesize`, {
            method: "Post",
            body: JSON.stringify({sequence: sequence, outfile: `${path}/${tempFileLocation.slice(1, tempFileLocation.length)}`})
        }).then(r=>r.text()).then(() => {
            toggleSpinnerButtons()
            keepSampleButton.dataset.newFileLocation = `./output/${game}/${voiceType}/${outputFileName}.wav`
            keepSampleButton.disabled = false
            samplePlay.dataset.tempFileLocation = tempFileLocation
            samplePlay.innerHTML = ""
            const audio = createElem("audio", {controls: true, style: {width:"150px"}},
                    createElem("source", {src: samplePlay.dataset.tempFileLocation, type: "audio/wav"}))
            samplePlay.appendChild(audio)
            audio.load()

            // Persistance across sessions
            localStorage.setItem("tempFileLocation", tempFileLocation)
        })
    }
})

keepSampleButton.addEventListener("click", () => {
    if (keepSampleButton.dataset.newFileLocation) {

        let fromLocation = samplePlay.dataset.tempFileLocation
        let toLocation = keepSampleButton.dataset.newFileLocation

        fromLocation = fromLocation.slice(1, fromLocation.length)
        toLocation = toLocation.slice(1, toLocation.length)

        fs.rename(`${__dirname}/${fromLocation}`, `${__dirname}/${toLocation}`, err => {
            voiceSamples.appendChild(makeSample(keepSampleButton.dataset.newFileLocation))
            keepSampleButton.disabled = true
        })
    }
})


gameDropdown.addEventListener("change", changeGame)

// Watch for new models being added, and load them into the app
fs.watch(`${path}/models`, {recursive: true, persistent: true}, (eventType, filename) => {

    if (eventType=="rename") {
        fileRenameCounter++
    }

    if (eventType=="change") {
        fileChangeCounter++
    }

    if (fileRenameCounter==(fileChangeCounter/3) || (!fileChangeCounter && fileRenameCounter>=4)) {
        setTimeout(() => {
            if (fileRenameCounter==(fileChangeCounter/3) || (!fileChangeCounter && fileRenameCounter>=4)) {
                fileRenameCounter = 0
                fileChangeCounter = 0
                loadAllModels().then(() => {
                    changeGame()
                })
            }
        })
    }

    // Reset the counter when audio preview files throw deletion counters off by one
    setTimeout(() => {
        if (fileRenameCounter==1 && fileChangeCounter==0) {
            fileRenameCounter = 0
        }
    }, 3000)
})

loadAllModels().then(() => {
    // Load the last selected game
    const lastGame = localStorage.getItem("lastGame")

    if (lastGame) {
        gameDropdown.value = lastGame
    }
    changeGame()
})




const createModal = (type, message) => {
    return new Promise(resolve => {
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span", message))
        modal.dataset.type = type

        if (type=="confirm") {
            const yesButton = createElem("button", {style: {background: `#${themeColour}`}})
            yesButton.innerHTML = "Yes"
            const noButton = createElem("button", {style: {background: `#${themeColour}`}})
            noButton.innerHTML = "No"
            modal.appendChild(createElem("div", yesButton, noButton))

            yesButton.addEventListener("click", () => {
                resolve(true)
                closeModal()
            })
            noButton.addEventListener("click", () => {
                resolve(false)
                closeModal()
            })
        } else if (type=="error") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = "Close"
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                resolve(false)
                closeModal()
            })
        } else {
            modal.appendChild(createElem("div.spinner", {style: {borderLeftColor: document.querySelector("button").style.background}}))
        }

        modalContainer.appendChild(modal)
        modalContainer.style.opacity = 0
        modalContainer.style.display = "flex"

        requestAnimationFrame(() => requestAnimationFrame(() => modalContainer.style.opacity = 1))
    })
}
const closeModal = () => {
    return new Promise(resolve => {
        modalContainer.style.opacity = 0
        setTimeout(() => {
            modalContainer.style.display = "none"
            activeModal.remove()
            resolve()
        }, 300)
    })
}

window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))

// Need to put a small delay to keep a user from immediatelly going to load a model, because the
// python server takes a few seconds to boot up, and the request would cause an infinite spinner
if (PRODUCTION) {
    let loadSpinnerClosed = false
    spinnerModal("Loading...")

    // So I'm watching for changes on the log file, which always gets written to, first when the server starts
    fs.watch(`${path}/`, {persistent: true}, (eventType, filename) => {
        if (filename=="server.log" && !loadSpinnerClosed) {
            loadSpinnerClosed = true
            closeModal()
        }
    })

    // Seems to sometimes take too long to detect the file change, so I'm placing a limit,
    // because the server IS up by this point, usually. Leaves some time padding for navigation, also
    setTimeout(() => {
        loadSpinnerClosed = true
        closeModal()
    }, 4000)
}

modalContainer.addEventListener("click", event => {
    if (event.target==modalContainer && activeModal.dataset.type!="spinner") {
        closeModal()
    }
})

dialogueInput.addEventListener("keyup", () => {
    localStorage.setItem("dialogueInput", dialogueInput.value)
})

const dialogueInputCache = localStorage.getItem("dialogueInput")

if (dialogueInputCache) {
    dialogueInput.value = dialogueInputCache
}

window.addEventListener("resize", e => {
    localStorage.setItem("customWindowSize", `${window.innerHeight},${window.innerWidth}`)
})

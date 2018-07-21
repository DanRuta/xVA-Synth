"use strict"

const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "./resources/app" : "."

const fs = require("fs")
const {shell, ipcRenderer} = require("electron")
const fetch = require("node-fetch")
const {text_to_sequence, english_cleaners} = require("./text.js")

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

let willExecute = false

const loadAllModels = () => {
    return new Promise(resolve => {
        fs.readdir(`${path}/models`, (err, gameDirs) => {
            gameDirs.filter(name => !name.includes(".")).forEach(game => {
                const files = fs.readdirSync(`${path}/models/${game}`).filter(f => f.endsWith(".json"))

                if (!files.length) {
                    return
                }

                files.forEach(fileName => {

                    if (!models.hasOwnProperty(`${game}/${fileName}`)) {

                        models[`${game}/${fileName}`] = null

                        if (!games.hasOwnProperty(game)) {
                            const gameAsset = fs.readdirSync(`${path}/assets`).find(f => f.startsWith(game))
                            const option = document.createElement("option")
                            option.value = gameAsset
                            option.innerHTML = gameAsset.split("-").reverse()[0].split(".")[0]
                            gameDropdown.appendChild(option)
                            games[game] = {
                                models: [],
                                gameAsset
                            }
                        }

                        games[game].models.push(`${game}/${fileName}`)
                    }
                })
                resolve()
            })
        })

        setTimeout(() => willExecute=false, 1000)
    })
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

        spinnerModal("Loading model...")
        fetch("http://localhost:8008/loadModel", {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            closeModal()
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = "Generate Voice"
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad
        })
    } else {
        toggleSpinnerButtons()

        const game = gameDropdown.value.split("-")[0]
        const voiceType = title.dataset.modelId

        const sequence = text_to_sequence(dialogueInput.value).join(",")
        const outputFileName = dialogueInput.value.slice(0, 50)

        try {fs.unlinkSync(localStorage.getItem("tempFileLocation"))} catch (e) {/*Do nothing*/}

        // For some reason, the samplePlay audio element does not update the source when the file name is the same
        const tempFileLocation = `./output/temp-${Math.random().toString().split(".")[1]}.wav`

        fetch("http://localhost:8008/synthesize", {
            method: "Post",
            body: JSON.stringify({sequence: sequence, outfile: `${path}/${tempFileLocation.slice(1, tempFileLocation.length)}`})
        }).then(r=>r.text()).then(() => {
            toggleSpinnerButtons()
            keepSampleButton.dataset.newFileLocation = `./output/${game}/${voiceType}/${outputFileName}.wav`
            keepSampleButton.disabled = false
            samplePlay.dataset.tempFileLocation = tempFileLocation
            samplePlay.innerHTML = ""
            const audio = createElem("audio", {controls: true, style: {width:"100px"}},
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

// Change game
const changeGame = () => {

    const meta = gameDropdown.value.split("-")
    generateVoiceButton.disabled = true
    generateVoiceButton.innerHTML = "Generate Voice"

    if (meta) {
        const background = `linear-gradient(0deg, grey 0, rgba(0,0,0,0)), url("assets/${meta.join("-")}"), grey`
        right.style.background = background
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.background = `#${meta[1]}`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.background = `#${meta[1]}`)
        Array.from(document.querySelectorAll(".spinner")).forEach(e => e.style.borderLeftColor = `#${meta[1]}`)
    }

    cssHack.innerHTML = `::selection {
        background: #${meta[1]};
    }`

    try {fs.mkdirSync(`${path}/output/${meta[0]}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", gameDropdown.value)

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""
    title.innerHTML = "Select Voice Type"

    games[meta[0]].models.forEach(model => {

        const modelMeta = JSON.parse(fs.readFileSync(`${path}/models/${model}`))

        const button = createElem("div.voiceType", modelMeta.name)
        button.style.background = `#${meta[1]}`
        button.dataset.modelId = modelMeta.id

        // Quick voice set preview, if there is a preview file
        button.addEventListener("contextmenu", () => {
            if (modelMeta.preview) {
                const audio = createElem("audio", {autoplay: true}, createElem("source", {
                    src: `${path}/models/${model.split("/")[0]}/${modelMeta.preview}`
                }))
                audio.play()
            }
        })

        button.addEventListener("click", () => {

            if (modelMeta.description) {
                description.innerHTML = modelMeta.description
                description.className = "withContent"
            } else {
                description.innerHTML = ""
                description.className = ""
            }

            try {fs.mkdirSync(`${path}/output/${meta[0]}/${modelMeta.id}`)} catch (e) {/*Do nothing*/}

            generateVoiceButton.dataset.modelQuery = null

            if (generateVoiceButton.dataset.modelIDLoaded != modelMeta.id) {
                generateVoiceButton.innerHTML = "Load model"
                generateVoiceButton.dataset.modelQuery = JSON.stringify({
                    outputs: parseInt(modelMeta.outputs),
                    model: `${path}/models/${meta[0]}/${modelMeta.id}`,
                    cmudict: modelMeta.cmudict
                })
                generateVoiceButton.dataset.modelIDToLoad = modelMeta.id
            }
            generateVoiceButton.disabled = false

            title.innerHTML = button.innerHTML
            title.dataset.modelId = modelMeta.id
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
        voiceTypeContainer.appendChild(button)
    })
}
gameDropdown.addEventListener("change", changeGame)


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

// Watch for new models being added, and load them into the app
fs.watch(`${path}/models`, {recursive: true, persistent: true}, (eventType, filename) => {
    if (!willExecute) {
        willExecute = true
        loadAllModels()
    }
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
            const yesButton = createElem("button", "Yes")
            const noButton = createElem("button", "No")
            modal.appendChild(createElem("div", yesButton, noButton))

            yesButton.addEventListener("click", () => {
                resolve(true)
                closeModal()
            })
            noButton.addEventListener("click", () => {
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
    modalContainer.style.opacity = 0
    setTimeout(() => {
        modalContainer.style.display = "none"
        activeModal.remove()
    }, 300)
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
    }, 3000)
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

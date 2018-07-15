"use strict"

const fs = require("fs")
const {shell} = require("electron")
const fetch = require("node-fetch")
const {text_to_sequence, english_cleaners} = require("./text.js")

window.games = {}
window.models = {}
window.text_to_sequence = text_to_sequence
window.english_cleaners = english_cleaners

// Set up folders
try {fs.mkdirSync("models")} catch (e) {/*Do nothing*/}
try {fs.mkdirSync("output")} catch (e) {/*Do nothing*/}
try {fs.mkdirSync("assets")} catch (e) {/*Do nothing*/}

let willExecute = false

const loadAllModels = () => {
    return new Promise(resolve => {
        fs.readdir("./models", (err, gameDirs) => {
            gameDirs.filter(name => !name.includes(".")).forEach(game => {
                const files = fs.readdirSync(`./models/${game}`).filter(f => f.endsWith(".json"))

                if (!files.length) {
                    return
                }

                files.forEach(fileName => {

                    if (!models.hasOwnProperty(`${game}/${fileName}`)) {

                        models[`${game}/${fileName}`] = null

                        if (!games.hasOwnProperty(game)) {
                            const gameAsset = fs.readdirSync("assets").find(f => f.startsWith(game))
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
    toggleSpinnerButtons()

    const game = gameDropdown.value.split("-")[0]
    const voiceType = title.dataset.modelId

    const sequence = text_to_sequence(dialogueInput.value).join(",")
    const outputFileName = dialogueInput.value.slice(0, 30)

    fetch("http://localhost:8008/synthesize", {
        method: "Post",
        body: JSON.stringify({sequence: sequence, outfile: `output/${game}/${voiceType}/${outputFileName}.wav`})
    }).then(r=>r.text()).then(() => {
        toggleSpinnerButtons()
    })
})

samplePlay.addEventListener("click", () => {
    // TODO
    console.log(samplePlay.innerHTML)
    samplePlay.innerHTML = samplePlay.innerHTML=="▶" ? "&#9632;" : "&#9654;"
})

// Change game
const changeGame = () => {

    const meta = gameDropdown.value.split("-")

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

    try {fs.mkdirSync(`output/${meta[0]}`)} catch (e) {/*Do nothing*/}
    localStorage.setItem("lastGame", gameDropdown.value)

    // Populate models
    voiceTypeContainer.innerHTML = ""
    voiceSamples.innerHTML = ""
    title.innerHTML = "Select Voice Type"

    games[meta[0]].models.forEach(model => {

        const modelMeta = JSON.parse(fs.readFileSync(`models/${model}`))

        const button = createElem("div.voiceType", modelMeta.name)
        button.style.background = `#${meta[1]}`
        button.dataset.modelId = modelMeta.id

        button.addEventListener("click", () => {

            title.innerHTML = button.innerHTML
            title.dataset.modelId = modelMeta.id
            keepSampleButton.style.display = "none"
            samplePlay.style.display = "none"

            spinnerModal("Loading model...")
            try {fs.mkdirSync(`output/${meta[0]}/${modelMeta.id}`)} catch (e) {/*Do nothing*/}

            fetch("http://localhost:8008/loadModel", {
                method: "Post",
                body: JSON.stringify({
                    outputs: parseInt(modelMeta.outputs),
                    model: `models/${meta[0]}/${modelMeta.id}`,
                    cmudict: modelMeta.cmudict
                })
            }).then(r=>r.text()).then(res => {
                closeModal()

                // Voice samples
                voiceSamples.innerHTML = ""
                fs.readdir(`output/${meta[0]}/${button.dataset.modelId}`, (err, files) => {

                    if (err) return

                    files.filter(f => f.endsWith(".wav")).forEach(file => {

                        const sample = createElem("div.sample", createElem("div", file.split(".wav")[0]))
                        const audioControls = createElem("div")
                        const audio = createElem("audio", {controls: true}, createElem("source", {
                            src: `output/${meta[0]}/${button.dataset.modelId}/${file}`,
                            type: "audio/wav"
                        }))
                        const openFileLocationButton = createElem("div", "&#10064;")
                        openFileLocationButton.addEventListener("click", () => {
                            const path = `${__dirname}/output/${meta[0]}/${button.dataset.modelId}/${file}`
                            shell.showItemInFolder(path)
                        })
                        const deleteFileButton = createElem("div", "&#10060;")
                        deleteFileButton.addEventListener("click", () => {
                            confirmModal("Are you sure you'd like to delete this file?").then(confirmation => {
                                if (confirmation) {
                                    fs.unlinkSync(`output/${meta[0]}/${button.dataset.modelId}/${file}`)
                                    sample.remove()
                                }
                            })
                        })
                        audioControls.appendChild(audio)
                        audioControls.appendChild(openFileLocationButton)
                        audioControls.appendChild(deleteFileButton)
                        sample.appendChild(audioControls)
                        voiceSamples.appendChild(sample)
                    })
                })
            })
        })
        voiceTypeContainer.appendChild(button)
    })
}
gameDropdown.addEventListener("change", changeGame)


// Watch for new models being added, and load them into the app
fs.watch("./models", {recursive: true, persistent: true}, (eventType, filename) => {
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

modalContainer.addEventListener("click", event => {
    if (event.target==modalContainer && activeModal.dataset.type!="spinner") {
        closeModal()
    }
})
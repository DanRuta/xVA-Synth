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
window.pitchEditor = {resetPitch: null, resetDurs: null, resetDursMult: null}

// Load user settings
window.userSettings = localStorage.getItem("userSettings") || {useGPU: false, customWindowSize:`${window.innerHeight},${window.innerWidth}`}
if ((typeof window.userSettings)=="string") {
    window.userSettings = JSON.parse(window.userSettings)
}
useGPUCbx.checked = window.userSettings.useGPU
const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
ipcRenderer.send("resize", {height, width})
localStorage.setItem("userSettings", JSON.stringify(window.userSettings))



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

                        const model = JSON.parse(fs.readFileSync(`${path}/models/${gameFolder}/${fileName}`, "utf8"))
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
                src: `./models/${audioPreviewPath}.wav`
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
    // }

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
                try {
                    fs.unlinkSync(`${path}${src.slice(1, src.length)}`)
                } catch (e) {}
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

        spinnerModal("Loading voice set<br>(may take a minute...)")
        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: generateVoiceButton.dataset.modelQuery
        }).then(r=>r.text()).then(res => {
            closeModal()
            generateVoiceButton.dataset.modelQuery = null
            generateVoiceButton.innerHTML = "Generate Voice"
            generateVoiceButton.dataset.modelIDLoaded = generateVoiceButton.dataset.modelIDToLoad
        }).catch(e => {
            console.log(e)
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

        // const sequence = text_to_sequence(dialogueInput.value.trim()).join(",")
        const sequence = dialogueInput.value.trim()
        const outputFileName = dialogueInput.value.slice(0, 260).replace("?", "")

        try {fs.unlinkSync(localStorage.getItem("tempFileLocation"))} catch (e) {/*Do nothing*/}

        // For some reason, the samplePlay audio element does not update the source when the file name is the same
        const tempFileLocation = `./output/temp-${Math.random().toString().split(".")[1]}.wav`
        let pitch = []
        let duration = []

        if (editor.innerHTML && editor.innerHTML.length && window.pitchEditor.sequence && sequence.length==window.pitchEditor.sequence.length) {
            pitch = window.pitchEditor.pitchNew
            duration = window.pitchEditor.dursNew
        }

        fetch(`http://localhost:8008/synthesize`, {
            method: "Post",
            body: JSON.stringify({sequence: sequence, pitch, duration, outfile: `${path}/${tempFileLocation.slice(1, tempFileLocation.length)}`})
        }).then(r=>r.text()).then(res => {
            res = res.split("\n")
            let pitchData = res[0]
            let durationsData = res[1]
            pitchData = pitchData.split(",").map(v => parseFloat(v))
            durationsData = durationsData.split(",").map(v => parseFloat(v))
            window.pitchEditor.sequence = sequence

            if (pitch.length==0) {
                window.pitchEditor.resetPitch = pitchData
                window.pitchEditor.resetDurs = durationsData
                window.pitchEditor.resetDursMult = window.pitchEditor.resetDurs.map(v=>1)
            }

            setPitchEditorValues(sequence.replace(/\s/g, "_").split(""), pitchData, durationsData)

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
        }).catch(res => {
            console.log(res)
            window.errorModal(`Something went wrong`)
            toggleSpinnerButtons()
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



let startingSplashInterval
let loadingStage = 0
startingSplashInterval = setInterval(() => {
    if (fs.existsSync("./FASTPITCH_LOADING")) {
        if (loadingStage==0) {
            spinnerModal("Loading...<br>May take a minute<br><br>Building FastPitch model...")
            loadingStage = 1
        }
    } else if (fs.existsSync("./WAVEGLOW_LOADING")) {
        if (loadingStage==1) {
            activeModal.children[0].innerHTML = "Loading...<br>May take a minute<br><br>Loading WaveGlow model..."
            loadingStage = 2
        }
    } else if (fs.existsSync("./SERVER_STARTING")) {
        if (loadingStage==2) {
            activeModal.children[0].innerHTML = "Loading...<br>May take a minute<br><br>Starting up the python backend..."
            loadingStage = 3
        }
    } else {
        closeModal()
        clearInterval(startingSplashInterval)
    }
}, 100)

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
const closeModal = (container=modalContainer) => {
    return new Promise(resolve => {
        container.style.opacity = 0
        setTimeout(() => {
            container.style.display = "none"
            try {
                activeModal.remove()
            } catch (e) {}
            resolve()
        }, 300)
    })
}

window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))
window.errorModal = message => new Promise(resolve => resolve(createModal("error", message)))


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
    window.userSettings.customWindowSize = `${window.innerHeight},${window.innerWidth}`
    localStorage.setItem("userSettings", JSON.stringify(userSettings))
})




const setPitchEditorValues = (letters, pitchOrig, lengthsOrig) => {

    editor.innerHTML = ""

    let lengthsMult = lengthsOrig.map(l => 1)
    let pacingMult = lengthsOrig.map(l => 1)

    window.pitchEditor.pitchNew = pitchOrig.map(p=>p)
    window.pitchEditor.dursNew = lengthsOrig.map(v=>v)


    const letterElems = []
    const css_hack_items = []
    const elemsWidths = []
    let letterFocus = -1
    let autoinfer_timer = null
    let has_been_changed = false

    const set_letter_display = (elem, elem_i, length=null, value=null) => {

        if (length != null) {
            const elem_length = length/2
            elem.style.width = `${parseInt(elem_length/2)}px`
            elem.children[1].style.height = `${elem_length}px`
            // elem.children[1].style.marginTop = `${-parseInt(elem_length/2)+100}px`
            elem.children[1].style.marginTop = `${-parseInt(elem_length/2)+65}px`
            css_hack_items[elem_i].innerHTML = `#slider_${elem_i}::-webkit-slider-thumb {height: ${elem_length}px;}`
            elemsWidths[elem_i] = elem_length
            elem.style.paddingLeft = `${parseInt(elem_length/2)}px`
            editor.style.width = `${parseInt(elemsWidths.reduce((p,c)=>p+c,1)*1.25)}px`
        }

        if (value != null) {
            elem.children[1].value = value
        }
    }


    letters.forEach((letter, l) => {

        const letterDiv = createElem("div.letter", createElem("div", letter))
        const slider = createElem(`input.slider#slider_${l}`, {
            type: "range",
            orient: "vertical",
            step: 0.001,
            min: -5,
            max:  5,
            value: pitchOrig[l]
        })
        letterDiv.appendChild(slider)

        slider.addEventListener("mousedown", () => {
            letterFocus = l
            letterLength.value = parseInt(window.pitchEditor.resetDursMult[letterFocus])
        })

        slider.addEventListener("change", () => {
            window.pitchEditor.pitchNew[l] = parseFloat(slider.value)
            has_been_changed = true
            if (autoplay_ckbx.checked) {
                generateVoiceButton.click()
            }
        })


        let length = lengthsOrig[l] * lengthsMult[l] * 10 + 50

        letterDiv.style.width = `${parseInt(length/2)}px`
        slider.style.height = `${length}px`

        slider.style.marginLeft = `${-83}px`
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


    const infer = () => {
        movingSlider = false
        generateVoiceButton.click()
    }


    let movingSlider = false
    letterLength.addEventListener("mousedown", () => movingSlider=true)
    letterLength.addEventListener("mouseup", () => movingSlider=false)
    letterLength.addEventListener("change", () => movingSlider=false)


    letterLength.addEventListener("mousemove", () => {
        if (letterFocus<0) {
            return
        }

        if (lengthsMult[letterFocus] != letterLength.value) {
            has_been_changed = true
        }
        lengthsMult[letterFocus] = parseFloat(letterLength.value)
        window.pitchEditor.resetDursMult[letterFocus] = parseFloat(letterLength.value)
        lengthsMult.forEach((v,vi) => window.pitchEditor.dursNew[vi] = lengthsOrig[vi]*v)

        const letterElem = letterElems[letterFocus]
        const newWidth = lengthsOrig[letterFocus] * lengthsMult[letterFocus] * pace_slid.value //* 100
        set_letter_display(letterElem, letterFocus, newWidth * 10 + 50)

        if (autoinfer_timer != null) {
            clearTimeout(autoinfer_timer)
            autoinfer_timer = null
        }
        if (autoplay_ckbx.checked) {
            autoinfer_timer = setTimeout(infer, 500)
        }
    })

    // Reset button
    reset_btn.addEventListener("click", () => {
        lengthsMult = lengthsOrig.map(l => 1)
        window.pitchEditor.dursNew = window.pitchEditor.resetDurs
        window.pitchEditor.pitchNew = window.pitchEditor.resetPitch.map(p=>p)
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, window.pitchEditor.resetDurs[l]*10+50, window.pitchEditor.pitchNew[l]))
    })
    amplify_btn.addEventListener("click", () => {
        window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map(p=>p*1.1)
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    })
    flatten_btn.addEventListener("click", () => {
        window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map(p=>p*0.9)
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    })
    increase_btn.addEventListener("click", () => {
        window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map(p=>p+=0.1)
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    })
    decrease_btn.addEventListener("click", () => {
        window.pitchEditor.pitchNew = window.pitchEditor.pitchNew.map(p=>p-=0.1)
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, null, window.pitchEditor.pitchNew[l]))
    })
    pace_slid.addEventListener("change", () => {
        const new_lengths = window.pitchEditor.resetDurs.map((v,l) => v * lengthsMult[l] * pace_slid.value)
        window.pitchEditor.dursNew = new_lengths
        letters.forEach((_, l) => set_letter_display(letterElems[l], l, new_lengths[l]* 10 + 50, null))
    })
}


// Settings
// ========
settingsCog.addEventListener("click", () => {
    settingsContainer.style.opacity = 0
    settingsContainer.style.display = "flex"
    requestAnimationFrame(() => requestAnimationFrame(() => settingsContainer.style.opacity = 1))
})
settingsContainer.addEventListener("click", event => {
    if (event.target==settingsContainer) {
        closeModal(settingsContainer)
    }
})
useGPUCbx.addEventListener("change", () => {
    spinnerModal("Changing device...")
    fetch(`http://localhost:8008/setDevice`, {
        method: "Post",
        body: JSON.stringify({device: useGPUCbx.checked ? "gpu" : "cpu"})
    }).then(r=>r.text()).then(res => {
        closeModal()
        window.userSettings.useGPU = useGPUCbx.checked
        localStorage.setItem("userSettings", JSON.stringify(window.userSettings))
    }).catch(e => {
        console.log(e)
        if (e.code =="ENOENT") {
            closeModal().then(() => {
                createModal("error", "There was a problem")
            })
        }
    })
})
"use strict"

window.xVASpeechState = {
    isReadingMic: false,
    elapsedRecording: 0,
    recorder: null,
    stream: null
}

// Populate the microphone dropdown with the available options
navigator.mediaDevices.enumerateDevices().then(devices => {
    devices = devices.filter(device => device.kind=="audioinput" && device.deviceId!="default" && device.deviceId!="communications")
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_mic_selection.appendChild(option)
    })

    setting_mic_selection.addEventListener("change", () => {
        window.userSettings.microphone = setting_mic_selection.value
        window.saveUserSettings()
        window.initMic()
    })

    if (Object.keys(window.userSettings).includes("microphone")) {
        setting_mic_selection.value = window.userSettings.microphone
    } else {
        window.userSettings.microphone = setting_mic_selection.value
        window.saveUserSettings()
    }
})




window.initMic = () => {
    return new Promise(resolve => {
        const deviceId = window.userSettings.microphone

        navigator.mediaDevices.getUserMedia({audio: {deviceId: deviceId}}).then(stream => {
            const audio_context = new AudioContext
            const input = audio_context.createMediaStreamSource(stream)
            window.xVASpeechState.stream = stream
            window.xVASpeechState.recorder = new Recorder(input)
            resolve()

        }).catch(err => {
            console.log(err)
            resolve()
        })
    })
}
window.initMic()


const animateRecordingProgress = () => {
    const percentDone = (Date.now() - window.xVASpeechState.elapsedRecording) / 10000

    if (percentDone >= 1 && percentDone!=Infinity) {
        window.stopRecord()
    } else {
        const circle = mic_progress_SVG_circle
        const radius = circle.r.baseVal.value
        const circumference = radius * 2 * Math.PI
        const offset = circumference - percentDone * circumference

        circle.style.strokeDasharray = `${circumference} ${circumference}`
        circle.style.strokeDashoffset = circumference
        circle.style.strokeDashoffset = Math.round(offset)

        requestAnimationFrame(animateRecordingProgress)
    }
}


const clearProgress = (percent=0) => {
    const circle = mic_progress_SVG_circle
    circle.style.stroke = "transparent"
    const radius = circle.r.baseVal.value
    const circumference = radius * 2 * Math.PI
    const offset = circumference - percent * circumference

    circle.style.strokeDasharray = `${circumference} ${circumference}`
    circle.style.strokeDashoffset = circumference
    circle.style.strokeDashoffset = Math.floor(offset)
}
window.clearProgress = clearProgress

window.startRecord = async () => {
    await window.initMic()
    window.xVASpeechState.recorder.record()

    window.xVASpeechState.isReadingMic = true
    window.xVASpeechState.elapsedRecording = Date.now()
    clearProgress()
    mic_progress_SVG_circle.style.stroke = "red"
    requestAnimationFrame(animateRecordingProgress)
}

const outputRecording = (outPath, callback) => {
    window.xVASpeechState.recorder.exportWAV(AudioBLOB => {
        const fileReader = new FileReader()
        fileReader.onload = function() {

            if (fs.existsSync(outPath)) {
                fs.unlinkSync(outPath)
            }

            fs.writeFileSync(outPath, Buffer.from(new Uint8Array(this.result)))

            if (callback) {
                callback()
            }
        }
        fileReader.readAsArrayBuffer(AudioBLOB)
        window.xVASpeechState.recorder.clear()
    })
}

const useWavFileForxVASpeech = (fileName) => {
    fetch(`http://localhost:8008/runSpeechToSpeech`, {
        method: "Post",
        body: JSON.stringify({
            input_path: fileName,
            doPitchShift: window.userSettings.s2s_prePitchShift,
            removeNoise: window.userSettings.s2s_removeNoise,
            removeNoiseStrength: window.userSettings.s2s_noiseRemStrength,
            modelPath: window.userSettings.s2s_voiceId.split(",")[2],
            voiceId: window.userSettings.s2s_voiceId.split(",")[0]
        })
    }).then(r=>r.text()).then(res => {
        mic_progress_SVG.style.animation = "none"
        clearProgress()
        res = res.split("\n")
        let pitchData = res[0]
        let durationsData = res[1]
        let textSequence = res[2]//.split(",").join("")
        pitchData = pitchData.split(",").map(v => parseFloat(v))
        const isFreshRegen = true
        durationsData = durationsData.split(",").map(v => isFreshRegen ? parseFloat(v) : parseFloat(v)/pace_slid.value)
        window.pitchEditor.inputSequence = textSequence
        window.pitchEditor.sequence = textSequence
        dialogueInput.value = textSequence

        window.pitchEditor.ampFlatCounter = 0
        window.pitchEditor.resetPitch = pitchData
        window.pitchEditor.resetDurs = durationsData
        window.pitchEditor.currentVoice = generateVoiceButton.dataset.modelIDLoaded

        window.pitchEditor.audioInput = true

        const pace = 1
        setPitchEditorValues(textSequence.replace(/\s/g, "_").split(""), pitchData, durationsData, isFreshRegen, pace)

        generateVoiceButton.innerHTML = "Generate Voice"
        keepSampleButton.style.display = "none"
        samplePlay.style.display = "none"

        if (window.userSettings.s2s_autogenerate) {
            generateVoiceButton.click()
        }
    }).catch(e => {
        console.log(e)
        mic_progress_SVG.style.animation = "none"
    })
}

window.stopRecord = (cancelled) => {

    window.xVASpeechState.recorder.stop()
    window.xVASpeechState.stream.getAudioTracks()[0].stop()

    if (!cancelled) {
        clearProgress(0.35)
        mic_progress_SVG.style.animation = "spin 1.5s linear infinite"
        mic_progress_SVG_circle.style.stroke = "white"
        const fileName = `${__dirname.replace(/\\/g,"/")}/output/recorded_file.wav`
        outputRecording(fileName, () => {
            useWavFileForxVASpeech(fileName)
        })
    }

    window.xVASpeechState.isReadingMic = false
    window.xVASpeechState.elapsedRecording = 0
    clearProgress()
}

const micClickHandler = () => {
    if (window.xVASpeechState.isReadingMic) {
        window.stopRecord()
    } else {

        if (!Object.keys(window.userSettings).includes("s2s_voiceId") || !window.userSettings.s2s_voiceId) {
            s2s_selectVoiceBtn.click()
        } else {
            const xvaspeechPath = window.userSettings.s2s_voiceId.split(",")[2]
            if (!fs.existsSync(xvaspeechPath)) {
                window.userSettings.s2s_voiceId = undefined
                micClickHandler()
                return
            }
            if (window.currentModel && generateVoiceButton.innerHTML == "Generate Voice") {
                window.startRecord()
            } else {
                window.errorModal("Please load a target voice from the panel on the left, first.")
            }
        }
    }
}
mic_SVG.addEventListener("click", () => micClickHandler())
mic_SVG.addEventListener("contextmenu", () => {
    if (window.xVASpeechState.isReadingMic) {
        window.stopRecord(true)
    } else {
        const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
            src: `${__dirname.replace(/\\/g,"/")}/output/recorded_file_post${window.userSettings.s2s_prePitchShift?"_praat":""}.wav`
        }))
    }
})
clearProgress()


const populateS2SVoiceList = () => {
    const models = []
    Object.values(window.games).forEach(game => {
        game.models.forEach(model => {
            if (model.xvaspeech) {

                model.model.games.forEach(modelGame => {

                    if (!s2sVLFemale.checked && modelGame.gender && modelGame.gender.toLowerCase()=="female") return
                    if (!s2sVLMale.checked && modelGame.gender && modelGame.gender.toLowerCase()=="male") return
                    if (!s2sVLOther.checked && modelGame.gender && modelGame.gender.toLowerCase()=="other") return

                    models.push([model.xvaspeech, model.audioPreviewPath, modelGame.voiceName, modelGame.voiceId, window.games[modelGame.gameId].gameAsset.split("-")[1]])
                })
            }
        })
    })

    s2sVoiceList.innerHTML = ""


    models.forEach(([xvaspeechPath, audioPreviewPath, voiceName, voiceId, themeColour]) => {
        const record = createElem("div")

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`

        record.appendChild(button)
        record.appendChild(createElem("audio", {controls: true}, createElem("source", {
            src: audioPreviewPath+".wav",
            type: `audio/wav`
        })))
        s2sVoiceList.appendChild(record)

        button.addEventListener("click", () => {
            window.userSettings.s2s_voiceId = `${voiceId},${voiceName},${xvaspeechPath}`
            window.saveUserSettings()
            s2s_selectedVoiceName.innerHTML = voiceName
            window.closeModal(s2sSelectContainer)
        })
    })

    if (s2sVoiceList.innerHTML=="") {
        s2sVoiceList.appendChild(createElem("div", "No xVASpeech models are installed"))
    }
}
window.populateS2SVoiceList = populateS2SVoiceList




if (Object.keys(window.userSettings).includes("s2s_voiceId")) {
    s2s_selectedVoiceName.innerHTML = window.userSettings.s2s_voiceId.split(",")[1]
}

s2sVLFemale.addEventListener("change", () => {
    window.userSettings.s2sVL_female = s2sVLFemale.checked
    populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_female")) {
    s2sVLFemale.checked = window.userSettings.s2sVL_female
}
s2sVLMale.addEventListener("change", () => {
    window.userSettings.s2sVL_male = s2sVLMale.checked
    populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_male")) {
    s2sVLMale.checked = window.userSettings.s2sVL_male
}
s2sVLOther.addEventListener("change", () => {
    window.userSettings.s2sVL_other = s2sVLOther.checked
    populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_other")) {
    s2sVLOther.checked = window.userSettings.s2sVL_other
}


const silenceFileName = `${__dirname.replace(/\\/g,"/")}/output/silence.wav`
s2sNoiseRecordSampleBtn.addEventListener("click", () => {
    if (window.xVASpeechState.isReadingMic) {
        return
    }
    window.xVASpeechState.recorder.record()
    window.xVASpeechState.isReadingMic = true

    const origButtonColour = s2sNoiseRecordSampleBtn.style.background
    s2sNoiseRecordSampleBtn.style.background = "red"

    let secondsElapsed = 1
    const interval = setInterval(() => {

        if (secondsElapsed>=5) {
            s2sNoiseSampleRecordTimer.innerHTML = ""
            window.xVASpeechState.isReadingMic = false
            clearInterval(interval)
            s2sNoiseRecordSampleBtn.style.background = origButtonColour
            window.xVASpeechState.recorder.stop()
            window.xVASpeechState.stream.getAudioTracks()[0].stop()
            outputRecording(silenceFileName, () => {
                s2sNoiseAudioContainer.innerHTML = ""
                s2sNoiseAudioContainer.appendChild(createElem("audio", {controls: true}, createElem("source", {
                    src: silenceFileName,
                    type: `audio/wav`
                })))
            })
        } else {
            s2sNoiseSampleRecordTimer.innerHTML = `${5-secondsElapsed}s`
            secondsElapsed++
        }
    }, 1000)
    s2sNoiseSampleRecordTimer.innerHTML = `5s`
})
if (fs.existsSync(silenceFileName)) {
    s2sNoiseAudioContainer.appendChild(createElem("audio", {controls: true}, createElem("source", {
        src: silenceFileName,
        type: `audio/wav`
    })))
}


s2sVLRecordSampleBtn.addEventListener("click", () => {

    if (window.xVASpeechState.isReadingMic) {
        return
    }

    window.xVASpeechState.recorder.record()
    window.xVASpeechState.isReadingMic = true

    const origButtonColour = s2sVLRecordSampleBtn.style.background
    s2sVLRecordSampleBtn.style.background = "red"

    let secondsElapsed = 1
    const interval = setInterval(() => {

        if (secondsElapsed>=5) {
            s2sVLSampleRecordTimer.innerHTML = ""
            window.xVASpeechState.isReadingMic = false
            clearInterval(interval)
            s2sVLRecordSampleBtn.style.background = origButtonColour
            window.xVASpeechState.recorder.stop()
            window.xVASpeechState.stream.getAudioTracks()[0].stop()
            const fileName = `${__dirname.replace(/\\/g,"/")}/output/temp-recsample.wav`
            outputRecording(fileName, () => {
                s2sVLVoiceSampleAudioContainer.innerHTML = ""
                s2sVLVoiceSampleAudioContainer.appendChild(createElem("audio", {controls: true}, createElem("source", {
                    src: fileName,
                    type: `audio/wav`
                })))
            })
        } else {
            s2sVLSampleRecordTimer.innerHTML = `${5-secondsElapsed}s`
            secondsElapsed++
        }
    }, 1000)
    s2sVLSampleRecordTimer.innerHTML = `5s`
})






// File dragging
const uploadS2SFile = (eType, event) => {

    if (["dragenter", "dragover"].includes(eType)) {
        clearProgress(1)
        mic_progress_SVG_circle.style.stroke = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        clearProgress()
    }

    event.preventDefault()
    event.stopPropagation()

    if (eType=="drop") {

        if (!Object.keys(window.userSettings).includes("s2s_voiceId")) {
            s2s_selectVoiceBtn.click()
        } else {
            if (window.currentModel && generateVoiceButton.innerHTML == "Generate Voice") {
                const dataTransfer = event.dataTransfer
                const files = Array.from(dataTransfer.files)
                const file = files[0]

                if (!file.name.endsWith(".wav")) {
                    window.errorModal("Only .wav files are supported for speech-to-speech file input at the moment.")
                    return
                }

                clearProgress(0.35)
                mic_progress_SVG.style.animation = "spin 1.5s linear infinite"
                mic_progress_SVG_circle.style.stroke = "white"

                const fileName = `${__dirname.replace(/\\/g,"/")}/output/recorded_file.wav`
                fs.copyFileSync(file.path, fileName)
                useWavFileForxVASpeech(fileName)
            } else {
                window.errorModal("Please load a target voice from the panel on the left, first.")
            }
        }
    }
}

micContainer.addEventListener("dragenter", event => uploadS2SFile("dragenter", event), false)
micContainer.addEventListener("dragleave", event => uploadS2SFile("dragleave", event), false)
micContainer.addEventListener("dragover", event => uploadS2SFile("dragover", event), false)
micContainer.addEventListener("drop", event => uploadS2SFile("drop", event), false)

// Disable page navigation on badly dropped file
window.document.addEventListener("dragover", event => event.preventDefault(), false)
window.document.addEventListener("drop", event => event.preventDefault(), false)
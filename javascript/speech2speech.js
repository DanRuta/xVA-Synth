"use strict"

window.speech2speechState = {
    isReadingMic: false,
    elapsedRecording: 0,
    recorder: null,
    stream: null,
    s2s_running: false
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
            window.speech2speechState.stream = stream
            window.speech2speechState.recorder = new Recorder(input)
            resolve()

        }).catch(err => {
            console.log(err)
            resolve()
        })
    })
}
window.initMic()


const animateRecordingProgress = () => {
    const percentDone = (Date.now() - window.speech2speechState.elapsedRecording) / 10000

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
    window.speech2speechState.recorder.record()

    window.speech2speechState.isReadingMic = true
    window.speech2speechState.elapsedRecording = Date.now()
    clearProgress()
    mic_progress_SVG_circle.style.stroke = "red"
    requestAnimationFrame(animateRecordingProgress)
}

window.outputS2SRecording = (outPath, callback) => {
    window.speech2speechState.recorder.exportWAV(AudioBLOB => {
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
        window.speech2speechState.recorder.clear()
    })
}

window.useWavFileForspeech2speech = (fileName) => {
    let sequence = dialogueInput.value.trim().replace("…", "...")
    doFetch(`http://localhost:8008/runSpeechToSpeech`, {
        method: "Post",
        body: JSON.stringify({
            input_path: fileName,
            modelType: window.currentModel.modelType,
            s2s_components: window.userSettings.s2s_components,
            text: sequence,
            doPitchShift: window.userSettings.s2s_prePitchShift,
            removeNoise: window.userSettings.s2s_removeNoise,
            removeNoiseStrength: window.userSettings.s2s_noiseRemStrength,
            n_speakers: window.userSettings.s2s_voiceId.split(",").reverse()[1],
            modelPath: window.userSettings.s2s_voiceId.split(",").reverse()[0],
            voiceId: window.userSettings.s2s_voiceId.split(",")[0]
        })
    }).then(r=>r.text()).then(res => {

        window.speech2speechState.s2s_running = true
        mic_progress_SVG.style.animation = "none"
        clearProgress()

        if (res.includes("Traceback")) {
            window.errorModal(`<h3>${window.i18n.SOMETHING_WENT_WRONG}</h3>${res.replaceAll("\n", "<br>")}`)

        } else if (res.includes("ERROR:APP_VERSION")) {
            const speech2speechModelVersion = "v"+res.split(",")[1]
            window.errorModal(`${window.i18n.ERR_XVASPEECH_MODEL_VERSION.replace("_1", speech2speechModelVersion)} ${window.appVersion}`)
        } else {
            res = res.split("\n")
            let pitchData = res[0]
            let durationsData = res[1]
            let energyData = res[2]
            let cleanedSequence = res[3].split("|").map(c=>c.replaceAll("{", "").replaceAll("}", "")).join("")

            if (!dialogueInput.value.length) {
                dialogueInput.value = cleanedSequence
            }

            sequence = cleanedSequence

            pitchData = pitchData.split(",").map(v => parseFloat(v))
            if (energyData.length) {
                energyData = energyData.split(",").map(v => parseFloat(v)).filter(v => !isNaN(v))
            } else {
                energyData = []
            }
            durationsData = durationsData.split(",").map(v => parseFloat(v))

            window.sequenceEditor.inputSequence = sequence
            window.sequenceEditor.sequence = cleanedSequence

            window.sequenceEditor.resetPitch = pitchData
            window.sequenceEditor.resetDurs = durationsData
            window.sequenceEditor.resetEnergy = energyData

            window.sequenceEditor.letters = cleanedSequence.replace(/\s/g, "_").split("")
            window.sequenceEditor.pitchNew = pitchData.map(p=>p)
            window.sequenceEditor.dursNew = durationsData.map(v=>v)
            window.sequenceEditor.energyNew = energyData.map(v=>v)
            window.sequenceEditor.init()
            window.sequenceEditor.update()

            window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
            window.sequenceEditor.autoInferTimer = null
            window.sequenceEditor.hasChanged = false

            generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE

            if (window.userSettings.s2s_autogenerate) {
                speech2speechState.s2s_autogenerate = true
                generateVoiceButton.click()
            }
        }

    }).catch(e => {
        console.log(e)
        mic_progress_SVG.style.animation = "none"
    })
}

window.stopRecord = (cancelled) => {

    window.speech2speechState.recorder.stop()
    window.speech2speechState.stream.getAudioTracks()[0].stop()

    if (!cancelled) {
        clearProgress(0.35)
        mic_progress_SVG.style.animation = "spin 1.5s linear infinite"
        mic_progress_SVG_circle.style.stroke = "white"
        const fileName = `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/output/recorded_file.wav`
        window.outputS2SRecording(fileName, () => {
            window.useWavFileForspeech2speech(fileName)
        })
    }

    window.speech2speechState.isReadingMic = false
    window.speech2speechState.elapsedRecording = 0
    clearProgress()
}

window.micClickHandler = (ctrlKey) => {

    if (ctrlKey) {
        s2s_selectVoiceBtn.click()
    } else {
        if (window.speech2speechState.isReadingMic) {
            window.stopRecord()
        } else {

            if (!Object.keys(window.userSettings).includes("s2s_voiceId") || !window.userSettings.s2s_voiceId) {
                s2s_selectVoiceBtn.click()
            } else {
                const speech2speechPath = window.userSettings.s2s_voiceId.split(",")[3]
                if (!fs.existsSync(speech2speechPath)) {
                    window.userSettings.s2s_voiceId = undefined
                    window.micClickHandler()
                    return
                }
                if (window.currentModel && generateVoiceButton.innerHTML == window.i18n.GENERATE_VOICE) {
                    window.startRecord()
                } else {
                    window.errorModal(window.i18n.LOAD_TARGET_MODEL)
                }
            }
        }
    }
}
mic_SVG.addEventListener("mouseenter", () => {
    s2s_voiceId_selected_label.style.display = "inline-block"
})
mic_SVG.addEventListener("mouseleave", () => {
    s2s_voiceId_selected_label.style.display = "none"
})
mic_SVG.addEventListener("click", event => window.micClickHandler(event.ctrlKey))
mic_SVG.addEventListener("contextmenu", () => {
    if (window.speech2speechState.isReadingMic) {
        window.stopRecord(true)
    } else {
        const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
            src: `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/output/recorded_file_post${window.userSettings.s2s_prePitchShift?"_praat":""}.wav`
        }))
        audioPreview.setSinkId(window.userSettings.base_speaker)
    }
})
clearProgress()


window.populateS2SVoiceList = () => {
    const models = []
    Object.values(window.games).forEach(game => {
        game.models.forEach(model => {

            model.variants.forEach(variant => {

                if (variant.modelType.toLowerCase()=="fastpitch1.1") {

                    if (!s2sVLFemale.checked && variant.gender && variant.gender.toLowerCase()=="female") return
                    if (!s2sVLMale.checked && variant.gender && variant.gender.toLowerCase()=="male") return
                    if (!s2sVLOther.checked && variant.gender && variant.gender.toLowerCase()=="other") return

                    const modelPath = window.userSettings[`modelspath_${model.gameId}`] + `/${variant.voiceId}.pt`

                    const themeColour = window.games[model.gameId].gameTheme.themeColourPrimary
                    models.push([modelPath, model.num_speakers||0, model.audioPreviewPath, model.voiceName, variant.voiceId, themeColour] )
                }
            })
        })
    })

    s2sVoiceList.innerHTML = ""

    models.forEach(([modelPath, num_speakers, audioPreviewPath, voiceName, voiceId, themeColour]) => {
        const record = createElem("div")

        const button = createElem("div.voiceType", voiceName)
        button.style.background = `#${themeColour}`

        record.appendChild(button)
        const audioElem = createElem("audio", {controls: true}, createElem("source", {
            src: audioPreviewPath+".wav",
            type: `audio/wav`
        }))
        audioElem.setSinkId(window.userSettings.base_speaker)
        record.appendChild(audioElem)
        s2sVoiceList.appendChild(record)

        button.addEventListener("click", () => {
            window.userSettings.s2s_voiceId = `${voiceId},${voiceName},${num_speakers},${modelPath}`
            window.saveUserSettings()
            s2s_selectedVoiceName.innerHTML = voiceName
            s2s_voiceId_selected_label.innerHTML = voiceName
            window.closeModal(s2sSelectContainer)
        })
    })

    if (s2sVoiceList.innerHTML=="") {
        s2sVoiceList.appendChild(createElem("div", window.i18n.NO_XVASPEECH_MODELS))
    }
}



if (Object.keys(window.userSettings).includes("s2s_voiceId")) {
    s2s_selectedVoiceName.innerHTML = window.userSettings.s2s_voiceId.split(",")[1]
    s2s_voiceId_selected_label.innerHTML = window.userSettings.s2s_voiceId.split(",")[1]
}

s2sVLFemale.addEventListener("change", () => {
    window.userSettings.s2sVL_female = s2sVLFemale.checked
    window.populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_female")) {
    s2sVLFemale.checked = window.userSettings.s2sVL_female
}
s2sVLMale.addEventListener("change", () => {
    window.userSettings.s2sVL_male = s2sVLMale.checked
    window.populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_male")) {
    s2sVLMale.checked = window.userSettings.s2sVL_male
}
s2sVLOther.addEventListener("change", () => {
    window.userSettings.s2sVL_other = s2sVLOther.checked
    window.populateS2SVoiceList()
    window.saveUserSettings()
})
if (Object.keys(window.userSettings).includes("s2sVL_other")) {
    s2sVLOther.checked = window.userSettings.s2sVL_other
}


const silenceFileName = `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/output/silence.wav`
s2sNoiseRecordSampleBtn.addEventListener("click", () => {
    if (window.speech2speechState.isReadingMic) {
        return
    }
    window.speech2speechState.recorder.record()
    window.speech2speechState.isReadingMic = true

    const origButtonColour = s2sNoiseRecordSampleBtn.style.background
    s2sNoiseRecordSampleBtn.style.background = "red"

    let secondsElapsed = 1
    const interval = setInterval(() => {

        if (secondsElapsed>=5) {
            s2sNoiseSampleRecordTimer.innerHTML = ""
            window.speech2speechState.isReadingMic = false
            clearInterval(interval)
            s2sNoiseRecordSampleBtn.style.background = origButtonColour
            window.speech2speechState.recorder.stop()
            window.speech2speechState.stream.getAudioTracks()[0].stop()
            window.outputS2SRecording(silenceFileName, () => {
                s2sNoiseAudioContainer.innerHTML = ""
                const audioElem = createElem("audio", {controls: true}, createElem("source", {
                    src: silenceFileName,
                    type: `audio/wav`
                }))
                audioElem.setSinkId(window.userSettings.base_speaker)
                s2sNoiseAudioContainer.appendChild(audioElem)


            })
        } else {
            s2sNoiseSampleRecordTimer.innerHTML = `${5-secondsElapsed}s`
            secondsElapsed++
        }
    }, 1000)
    s2sNoiseSampleRecordTimer.innerHTML = `5s`
})
if (fs.existsSync(silenceFileName)) {
    const audioElem = createElem("audio", {controls: true}, createElem("source", {
        src: silenceFileName,
        type: `audio/wav`
    }))
    audioElem.setSinkId(window.userSettings.base_speaker)
    s2sNoiseAudioContainer.appendChild(audioElem)
}


s2sVLRecordSampleBtn.addEventListener("click", () => {

    if (window.speech2speechState.isReadingMic) {
        return
    }

    window.speech2speechState.recorder.record()
    window.speech2speechState.isReadingMic = true

    const origButtonColour = s2sVLRecordSampleBtn.style.background
    s2sVLRecordSampleBtn.style.background = "red"

    let secondsElapsed = 1
    const interval = setInterval(() => {

        if (secondsElapsed>=5) {
            s2sVLSampleRecordTimer.innerHTML = ""
            window.speech2speechState.isReadingMic = false
            clearInterval(interval)
            s2sVLRecordSampleBtn.style.background = origButtonColour
            window.speech2speechState.recorder.stop()
            window.speech2speechState.stream.getAudioTracks()[0].stop()
            const fileName = `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/output/temp-recsample.wav`
            window.outputS2SRecording(fileName, () => {
                s2sVLVoiceSampleAudioContainer.innerHTML = ""
                const audioElem = createElem("audio", {controls: true}, createElem("source", {
                    src: fileName,
                    type: `audio/wav`
                }))
                audioElem.setSinkId(window.userSettings.base_speaker)
                s2sVLVoiceSampleAudioContainer.appendChild(audioElem)


            })
        } else {
            s2sVLSampleRecordTimer.innerHTML = `${5-secondsElapsed}s`
            secondsElapsed++
        }
    }, 1000)
    s2sVLSampleRecordTimer.innerHTML = `5s`
})



// File dragging
window.uploadS2SFile = (eType, event) => {

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
            if (window.currentModel && generateVoiceButton.innerHTML == window.i18n.GENERATE_VOICE) {
                const dataTransfer = event.dataTransfer
                const files = Array.from(dataTransfer.files)
                const file = files[0]

                if (!file.name.endsWith(".wav")) {
                    window.errorModal(window.i18n.ONLY_WAV_S2S)
                    return
                }

                clearProgress(0.35)
                mic_progress_SVG.style.animation = "spin 1.5s linear infinite"
                mic_progress_SVG_circle.style.stroke = "white"

                const fileName = `${__dirname.replace("\\javascript", "").replace(/\\/g,"/")}/output/recorded_file.wav`
                fs.copyFileSync(file.path, fileName)
                window.useWavFileForspeech2speech(fileName)
            } else {
                window.errorModal(window.i18n.LOAD_TARGET_MODEL)
            }
        }
    }
}

micContainer.addEventListener("dragenter", event => window.uploadS2SFile("dragenter", event), false)
micContainer.addEventListener("dragleave", event => window.uploadS2SFile("dragleave", event), false)
micContainer.addEventListener("dragover", event => window.uploadS2SFile("dragover", event), false)
micContainer.addEventListener("drop", event => window.uploadS2SFile("drop", event), false)

// Disable page navigation on badly dropped file
window.document.addEventListener("dragover", event => event.preventDefault(), false)
window.document.addEventListener("drop", event => event.preventDefault(), false)
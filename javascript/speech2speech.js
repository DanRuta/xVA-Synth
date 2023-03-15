"use strict"

window.speech2speechState = {
    isReadingMic: false,
    elapsedRecording: 0,
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
        if (window.speech2speechState.isReadingMic) {
            window.stopRecord()
        }
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

    doFetch(`http://localhost:8008/start_microphone_recording`, {
        method: "Post"
    })

    window.speech2speechState.isReadingMic = true
    window.speech2speechState.elapsedRecording = Date.now()
    clearProgress()
    mic_progress_SVG_circle.style.stroke = "red"
    requestAnimationFrame(animateRecordingProgress)
}

window.outputS2SRecording = (outPath, callback) => {
    doFetch(`http://localhost:8008/move_recorded_file`, {
        method: "Post",
        body: JSON.stringify({
            file_path: outPath
        })
    }).then(r=>r.text()).then(res => {
        callback()
    })
}

window.useWavFileForspeech2speech = (fileName) => {
    // let sequence = dialogueInput.value.trim().replace("…", "...")

    // For some reason, the samplePlay audio element does not update the source when the file name is the same
    const tempFileNum = `${Math.random().toString().split(".")[1]}`
    let tempFileLocation = `${path}/output/temp-${tempFileNum}.wav`

    let style_emb = window.currentModel.audioPreviewPath // Default to the preview audio file, if an embedding can't be found in the json - This shouldn't happen
    try {
        style_emb = window.currentModel.games[0].base_speaker_emb // If this fails, the json isn't complete
    } catch (e) {
        console.log(e)
    }

    if (window.wavesurfer) {
        window.wavesurfer.stop()
        wavesurferContainer.style.opacity = 0
    }
    window.tempFileLocation = `${__dirname.replace("/javascript", "").replace("\\javascript", "")}/output/temp-${tempFileNum}.wav`
    toggleSpinnerButtons()

    const options = {
        hz: window.userSettings.audio.hz,
        padStart: window.userSettings.audio.padStart,
        padEnd: window.userSettings.audio.padEnd,
        bit_depth: window.userSettings.audio.bitdepth,
        amplitude: window.userSettings.audio.amplitude,
        pitchMult: window.userSettings.audio.pitchMult,
        tempo: window.userSettings.audio.tempo,
        deessing: window.userSettings.audio.deessing,
        nr: window.userSettings.audio.nr,
        nf: window.userSettings.audio.nf,
        useNR: window.userSettings.audio.useNR,
        useSR: useSRCkbx.checked,
        useCleanup: useCleanupCkbx.checked
    }

    doFetch(`http://localhost:8008/runSpeechToSpeech`, {
        method: "Post",
        body: JSON.stringify({
            input_path: fileName,
            useSR: useSRCkbx.checked,
            isBatchMode: false,

            style_emb: window.currentModel.games[0].base_speaker_emb,
            audio_out_path: tempFileLocation,

            doPitchShift: window.userSettings.s2s_prePitchShift,
            removeNoise: window.userSettings.s2s_removeNoise,
            removeNoiseStrength: window.userSettings.s2s_noiseRemStrength,
            n_speakers: undefined,
            modelPath: undefined,
            voiceId: undefined,

            options: JSON.stringify(options)
        })
    }).then(r=>r.text()).then(res => {
        // This block of code sometimes gets called before the audio file has actually finished flushing to file
        // I need a better way to make sure that this doesn't get called until it IS finished, but "for now",
        // I've set up some recursive re-attempts, below doTheRest

        let hasLoaded = false
        let numRetries = 0
        toggleSpinnerButtons()

        const doTheRest = () => {
            if (hasLoaded) {
                return
            }
            // window.wavesurfer = undefined
            tempFileLocation = tempFileLocation.replaceAll(/\\/, "/")
            tempFileLocation = tempFileLocation.replaceAll('/resources/app/resources/app', "/resources/app")
            tempFileLocation = tempFileLocation.replaceAll('/resources/app', "")

            dialogueInput.value = ""
            window.isGenerating = false

            window.speech2speechState.s2s_running = true
            mic_progress_SVG.style.animation = "none"
            clearProgress()

            if (res.includes("Traceback")) {
                window.errorModal(`<h3>${window.i18n.SOMETHING_WENT_WRONG}</h3>${res.replaceAll("\n", "<br>")}`)

            } else if (res.includes("ERROR:APP_VERSION")) {
                const speech2speechModelVersion = "v"+res.split(",")[1]
                window.errorModal(`${window.i18n.ERR_XVASPEECH_MODEL_VERSION.replace("_1", speech2speechModelVersion)} ${window.appVersion}`)
            } else {

                keepSampleButton.disabled = false
                window.tempFileLocation = tempFileLocation

                // Wavesurfer
                if (!window.wavesurfer) {
                    window.initWaveSurfer(window.tempFileLocation)
                } else {
                    window.wavesurfer.load(window.tempFileLocation)
                }
                window.wavesurfer.on("ready",  () => {

                    hasLoaded = true
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
                generateVoiceButton.innerHTML = window.i18n.GENERATE_VOICE

                if (window.userSettings.s2s_autogenerate) {
                    speech2speechState.s2s_autogenerate = true
                    generateVoiceButton.click()
                }

                keepSampleButton.dataset.newFileLocation = `${window.userSettings[`outpath_${window.currentGame.gameId}`]}/${title.dataset.modelId}/vc_${tempFileNum}.wav`
                keepSampleButton.disabled = false
                keepSampleButton.style.display = "block"
                samplePlayPause.style.display = "block"

                setTimeout(doTheRest, 100)
            }

        }

        doTheRest()
    }).catch(e => {
        console.log(e)
        window.errorModal(`<h3>${window.i18n.SOMETHING_WENT_WRONG}</h3>`)
        mic_progress_SVG.style.animation = "none"
    })
}

window.stopRecord = (cancelled) => {
    fs.writeFileSync(`${window.path}/python/temp_stop_recording`, "")

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
    if (window.speech2speechState.isReadingMic) {
        window.stopRecord()
    } else {
        if (window.currentModel && generateVoiceButton.innerHTML == window.i18n.GENERATE_VOICE) {
            if (window.currentModel.modelType.toLowerCase()=="xvapitch") {
                window.startRecord()
            }
        } else {
            window.errorModal(window.i18n.LOAD_TARGET_MODEL)
        }
    }
}
mic_SVG.addEventListener("mouseenter", () => {
    if (!window.currentModel || window.currentModel.modelType.toLowerCase()!="xvapitch") {
        s2s_voiceId_selected_label.style.display = "inline-block"
    }
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

micContainer.addEventListener("dragenter", event => window.uploadS2SFile("dragenter", event), false)
micContainer.addEventListener("dragleave", event => window.uploadS2SFile("dragleave", event), false)
micContainer.addEventListener("dragover", event => window.uploadS2SFile("dragover", event), false)
micContainer.addEventListener("drop", event => window.uploadS2SFile("drop", event), false)

// Disable page navigation on badly dropped file
window.document.addEventListener("dragover", event => event.preventDefault(), false)
window.document.addEventListener("drop", event => event.preventDefault(), false)
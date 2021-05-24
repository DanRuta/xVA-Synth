"use strict"

const saveUserSettings = () => localStorage.setItem("userSettings", JSON.stringify(window.userSettings))

const deleteFolderRecursive = function (directoryPath) {
    if (fs.existsSync(directoryPath)) {
        fs.readdirSync(directoryPath).forEach((file, index) => {
          const curPath = `${directoryPath}/${file}`;
          if (fs.lstatSync(curPath).isDirectory()) {
           // recurse
            deleteFolderRecursive(curPath);
          } else {
            // delete file
            fs.unlinkSync(curPath);
          }
        });
        fs.rmdirSync(directoryPath);
      }
    };

// Load user settings
window.userSettings = localStorage.getItem("userSettings") ||
    {
        useGPU: false,
        customWindowSize:`${window.innerHeight},${window.innerWidth}`,
        autoplay: false,
        autoPlayGen: false,
        audio: {
            format: "wav"
        },
        plugins: {

        }
    }
if ((typeof window.userSettings)=="string") {
    window.userSettings = JSON.parse(window.userSettings)
}
if (!Object.keys(window.userSettings).includes("audio")) { // For backwards compatibility
    window.userSettings.audio = {format: "wav", hz: 22050, padStart: 0, padEnd: 0}
}
if (!Object.keys(window.userSettings).includes("sliderTooltip")) { // For backwards compatibility
    window.userSettings.sliderTooltip = true
}
if (!Object.keys(window.userSettings).includes("darkPrompt")) { // For backwards compatibility
    window.userSettings.darkPrompt = false
}
if (!Object.keys(window.userSettings).includes("autoReloadVoices")) { // For backwards compatibility
    window.userSettings.autoReloadVoices = false
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("hz")) { // For backwards compatibility
    window.userSettings.audio.hz = 22050
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("padStart")) { // For backwards compatibility
    window.userSettings.audio.padStart = 0
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("padEnd")) { // For backwards compatibility
    window.userSettings.audio.padEnd = 0
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("ffmpeg")) { // For backwards compatibility
    window.userSettings.audio.ffmpeg = false
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("bitdepth")) { // For backwards compatibility
    window.userSettings.audio.bitdepth = "pcm_s32le"
}
if (!Object.keys(window.userSettings).includes("vocoder")) { // For backwards compatibility
    window.userSettings.vocoder = "256_waveglow"
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("amplitude")) { // For backwards compatibility
    window.userSettings.audio.amplitude = 1
}
if (!Object.keys(window.userSettings).includes("keepPaceOnNew")) { // For backwards compatibility
    window.userSettings.keepPaceOnNew = true
}
if (!Object.keys(window.userSettings).includes("batchOutFolder")) { // For backwards compatibility
    window.userSettings.batchOutFolder = `${__dirname.replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
}
if (!Object.keys(window.userSettings).includes("batch_clearDirFirst")) { // For backwards compatibility
    window.userSettings.batch_clearDirFirst = false
}
if (!Object.keys(window.userSettings).includes("batch_fastMode")) { // For backwards compatibility
    window.userSettings.batch_fastMode = false
}
if (!Object.keys(window.userSettings).includes("batch_skipExisting")) { // For backwards compatibility
    window.userSettings.batch_skipExisting = true
}
if (!Object.keys(window.userSettings).includes("defaultToHiFi")) { // For backwards compatibility
    window.userSettings.defaultToHiFi = true
}
if (!Object.keys(window.userSettings).includes("batch_batchSize")) { // For backwards compatibility
    window.userSettings.batch_batchSize = 1
}
if (!Object.keys(window.userSettings).includes("autoPlayGen")) { // For backwards compatibility
    window.userSettings.autoPlayGen = true
}
if (!Object.keys(window.userSettings).includes("outputJSON")) { // For backwards compatibility
    window.userSettings.outputJSON = true
}
if (!Object.keys(window.userSettings).includes("keepEditorOnVoiceChange")) { // For backwards compatibility
    window.userSettings.keepEditorOnVoiceChange = false
}
if (!Object.keys(window.userSettings).includes("filenameNumericalSeq")) { // For backwards compatibility
    window.userSettings.filenameNumericalSeq = false
}
if (!Object.keys(window.userSettings).includes("plugins")) { // For backwards compatibility
    window.userSettings.plugins = {}
}
if (!Object.keys(window.userSettings.plugins).includes("loadOrder")) { // For backwards compatibility
    window.userSettings.plugins.loadOrder = ""
}
if (!Object.keys(window.userSettings).includes("externalAudioEditor")) { // For backwards compatibility
    window.userSettings.externalAudioEditor = ""
}

const updateUIWithSettings = () => {
    useGPUCbx.checked = window.userSettings.useGPU
    autoplay_ckbx.checked = window.userSettings.autoplay
    setting_slidersTooltip.checked = window.userSettings.sliderTooltip
    setting_defaultToHiFi.checked = window.userSettings.defaultToHiFi
    setting_keepPaceOnNew.checked = window.userSettings.keepPaceOnNew
    setting_autoplaygenCbx.checked = window.userSettings.autoPlayGen
    setting_darkprompt.checked = window.userSettings.darkPrompt
    setting_areload_voices.checked = window.userSettings.autoReloadVoices
    setting_output_json.checked = window.userSettings.outputJSON
    setting_output_num_seq.checked = window.userSettings.filenameNumericalSeq
    setting_keepEditorOnVoiceChange.checked = window.userSettings.keepEditorOnVoiceChange

    setting_external_audio_editor.value = window.userSettings.externalAudioEditor
    setting_audio_ffmpeg.checked = window.userSettings.audio.ffmpeg
    setting_audio_format.value = window.userSettings.audio.format
    setting_audio_hz.value = window.userSettings.audio.hz
    setting_audio_pad_start.value = window.userSettings.audio.padStart
    setting_audio_pad_end.value = window.userSettings.audio.padEnd
    setting_audio_bitdepth.value = window.userSettings.audio.bitdepth
    setting_audio_amplitude.value = window.userSettings.audio.amplitude

    setting_batch_fastmode.checked = window.userSettings.batch_fastMode

    batch_batchSizeInput.value = parseInt(window.userSettings.batch_batchSize)
    batch_skipExisting.checked = window.userSettings.batch_skipExisting
    batch_clearDirFirstCkbx.checked = window.userSettings.batch_clearDirFirst

    const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
    ipcRenderer.send("resize", {height, width})
}
updateUIWithSettings()
saveUserSettings()



// Settings Menu
// =============
useGPUCbx.addEventListener("change", () => {
    spinnerModal("Changing device...")
    fetch(`http://localhost:8008/setDevice`, {
        method: "Post",
        body: JSON.stringify({device: useGPUCbx.checked ? "gpu" : "cpu"})
    }).then(r=>r.text()).then(res => {
        window.closeModal()
        window.userSettings.useGPU = useGPUCbx.checked
        saveUserSettings()
    }).catch(e => {
        console.log(e)
        if (e.code =="ENOENT") {
            window.closeModal().then(() => {
                createModal("error", "There was a problem")
            })
        }
    })
})
setting_autoplaygenCbx.addEventListener("click", () => {
    window.userSettings.autoPlayGen = setting_autoplaygenCbx.checked
    saveUserSettings()
})
setting_slidersTooltip.addEventListener("click", () => {
    window.userSettings.sliderTooltip = setting_slidersTooltip.checked
    saveUserSettings()
})
setting_defaultToHiFi.addEventListener("click", () => {
    window.userSettings.defaultToHiFi = setting_defaultToHiFi.checked
    saveUserSettings()
})
setting_keepPaceOnNew.addEventListener("click", () => {
    window.userSettings.keepPaceOnNew = setting_keepPaceOnNew.checked
    saveUserSettings()
})
setting_areload_voices.addEventListener("click", () => {
    window.userSettings.autoReloadVoices = setting_areload_voices.checked
    saveUserSettings()
})
setting_output_json.addEventListener("click", () => {
    window.userSettings.outputJSON = setting_output_json.checked
    saveUserSettings()
})
setting_output_num_seq.addEventListener("click", () => {
    window.userSettings.filenameNumericalSeq = setting_output_num_seq.checked
    saveUserSettings()
})

const setTheme = () => {
    if (window.userSettings.darkPrompt) {
        dialogueInput.style.backgroundColor = "rgba(25,25,25,0.9)"
        dialogueInput.style.color = "white"
    } else {
        dialogueInput.style.backgroundColor = "rgba(255,255,255,0.9)"
        dialogueInput.style.color = "black"
    }
}
setting_darkprompt.addEventListener("click", () => {
    window.userSettings.darkPrompt = setting_darkprompt.checked
    setTheme()
    saveUserSettings()
})
setTheme()
setting_batch_fastmode.addEventListener("click", () => {
    window.userSettings.batch_fastMode = setting_batch_fastmode.checked
    saveUserSettings()
})

setting_external_audio_editor.addEventListener("change", () => {
    window.userSettings.externalAudioEditor = setting_external_audio_editor.value
    saveUserSettings()
})
setting_audio_ffmpeg.addEventListener("click", () => {
    window.userSettings.audio.ffmpeg = setting_audio_ffmpeg.checked
    setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
    saveUserSettings()
})
setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
setting_audio_format.addEventListener("change", () => {
    window.userSettings.audio.format = setting_audio_format.value
    saveUserSettings()
})
setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
setting_audio_hz.addEventListener("change", () => {
    window.userSettings.audio.hz = setting_audio_hz.value
    saveUserSettings()
})
setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_start.addEventListener("change", () => {
    window.userSettings.audio.padStart = parseInt(setting_audio_pad_start.value)
    saveUserSettings()
})
setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_end.addEventListener("change", () => {
    window.userSettings.audio.padEnd = parseInt(setting_audio_pad_end.value)
    saveUserSettings()
})
setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
setting_audio_bitdepth.addEventListener("change", () => {
    window.userSettings.audio.bitdepth = setting_audio_bitdepth.value
    saveUserSettings()
})
setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
setting_audio_amplitude.addEventListener("change", () => {
    window.userSettings.audio.amplitude = setting_audio_amplitude.value
    saveUserSettings()
})

// Output path
fs.readdir(`${path}/models`, (err, gameDirs) => {
    gameDirs.filter(name => !name.includes(".")).forEach(gameFolder => {
        // Initialize the default output directory setting for this game
        if (!Object.keys(window.userSettings).includes(`outpath_${gameFolder}`)) {
            window.userSettings[`outpath_${gameFolder}`] = `${__dirname.replace(/\\/g,"/")}/output/${gameFolder}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
            saveUserSettings()
        }
    })
})
setting_out_path_input.addEventListener("change", () => {
    const gameFolder = window.currentGame[0]

    setting_out_path_input.value = setting_out_path_input.value.replace(/\/\//g, "/").replace(/\\/g,"/")
    window.userSettings[`outpath_${gameFolder}`] = setting_out_path_input.value
    saveUserSettings()
    if (window.currentModelButton) {
        window.currentModelButton.click()
    }
})
// Models path
fs.readdir(`${path}/assets`, (err, assetFiles) => {
    assetFiles.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png"))&&fn.split("-").length==4).forEach(assetFileName => {
        const gameId = assetFileName.split("-")[0]
        const gameName = assetFileName.split("-").reverse()[0].split(".")[0]
        // Initialize the default models directory setting for this game
        if (!Object.keys(window.userSettings).includes(`modelspath_${gameId}`)) {
            window.userSettings[`modelspath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/models/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
            saveUserSettings()
        }
    })
})


// Batch stuff
// Output folder
batch_outputFolderInput.addEventListener("change", () => {
    if (batch_outputFolderInput.value.length==0) {
        window.errorModal("Please enter a directory path")
        batch_outputFolderInput.value = window.userSettings.batchOutFolder
    } else {
        window.userSettings.batchOutFolder = batch_outputFolderInput.value
        saveUserSettings()
    }
})
batch_outputFolderInput.value = window.userSettings.batchOutFolder
// Clear out dir first
batch_clearDirFirstCkbx.addEventListener("change", () => {
    window.userSettings.batch_clearDirFirst = batch_clearDirFirstCkbx.checked
    saveUserSettings()
})
// Ignore existing output
batch_skipExisting.addEventListener("change", () => {
    window.userSettings.batch_skipExisting = batch_skipExisting.checked
    saveUserSettings()
})
// Batch size
batch_batchSizeInput.addEventListener("change", () => {
    window.userSettings.batch_batchSize = parseInt(batch_batchSizeInput.value)
    saveUserSettings()
})
// ======



reset_settings_btn.addEventListener("click", () => {
    window.confirmModal(`Are you sure you'd like to reset your settings?`).then(confirmation => {
        if (confirmation) {
            window.userSettings.audio.format = "wav"
            window.userSettings.audio.hz = 22050
            window.userSettings.audio.padStart = 0
            window.userSettings.audio.padEnd = 0
            window.userSettings.audio.ffmpeg = false
            window.userSettings.autoPlayGen = false
            window.userSettings.autoReloadVoices = false
            window.userSettings.autoplay = true
            window.userSettings.darkPrompt = false
            window.userSettings.defaultToHiFi = true
            window.userSettings.keepPaceOnNew = true
            window.userSettings.sliderTooltip = true
            window.userSettings.audio.bitdepth = "pcm_s32le"

            window.userSettings.batch_batchSize = 1
            window.userSettings.batch_clearDirFirst= false
            window.userSettings.batch_fastMode = false
            window.userSettings.batch_skipExisting = true
            updateUIWithSettings()
            saveUserSettings()
        }
    })
})
reset_paths_btn.addEventListener("click", () => {
    window.confirmModal(`Are you sure you'd like to reset your paths? This includes the paths for models, and output.`).then(confirmation => {
        if (confirmation) {
            const currGame = window.currentGame[0]

            // Models paths
            const assetFiles = fs.readdirSync(`${path}/assets`)
            assetFiles.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png"))&&fn.split("-").length==4).forEach(assetFileName => {
                const gameId = assetFileName.split("-")[0]
                window.userSettings[`modelspath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/models/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
                if (gameId==currGame) {
                    setting_models_path_input.value = window.userSettings[`modelspath_${gameId}`]
                }
            })

            // Output paths
            const gameDirs = fs.readdirSync(`${path}/models`)
            gameDirs.filter(name => !name.includes(".")).forEach(gameId => {
                window.userSettings[`outpath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/output/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
                if (gameId==currGame) {
                    setting_out_path_input.value = window.userSettings[`outpath_${gameId}`]
                }
            })

            if (window.currentModelButton) {
                window.currentModelButton.click()
            }

            window.userSettings.batchOutFolder = `${__dirname.replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
            batch_outputFolderInput.value = window.userSettings.batchOutFolder

            window.loadAllModels().then(() => window.changeGame(window.currentGame.join("-")))
            saveUserSettings()
        }
    })
})


window.saveUserSettings = saveUserSettings
exports.saveUserSettings = saveUserSettings
exports.deleteFolderRecursive = deleteFolderRecursive

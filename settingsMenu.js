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
if (!Object.keys(window.userSettings).includes("prompt_fontSize")) { // For backwards compatibility
    window.userSettings.prompt_fontSize = 13
}
if (!Object.keys(window.userSettings).includes("bg_gradient_opacity")) { // For backwards compatibility
    window.userSettings.bg_gradient_opacity = 13
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
if (!Object.keys(window.userSettings).includes("s2s_autogenerate")) { // For backwards compatibility
    window.userSettings.s2s_autogenerate = true
}
if (!Object.keys(window.userSettings).includes("s2s_prePitchShift")) { // For backwards compatibility
    window.userSettings.s2s_prePitchShift = false
}
if (!Object.keys(window.userSettings).includes("s2s_removeNoise")) { // For backwards compatibility
    window.userSettings.s2s_removeNoise = false
}
if (!Object.keys(window.userSettings).includes("s2s_noiseRemStrength")) { // For backwards compatibility
    window.userSettings.s2s_noiseRemStrength = 0.25
}

const updateUIWithSettings = () => {
    useGPUCbx.checked = window.userSettings.useGPU
    autoplay_ckbx.checked = window.userSettings.autoplay
    setting_slidersTooltip.checked = window.userSettings.sliderTooltip
    setting_defaultToHiFi.checked = window.userSettings.defaultToHiFi
    setting_keepPaceOnNew.checked = window.userSettings.keepPaceOnNew
    setting_autoplaygenCbx.checked = window.userSettings.autoPlayGen
    setting_darkprompt.checked = window.userSettings.darkPrompt
    setting_prompt_fontSize.value = window.userSettings.prompt_fontSize
    setting_bg_gradient_opacity.value = window.userSettings.bg_gradient_opacity
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

    setting_s2s_autogenerate.checked = window.userSettings.s2s_autogenerate
    setting_s2s_prePitchShift.checked = window.userSettings.s2s_prePitchShift
    setting_s2s_removeNoise.checked = window.userSettings.s2s_removeNoise
    setting_s2s_noiseRemStrength.value = window.userSettings.s2s_noiseRemStrength

    setting_batch_fastmode.checked = window.userSettings.batch_fastMode

    batch_batchSizeInput.value = parseInt(window.userSettings.batch_batchSize)
    batch_skipExisting.checked = window.userSettings.batch_skipExisting
    batch_clearDirFirstCkbx.checked = window.userSettings.batch_clearDirFirst

    const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
    ipcRenderer.send("resize", {height, width})
}
updateUIWithSettings()
saveUserSettings()


// Audio hardware
// ==============
navigator.mediaDevices.enumerateDevices().then(devices => {
    devices = devices.filter(device => device.kind=="audiooutput" && device.deviceId!="communications")

    // Base device
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_base_speaker.appendChild(option)
    })
    setting_base_speaker.addEventListener("change", () => {
        window.userSettings.base_speaker = setting_base_speaker.value
        window.saveUserSettings()

        window.document.querySelectorAll("audio").forEach(audioElem => {
            audioElem.setSinkId(window.userSettings.base_speaker)
        })
    })
    if (Object.keys(window.userSettings).includes("base_speaker")) {
        setting_base_speaker.value = window.userSettings.base_speaker
    } else {
        window.userSettings.base_speaker = setting_base_speaker.value
        window.saveUserSettings()
    }

    // Alternate device
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_alt_speaker.appendChild(option)
    })
    setting_alt_speaker.addEventListener("change", () => {
        window.userSettings.alt_speaker = setting_alt_speaker.value
        window.saveUserSettings()
    })
    if (Object.keys(window.userSettings).includes("alt_speaker")) {
        setting_alt_speaker.value = window.userSettings.alt_speaker
    } else {
        window.userSettings.alt_speaker = setting_alt_speaker.value
        window.saveUserSettings()
    }
})



// Settings Menu
// =============
useGPUCbx.addEventListener("change", () => {
    spinnerModal(window.i18n.CHANGING_DEVICE)
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
                createModal("error", window.i18n.THERE_WAS_A_PROBLEM)
            })
        }
    })
})


const initMenuSetting = (elem, setting, type, callback=undefined, valFn=undefined) => {

    valFn = valFn ? valFn : x=>x

    if (type=="checkbox") {
        elem.addEventListener("click", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.checked)
            } else {
                window.userSettings[setting] = valFn(elem.checked)
            }
            saveUserSettings()
            if (callback) callback()
        })
    } else {
        elem.addEventListener("change", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.value)
            } else {
                window.userSettings[setting] = valFn(elem.value)
            }
            saveUserSettings()
            if (callback) callback()
        })
    }
}

const setPromptTheme = () => {
    if (window.userSettings.darkPrompt) {
        dialogueInput.style.backgroundColor = "rgba(25,25,25,0.9)"
        dialogueInput.style.color = "white"
    } else {
        dialogueInput.style.backgroundColor = "rgba(255,255,255,0.9)"
        dialogueInput.style.color = "black"
    }
}
const setPromptFontSize = () => {
    dialogueInput.style.fontSize = `${window.userSettings.prompt_fontSize}pt`
}
const updateBackground = () => {
    const background = `linear-gradient(0deg, rgba(128,128,128,${window.userSettings.bg_gradient_opacity}) 0px, rgba(0,0,0,0)), url("assets/${window.currentGame.join("-")}")`
    // Fade the background image transition
    rightBG1.style.background = background
    rightBG2.style.opacity = 0
    setTimeout(() => {
        rightBG2.style.background = rightBG1.style.background
        rightBG2.style.opacity = 1
    }, 1000)
}

initMenuSetting(setting_autoplaygenCbx, "autoPlayGen", "checkbox")
initMenuSetting(setting_slidersTooltip, "sliderTooltip", "checkbox")
initMenuSetting(setting_defaultToHiFi, "defaultToHiFi", "checkbox")
initMenuSetting(setting_keepPaceOnNew, "keepPaceOnNew", "checkbox")
initMenuSetting(setting_areload_voices, "autoReloadVoices", "checkbox")
initMenuSetting(setting_output_json, "outputJSON", "checkbox")
initMenuSetting(setting_keepEditorOnVoiceChange, "keepEditorOnVoiceChange", "checkbox")
initMenuSetting(setting_output_num_seq, "filenameNumericalSeq", "checkbox")
initMenuSetting(setting_darkprompt, "darkPrompt", "checkbox", setPromptTheme)
initMenuSetting(setting_prompt_fontSize, "prompt_fontSize", "number", setPromptFontSize)
initMenuSetting(setting_bg_gradient_opacity, "bg_gradient_opacity", "number", updateBackground)

initMenuSetting(setting_batch_fastmode, "batch_fastMode", "checkbox")


initMenuSetting(setting_external_audio_editor, "externalAudioEditor", "text")
initMenuSetting(setting_audio_ffmpeg, "audio.ffmpeg", "checkbox", () => {
    setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
})
initMenuSetting(setting_audio_format, "audio.format", "text")
initMenuSetting(setting_audio_hz, "audio.hz", "text", undefined, parseInt)
initMenuSetting(setting_audio_pad_start, "audio.padStart", "text", undefined, parseInt)
initMenuSetting(setting_audio_pad_end, "audio.padEnd", "text", undefined, parseInt)
initMenuSetting(setting_audio_bitdepth, "audio.bitdepth", "select")
initMenuSetting(setting_audio_amplitude, "audio.amplitude", "select", parseFloat)

initMenuSetting(setting_s2s_autogenerate, "s2s_autogenerate", "checkbox")
initMenuSetting(setting_s2s_prePitchShift, "s2s_prePitchShift", "checkbox")
initMenuSetting(setting_s2s_removeNoise, "s2s_removeNoise", "checkbox")
initMenuSetting(setting_s2s_noiseRemStrength, "s2s_noiseRemStrength", "number", undefined, parseFloat)

initMenuSetting(batch_clearDirFirstCkbx, "batch_clearDirFirst", "checkbox")
initMenuSetting(batch_skipExisting, "batch_skipExisting", "checkbox")
initMenuSetting(batch_batchSizeInput, "batch_batchSize", "text", undefined, parseInt)

setPromptTheme()
setPromptFontSize()

setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg

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
        window.errorModal(window.i18n.ENTER_DIR_PATH)
        batch_outputFolderInput.value = window.userSettings.batchOutFolder
    } else {
        window.userSettings.batchOutFolder = batch_outputFolderInput.value
        saveUserSettings()
    }
})
batch_outputFolderInput.value = window.userSettings.batchOutFolder
// ======



reset_settings_btn.addEventListener("click", () => {
    window.confirmModal(window.i18n.SURE_RESET_SETTINGS).then(confirmation => {
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
    window.confirmModal(window.i18n.SURE_RESET_PATHS).then(confirmation => {
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

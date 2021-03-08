"use strict"

const saveUserSettings = () => localStorage.setItem("userSettings", JSON.stringify(window.userSettings))

// Load user settings
window.userSettings = localStorage.getItem("userSettings") ||
    {
        useGPU: false,
        customWindowSize:`${window.innerHeight},${window.innerWidth}`,
        autoplay: false,
        autoPlayGen: false,
        audio: {
            format: "wav"
        }
    }
if ((typeof window.userSettings)=="string") {
    window.userSettings = JSON.parse(window.userSettings)
}
if (!window.userSettings.audio) { // For backwards compatibility
    window.userSettings.audio = {format: "wav", hz: 22050, padStart: 0, padEnd: 0}
}
if (!window.userSettings.sliderTooltip) { // For backwards compatibility
    window.userSettings.sliderTooltip = true
}
if (!window.userSettings.darkPrompt) { // For backwards compatibility
    window.userSettings.darkPrompt = false
}
if (!window.userSettings.audio.hz) { // For backwards compatibility
    window.userSettings.audio.hz = 22050
}
if (!window.userSettings.audio.padStart) { // For backwards compatibility
    window.userSettings.audio.padStart = 0
}
if (!window.userSettings.audio.padEnd) { // For backwards compatibility
    window.userSettings.audio.padEnd = 0
}
if (!window.userSettings.audio.ffmpeg) { // For backwards compatibility
    window.userSettings.audio.ffmpeg = false
}
if (!window.userSettings.audio.bitdepth) { // For backwards compatibility
    window.userSettings.audio.bitdepth = "pcm_s32le"
}
if (!window.userSettings.vocoder) { // For backwards compatibility
    window.userSettings.vocoder = "256_waveglow"
}
if (!window.userSettings.audio.amplitude) { // For backwards compatibility
    window.userSettings.audio.amplitude = 1
}
if (!window.userSettings.keepPaceOnNew) { // For backwards compatibility
    window.userSettings.keepPaceOnNew = false
}



useGPUCbx.checked = window.userSettings.useGPU
autoplay_ckbx.checked = window.userSettings.autoplay
setting_slidersTooltip.checked = window.userSettings.sliderTooltip
setting_keepPaceOnNew.checked = window.userSettings.keepPaceOnNew
setting_autoplaygenCbx.checked = window.userSettings.autoPlayGen
setting_audio_ffmpeg.checked = window.userSettings.audio.ffmpeg
setting_audio_format.value = window.userSettings.audio.format
setting_audio_hz.value = window.userSettings.audio.hz
setting_audio_pad_start.value = window.userSettings.audio.padStart
setting_audio_pad_end.value = window.userSettings.audio.padEnd
setting_audio_bitdepth.value = window.userSettings.audio.bitdepth
setting_audio_amplitude.value = window.userSettings.audio.amplitude

const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
ipcRenderer.send("resize", {height, width})

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
setting_keepPaceOnNew.addEventListener("click", () => {
    window.userSettings.keepPaceOnNew = setting_keepPaceOnNew.checked
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

setting_audio_ffmpeg.addEventListener("click", () => {
    window.userSettings.audio.ffmpeg = setting_audio_ffmpeg.checked
    setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
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

// Populate the game directories
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


exports.saveUserSettings = saveUserSettings
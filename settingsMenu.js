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
    window.userSettings.audio = {format: "wav"}
}



useGPUCbx.checked = window.userSettings.useGPU
autoplay_ckbx.checked = window.userSettings.autoplay
setting_autoplaygenCbx.checked = window.userSettings.autoPlayGen
setting_audio_format.value = window.userSettings.audio.format

const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
ipcRenderer.send("resize", {height, width})

saveUserSettings()



// Settings Menu
// =============
settingsCog.addEventListener("click", () => {
    settingsContainer.style.opacity = 0
    settingsContainer.style.display = "flex"
    chrome.style.opacity = 0.88
    requestAnimationFrame(() => requestAnimationFrame(() => settingsContainer.style.opacity = 1))
    requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
})
settingsContainer.addEventListener("click", event => {
    if (event.target==settingsContainer) {
        window.closeModal(settingsContainer)
    }
})
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
setting_audio_format.addEventListener("change", () => {
    if (!window.userSettings.audio) { // For backwards compatibility
        window.userSettings.audio = {}
    }
    window.userSettings.audio.format = setting_audio_format.value
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
        // Create and populate the settings menu entry for this
        const outPathElem = createElem("input", {value: window.userSettings[`outpath_${gameFolder}`]})
        outPathElem.addEventListener("change", () => {
            outPathElem.value = outPathElem.value.replace(/\/\//g, "/").replace(/\\/g,"/")
            window.userSettings[`outpath_${gameFolder}`] = outPathElem.value
            saveUserSettings()
            if (window.currentModelButton) {
                window.currentModelButton.click()
            }
        })
        const gameName = fs.readdirSync(`${path}/assets`).find(f => f.startsWith(gameFolder)).split("-").reverse()[0].split(".")[0]
        settingsOptionsContainer.appendChild(createElem("div", [createElem("div", `${gameName} output folder`), createElem("div", outPathElem)]))
    })
})


exports.saveUserSettings = saveUserSettings
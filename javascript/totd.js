"use strict"
// tip of the day

const tips = {
    "1": "You can right-click a voice on the left to hear a preview of the voice",
    "2": "You can right-click the microphone icon after a recording, to hear back the audio you recorded/inserted",
    "3": "There are a number of keyboard shortcuts you can use. Check the info tab for a reference",
    "4": "Check the community guide for tips for how to get the best quality out of the tool. This is linked in the info (i) menu",
    "5": "You can create a multi-letter selection in the editor by Ctrl+clicking several letters",
    "6": "You can shift-click the 'Keep Sample' button (or Ctrl+Shift+S) to first give your file a custom name before saving",
    "7": "You can alt+click editor letters to make a multi-letter selection for the entire word you click on",
    "8": "You can drag+drop multiple .csv or .txt files into batch mode",
    "9": "You can use .txt files in batch mode instead of .csv files, if you first click a voice in the main app to assign the lines to",
    "10": "If you have a compatible NVIDIA GPU, and CUDA installed, you can switch to the CPU+GPU installation. Using the GPU is much faster, especially for batch mode.",
    "11": "The HiFi-GAN vocoder is normally the best quality, but you can also download and use WaveGlow vocoders, if you'd like.",
    "12": "If the 'Keep editor state on voice changes' option is ticked on, you can generate a line using one voice, then switch to a different voice, and click the 'Generate Voice' button again to generate a line using the new voice, but using a similar speaking style to the first voice.",
    "13": "If you set the 'Alternative Output device' to something other than the default device, you can Ctrl-click when playing audio, to have it play on a different speaker. You can couple this with something like Voicemeeter Banana split, to have the app speak for you over the microphone, for voice chat, or other audio recording.",
    "14": "If you add the path to an audio editing program to the 'External Program for Editing audio' setting, you can open generated audio straight in that program in one click, from the output records on the main page",
    "15": "If you install ffmpeg (at least version 4.3), you can automatically directly apply a few different audio post processing tasks on the generated audio. This can include Hz resampling, silence padding to the start and/or end of the audio, bit depth, loudness, and different audio formats. You can also tick on the option to pre-apply these to the temporary preview audio sample.",
    "16": "You can tick on the 'Fast mode' for batch mode to parallelize the audio generation and the audio output (via ffmpeg for example)",
    "17": "You can enable multiprocessing for ffmpeg file output in batch mode, to speed up the output process. This is especially useful if you use a large batch size, and your CPU has plenty of threads. This can be used together with Fast Mode.",
    "18": "If you're having trouble formatting a .csv file for batch mode, you can change the delimiter in the settings to something else (for example a pipe symbol '|')",
    "19": "You can change the folder location of your output files, as well as the models. I'd recommend keeping your model files on an SSD, to reduce the loading time.",
    "20": "Use the voice embeddings search menu to get a 3D visualisation of all the voices in the app (including some 'officially' trained voices not downloaded yet). You can use this as a reference for voice similarly search, to see what other voices there are, which sound similar to a particular voice.",
    "21": "You can right click on the points in the 3D voice embeddings visualisation, to hear a preview of that voice. This will only work for the voices you have installed, locally.",
    "22": "The app is customisable via third-party plugins. Plugins can be managed from the plugins menu, and they can change, or add to the front end app functionality/looks (the UI), as well as the python back-end (the machine learning code). If you're interested in developing such a plugin, there is a full developer reference on the GitHub wiki, here: https://github.com/DanRuta/xvasynth-community-guide",
    "23": "If you log into nexusmods.com from within the app, you can check for new and updated voice models on your chosen Nexus pages. You can also endorse these, as well as any plugins configured with a nexus link. If you have a premium membership for the Nexus, you can also download (or batch download) all available voices, and have them installed automatically.",
    "24": "You can manage the list of Nexus pages to check for voice models by clicking the 'Manage Repos' button in the Nexus menu, or by editing the repositories.txt file",
    "25": "You can enable/disable error sounds in the settings. You can also pick a different sound, if you'd prefer something else",
    "26": "You can resize the window by dragging one of the bottom corners"
}

window.totd_state = {
    startupChecked: false,
    filteredIDs: [],
    tipPageIndex: 0
}

const initTipOfTheDayMenu = (now, tipIDs) => {

    window.totd_state.filteredIDs = tipIDs

    totdContainer.style.opacity = 1
    totdContainer.style.display = "flex"
    chrome.style.opacity = 1
    requestAnimationFrame(() => requestAnimationFrame(() => totdContainer.style.opacity = 1))

    return new Promise(resolve => {
        // Close button
        totd_close.addEventListener("click", () => {
            closeModal(totdContainer)
            resolve()
        })

        localStorage.setItem("totd_lastDate", now.toJSON(now))
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[0]]

        totd_counter.innerHTML = `1/${window.totd_state.filteredIDs.length}`

        saveSeenTip(window.totd_state.filteredIDs[0])
    })
}


const saveSeenTip = ID => {
    let seenTipIDs = localStorage.getItem("totd_seenIDs")
    seenTipIDs = seenTipIDs ? seenTipIDs.split(",") : []
    seenTipIDs = new Set(seenTipIDs)
    seenTipIDs.add(ID)
    localStorage.setItem("totd_seenIDs", Array.from(seenTipIDs).join(","))
}

setting_btnShowTOTD.addEventListener("click", () => {
    window.showTipIfEnabledAndNewDay(true)
})


totdPrevTipBtn.addEventListener("click", () => {
    const newIndex = Math.max(0, window.totd_state.tipPageIndex-1)
    if (newIndex!=window.totd_state.tipPageIndex) {
        window.totd_state.tipPageIndex = newIndex
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[window.totd_state.tipPageIndex]]
        saveSeenTip(window.totd_state.filteredIDs[window.totd_state.tipPageIndex])
        totd_counter.innerHTML = `${window.totd_state.tipPageIndex+1}/${window.totd_state.filteredIDs.length}`
    }
})

totdNextTipBtn.addEventListener("click", () => {
    const newIndex = Math.min(window.totd_state.filteredIDs.length-1, window.totd_state.tipPageIndex+1)
    if (newIndex!=window.totd_state.tipPageIndex) {
        window.totd_state.tipPageIndex = newIndex
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[window.totd_state.tipPageIndex]]
        saveSeenTip(window.totd_state.filteredIDs[window.totd_state.tipPageIndex])
        totd_counter.innerHTML = `${window.totd_state.tipPageIndex+1}/${window.totd_state.filteredIDs.length}`
    }
})


window.showTipIfEnabledAndNewDay = (justShowIt) => {

    window.totd_state.startupChecked = true

    return new Promise(async resolve => {
        const lastDate = localStorage.getItem("totd_lastDate")
        const now = new Date()

        // If this has never happened before, or the last date is not today, then show the tip menu
        if (justShowIt || !lastDate || lastDate.split("T")[0]!=now.toJSON().split("T")[0]) {

            // If the tips of the day are enabled
            if (justShowIt || window.userSettings.showTipOfTheDay) {

                let shuffledTipIDs = window.shuffle(Object.keys(tips))

                // If only new/unseen tips are to be shown, get the seen list, and filter out the seen tips
                if (window.userSettings.showUnseenTipOfTheDay) {
                    let seenTipIDs = localStorage.getItem("totd_seenIDs")
                    if (seenTipIDs) {
                        seenTipIDs = seenTipIDs.split(",")
                        shuffledTipIDs = shuffledTipIDs.filter(id => !seenTipIDs.includes(id))
                    }
                }

                // If there are any tips remaining, after any filtering, then show the menu
                if (shuffledTipIDs && shuffledTipIDs.length) {
                    await initTipOfTheDayMenu(now, shuffledTipIDs)
                    resolve()
                } else if (justShowIt) {
                    window.errorModal("There are no unseen tips left to show. Untick the 'Only show unseen tips' setting to show all tips.")
                }
            } else {
                resolve()
            }
        } else {
            resolve()
        }
    })
}


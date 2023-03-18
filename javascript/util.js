"use strict"

const er = require('@electron/remote')

/**
 * String.prototype.replaceAll() polyfill
 * https://gomakethings.com/how-to-replace-a-section-of-a-string-with-another-one-with-vanilla-js/
 * @author Chris Ferdinandi
 * @license MIT
 */
if (!String.prototype.replaceAll) {
    String.prototype.replaceAll = function(str, newStr){

        // If a regex pattern
        if (Object.prototype.toString.call(str).toLowerCase() === '[object regexp]') {
            return this.replace(str, newStr);
        }

        // If a string
        return this.replace(new RegExp(str, 'g'), newStr);

    };
}


window.toggleSpinnerButtons = () => {
    const spinnerVisible = window.getComputedStyle(spinner).display == "block"
    spinner.style.display = spinnerVisible ? "none" : "block"
    keepSampleButton.style.display = spinnerVisible ? "block" : "none"
    generateVoiceButton.style.display = spinnerVisible ? "block" : "none"
    samplePlayPause.style.display = spinnerVisible ? "flex" : "none"
}

window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))
window.errorModal = message => {
    window.errorModalHasOpened = true
    if (window.userSettings.useErrorSound) {
        const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
            src: window.userSettings.errorSoundFile
        }))
        audioPreview.setSinkId(window.userSettings.base_speaker)
    }
    window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent?window.batch_state.taskBarPercent:1, {mode: "error"})
    return new Promise(topResolve => {
        createModal("error", message).then(() => {
            window.electronBrowserWindow.setProgressBar(window.batch_state.taskBarPercent?window.batch_state.taskBarPercent:-1, {mode: window.batch_state.state?"normal":"paused"})
            topResolve()
        })
    })
}
window.createModal = (type, message) => {
    dialogueInput.blur()
    return new Promise(resolve => {
        modalContainer.innerHTML = ""
        const displayMessage = message.prompt ? message.prompt : message
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span.createModalContents", displayMessage))
        modal.dataset.type = type

        if (type=="confirm") {
            const yesButton = createElem("button", {style: {background: `#${themeColour}`}})
            yesButton.innerHTML = window.i18n.YES
            const noButton = createElem("button", {style: {background: `#${themeColour}`}})
            noButton.innerHTML = window.i18n.NO
            modal.appendChild(createElem("div", yesButton, noButton))

            yesButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(true)
                })
            })
            noButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="error") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = window.i18n.CLOSE
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="prompt") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = window.i18n.SUBMIT
            const inputElem = createElem("input", {type: "text", value: message.value})
            modal.appendChild(createElem("div", inputElem))
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(inputElem.value)
                })
            })
        } else {
            modal.appendChild(createElem("div.spinner", {style: {borderLeftColor: document.querySelector("button").style.background}}))
        }

        modalContainer.appendChild(modal)
        modalContainer.style.opacity = 0
        modalContainer.style.display = "flex"

        requestAnimationFrame(() => requestAnimationFrame(() => modalContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
    })
}
window.closeModal = (container=undefined, notThisOne=undefined, skipIfErrorOpen=false) => {
    return new Promise(resolve => {
        if (window.errorModalHasOpened && skipIfErrorOpen) {
            return resolve()
        }
        window.errorModalHasOpened = false
        const allContainers = [batchGenerationContainer, gameSelectionContainer, updatesContainer, infoContainer, settingsContainer, patreonContainer, pluginsContainer, modalContainer, nexusContainer, embeddingsContainer, totdContainer, nexusReposContainer, EULAContainer, arpabetContainer, styleEmbeddingsContainer]
        const containers = container==undefined ? allContainers : (Array.isArray(container) ? container.filter(c=>c!=undefined) : [container])

        notThisOne = Array.isArray(notThisOne) ? notThisOne : (notThisOne==undefined ? [] : [notThisOne])

        containers.forEach(cont => {
            // Fade out the containers except the exceptions
            if (cont!=undefined && !notThisOne.includes(cont)) {
                cont.style.opacity = 0
            }
        })

        const someOpenContainer = allContainers.filter(c=>c!=undefined).find(cont => cont.style.opacity==1 && cont.style.display!="none" && cont!=modalContainer)
        if (!someOpenContainer || someOpenContainer==container) {
            chromeBar.style.opacity = 0.88
        }

        setTimeout(() => {
            if (window.errorModalHasOpened && skipIfErrorOpen) {
            } else {
                containers.forEach(cont => {
                    // Hide the containers except the exceptions
                    if (cont!=undefined && !notThisOne.includes(cont)) {
                        cont.style.display = "none"
                        const someOpenContainer2 = allContainers.filter(c=>c!=undefined).find(cont => cont.style.opacity==1 && cont.style.display!="none" && cont!=modalContainer)
                        if (!someOpenContainer2 || someOpenContainer2==container) {
                            chromeBar.style.opacity = 0.88
                        }
                    }
                })
                window.errorModalHasOpened = false
            }
            // resolve()
        }, 200)
        try {
            activeModal.remove()
        } catch (e) {}
        resolve()
    })
}


window.setTheme = (meta) => {

    const primaryColour = meta.themeColourPrimary
    const secondaryColour = meta.themeColourSecondary
    const gameName = meta.gameName

    if (window.userSettings.showDiscordStatus) {
        ipcRenderer.send('updateDiscord', {details: gameName})
    }

    // Change batch panel colours, if it is initialized
    try {
        Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(pluginsRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(styleembsRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(nexusRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(nexusSearchHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(nexusReposUsedHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(embeddingsRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(arpabetWordsListHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        window.sequenceEditor.grabbers.forEach(grabber => grabber.fillStyle = `#${primaryColour}`)
        window.sequenceEditor.energyGrabbers.forEach(grabber => grabber.fillStyle = `#${primaryColour}`)
    } catch (e) {}

    const background = `linear-gradient(0deg, rgba(128,128,128,${window.userSettings.bg_gradient_opacity}) 0px, rgba(0,0,0,0)), url("assets/${meta.assetFile}")`
    Array.from(document.querySelectorAll("button:not(.fixedColour)")).forEach(e => e.style.background = `#${primaryColour}`)
    Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.background = `#${primaryColour}`)
    Array.from(document.querySelectorAll(".spinner")).forEach(e => e.style.borderLeftColor = `#${primaryColour}`)

    if (secondaryColour) {
        Array.from(document.querySelectorAll("button:not(.fixedColour)")).forEach(e => e.style.color = `#${secondaryColour}`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.color = `#${secondaryColour}`)
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.textShadow  = `none`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.textShadow  = `none`)
    } else {
        Array.from(document.querySelectorAll("button:not(.fixedColour)")).forEach(e => e.style.color = `white`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.color = `white`)
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.textShadow = `0 0 2px black`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.textShadow = `0 0 2px black`)
    }

    if (window.wavesurfer) {
        window.wavesurfer.setWaveColor(`#${window.currentGame.themeColourPrimary}`)
    }

    // Fade the background image transition
    rightBG1.style.background = background
    rightBG2.style.opacity = 0
    setTimeout(() => {
        rightBG2.style.background = rightBG1.style.background
        rightBG2.style.opacity = 1
    }, 1000)

    cssHack.innerHTML = `::selection {
        background: #${primaryColour};
    }
    ::-webkit-scrollbar-thumb {
        background-color: #${primaryColour} !important;
    }
    .slider::-webkit-slider-thumb {
        background-color: #${primaryColour} !important;
    }
    a {color: #${primaryColour}};
    #batchRecordsHeader > div {background-color: #${primaryColour} !important;}
    #pluginsRecordsHeader > div {background-color: #${primaryColour} !important;}

    .invertedButton {
       background: none !important;
       border: 2px solid #${primaryColour} !important;
    }

    `
    if (secondaryColour) {
        cssHack.innerHTML += `
            #batchRecordsHeader > div {color: #${secondaryColour} !important;text-shadow: none}
            #pluginsRecordsHeader > div {color: #${secondaryColour} !important;text-shadow: none}
        `
    } else {
        cssHack.innerHTML += `
            #batchRecordsHeader > div {color: white !important;text-shadow: 0 0 2px black;}
            #pluginsRecordsHeader > div {color: white !important;text-shadow: 0 0 2px black;}
        `
    }
}

window.getAudioPlayTriangleSVG = () => {
    const div = createElem("div", `<svg class="renameSVG" version="1.0" xmlns="http://www.w3.org/2000/svg" width="770.000000pt" height="980.000000pt" viewBox="0 0 770.000000 980.000000"
 preserveAspectRatio="xMidYMid meet"><g transform="translate(0.000000,980.000000) scale(0.100000,-0.100000)"fill="#555555" stroke="none">
        <path d="M26 9718 c-3 -46 -9 -2249 -12 -4897 -5 -4085 -4 -4813 8 -4809 8 3
        389 244 848 535 459 291 1598 1013 2530 1603 2872 1819 3648 2311 4182 2656
        60 38 108 75 108 81 0 55 -595 448 -2855 1886 -1259 801 -4552 2877 -4792
        3020 -8 5 -13 -16 -17 -75z"/>
        </g>
        </svg>`)
    return div
}


window.addEventListener("resize", e => {
    window.userSettings.customWindowSize = `${window.innerHeight},${window.innerWidth}`
    saveUserSettings()
})

// Keyboard actions
// ================
window.addEventListener("keyup", event => {
    if (!event.ctrlKey) {
        window.ctrlKeyIsPressed = false
    }
    if (!event.shiftKey) {
        window.shiftKeyIsPressed = false
    }
})

window.addEventListener("keydown", event => {

    if (event.ctrlKey) {
        window.ctrlKeyIsPressed = true
    }
    if (event.shiftKey) {
        window.shiftKeyIsPressed = true
    }

    if (event.ctrlKey && event.key.toLowerCase()=="r") {
        location.reload()
    }

    if (event.ctrlKey && event.shiftKey && event.key.toLowerCase()=="i") {
        window.electron = require("electron")
        er.BrowserWindow.getFocusedWindow().webContents.openDevTools()
        return
    }

    if (event.ctrlKey) {
        window.ctrlKeyIsPressed = true
    }

    // The Enter key to submit text input prompts in modals
    if (event.key=="Enter" && modalContainer.style.display!="none" && event.target.tagName=="INPUT") {
        activeModal.querySelector("button").click()
    }
    const key = event.key.toLowerCase()

    // CTRL-S: Keep sample
    if (key=="s" && event.ctrlKey && !event.shiftKey) {
        console.log("here")
        keepSampleFunction(false)
    }
    // CTRL-SHIFT-S: Keep sample (but with rename prompt)
    if (key=="s" && event.ctrlKey && event.shiftKey) {
        keepSampleFunction(true)
    }

    // Disable keyboard controls while in a text input
    if (event.target.tagName=="INPUT" && event.target.tagName!=dialogueInput) {
        return
    }

    if (event.target==dialogueInput || event.target==letterPitchNumb || event.target==letterLengthNumb) {
        // Enter: Generate sample
        if (event.key=="Enter") {
            generateVoiceButton.click()
            event.preventDefault()
        }
        return
    }

    // Escape: close modals
    if (key=="escape") {
        closeModal()
    }
    // Space: bring focus to the input textarea
    if (key==" ") {
        setTimeout(() => dialogueInput.focus(), 0)
    }
    // Create selection for all of the editor letters
    if (key=="a" && event.ctrlKey && !event.shiftKey) {
        window.sequenceEditor.letterFocus = []
        window.sequenceEditor.dursNew.forEach((_,i) => {
            window.sequenceEditor.letterFocus.push(i)
            window.sequenceEditor.setLetterFocus(i, true)
        })
        event.preventDefault()
    }
    // Y/N for prompt modals
    if (key=="y" || key=="n" || key==" ") {
        if (document.querySelector("#activeModal")) {
            const buttons = Array.from(document.querySelector("#activeModal").querySelectorAll("button"))
            const yesBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="yes")
            const noBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="no")
            if (key=="y") yesBtn.click()
            if (key==" ") yesBtn.click()
            if (key=="n") noBtn.click()
        }
    }
    // Left/Right arrows: Move between letter focused (clears multi-select, picks the min(0,first-1) if left, max($L, last+1) if right)
    // SHIFT-Left/Right: multi-letter create selection range
    if ((key=="arrowleft" || key=="arrowright") && !event.ctrlKey) {
        event.preventDefault()
        if (window.sequenceEditor.letterFocus.length==0) {
            window.sequenceEditor.setLetterFocus(0)
        } else if (window.sequenceEditor.letterFocus.length==1) {
            const curr_l = window.sequenceEditor.letterFocus[0]
            const newIndex = key=="arrowleft" ? Math.max(0, curr_l-1) : Math.min(curr_l+1, window.sequenceEditor.letters.length-1)
            window.sequenceEditor.setLetterFocus(newIndex, event.shiftKey)
        } else {
            if (key=="arrowleft") {
                window.sequenceEditor.setLetterFocus(Math.max(0, Math.min(...window.sequenceEditor.letterFocus)-1), event.shiftKey)
            } else {
                window.sequenceEditor.setLetterFocus(Math.min(Math.max(...window.sequenceEditor.letterFocus)+1, window.sequenceEditor.letters.length-1), event.shiftKey)
            }
        }
    }

    // Up/Down arrows: Move pitch up/down for the letter(s) selected
    if ((key=="arrowup" || key=="arrowdown") && !event.ctrlKey) {
        event.preventDefault()
        if (window.sequenceEditor.letterFocus.length) {
            window.sequenceEditor.letterFocus.forEach(li => {
                window.sequenceEditor.pitchNew[li] += (key=="arrowup" ? 0.1 : -0.1)
                window.sequenceEditor.grabbers[li].setValueFromValue(window.sequenceEditor.pitchNew[li])
                window.sequenceEditor.hasChanged = true
            })
            if (window.sequenceEditor.autoInferTimer != null) {
                clearTimeout(window.sequenceEditor.autoInferTimer)
                window.sequenceEditor.autoInferTimer = null
            }
            if (autoplay_ckbx.checked) {
                window.sequenceEditor.autoInferTimer = setTimeout(infer, 500)
            }

            if (window.sequenceEditor.letterFocus.length==1) {
                letterPitchNumb.value = window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]
            }
        }
    }

    // CTRL+Left/Right arrows: change the sequence-wide pacing
    if ((key=="arrowleft" || key=="arrowright") && event.ctrlKey) {
        if (event.altKey) {
            window.sequenceEditor.letterFocus.forEach(li => {
                window.sequenceEditor.dursNew[li] = window.sequenceEditor.dursNew[li] + (key=="arrowleft"? -0.1 : 0.1)
                window.sequenceEditor.hasChanged = true
            })
            if (window.sequenceEditor.autoInferTimer != null) {
                clearTimeout(window.sequenceEditor.autoInferTimer)
                window.sequenceEditor.autoInferTimer = null
            }
            if (autoplay_ckbx.checked) {
                window.sequenceEditor.autoInferTimer = setTimeout(infer, 500)
            }
            window.sequenceEditor.sliderBoxes.forEach((box, i) => box.setValueFromValue(window.sequenceEditor.dursNew[i]))

            if (window.sequenceEditor.letterFocus.length==1) {
                letterLengthNumb.value = parseInt(window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]]*100)/100
            }
        } else {
            pace_slid.value = parseFloat(pace_slid.value) + (key=="arrowleft"? -0.01 : 0.01)
            paceNumbInput.value = pace_slid.value
            const new_lengths = window.sequenceEditor.dursNew.map((v,l) => v * pace_slid.value)
            window.sequenceEditor.sliderBoxes.forEach((box, i) => box.setValueFromValue(window.sequenceEditor.dursNew[i]))
            window.sequenceEditor.pacing = parseFloat(pace_slid.value)
            window.sequenceEditor.init()

        }
    }

    // CTRL+Up/Down arrows: increase/decrease buttons
    if (key=="arrowup" && event.ctrlKey && !event.shiftKey) {
        increase_btn.click()
    }
    if (key=="arrowdown" && event.ctrlKey && !event.shiftKey) {
        decrease_btn.click()
    }
    // CTRL+SHIFT+Up/Down arrows: amplify/flatten buttons
    if (key=="arrowup" && event.ctrlKey && event.shiftKey) {
        amplify_btn.click()
    }
    if (key=="arrowdown" && event.ctrlKey && event.shiftKey) {
        flatten_btn.click()
    }
})



window.setupModal = (openingButton, modalContainerElem, callback, exitCallback) => {
    if (openingButton) {
        openingButton.addEventListener("click", () => {
            closeModal(undefined, modalContainerElem).then(() => {
                modalContainerElem.style.opacity = 0
                modalContainerElem.style.display = "flex"
                requestAnimationFrame(() => requestAnimationFrame(() => {
                    modalContainerElem.style.opacity = 1
                    chromeBar.style.opacity = 1
                    requestAnimationFrame(() => {
                        setTimeout(() => {
                            if (callback) {
                                callback()
                            }
                        }, 250)
                    })
                }))
            })
        })
    }
    modalContainerElem.addEventListener("click", event => {
        if (event.target==modalContainerElem) {
            if (exitCallback) {
                exitCallback()
            }
            window.closeModal(modalContainerElem)
        }
    })
}

window.checkVersionRequirements = (requirements, appVersion, checkMax=false) => {

    if (!requirements) {
        return true
    }

    const appVersionRequirement = requirements.toString().split(".").map(v=>parseInt(v))
    const appVersionInts = appVersion.replace("v", "").split(".").map(v=>parseInt(v))
    let appVersionOk = true

    if (checkMax) {

        if (appVersionRequirement[0] >= appVersionInts[0] ) {
            if (appVersionRequirement.length>1 && parseInt(appVersionRequirement[0]) == appVersionInts[0]) {
                if (appVersionRequirement[1] >= appVersionInts[1] ) {
                    if (appVersionRequirement.length>2 && parseInt(appVersionRequirement[1]) == appVersionInts[1]) {
                        if (appVersionRequirement[2] >= appVersionInts[2] ) {
                        } else {
                            appVersionOk = false
                        }
                    }
                } else {
                    appVersionOk = false
                }
            }
        } else {
            appVersionOk = false
        }


    } else {
        if (appVersionRequirement[0] <= appVersionInts[0] ) {
            if (appVersionRequirement.length>1 && parseInt(appVersionRequirement[0]) == appVersionInts[0]) {
                if (appVersionRequirement[1] <= appVersionInts[1] ) {
                    if (appVersionRequirement.length>2 && parseInt(appVersionRequirement[1]) == appVersionInts[1]) {
                        if (appVersionRequirement[2] <= appVersionInts[2] ) {
                        } else {
                            appVersionOk = false
                        }
                    }
                } else {
                    appVersionOk = false
                }
            }
        } else {
            appVersionOk = false
        }
    }
    return appVersionOk
}


// https://stackoverflow.com/questions/18052762/remove-directory-which-is-not-empty
const path = require('path')
window.deleteFolderRecursive = function (directoryPath, keepRoot=false) {
if (fs.existsSync(directoryPath)) {
    fs.readdirSync(directoryPath).forEach((file, index) => {
      const curPath = path.join(directoryPath, file);
      if (fs.lstatSync(curPath).isDirectory()) {
       // recurse
        window.deleteFolderRecursive(curPath);
      } else {
        // delete file
        fs.unlinkSync(curPath);
      }
    });
    if (!keepRoot) {
        fs.rmdirSync(directoryPath);
    }
  }
};

window.createFolderRecursive = (pathToMake) => {
    console.log("createFolderRecursive", pathToMake)
    pathToMake.split('/').reduce((directories, directory) => {
        directories += `${directory}/`

        if (!fs.existsSync(directories)) {
          fs.mkdirSync(directories)
        }

        return directories
      }, '')
}

window.uuidv4 = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8)
        return v.toString(16)
    })
}

// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
window.shuffle = (array) => {
    var currentIndex = array.length,  randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
          array[randomIndex], array[currentIndex]];
    }

    return array;
}


// Just for easier packaging of the voice models for publishing - yes, lazy
window.packageVoice_variantIndex = 0
window.packageVoice = (doVoicePreviewCreate, variants, {modelsPath, gameId}={}) => {

    const {voiceId, hifi} = variants[window.packageVoice_variantIndex]

    if (doVoicePreviewCreate) {
        const files = fs.readdirSync(`./output`).filter(fname => fname.includes("temp-") && fname.includes(".wav"))
        if (files.length) {
            const options = {
                hz: window.userSettings.audio.hz,
                padStart: window.userSettings.audio.padStart,
                padEnd: window.userSettings.audio.padEnd,
                bit_depth: window.userSettings.audio.bitdepth,
                amplitude: window.userSettings.audio.amplitude,
                pitchMult: window.userSettings.audio.pitchMult,
                tempo: window.userSettings.audio.tempo
            }

            doFetch(`http://localhost:8008/outputAudio`, {
                method: "Post",
                body: JSON.stringify({
                    input_path: `./output/${files[0]}`,
                    isBatchMode: false,
                    output_path: `${modelsPath}/${voiceId}_raw.wav`,
                    options: JSON.stringify(options)
                })
            }).then(r=>r.text()).then(() => {
                try {
                    fs.unlinkSync(`${modelsPath}/${voiceId}.wav`)
                } catch (e) {}
                doFetch(`http://localhost:8008/normalizeAudio`, {
                    method: "Post",
                    body: JSON.stringify({
                        input_path: `${modelsPath}/${voiceId}_raw.wav`,
                        output_path: `${modelsPath}/${voiceId}.wav`
                    })
                }).then(r=>r.text()).then((resp) => {
                    console.log(resp)
                    fs.unlinkSync(`${modelsPath}/${voiceId}_raw.wav`)
                })
            })
        }
    } else {
        fs.mkdirSync(`./build/${voiceId}`)
        fs.mkdirSync(`./build/${voiceId}/resources`)
        fs.mkdirSync(`./build/${voiceId}/resources/app`)
        fs.mkdirSync(`./build/${voiceId}/resources/app/models`)
        fs.mkdirSync(`./build/${voiceId}/resources/app/models/${gameId}`)
        fs.copyFileSync(`${modelsPath}/${voiceId}.json`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.json`)
        fs.copyFileSync(`${modelsPath}/${voiceId}.wav`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.wav`)
        fs.copyFileSync(`${modelsPath}/${voiceId}.pt`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.pt`)
        if (hifi) {
            fs.copyFileSync(`${modelsPath}/${voiceId}.hg.pt`, `./build/${voiceId}/resources/app/models/${gameId}/${voiceId}.hg.pt`)
        }
        zipdir(`./build/${voiceId}`, {saveTo: `./build/${voiceId}.zip`}, (err, buffer) => deleteFolderRecursive(`./build/${voiceId}`))
    }
}


const assetFiles = fs.readdirSync(`${window.path}/assets`)
const jsonFiles = fs.readdirSync(`${window.path}/assets`).filter(fn => fn.endsWith(".json"))
const missingAssetFiles = jsonFiles.filter(jsonFile => !(assetFiles.includes(jsonFile.replace(".json", ".png"))||assetFiles.includes(jsonFile.replace(".json", ".jpg"))) )
if (missingAssetFiles.length) {
    noAssetFilesFoundMessage.style.display = "block"
    assetDirLink.addEventListener("click", () => {
        shell.showItemInFolder((require("path")).resolve(`${window.path}/assets/other.jpg`))
    })
}



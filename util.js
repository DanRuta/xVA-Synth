window.toggleSpinnerButtons = () => {
    const spinnerVisible = window.getComputedStyle(spinner).display == "block"
    spinner.style.display = spinnerVisible ? "none" : "block"
    keepSampleButton.style.display = spinnerVisible ? "block" : "none"
    generateVoiceButton.style.display = spinnerVisible ? "block" : "none"
    samplePlay.style.display = spinnerVisible ? "flex" : "none"
}

window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))
window.errorModal = message => new Promise(resolve => resolve(createModal("error", message)))
window.createModal = (type, message) => {
    return new Promise(resolve => {
        modalContainer.innerHTML = ""
        const displayMessage = message.prompt ? message.prompt : message
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span", displayMessage))
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
        requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
    })
}
window.closeModal = (container=undefined, notThisOne=undefined) => {
    return new Promise(resolve => {
        const allContainers = [batchGenerationContainer, gameSelectionContainer, updatesContainer, infoContainer, settingsContainer, patreonContainer, container, pluginsContainer, modalContainer, s2sSelectContainer]
        const containers = container==undefined ? allContainers : [container]
        containers.forEach(cont => {
            if ((notThisOne!=undefined&&notThisOne!=cont) && (notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                cont.style.opacity = 0
            }
        })

        const someOpenContainer = allContainers.find(container => container!=undefined && container.style.opacity==1 && container.style.display!="none" && container!=modalContainer)
        if (!someOpenContainer || someOpenContainer==container) {
            chrome.style.opacity = 0.88
        }

        containers.forEach(cont => {
            if ((notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                cont.style.opacity = 0
            }
        })

        setTimeout(() => {
            containers.forEach(cont => {
                if ((notThisOne==undefined || notThisOne!=cont) && cont!=undefined) {
                    cont.style.display = "none"
                    const someOpenContainer2 = allContainers.find(container => container!=undefined && container.style.opacity==1 && container.style.display!="none" && container!=modalContainer)
                    if (!someOpenContainer2 || someOpenContainer2==container) {
                        chrome.style.opacity = 0.88
                    }
                }
            })
        }, 200)
        try {
            activeModal.remove()
        } catch (e) {}
        resolve()
    })
}


window.setTheme = (meta) => {
    const primaryColour = meta[1]
    const secondaryColour = meta.length==5?meta[2]:undefined

    // Change batch panel colours, if it is initialized
    try {
        Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}
    try {
        Array.from(pluginsRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${primaryColour}`)
    } catch (e) {}

    const background = `linear-gradient(0deg, rgba(128,128,128,${window.userSettings.bg_gradient_opacity}) 0px, rgba(0,0,0,0)), url("assets/${meta.join("-")}")`
    Array.from(document.querySelectorAll("button")).forEach(e => e.style.background = `#${primaryColour}`)
    Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.background = `#${primaryColour}`)
    Array.from(document.querySelectorAll(".spinner")).forEach(e => e.style.borderLeftColor = `#${primaryColour}`)

    if (secondaryColour) {
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.color = `#${secondaryColour}`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.color = `#${secondaryColour}`)
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.textShadow  = `none`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.textShadow  = `none`)
    } else {
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.color = `white`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.color = `white`)
        Array.from(document.querySelectorAll("button")).forEach(e => e.style.textShadow = `0 0 2px black`)
        Array.from(document.querySelectorAll(".voiceType")).forEach(e => e.style.textShadow = `0 0 2px black`)
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
})

window.addEventListener("keydown", event => {

    if (event.ctrlKey) {
        window.ctrlKeyIsPressed = true
    }

    // The Enter key to submit text input prompts in modals
    if (event.key=="Enter" && modalContainer.style.display!="none" && event.target.tagName=="INPUT") {
        activeModal.querySelector("button").click()
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

    const key = event.key.toLowerCase()

    // Escape: close modals
    if (key=="escape") {
        closeModal()
    }
    // Space: bring focus to the input textarea
    if (key==" ") {
        setTimeout(() => dialogueInput.focus(), 0)
    }
    // CTRL-S: Keep sample
    if (key=="s" && event.ctrlKey && !event.shiftKey) {
        keepSampleFunction(false)
    }
    // CTRL-SHIFT-S: Keep sample (but with rename prompt)
    if (key=="s" && event.ctrlKey && event.shiftKey) {
        keepSampleFunction(true)
    }
    // Create selection for all of the editor letters
    if (key=="a" && event.ctrlKey && !event.shiftKey) {
        window.pitchEditor.letterFocus = []
        window.pitchEditor.dursNew.forEach((_,i) => {
            window.pitchEditor.letterFocus.push(i)
            setLetterFocus(i, true)
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
        if (window.pitchEditor.letterFocus.length==0) {
            setLetterFocus(0)
        } else if (window.pitchEditor.letterFocus.length==1) {
            const curr_l = window.pitchEditor.letterFocus[0]
            const newIndex = key=="arrowleft" ? Math.max(0, curr_l-1) : Math.min(curr_l+1, window.pitchEditor.letters.length-1)
            setLetterFocus(newIndex, event.shiftKey)
        } else {
            if (key=="arrowleft") {
                setLetterFocus(Math.max(0, Math.min(...window.pitchEditor.letterFocus)-1), event.shiftKey)
            } else {
                setLetterFocus(Math.min(Math.max(...window.pitchEditor.letterFocus)+1, window.pitchEditor.letters.length-1), event.shiftKey)
            }
        }
    }

    // Up/Down arrows: Move pitch up/down for the letter(s) selected
    if ((key=="arrowup" || key=="arrowdown") && !event.ctrlKey) {
        event.preventDefault()
        if (window.pitchEditor.letterFocus.length) {
            window.pitchEditor.letterFocus.forEach(li => {
                sliders[li].value = parseFloat(sliders[li].value) + (key=="arrowup" ? 0.1 : -0.1)
                window.pitchEditor.pitchNew[li] = parseFloat(sliders[li].value)
                has_been_changed = true
            })
            if (autoinfer_timer != null) {
                clearTimeout(autoinfer_timer)
                autoinfer_timer = null
            }
            if (autoplay_ckbx.checked) {
                autoinfer_timer = setTimeout(infer, 500)
            }

            if (window.pitchEditor.letterFocus.length==1) {
                letterPitchNumb.value = sliders[window.pitchEditor.letterFocus[0]].value
            }
        }
    }

    // CTRL+Left/Right arrows: change the sequence-wide pacing
    if ((key=="arrowleft" || key=="arrowright") && event.ctrlKey) {
        if (event.altKey) {
            window.pitchEditor.letterFocus.forEach(li => {
                window.pitchEditor.dursNew[li] = window.pitchEditor.dursNew[li] + (key=="arrowleft"? -0.1 : 0.1)
                has_been_changed = true
            })
            if (autoinfer_timer != null) {
                clearTimeout(autoinfer_timer)
                autoinfer_timer = null
            }
            if (autoplay_ckbx.checked) {
                autoinfer_timer = setTimeout(infer, 500)
            }
            window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, window.pitchEditor.dursNew[l]* 10 + 50, null))
        } else {
            pace_slid.value = parseFloat(pace_slid.value) + (key=="arrowleft"? -0.01 : 0.01)
            paceNumbInput.value = pace_slid.value
            const new_lengths = window.pitchEditor.dursNew.map((v,l) => v * pace_slid.value)
            window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, new_lengths[l]* 10 + 50, null))
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



window.setupModal = (openingButton, modalContainerElem, callback) => {
    openingButton.addEventListener("click", () => {
        if (callback) {
            callback()
        }
        closeModal(undefined, modalContainerElem).then(() => {
            modalContainerElem.style.opacity = 0
            modalContainerElem.style.display = "flex"
            requestAnimationFrame(() => requestAnimationFrame(() => modalContainerElem.style.opacity = 1))
            requestAnimationFrame(() => requestAnimationFrame(() => chrome.style.opacity = 1))
        })
    })
    modalContainerElem.addEventListener("click", event => {
        if (event.target==modalContainerElem) {
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
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
const createModal = (type, message) => {
    return new Promise(resolve => {
        modalContainer.innerHTML = ""
        const displayMessage = message.prompt ? message.prompt : message
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span", displayMessage))
        modal.dataset.type = type

        if (type=="confirm") {
            const yesButton = createElem("button", {style: {background: `#${themeColour}`}})
            yesButton.innerHTML = "Yes"
            const noButton = createElem("button", {style: {background: `#${themeColour}`}})
            noButton.innerHTML = "No"
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
            closeButton.innerHTML = "Close"
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="prompt") {
            const closeButton = createElem("button", {style: {background: `#${themeColour}`}})
            closeButton.innerHTML = "Submit"
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


window.addEventListener("resize", e => {
    window.userSettings.customWindowSize = `${window.innerHeight},${window.innerWidth}`
    saveUserSettings()
})

// Keyboard actions
// ================
window.addEventListener("keydown", event => {

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
    // Y/N for prompt modals
    if (key=="y" || key=="n") {
        if (document.querySelector("#activeModal")) {
            const buttons = Array.from(document.querySelector("#activeModal").querySelectorAll("button"))
            const yesBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="yes")
            const noBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="no")
            if (key=="y") yesBtn.click()
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
        pace_slid.value = parseFloat(pace_slid.value) + (key=="arrowleft"? -0.01 : 0.01)
        paceNumbInput.value = pace_slid.value
        const new_lengths = window.pitchEditor.dursNew.map((v,l) => v * pace_slid.value)
        window.pitchEditor.letters.forEach((_, l) => set_letter_display(letterElems[l], l, new_lengths[l]* 10 + 50, null))
        // if (autoinfer_timer != null) {
        //     clearTimeout(autoinfer_timer)
        //     autoinfer_timer = null
        // }
        // if (autoplay_ckbx.checked) {
        //     autoinfer_timer = setTimeout(infer, 500)
        // }
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
            // chrome.style.opacity = 0.88
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
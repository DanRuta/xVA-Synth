"use strict"


class Editor {

    constructor () {

        this.isCreated = false

        this.hasChanged = false
        this.autoInferTimer = null

        this.adjustedLetters = new Set()

        this.LEFT_RIGHT_SEQ_PADDING = 20
        this.EDITOR_HEIGHT = 150
        this.LETTERS_Y_OFFSET = 40
        this.SLIDER_GRABBER_H = 15
        this.MIN_LETTER_LENGTH = 20
        this.MAX_LETTER_LENGTH = 100
        this.SPACE_BETWEEN_LETTERS = 5

        this.default_pitchSliderRange = 4
        this.pitchSliderRange = 4
        this.duration_visual_size_multiplier = 1

        this.default_MIN_ENERGY = 3.45
        this.MIN_ENERGY = 3.45
        this.default_MAX_ENERGY = 4.35
        this.MAX_ENERGY = 4.35
        this.ENERGY_GRABBER_RADIUS = 8
        this.EMOTION_STYLE_GRABBER_RADIUS = 8

        this.MIN_EMOTIONS = 0
        this.MAX_EMOTIONS = 1.06

        this.MIN_STYLES = 0
        this.MAX_STYLES = 1.06

        this.clear() // And thus init

        this.historyState = [] // TODO, add support for undo/redo across all editor functions
    }

    clear () {
        this.sliderBoxes = []
        this.grabbers = []
        this.energyGrabbers = []
        this.emAngryGrabbers = []
        this.emHappyGrabbers = []
        this.emSadGrabbers = []
        this.emSurpriseGrabbers = []

        this.styleGrabbers = {}
        this.styleValuesNew = {}
        this.styleValuesReset = {}
        this.multiLetterStyleDelta = {}
        this.multiLetterStartStyleVals = {}

        this.letters = []
        this.pitchNew = []
        this.dursNew = []
        this.energyNew = []
        this.emAngryNew = []
        this.emHappyNew = []
        this.emSadNew = []
        this.emSurpriseNew = []
        this.pacing = 1
        this.ampFlatCounter = 0

        this.inputSequence = undefined
        this.currentVoice = undefined
        this.letterFocus = []
        this.lastSelected = 0
        this.letterClasses = []
        this.resetDurs = []
        this.resetPitch = []
        this.resetEnergy = []
        this.resetEmAngry = []
        this.resetEmHappy = []
        this.resetEmSad = []
        this.resetEmSurprise = []

        this.multiLetterPitchDelta = undefined
        this.multiLetterStartPitchVals = []
        this.multiLetterStartDursVals = []
        this.multiLetterEnergyDelta = undefined
        this.multiLetterStartEnergyVals = []
        this.multiLetterEmotionDelta = undefined
        this.multiLetterStartEmotionVals = []

        this.multiLetterLengthDelta = undefined
        this.multiLetterStartLengthVals = []
    }

    loadStylesData (editorStyles) {
        this.registeredStyleKeys = []
        Object.keys(window.appState.currentModelEmbeddings).forEach(styleKey => {
            if (styleKey=="default") return
            this.registeredStyleKeys.push(styleKey)

            this.styleGrabbers[styleKey] = []
            this.styleValuesReset[styleKey] = this.resetPitch.map(v=>0)
            this.styleValuesNew[styleKey] = (editorStyles&&editorStyles[styleKey]&&editorStyles[styleKey].sliders) ? editorStyles[styleKey].sliders : this.resetPitch.map(v=>0)
            this.multiLetterStyleDelta[styleKey] = undefined
            this.multiLetterStartStyleVals[styleKey] = []
        })
    }

    init () {

        // Clear away old instance
        if (this.isCreated) {
            editorContainer.innerHTML = ""
            delete this.canvas
            delete this.context
        }


        let canvasWidth = this.LEFT_RIGHT_SEQ_PADDING*2 // Padding
        this.dursNew.forEach((dur,di) => {
            if (di) {
                canvasWidth += this.SPACE_BETWEEN_LETTERS
            }

            let value = dur
            value = value * this.pacing
            value = Math.max(0.1, value)
            value = Math.min(value, 20)
            const percentAcross = value/20
            const width = percentAcross * (this.MAX_LETTER_LENGTH-this.MIN_LETTER_LENGTH) + this.MIN_LETTER_LENGTH
            canvasWidth += width
        })

        this.canvas = document.createElement("canvas")
        this.context = this.canvas.getContext("2d")
        this.context.textAlign  = "center"
        this.canvas.width = canvasWidth
        this.canvas.height = 200

        editorContainer.appendChild(this.canvas)


        // Mouse cursor
        this.canvas.addEventListener("mousemove", event => {
            const mouseX = parseInt(event.offsetX)
            const mouseY = parseInt(event.offsetY)
            this.canvas.style.cursor = "default"

            // Check energy grabber hover
            const isOnEGrabber = seq_edit_view_select.value.includes("energy") && this.energyGrabbers.find((eGrabber, egi) => {
                if (!this.enabled_disabled_items[egi]) return
                const grabberX = eGrabber.getXLeft()+eGrabber.sliderBox.width/2-this.ENERGY_GRABBER_RADIUS
                return (mouseX>grabberX && mouseX<grabberX+this.ENERGY_GRABBER_RADIUS*2+4) && (mouseY>eGrabber.topLeftY-this.ENERGY_GRABBER_RADIUS-2 && mouseY<eGrabber.topLeftY+this.ENERGY_GRABBER_RADIUS+2)
            })
            if (isOnEGrabber && isOnEGrabber!=undefined) {
                this.canvas.style.cursor = "row-resize"
                return
            }
            // Check emotion grabber hover
            const isHoveringOverEmotionGrabber = emotionGrabbers => {
                return emotionGrabbers.find((eGrabber, egi) => {
                    if (!this.enabled_disabled_items[egi]) return
                    const grabberX = eGrabber.getXLeft()+eGrabber.sliderBox.width/2-this.EMOTION_STYLE_GRABBER_RADIUS
                    return (mouseX>grabberX && mouseX<grabberX+this.EMOTION_STYLE_GRABBER_RADIUS*2+4) && (mouseY>eGrabber.topLeftY-this.EMOTION_STYLE_GRABBER_RADIUS-2 && mouseY<eGrabber.topLeftY+this.EMOTION_STYLE_GRABBER_RADIUS+2)
                })
            }
            if (window.currentModel.modelType=="xVAPitch") {
                const isOnEmGrabber = seq_edit_view_select.value.startsWith("em") && (isHoveringOverEmotionGrabber(this.emAngryGrabbers) || isHoveringOverEmotionGrabber(this.emHappyGrabbers) || isHoveringOverEmotionGrabber(this.emSadGrabbers) || isHoveringOverEmotionGrabber(this.emSurpriseGrabbers))
                if (isOnEmGrabber && isOnEmGrabber!=undefined) {
                    this.canvas.style.cursor = "row-resize"
                    return
                }
            }
            // Check pitch grabber hover
            const isOnGrabber = seq_edit_view_select.value.includes("pitch") && this.grabbers.find((grabber, gi) => {
                if (!this.enabled_disabled_items[gi]) return
                const grabberX = grabber.getXLeft()
                return (mouseX>grabberX && mouseX<grabberX+grabber.width) && (mouseY>grabber.topLeftY && mouseY<grabber.topLeftY+grabber.height)
            })
            if (isOnGrabber && isOnGrabber!=undefined) {
                this.canvas.style.cursor = "n-resize"
                return
            }

            // Check styles grabbers
            if (this.registeredStyleKeys && this.registeredStyleKeys.length) {
                let isOnStyleGrabber
                this.registeredStyleKeys.forEach(styleKey => {
                    if (isOnStyleGrabber) return // Skip unnecessary work if already found
                    if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
                        isOnStyleGrabber = this.styleGrabbers[styleKey].find((grabber, gi) => {
                            if (!this.enabled_disabled_items[gi]) return
                            const grabberX = grabber.getXLeft()
                            return (mouseX>grabberX && mouseX<grabberX+grabber.width) && (mouseY>grabber.topLeftY && mouseY<grabber.topLeftY+grabber.height)
                        })
                        if (isOnStyleGrabber) {
                            this.canvas.style.cursor = "row-resize"
                            return
                        }
                    }
                })
            }

            // Check letter hover
            const isOnLetter = this.letterClasses.find((letter, l) => {
                if (!this.enabled_disabled_items[l]) return
                return (mouseY<this.LETTERS_Y_OFFSET) && (mouseX>this.sliderBoxes[l].getXLeft() && mouseX<this.sliderBoxes[l].getXLeft()+this.sliderBoxes[l].width)
            })
            if (isOnLetter!=undefined) {
                this.canvas.style.cursor = "pointer"
                return
            }
            // Check box length dragger
            const isBetweenBoxes = this.sliderBoxes.find((box, bi) => {
                if (!this.enabled_disabled_items[bi]) return
                const boxX = box.getXLeft()
                return (mouseY>box.topY && mouseY<box.topY+box.height) && (mouseX>(boxX+box.width-10) && mouseX<(boxX+box.width+10)+5)
            })
            if (isBetweenBoxes!=undefined) {
                this.canvas.style.cursor = "w-resize"
                return
            }
        })

        let elemDragged = undefined
        let mouseDownStart = {x: undefined, y: undefined}
        this.canvas.addEventListener("mousedown", event => {
            const mouseX = parseInt(event.offsetX)
            const mouseY = parseInt(event.offsetY)
            mouseDownStart.x = mouseX
            mouseDownStart.y = mouseY

            // Check up-down emotion dragging
            const findGrabber = emotionGrabbers => {
                return emotionGrabbers.find((eGrabber, egi) => {
                    if (!this.enabled_disabled_items[egi]) return
                    const grabberX = eGrabber.getXLeft()+eGrabber.sliderBox.width/2-this.EMOTION_STYLE_GRABBER_RADIUS
                    return (mouseX>grabberX && mouseX<grabberX+this.EMOTION_STYLE_GRABBER_RADIUS*2+4) && (mouseY>eGrabber.topLeftY-this.EMOTION_STYLE_GRABBER_RADIUS-2 && mouseY<eGrabber.topLeftY+this.EMOTION_STYLE_GRABBER_RADIUS+2)
                })
            }
            const handleEmGrabber = (emGrabber, grabbersList) => {
                if (this.letterFocus.length <= 1 || (!this.letterFocus.includes(emGrabber.index))) {
                    this.setLetterFocus(grabbersList.indexOf(emGrabber), event.ctrlKey, event.shiftKey, event.altKey)
                }
                this.multiLetterEmotionDelta = emGrabber.topLeftY
                this.multiLetterStartEmotionVals = grabbersList.map(emGrabber => emGrabber.topLeftY)

                return emGrabber
            }
            const handleStyleGrabber = (styleGrabber, grabbersList, styleKey) => {
                if (this.letterFocus.length <= 1 || (!this.letterFocus.includes(styleGrabber.index))) {
                    this.setLetterFocus(grabbersList.indexOf(styleGrabber), event.ctrlKey, event.shiftKey, event.altKey)
                }
                this.multiLetterStyleDelta[styleKey] = styleGrabber.topLeftY
                this.multiLetterStartStyleVals[styleKey] = grabbersList.map(styleGrabber => styleGrabber.topLeftY)
                return styleGrabber
            }
            if (window.currentModel.modelType=="xVAPitch") {
                if (seq_edit_view_select.value.startsWith("style_") && this.registeredStyleKeys.length) {
                    this.registeredStyleKeys.forEach(styleKey => {
                        if (seq_edit_view_select.value.includes(styleKey)) {
                            const isOnStyleGrabber = findGrabber(this.styleGrabbers[styleKey])
                            if (isOnStyleGrabber) {
                                elemDragged = handleStyleGrabber(isOnStyleGrabber, this.styleGrabbers[styleKey], styleKey)
                                return
                            }
                        }
                    })
                }
                const isOnEmAngryGrabber = seq_edit_view_select.value=="emAngry" && findGrabber(this.emAngryGrabbers)
                if (isOnEmAngryGrabber) {
                    elemDragged = handleEmGrabber(isOnEmAngryGrabber, this.emAngryGrabbers)
                    return
                }
                const isOnEmHappyGrabber = seq_edit_view_select.value=="emHappy" && findGrabber(this.emHappyGrabbers)
                if (isOnEmHappyGrabber) {
                    elemDragged = handleEmGrabber(isOnEmHappyGrabber, this.emHappyGrabbers)
                    return
                }
                const isOnEmSadGrabber = seq_edit_view_select.value=="emSad" && findGrabber(this.emSadGrabbers)
                if (isOnEmSadGrabber) {
                    elemDragged = handleEmGrabber(isOnEmSadGrabber, this.emSadGrabbers)
                    return
                }
                const isOnEmSurpriseGrabber = seq_edit_view_select.value=="emSurprise" && findGrabber(this.emSurpriseGrabbers)
                if (isOnEmSurpriseGrabber) {
                    elemDragged = handleEmGrabber(isOnEmSurpriseGrabber, this.emSurpriseGrabbers)
                    return
                }

            }

            // Check up-down energy dragging
            const isOnEGrabber = seq_edit_view_select.value.includes("energy") && this.energyGrabbers.find((eGrabber, egi) => {
                if (!this.enabled_disabled_items[egi]) return
                const grabberX = eGrabber.getXLeft()+eGrabber.sliderBox.width/2-this.ENERGY_GRABBER_RADIUS
                return (mouseX>grabberX && mouseX<grabberX+this.ENERGY_GRABBER_RADIUS*2+4) && (mouseY>eGrabber.topLeftY-this.ENERGY_GRABBER_RADIUS-2 && mouseY<eGrabber.topLeftY+this.ENERGY_GRABBER_RADIUS+2)
            })
            if (isOnEGrabber) {

                const eGrabber = isOnEGrabber
                if (this.letterFocus.length <= 1 || (!this.letterFocus.includes(eGrabber.index))) {
                    this.setLetterFocus(this.energyGrabbers.indexOf(eGrabber), event.ctrlKey, event.shiftKey, event.altKey)
                }
                this.multiLetterEnergyDelta = eGrabber.topLeftY
                this.multiLetterStartEnergyVals = this.energyGrabbers.map(eGrabber => eGrabber.topLeftY)

                elemDragged = isOnEGrabber
                return
            }
            // Check up-down pitch dragging
            const isOnGrabber = seq_edit_view_select.value.includes("pitch") && this.grabbers.find((grabber, gi) => {
                if (!this.enabled_disabled_items[gi]) return
                const grabberX = grabber.getXLeft()
                return (mouseX>grabberX && mouseX<grabberX+grabber.width) && (mouseY>grabber.topLeftY && mouseY<grabber.topLeftY+grabber.height)
            })
            if (isOnGrabber) {

                const slider = isOnGrabber
                if (this.letterFocus.length <= 1 || (!this.letterFocus.includes(slider.index))) {
                    this.setLetterFocus(this.grabbers.indexOf(slider), event.ctrlKey, event.shiftKey, event.altKey)
                }
                this.multiLetterPitchDelta = slider.topLeftY
                this.multiLetterStartPitchVals = this.grabbers.map(slider => slider.topLeftY)

                elemDragged = isOnGrabber
                return
            }
            // Check sideways dragging
            const isBetweenBoxes = this.sliderBoxes.find((box, bi) => {
                if (!this.enabled_disabled_items[bi]) return
                const boxX = box.getXLeft()
                return (mouseY>box.topY && mouseY<box.topY+box.height) && (mouseX>(boxX+box.width-10) && mouseX<(boxX+box.width+10)+5)
            })
            if (isBetweenBoxes) {
                this.multiLetterStartDursVals = this.sliderBoxes.map(box => box.width)

                isBetweenBoxes.dragStart.width = isBetweenBoxes.width
                elemDragged = isBetweenBoxes
                return
            }

            // Check clicking on the top letters
            const isOnLetter = this.letterClasses.find((letter, l) => {
                if (!this.enabled_disabled_items[l]) return
                return (mouseY<this.LETTERS_Y_OFFSET) && (mouseX>this.sliderBoxes[l].getXLeft() && mouseX<this.sliderBoxes[l].getXLeft()+this.sliderBoxes[l].width)
            })
            if (isOnLetter) {
                this.setLetterFocus(this.letterClasses.indexOf(isOnLetter), event.ctrlKey, event.shiftKey, event.altKey)
            }


        })

        this.canvas.addEventListener("mouseup", event => {
            mouseDownStart = {x: undefined, y: undefined}
            if (autoplay_ckbx.checked && this.hasChanged) {
                generateVoiceButton.click()
            }
            this.init()
        })

        this.canvas.addEventListener("mousemove", event => {
            if (mouseDownStart.x && mouseDownStart.y) {

                if (elemDragged && (parseInt(event.offsetX)-mouseDownStart.x || parseInt(event.offsetY)-mouseDownStart.y)) {
                    this.hasChanged = true
                    this.letterFocus.forEach(index => this.adjustedLetters.add(index))
                }

                if (elemDragged) {
                    if (elemDragged.type=="slider") { // Pitch sliders, specifically

                        elemDragged.setValueFromCoords(parseInt(event.offsetY)-elemDragged.height/2)

                        // If there's a multi-selection, update all of their values, otherwise update the numerical input
                        if (this.letterFocus.length>1) {
                            this.letterFocus.forEach(li => {
                                if (li!=elemDragged.index) {
                                    this.grabbers[li].setValueFromCoords(this.multiLetterStartPitchVals[li]+(elemDragged.topLeftY-this.multiLetterPitchDelta))
                                }
                            })
                        } else {
                            letterPitchNumb.value = parseInt(this.pitchNew[elemDragged.index]*100)/100
                        }


                    } else if (elemDragged.type=="box") { // Durations being dragged sideways

                        // If there's a multi-selection, update all of their values, otherwise update the numerical input
                        if (this.letterFocus.length>1) {
                            this.letterFocus.forEach(li => {
                                let newWidth = this.multiLetterStartDursVals[li] + parseInt(elemDragged.width - elemDragged.dragStart.width)
                                newWidth = Math.max(20, newWidth)
                                newWidth = Math.min(newWidth, this.MAX_LETTER_LENGTH)

                                this.sliderBoxes[li].width = newWidth

                                this.sliderBoxes[li].percentAcross = (this.sliderBoxes[li].width-20) / (this.MAX_LETTER_LENGTH-20)
                                this.dursNew[this.sliderBoxes[li].index] = Math.max(0.1, this.sliderBoxes[li].percentAcross*20)

                                this.sliderBoxes[li].grabber.width = this.sliderBoxes[li].width-2
                                this.sliderBoxes[li].letter.centerX = this.sliderBoxes[li].leftX + this.sliderBoxes[li].width/2
                            })
                        } else {

                            letterLengthNumb.value = parseInt(this.dursNew[elemDragged.index]*100)/100
                        }


                        let newWidth = elemDragged.dragStart.width + parseInt(event.offsetX)-mouseDownStart.x
                        newWidth = Math.max(20, newWidth)
                        newWidth = Math.min(newWidth, this.MAX_LETTER_LENGTH)
                        elemDragged.width = newWidth

                        elemDragged.percentAcross = (elemDragged.width-20) / (this.MAX_LETTER_LENGTH-20)
                        this.dursNew[elemDragged.index] = Math.max(0.1, elemDragged.percentAcross*20)

                        elemDragged.grabber.width = elemDragged.width-2
                        elemDragged.letter.centerX = elemDragged.leftX + elemDragged.width/2

                    } else if (elemDragged.type=="energy_slider") { // Energy sliders

                        elemDragged.setValueFromCoords(parseInt(event.offsetY)-elemDragged.height/2)

                        // If there's a multi-selection, update all of their values, otherwise update the numerical input
                        if (this.letterFocus.length>1) {
                            this.letterFocus.forEach(li => {
                                if (li!=elemDragged.index) {
                                    this.energyGrabbers[li].setValueFromCoords(this.multiLetterStartEnergyVals[li]+(elemDragged.topLeftY-this.multiLetterEnergyDelta))
                                }
                            })
                        } else {
                            letterEnergyNumb.value = parseInt(this.energyNew[elemDragged.index]*100)/100
                        }

                    } else if (elemDragged.type=="emotion_slider") { // Emotion sliders

                        elemDragged.setValueFromCoords(parseInt(event.offsetY)-elemDragged.height/2)

                        // If there's a multi-selection, update all of their values, otherwise update the numerical input
                        if (this.letterFocus.length>1) {
                            this.letterFocus.forEach(li => {
                                if (li!=elemDragged.index) {
                                    if (seq_edit_view_select.value=="emAngry") {
                                        this.emAngryGrabbers[li].setValueFromCoords(this.multiLetterStartEmotionVals[li]+(elemDragged.topLeftY-this.multiLetterEmotionDelta))
                                    } else if (seq_edit_view_select.value=="emHappy") {
                                        this.emHappyGrabbers[li].setValueFromCoords(this.multiLetterStartEmotionVals[li]+(elemDragged.topLeftY-this.multiLetterEmotionDelta))
                                    } else if (seq_edit_view_select.value=="emSad") {
                                        this.emSadGrabbers[li].setValueFromCoords(this.multiLetterStartEmotionVals[li]+(elemDragged.topLeftY-this.multiLetterEmotionDelta))
                                    } else if (seq_edit_view_select.value=="emSurprise") {
                                        this.emSurpriseGrabbers[li].setValueFromCoords(this.multiLetterStartEmotionVals[li]+(elemDragged.topLeftY-this.multiLetterEmotionDelta))
                                    }
                                }
                            })
                        } else {
                            if (seq_edit_view_select.value=="emAngry") {
                                letterEmotionNumb.value = parseFloat(this.emAngryNew[elemDragged.index]*100)/100
                            } else if (seq_edit_view_select.value=="emHappy") {
                                letterEmotionNumb.value = parseFloat(this.emHappyNew[elemDragged.index]*100)/100
                            } else if (seq_edit_view_select.value=="emSad") {
                                letterEmotionNumb.value = parseFloat(this.emSadNew[elemDragged.index]*100)/100
                            } else if (seq_edit_view_select.value=="emSurprise") {
                                letterEmotionNumb.value = parseFloat(this.emSurpriseNew[elemDragged.index]*100)/100
                            }
                        }
                    } else if (elemDragged.type=="style_slider") { // Style sliders

                        elemDragged.setValueFromCoords(parseInt(event.offsetY)-elemDragged.height/2)

                        if (this.registeredStyleKeys.length) {
                            this.registeredStyleKeys.forEach(styleKey => {
                                if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
                                    // If there's a multi-selection, update all of their values, otherwise update the numerical input
                                    if (this.letterFocus.length>1) {
                                        this.letterFocus.forEach(li => {
                                            if (li!=elemDragged.index) {
                                                this.styleGrabbers[styleKey][li].setValueFromCoords(this.multiLetterStartStyleVals[styleKey][li]+(elemDragged.topLeftY-this.multiLetterStyleDelta[styleKey]))
                                            }
                                        })
                                    } else {
                                        letterStyleNumb.value = parseInt(this.styleValuesNew[styleKey][elemDragged.index]*100)/100
                                    }
                                }
                            })
                        }
                    }
                }
            }
        })

        if (!this.isCreated) {
            this.render()
        }
        this.isCreated = true
    }




    render () {
        if (this.context!=undefined) {
            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height)
            this.letterClasses.forEach((letter, li) => {
                if (this.letters[li]=="<PAD>") return
                letter.context = this.context
                letter.render()
            })
            this.sliderBoxes.forEach((sliderBox, sbi) => {
                if (this.letters[sbi]=="<PAD>") return
                sliderBox.context = this.context
                sliderBox.render()
            })
            if (seq_edit_view_select.value=="pitch_energy" || seq_edit_view_select.value=="pitch") {
                this.grabbers.forEach((grabber,gi) => {
                    if (this.letters[gi]=="<PAD>") return
                    grabber.context = this.context
                    grabber.render()
                })
            }
            if (seq_edit_view_select.value=="pitch_energy" || seq_edit_view_select.value=="energy") {
                this.energyGrabbers.forEach((eGrabber, egi) => {
                    if (this.letters[egi]=="<PAD>") return
                    eGrabber.context = this.context
                    eGrabber.render()
                })
            }
            if (window.currentModel.modelType=="xVAPitch") {
                if (seq_edit_view_select.value=="emAngry") {
                    this.emAngryGrabbers.forEach((eGrabber, egi) => {
                        if (this.letters[egi]=="<PAD>") return
                        eGrabber.context = this.context
                        eGrabber.render()
                    })
                }
                if (seq_edit_view_select.value=="emHappy") {
                    this.emHappyGrabbers.forEach((eGrabber, egi) => {
                        if (this.letters[egi]=="<PAD>") return
                        eGrabber.context = this.context
                        eGrabber.render()
                    })
                }
                if (seq_edit_view_select.value=="emSad") {
                    this.emSadGrabbers.forEach((eGrabber, egi) => {
                        if (this.letters[egi]=="<PAD>") return
                        eGrabber.context = this.context
                        eGrabber.render()
                    })
                }
                if (seq_edit_view_select.value=="emSurprise") {
                    this.emSurpriseGrabbers.forEach((eGrabber, egi) => {
                        if (this.letters[egi]=="<PAD>") return
                        eGrabber.context = this.context
                        eGrabber.render()
                    })
                }
                if (this.registeredStyleKeys && this.registeredStyleKeys.length) {
                    this.registeredStyleKeys.forEach(styleKey => {
                        if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
                            this.styleGrabbers[styleKey].forEach((styleGrabber, sgi) => {
                                if (this.letters[sgi]=="<PAD>") return
                                styleGrabber.context = this.context
                                styleGrabber.render()
                            })
                        }
                    })
                }
            }
        }
        requestAnimationFrame(() => {this.render()})
    }




    update (modelType=undefined, sliderRange=undefined) {

        self.modelType = modelType

        // Make model-specific adjustments
        if (modelType=="xVAPitch") {
            this.default_pitchSliderRange = 6
            this.pitchSliderRange = sliderRange || 6
            this.duration_visual_size_multiplier = 1
            this.MAX_LETTER_LENGTH = 200
            this.default_MIN_ENERGY = 0
            this.MIN_ENERGY = 0
            this.default_MAX_ENERGY = 1.07
            this.MAX_ENERGY = 1.07

            this.styleGrabbers = {}
            this.registeredStyleKeys.forEach(styleKey => {
                this.styleGrabbers[styleKey] = []
            })

        } else {
            this.default_pitchSliderRange = 4
            this.pitchSliderRange = sliderRange || 4
            this.duration_visual_size_multiplier = 1
            this.MAX_LETTER_LENGTH = 100
        }

        this.letterClasses = []
        this.sliderBoxes = []
        this.grabbers = []
        this.energyGrabbers = []
        this.emAngryGrabbers = []
        this.emHappyGrabbers = []
        this.emSadGrabbers = []
        this.emSurpriseGrabbers = []

        this.enabled_disabled_items = []

        let xCounter = 0
        let lastBox = undefined
        let letter_counter = 0
        this.letters.forEach((letter, li) => {

            if (letter=="<PAD>") {
                this.enabled_disabled_items.push(false)
                letter_counter += 1
            } else {
                this.enabled_disabled_items.push(true)
            }
            letter_counter += 1

            const dur = this.dursNew[li]
            const width = Math.max(25, dur*10)


            // Slider box
            const sliderBox = new SliderBox(this.context, li, lastBox, this.LETTERS_Y_OFFSET, this.EDITOR_HEIGHT, this.MIN_LETTER_LENGTH, this.MAX_LETTER_LENGTH, letter_counter%2==0)
            sliderBox.render()
            if (lastBox) {
                lastBox.rightBox = sliderBox
            }
            lastBox = sliderBox
            this.sliderBoxes.push(sliderBox)

            // Letter text
            const letterClass = new Letter(this.context, li, letter, sliderBox, 20, 20+xCounter, width)
            if (this.letterFocus.includes(li)) {
                letterClass.colour = "red"
            }
            letterClass.render()
            this.letterClasses.push(letterClass)

            // Slider grabber thing
            const pitchPercent = 1-(this.pitchNew[li]+this.pitchSliderRange)/(this.pitchSliderRange*2)
            const grabber = new SliderGrabber(this.context, li, sliderBox, (this.LETTERS_Y_OFFSET+1)+(this.SLIDER_GRABBER_H/2)+((this.EDITOR_HEIGHT-2)-this.SLIDER_GRABBER_H)*pitchPercent-this.SLIDER_GRABBER_H/2, width-2, this.SLIDER_GRABBER_H, this.pitchSliderRange)
            grabber.render()
            this.grabbers.push(grabber)

            if (this.energyNew && this.energyNew.length) {
                // Energy round grabber
                let energyPercent
                if (modelType=="xVAPitch") {
                    energyPercent = ( (this.energyNew[li]-this.MIN_ENERGY) / (this.MAX_ENERGY-this.MIN_ENERGY)  )
                } else {
                    energyPercent = 1 - ( (this.energyNew[li]-this.MIN_ENERGY) / (this.MAX_ENERGY-this.MIN_ENERGY)  )
                }
                energyPercent = Math.max(0, energyPercent)
                energyPercent = Math.min(energyPercent, 1)

                let topLeftY = (1 - energyPercent) * (this.EDITOR_HEIGHT-2-this.ENERGY_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                const energyGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.ENERGY_GRABBER_RADIUS, undefined, modelType, this.ENERGY_GRABBER_RADIUS, "energy")
                energyGrabber.render()
                this.energyGrabbers.push(energyGrabber)
            }

            if (modelType=="xVAPitch") {
                if (this.emAngryNew && this.emAngryNew.length) {
                    let emotionPercent = ( (this.emAngryNew[li]-this.MIN_EMOTIONS) / (this.MAX_EMOTIONS-this.MIN_EMOTIONS)  )
                    emotionPercent = Math.max(0, emotionPercent)
                    emotionPercent = Math.min(emotionPercent, 1)

                    let topLeftY = (1 - emotionPercent) * (this.EDITOR_HEIGHT-2-this.EMOTION_STYLE_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                    const emAngryGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.EMOTION_STYLE_GRABBER_RADIUS, undefined, modelType, this.EMOTION_STYLE_GRABBER_RADIUS, "emotion")
                    emAngryGrabber.render()
                    this.emAngryGrabbers.push(emAngryGrabber)
                }
                if (this.emHappyNew && this.emHappyNew.length) {
                    let emotionPercent = ( (this.emHappyNew[li]-this.MIN_EMOTIONS) / (this.MAX_EMOTIONS-this.MIN_EMOTIONS)  )
                    emotionPercent = Math.max(0, emotionPercent)
                    emotionPercent = Math.min(emotionPercent, 1)

                    let topLeftY = (1 - emotionPercent) * (this.EDITOR_HEIGHT-2-this.EMOTION_STYLE_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                    const emHappyGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.EMOTION_STYLE_GRABBER_RADIUS, undefined, modelType, this.EMOTION_STYLE_GRABBER_RADIUS, "emotion")
                    emHappyGrabber.render()
                    this.emHappyGrabbers.push(emHappyGrabber)
                }
                if (this.emSadNew && this.emSadNew.length) {
                    let emotionPercent = ( (this.emSadNew[li]-this.MIN_EMOTIONS) / (this.MAX_EMOTIONS-this.MIN_EMOTIONS)  )
                    emotionPercent = Math.max(0, emotionPercent)
                    emotionPercent = Math.min(emotionPercent, 1)

                    let topLeftY = (1 - emotionPercent) * (this.EDITOR_HEIGHT-2-this.EMOTION_STYLE_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                    const emSadGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.EMOTION_STYLE_GRABBER_RADIUS, undefined, modelType, this.EMOTION_STYLE_GRABBER_RADIUS, "emotion")
                    emSadGrabber.render()
                    this.emSadGrabbers.push(emSadGrabber)
                }
                if (this.emSurpriseNew && this.emSurpriseNew.length) {
                    let emotionPercent = ( (this.emSurpriseNew[li]-this.MIN_EMOTIONS) / (this.MAX_EMOTIONS-this.MIN_EMOTIONS)  )
                    emotionPercent = Math.max(0, emotionPercent)
                    emotionPercent = Math.min(emotionPercent, 1)

                    let topLeftY = (1 - emotionPercent) * (this.EDITOR_HEIGHT-2-this.EMOTION_STYLE_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                    const emSurpriseGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.EMOTION_STYLE_GRABBER_RADIUS, undefined, modelType, this.EMOTION_STYLE_GRABBER_RADIUS, "emotion")
                    emSurpriseGrabber.render()
                    this.emSurpriseGrabbers.push(emSurpriseGrabber)
                }

                // Initialize grabbers dynamically for every style
                this.registeredStyleKeys.forEach(styleKey => {
                    let stylePercent = ( (this.styleValuesNew[styleKey][li]-this.MIN_STYLES) / (this.MAX_STYLES-this.MIN_STYLES)  )
                    stylePercent = Math.max(0, stylePercent)
                    stylePercent = Math.min(stylePercent, 1)

                    let topLeftY = (1 - stylePercent) * (this.EDITOR_HEIGHT-2-this.EMOTION_STYLE_GRABBER_RADIUS) + (this.LETTERS_Y_OFFSET)
                    const styleGrabber = new EnergyEmotionGrabber(this.context, li, sliderBox, topLeftY, width-2, this.EMOTION_STYLE_GRABBER_RADIUS, undefined, modelType, this.EMOTION_STYLE_GRABBER_RADIUS, "style")
                    styleGrabber.render()
                    this.styleGrabbers[styleKey].push(styleGrabber)
                })
            }

            sliderBox.letter = letterClass
            sliderBox.grabber = grabber

            sliderBox.setValueFromValue(dur)

            xCounter += width + 5

        })
    }

    setLetterFocus (l, ctrlKey, shiftKey, altKey) {

        // NONE = Clear selection, add l to selection
        // Ctrl = Add l to existing selection
        // Shift = Add all letters from the last selected letter up to and including l to existing selection
        // Ctrl + Shift = (See Shift)
        // Alt = Clear selection and select word surrounding l (space delimited)
        // Ctrl + Alt = Add word surrounding l to existing selection
        // Shift + Alt = Same as shift, then afterwards add word (space delimited) around l to selection
        // Ctrl + Shift + Alt = (See Shift + Alt)

        // If nothing is selected and we hold shift, we assume we start from the first letter: at position 0

        // If we don't press shift or ctrl, we can clear our current selection.
        if (!(ctrlKey || shiftKey) && this.letterFocus.length){
                this.letterFocus.forEach(li => {
                    this.letterClasses[li].colour = "black"
                })
                this.letterFocus = []
                this.lastSelected = 0
        }
        if (shiftKey){
            if (l>this.lastSelected) {
                for (let i=this.lastSelected; i<=l; i++) {
                    this.letterFocus.push(i)
                }
            } else {
                for (let i=l; i<=this.lastSelected; i++) {
                    this.letterFocus.push(i)
                }
            }
        }
        this.letterFocus.push(l) // Push l
        this.lastSelected = l
        if (altKey){
            let l2 = l
                // Looking backwards
                while (l2>=0) {
                    let prevLetter = this.letters[l2]
                    if (prevLetter!="_") {
                        this.letterFocus.push(l2)
                    } else {
                        break
                    }
                    l2--
                }
            l2 = l
            // Looking forward
            while (l2<this.letters.length) {
                let nextLetter = this.letters[l2]
                if (nextLetter!="_") {
                    this.letterFocus.push(l2)
                } else {
                    break
                }
                l2++
            }
        }

        this.letterFocus = Array.from(new Set(this.letterFocus.sort()))
        this.letterFocus.forEach(li => {
            this.letterClasses[li].colour = "red"
        })


        letterStyleNumb.value = ""
        letterStyleNumb.disabled = true
        if (this.letterFocus.length==1) {
            if (this.energyNew.length) {
                letterEnergyNumb.value = parseFloat(this.energyNew[this.letterFocus[0]])
                letterEnergyNumb.disabled = false
            }
            if (this.emAngryNew && this.emAngryNew.length) {
                letterEmotionNumb.value = parseFloat(this.emAngryNew[this.letterFocus[0]])
                letterEmotionNumb.disabled = false
            }
            if (this.emHappyNew && this.emHappyNew.length) {
                letterEmotionNumb.value = parseFloat(this.emHappyNew[this.letterFocus[0]])
                letterEmotionNumb.disabled = false
            }
            if (this.emSadNew && this.emSadNew.length) {
                letterEmotionNumb.value = parseFloat(this.emSadNew[this.letterFocus[0]])
                letterEmotionNumb.disabled = false
            }
            if (this.emSurpriseNew && this.emSurpriseNew.length) {
                letterEmotionNumb.value = parseFloat(this.emSurpriseNew[this.letterFocus[0]])
                letterEmotionNumb.disabled = false
            }
            if (this.registeredStyleKeys) {
                this.registeredStyleKeys.forEach(styleKey => {
                    if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
                        letterStyleNumb.value = parseFloat(this.styleValuesNew[styleKey][this.letterFocus[0]])
                        letterStyleNumb.disabled = false
                    }
                })
            }
            letterPitchNumb.value = parseInt(this.pitchNew[this.letterFocus[0]]*100)/100
            letterLengthNumb.value = parseInt(parseFloat(this.dursNew[this.letterFocus[0]])*100)/100

            letterPitchNumb.disabled = false
            letterLengthNumb.disabled = false
        } else {
            letterEnergyNumb.disabled = true
            letterEnergyNumb.value = ""
            letterEmotionNumb.disabled = true
            letterEmotionNumb.value = ""
            letterPitchNumb.disabled = true
            letterPitchNumb.value = ""
            letterLengthNumb.disabled = true
            letterLengthNumb.value = ""
        }
    }


    getChangedTimeStamps (startI, endI, audioSDuration) {

        const adjustedLetters = Array.from(this.adjustedLetters)

        // Skip this if start/end indexes are not found (new sample)
        if ((startI==-1 || endI==-1) && !adjustedLetters.length) {
            return undefined
        }
        startI = startI==-1 ? this.letters.length : parseInt(startI)
        endI = endI==-1 ? 0 : parseInt(endI)

        // Check OUTSIDE of the given changed indexes for TEXT, to see if there were other changes, to eg pitch/duration
        if (adjustedLetters.length) {
            startI = Math.min(startI, Math.min(adjustedLetters))
            endI = Math.max(endI, Math.max(adjustedLetters))
        }

        const newStartI = startI
        const newEndI = endI

        // Then, look through the duration values of the audio, and get a percent into the audio where those new start/end points are
        const totalDuration = this.dursNew.reduce((p,c)=>p+c,0)
        const durAtStart = this.dursNew.filter((v,vi) => vi<=newStartI).reduce((p,c)=>p+c,0)
        const durAtEnd = this.dursNew.filter((v,vi) => vi<=newEndI).reduce((p,c)=>p+c,0)
        const startPercent = durAtStart/totalDuration
        const endPercent = durAtEnd/totalDuration

        // Then, multiply this by the seconds duration of the generated audio, and pad with ~500ms, to get the final start/end of the section of the audio to play
        const startSeconds = Math.max(0, startPercent*audioSDuration-0.5)
        const endSeconds = Math.min(audioSDuration, endPercent*audioSDuration+0.5)

        return [startSeconds, endSeconds]
    }
}

class Letter {
    constructor (context, index, letter, sliderBox, centerY, left, width) {
        this.type = "letter"
        this.context = context
        this.letter = letter
        this.sliderBox = sliderBox
        this.centerY = centerY
        this.index = index

        this.left = left
        this.width = width
        this.colour = "black"
    }

    render () {
        this.context.fillStyle = this.colour
        this.context.font = "20pt Arial"
        this.context.textAlign = "center"
        this.context.textBaseline = "middle"
        this.context.fillText(this.letter, this.sliderBox.getXLeft()+this.sliderBox.width/2, this.centerY)
    }
}

class SliderGrabber {

    constructor (context, index, sliderBox, topLeftY, width, height, sliderRange) {
        this.type = "slider"
        this.context = context
        this.sliderBox = sliderBox
        this.topLeftY = topLeftY
        this.width = width
        this.height = height
        this.index = index
        this.sliderRange = sliderRange

        this.isBeingDragged = false
        this.dragStart = {x: undefined, y: undefined}

        this.fillStyle = `#${window.currentGame.themeColourPrimary}`
    }

    render () {

        this.context.beginPath()
        this.context.rect(this.sliderBox.getXLeft()+1, this.topLeftY, this.width, this.height)
        this.context.stroke()

        this.context.fillStyle = this.fillStyle
        this.context.fillRect(this.sliderBox.getXLeft()+1, this.topLeftY, this.width, this.height)
    }

    getXLeft () {
        return this.sliderBox.getXLeft()
    }

    setValueFromCoords (topLeftY) {

        this.topLeftY = topLeftY
        this.topLeftY = Math.max(window.sequenceEditor.LETTERS_Y_OFFSET+1, this.topLeftY)
        this.topLeftY = Math.min(this.topLeftY, window.sequenceEditor.LETTERS_Y_OFFSET+window.sequenceEditor.EDITOR_HEIGHT-this.height-1)

        this.percentUp = (this.topLeftY-window.sequenceEditor.LETTERS_Y_OFFSET) / (window.sequenceEditor.EDITOR_HEIGHT-this.height)
        window.sequenceEditor.pitchNew[this.index] = (1-this.percentUp)*(this.sliderRange*2)-this.sliderRange
    }

    setValueFromValue (value) {
        value = Math.max(-this.sliderRange, value)
        value = Math.min(value, this.sliderRange)
        this.percentUp = (value+this.sliderRange)/(this.sliderRange*2)

        this.topLeftY = (1-this.percentUp) * (window.sequenceEditor.EDITOR_HEIGHT-this.height) + window.sequenceEditor.LETTERS_Y_OFFSET
    }

}


class EnergyEmotionGrabber extends SliderGrabber {

    constructor (context, index, sliderBox, topLeftY, width, height, sliderRange, modelType, radius, sliderType) {
        super(context, index, sliderBox, topLeftY, width, height, sliderRange)
        this.type = `${sliderType}_slider`
        this.modelType = modelType
        this.radius = radius
    }

    render () {
        this.context.fillStyle = this.fillStyle
        this.context.beginPath()
        this.context.lineWidth = 1
        let x = this.sliderBox.getXLeft()+1 + this.sliderBox.width/2 // Centered
        let y = this.topLeftY
        this.context.arc(x, y, this.radius, 0, 2 * Math.PI)
        this.context.fill()
        this.context.stroke()
        this.context.lineWidth = 1
    }

    setValueFromCoords (topLeftY) {

        this.topLeftY = topLeftY
        this.topLeftY = Math.max(window.sequenceEditor.LETTERS_Y_OFFSET+this.radius, this.topLeftY)
        this.topLeftY = Math.min(this.topLeftY, window.sequenceEditor.LETTERS_Y_OFFSET+(window.sequenceEditor.EDITOR_HEIGHT-2-this.radius/2))

        if (this.type=="energy_slider") {
            if (this.modelType=="xVAPitch") {
                this.percentUp = (this.topLeftY-window.sequenceEditor.LETTERS_Y_OFFSET)/(window.sequenceEditor.EDITOR_HEIGHT-this.radius)
            } else {
                this.percentUp = 1-(this.topLeftY-window.sequenceEditor.LETTERS_Y_OFFSET)/(window.sequenceEditor.EDITOR_HEIGHT-this.radius)
            }
            window.sequenceEditor.energyNew[this.index] = window.sequenceEditor.MAX_ENERGY - (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)*this.percentUp
        } else if (this.type=="style_slider") {

            this.percentUp = (this.topLeftY-window.sequenceEditor.LETTERS_Y_OFFSET)/(window.sequenceEditor.EDITOR_HEIGHT-this.radius)

            window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
                if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
                    window.sequenceEditor.styleValuesNew[styleKey][this.index] = window.sequenceEditor.MAX_STYLES - (window.sequenceEditor.MAX_STYLES-window.sequenceEditor.MIN_STYLES)*this.percentUp
                }
            })

        } else {
            this.percentUp = (this.topLeftY-window.sequenceEditor.LETTERS_Y_OFFSET)/(window.sequenceEditor.EDITOR_HEIGHT-this.radius)
            if (seq_edit_view_select.value=="emAngry") {
                window.sequenceEditor.emAngryNew[this.index] = window.sequenceEditor.MAX_EMOTIONS - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_ENERGY)*this.percentUp
            } else if (seq_edit_view_select.value=="emHappy") {
                window.sequenceEditor.emHappyNew[this.index] = window.sequenceEditor.MAX_EMOTIONS - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_ENERGY)*this.percentUp
            } else if (seq_edit_view_select.value=="emSad") {
                window.sequenceEditor.emSadNew[this.index] = window.sequenceEditor.MAX_EMOTIONS - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_ENERGY)*this.percentUp
            } else if (seq_edit_view_select.value=="emSurprise") {
                window.sequenceEditor.emSurpriseNew[this.index] = window.sequenceEditor.MAX_EMOTIONS - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_ENERGY)*this.percentUp
            }
        }
    }

    setValueFromValue (value) {
        if (this.type=="energy_slider") {
            value = Math.max(window.sequenceEditor.MIN_ENERGY, value)
            value = Math.min(value, window.sequenceEditor.MAX_ENERGY)
            if (this.modelType=="xVAPitch") {
                this.percentUp = ( (value-window.sequenceEditor.MIN_ENERGY) / (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)  )
            } else {
                this.percentUp = 1 - ( (value-window.sequenceEditor.MIN_ENERGY) / (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)  )
            }
        } else if (this.type=="style_slider") {
            value = Math.max(window.sequenceEditor.MIN_STYLES, value)
            value = Math.min(value, window.sequenceEditor.MAX_STYLES)
            this.percentUp = ( (value-window.sequenceEditor.MIN_STYLES) / (window.sequenceEditor.MAX_STYLES-window.sequenceEditor.MIN_STYLES)  )
        } else {
            value = Math.max(window.sequenceEditor.MIN_EMOTIONS, value)
            value = Math.min(value, window.sequenceEditor.MAX_EMOTIONS)
            this.percentUp = ( (value-window.sequenceEditor.MIN_EMOTIONS) / (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_EMOTIONS)  )
        }

        this.topLeftY = (1 - this.percentUp) * (window.sequenceEditor.EDITOR_HEIGHT-2-this.radius*2) + (window.sequenceEditor.LETTERS_Y_OFFSET+1)
    }
}


class SliderBox {
    constructor (context, index, leftBox, topY, height, minLetterLength, maxLetterLength, alternateColour=false) {
        this.type = "box"
        this.context = context
        this.leftBox = leftBox
        this.topY = topY
        this.height = height
        this.index = index
        this.alternateColour = alternateColour

        this.LEFT_RIGHT_SEQ_PADDING = 20
        this.MIN_LETTER_LENGTH = minLetterLength
        this.MAX_LETTER_LENGTH = maxLetterLength

        this.isBeingDragged = false
        this.dragStart = {width: undefined, y: undefined}
    }

    render () {
        this.context.globalAlpha = 0.3
        this.context.fillStyle = this.alternateColour ? "white" : "black"
        this.context.fillRect(this.LEFT_RIGHT_SEQ_PADDING+ (this.getLeftBox()?this.getLeftBox().getX():0), this.topY, this.width, this.height)

        this.context.beginPath()
        this.context.rect(this.LEFT_RIGHT_SEQ_PADDING+ (this.getLeftBox()?this.getLeftBox().getX():0), this.topY, this.width, this.height)
        this.context.stroke()


        this.context.globalAlpha = 1
    }

    setValueFromValue (value) {

        value = value * window.sequenceEditor.pacing
        value = Math.max(0.1, value)
        value = Math.min(value, 20)

        this.percentAcross = value/20
        this.width = this.percentAcross * (this.MAX_LETTER_LENGTH-this.MIN_LETTER_LENGTH) + this.MIN_LETTER_LENGTH

        this.grabber.width = this.width-2
        this.letter.centerX = this.leftX + this.width/2
    }

    getLeftBox () {
        if (this.leftBox) {
            if (window.sequenceEditor.enabled_disabled_items[this.leftBox.index]) {
                return this.leftBox
            } else {
                return this.leftBox.leftBox
            }
        }
    }


    getX () {
        if (this.leftBox) {
            return this.getLeftBox().getX() + this.width + 5
        }
        return 0 + this.width + 5
    }
    getXLeft () {
        if (this.leftBox) {
            return this.LEFT_RIGHT_SEQ_PADDING+this.getLeftBox().getX()
        }
        return this.LEFT_RIGHT_SEQ_PADDING
    }
}







const infer = () => {
    window.sequenceEditor.hasChanged = false
    if (!isGenerating) {
        generateVoiceButton.click()
    }
}
const kickOffAutoInferTimer = () => {
    if (window.sequenceEditor.autoInferTimer != null) {
        clearTimeout(window.sequenceEditor.autoInferTimer)
        window.sequenceEditor.autoInferTimer = null
    }
    if (autoplay_ckbx.checked) {
        window.sequenceEditor.autoInferTimer = setTimeout(infer, 500)
    }
}


// Un-select letters when clicking anywhere else
right.addEventListener("click", event => {
    if (event.target.nodeName=="BUTTON" || event.target.nodeName=="INPUT" || event.target.nodeName=="SVG" || event.target.nodeName=="IMG" || event.target.nodeName=="path" || event.target == window.sequenceEditor.canvas) {
        return
    }
    window.sequenceEditor.letterFocus.forEach(li => {
        window.sequenceEditor.letterClasses[li].colour = "black"
    })
    window.sequenceEditor.letterFocus = []

    letterEnergyNumb.disabled = true
    letterEnergyNumb.value = ""
    letterPitchNumb.disabled = true
    letterPitchNumb.value = ""
    letterLengthNumb.disabled = true
    letterLengthNumb.value = ""
    letterEmotionNumb.disabled = true
    letterEmotionNumb.value = ""
    letterStyleNumb.disabled = true
    letterStyleNumb.value = ""
})

letterEnergyNumb.addEventListener("click", () => {
    const lpnValue = parseFloat(letterEnergyNumb.value) || 0
    if (window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
})
letterEnergyNumb.addEventListener("input", () => {
    const lpnValue = parseFloat(letterEnergyNumb.value) || 0
    if (window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]] = lpnValue
    window.sequenceEditor.energyGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    kickOffAutoInferTimer()
})
letterEnergyNumb.addEventListener("change", () => {
    const lpnValue = parseFloat(letterEnergyNumb.value) || 0
    if (window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]] = lpnValue
    window.sequenceEditor.energyGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    kickOffAutoInferTimer()
})

letterEmotionNumb.addEventListener("click", () => {
    const lpnValue = parseFloat(letterEmotionNumb.value) || 0
    let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
    if (data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
})
letterEmotionNumb.addEventListener("input", () => {
    const lpnValue = parseFloat(letterEmotionNumb.value) || 0
    let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
    if (window.sequenceEditor.data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.data[window.sequenceEditor.letterFocus[0]] = lpnValue
    grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    kickOffAutoInferTimer()
})
letterEmotionNumb.addEventListener("change", () => {
    const lpnValue = parseFloat(letterEmotionNumb.value) || 0
    let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
    if (data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    data[window.sequenceEditor.letterFocus[0]] = lpnValue
    grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    kickOffAutoInferTimer()
})

const getSelectedStyleDataAndGrabbers = () => {
    let data, grabbers
    window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
        if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
            data = window.sequenceEditor.styleValuesNew[styleKey]
            grabbers = window.sequenceEditor.styleGrabbers[styleKey]
        }
    })
    return [data, grabbers]
}

letterStyleNumb.addEventListener("click", () => {
    const lpnValue = parseFloat(letterStyleNumb.value) || 0
    let [data, grabbers] = getSelectedStyleDataAndGrabbers()
    if (data && data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
})
letterStyleNumb.addEventListener("input", () => {
    const lpnValue = parseFloat(letterStyleNumb.value) || 0
    let [data, grabbers] = getSelectedStyleDataAndGrabbers()
    if (data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    data[window.sequenceEditor.letterFocus[0]] = lpnValue
    grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    setNewDataToSelectedStyle(data)
    kickOffAutoInferTimer()
})
letterStyleNumb.addEventListener("change", () => {
    const lpnValue = parseFloat(letterStyleNumb.value) || 0
    let [data, grabbers] = getSelectedStyleDataAndGrabbers()
    if (data[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    data[window.sequenceEditor.letterFocus[0]] = lpnValue
    grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(lpnValue)
    setNewDataToSelectedStyle(data)
    kickOffAutoInferTimer()
})


letterPitchNumb.addEventListener("click", () => {
    const lpnValue = parseFloat(letterPitchNumb.value) || 0
    if (window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
})
letterPitchNumb.addEventListener("input", () => {
    const lpnValue = parseFloat(letterPitchNumb.value) || 0
    if (window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]] = lpnValue
    window.sequenceEditor.grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(letterPitchNumb.value)
    kickOffAutoInferTimer()
})
letterPitchNumb.addEventListener("change", () => {
    const lpnValue = parseFloat(letterPitchNumb.value) || 0
    if (window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]!=lpnValue) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]] = lpnValue
    window.sequenceEditor.grabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(letterPitchNumb.value)
    kickOffAutoInferTimer()
})

resetLetter_btn.addEventListener("click", () => {
    if (window.sequenceEditor.letterFocus.length==0) {
        return
    }

    window.sequenceEditor.letterFocus.forEach(l => {
        if (window.sequenceEditor.dursNew[l] != window.sequenceEditor.resetDurs[l]) {
            window.sequenceEditor.hasChanged = true
        }
        window.sequenceEditor.dursNew[l] = window.sequenceEditor.resetDurs[l]
        window.sequenceEditor.pitchNew[l] = window.sequenceEditor.resetPitch[l]

        window.sequenceEditor.grabbers[l].setValueFromValue(window.sequenceEditor.resetPitch[l])
        window.sequenceEditor.sliderBoxes[l].setValueFromValue(window.sequenceEditor.resetDurs[l])
    })

    if (window.sequenceEditor.letterFocus.length==1) {
        letterLengthNumb.value = parseFloat(window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]])
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]])

        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()

        window.sequenceEditor.emAngryNew[window.sequenceEditor.letterFocus[0]] = window.sequenceEditor.resetEmAngry[window.sequenceEditor.letterFocus[0]]
        window.sequenceEditor.emAngryGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(window.sequenceEditor.emAngryNew[window.sequenceEditor.letterFocus[0]])
        window.sequenceEditor.emHappyNew[window.sequenceEditor.letterFocus[0]] = window.sequenceEditor.resetEmHappy[window.sequenceEditor.letterFocus[0]]
        window.sequenceEditor.emHappyGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(window.sequenceEditor.emHappyNew[window.sequenceEditor.letterFocus[0]])
        window.sequenceEditor.emSadNew[window.sequenceEditor.letterFocus[0]] = window.sequenceEditor.resetEmSad[window.sequenceEditor.letterFocus[0]]
        window.sequenceEditor.emSadGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(window.sequenceEditor.emSadNew[window.sequenceEditor.letterFocus[0]])
        window.sequenceEditor.emSurpriseNew[window.sequenceEditor.letterFocus[0]] = window.sequenceEditor.resetEmSurprise[window.sequenceEditor.letterFocus[0]]
        window.sequenceEditor.emSurpriseGrabbers[window.sequenceEditor.letterFocus[0]].setValueFromValue(window.sequenceEditor.emSurpriseNew[window.sequenceEditor.letterFocus[0]])

        if (data) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]])
        }

        window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
            window.sequenceEditor.styleValuesNew[styleKey][window.sequenceEditor.letterFocus[0]] = window.sequenceEditor.styleValuesReset[styleKey][window.sequenceEditor.letterFocus[0]]
            window.sequenceEditor.styleGrabbers[styleKey][window.sequenceEditor.letterFocus[0]].setValueFromValue(window.sequenceEditor.styleValuesNew[styleKey][window.sequenceEditor.letterFocus[0]])
        })

        let [styleData, styleGrabbers] = getSelectedStyleDataAndGrabbers()
        if (styleData) {
            letterStyleNumb.value = parseInt(styleData[window.sequenceEditor.letterFocus[0]])
        }
    }
})
const updateLetterLengthFromInput = () => {
    if (window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]] != letterLengthNumb.value) {
        window.sequenceEditor.hasChanged = true
    }
    window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]] = parseFloat(letterLengthNumb.value)

    window.sequenceEditor.letterFocus.forEach(l => {
        window.sequenceEditor.sliderBoxes[l].setValueFromValue(window.sequenceEditor.dursNew[l])
    })
    kickOffAutoInferTimer()
}
letterLengthNumb.addEventListener("input", () => {
    updateLetterLengthFromInput()
})
letterLengthNumb.addEventListener("change", () => {
    updateLetterLengthFromInput()
})

// Reset button
window.resetEnergy = () => {
    window.sequenceEditor.energyNew = window.sequenceEditor.resetEnergy.map(v => v)
    window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
        slider.setValueFromValue(window.sequenceEditor.energyNew[l])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
    if (window.sequenceEditor.letterFocus.length==1) {
        letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]])
    }
}
window.resetStyle = () => {
    window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
        window.sequenceEditor.styleValuesNew[styleKey] = window.sequenceEditor.styleValuesReset[styleKey].map(v => v)
        window.sequenceEditor.styleGrabbers[styleKey].forEach((slider, l) => slider.setValueFromValue(window.sequenceEditor.styleValuesNew[styleKey][l]))

    })
    let [data, grabbers] = getSelectedStyleDataAndGrabbers()
    if (data) {
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]])
        }
    }
}
window.resetEmotion = () => {
    window.sequenceEditor.emAngryNew = window.sequenceEditor.resetEmAngry.map(v => v)
    window.sequenceEditor.emHappyNew = window.sequenceEditor.resetEmHappy.map(v => v)
    window.sequenceEditor.emSadNew = window.sequenceEditor.resetEmSad.map(v => v)
    window.sequenceEditor.emSurpriseNew = window.sequenceEditor.resetEmSurprise.map(v => v)

    let [data, grabbers] = getSelectedEmotionDataAndGrabbers()

    window.sequenceEditor.emAngryGrabbers.forEach((slider, l) => slider.setValueFromValue(window.sequenceEditor.emAngryNew[l]))
    window.sequenceEditor.emHappyGrabbers.forEach((slider, l) => slider.setValueFromValue(window.sequenceEditor.emHappyNew[l]))
    window.sequenceEditor.emSadGrabbers.forEach((slider, l) => slider.setValueFromValue(window.sequenceEditor.emSadNew[l]))
    window.sequenceEditor.emSurpriseGrabbers.forEach((slider, l) => slider.setValueFromValue(window.sequenceEditor.emSurpriseNew[l]))

    if (data && window.sequenceEditor.letterFocus.length==1) {
        letterEmotionNumb.value = parseFloat(data[window.sequenceEditor.letterFocus[0]])
    }
}
window.resetPitch = () => {
    window.sequenceEditor.pitchNew = window.sequenceEditor.resetPitch.map(p=>p)
    // Update the editor pitch values
    window.sequenceEditor.grabbers.forEach((slider, i) => {
        slider.setValueFromValue(window.sequenceEditor.pitchNew[i])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
}
window.resetDursPace = () => {

    pace_slid.value = 1
    paceNumbInput.value = 1
    window.sequenceEditor.pacing = parseFloat(pace_slid.value)

    window.sequenceEditor.dursNew = window.sequenceEditor.resetDurs.map(v => v)
    // Update the editor lengths
    window.sequenceEditor.sliderBoxes.forEach((box,i) => {
        box.setValueFromValue(window.sequenceEditor.dursNew[i])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterLengthNumb.value = parseFloat(window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]])
    }

}
reset_btn.addEventListener("click", () => {

    if (window.shiftKeyIsPressed) {
        if (seq_edit_edit_select.value=="energy") {
            resetEnergy()
            resetDursPace()

        } else if (seq_edit_edit_select.value=="pitch") {
            resetPitch()
            resetDursPace()

        } else if (seq_edit_edit_select.value=="emotion") {
            resetEmotion()
            resetDursPace()

        } else if (seq_edit_edit_select.value=="style") {
            resetStyle()
            resetDursPace()
        }

        window.sequenceEditor.init()
    } else {
        reset_what_open_btn.click()
    }
})
reset_what_confirm_btn.addEventListener("click", () => {
    resetContainer.click()
    if (reset_what_pitch.checked) {
        resetPitch()
    }
    if (reset_what_energy.checked) {
        resetEnergy()
    }
    if (reset_what_duration.checked) {
        resetDursPace()
    }
    if (reset_what_emotion.checked) {
        resetEmotion()
    }
    if (reset_what_style.checked) {
        resetStyle()
    }
    window.sequenceEditor.init()
})

const getSelectedEmotionDataAndGrabbers = () => {
    let data, grabbers
    if (seq_edit_view_select.value=="emAngry") {
        data = window.sequenceEditor.emAngryNew
        grabbers = window.sequenceEditor.emAngryGrabbers
    } else if (seq_edit_view_select.value=="emHappy") {
        data = window.sequenceEditor.emHappyNew
        grabbers = window.sequenceEditor.emHappyGrabbers
    } else if (seq_edit_view_select.value=="emSad") {
        data = window.sequenceEditor.emSadNew
        grabbers = window.sequenceEditor.emSadGrabbers
    } else if (seq_edit_view_select.value=="emSurprise") {
        data = window.sequenceEditor.emSurpriseNew
        grabbers = window.sequenceEditor.emSurpriseGrabbers
    }
    return [data, grabbers]
}
const setNewDataToSelectedEmotion = (data) => {
    if (seq_edit_view_select.value=="emAngry") {
        window.sequenceEditor.emAngryNew = data
    } else if (seq_edit_view_select.value=="emHappy") {
        window.sequenceEditor.emHappyNew = data
    } else if (seq_edit_view_select.value=="emSad") {
        window.sequenceEditor.emSadNew = data
    } else if (seq_edit_view_select.value=="emSurprise") {
        window.sequenceEditor.emSurpriseNew = data
    }
}
const setNewDataToSelectedStyle = (data) => {
    window.sequenceEditor.registeredStyleKeys.forEach(styleKey => {
        if (seq_edit_view_select.value.startsWith("style_") && seq_edit_view_select.value.includes(styleKey)) {
            window.sequenceEditor.styleValuesNew[styleKey] = data
        }
    })
}

amplify_btn.addEventListener("click", () => {
    if (seq_edit_edit_select.value=="pitch") {
        window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p, pi) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
                return p
            }
            const newVal = p*1.025
            return newVal>0 ? Math.min(window.sequenceEditor.pitchSliderRange, newVal) : Math.max(-window.sequenceEditor.pitchSliderRange, newVal)
        })
        window.sequenceEditor.grabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_edit_select.value=="energy") {
        window.sequenceEditor.energyNew = window.sequenceEditor.energyNew.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_ENERGY) - (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)/2
            const newVal = e + distFromMiddle*0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_ENERGY, newVal) : Math.max(window.sequenceEditor.MIN_ENERGY, newVal)
        })
        window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.energyNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_view_select.value.startsWith("style_")) {

        let [data, grabbers] = getSelectedStyleDataAndGrabbers()

        data = data.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_STYLES) - (window.sequenceEditor.MAX_STYLES-window.sequenceEditor.MIN_STYLES)/2
            const newVal = e + distFromMiddle*0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_STYLES, newVal) : Math.max(window.sequenceEditor.MIN_STYLES, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedStyle(data)

    } else if (seq_edit_edit_select.value=="emotion") {

        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()

        data = data.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_EMOTIONS) - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_EMOTIONS)/2
            const newVal = e + distFromMiddle*0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_EMOTIONS, newVal) : Math.max(window.sequenceEditor.MIN_EMOTIONS, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedEmotion(data)
    }
    kickOffAutoInferTimer()
})
flatten_btn.addEventListener("click", () => {
    if (seq_edit_edit_select.value=="pitch") {
        window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p,pi) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
                return p
            }
            return p*(1-0.025)
        })
        window.sequenceEditor.grabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }

    } else if (seq_edit_edit_select.value=="energy") {
        window.sequenceEditor.energyNew = window.sequenceEditor.energyNew.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_ENERGY) - (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)/2
            const newVal = e + distFromMiddle*-0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_ENERGY, newVal) : Math.max(window.sequenceEditor.MIN_ENERGY, newVal)
        })
        window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.energyNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_view_select.value.startsWith("style_")) {

        let [data, grabbers] = getSelectedStyleDataAndGrabbers()

        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_STYLES) - (window.sequenceEditor.MAX_STYLES-window.sequenceEditor.MIN_STYLES)/2
            const newVal = e + distFromMiddle*-0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_STYLES, newVal) : Math.max(window.sequenceEditor.MIN_STYLES, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedStyle(data)


    } else if (seq_edit_edit_select.value=="emotion") {
        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()

        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_EMOTIONS) - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_EMOTIONS)/2
            const newVal = e + distFromMiddle*-0.025
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_EMOTIONS, newVal) : Math.max(window.sequenceEditor.MIN_EMOTIONS, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedEmotion(data)
    }
    kickOffAutoInferTimer()
})


jitter_btn.addEventListener("click", () => {
    if (seq_edit_edit_select.value=="pitch") {
        window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p, pi) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
                return p
            }
            let newVal
            if (p==0) {
                newVal = 1*(1+ (Math.random()*0.4+0.05) * ((Math.random()-0.5)>0 ? 1 : -1)  )
                newVal -= 1
            } else {
                newVal = p*(1+ (Math.random()*0.2+0.05) * ((Math.random()-0.5)>0 ? 1 : -1)  )
            }
            return newVal>0 ? Math.min(window.sequenceEditor.pitchSliderRange, newVal) : Math.max(-window.sequenceEditor.pitchSliderRange, newVal)
        })
        window.sequenceEditor.grabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_edit_select.value=="energy") {
        window.sequenceEditor.energyNew = window.sequenceEditor.energyNew.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_ENERGY) - (window.sequenceEditor.MAX_ENERGY-window.sequenceEditor.MIN_ENERGY)/2
            const newVal = e + distFromMiddle*(Math.random()*0.1+0.05) * ((Math.random()-0.5)>0 ? 1 : -1)
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_ENERGY, newVal) : Math.max(window.sequenceEditor.MIN_ENERGY, newVal)
        })
        window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.energyNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_view_select.value.startsWith("style_")) {

        let [data, grabbers] = getSelectedStyleDataAndGrabbers()
        data = data.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_STYLES) - (window.sequenceEditor.MAX_STYLES-window.sequenceEditor.MIN_STYLES)/2
            const newVal = e + distFromMiddle*(Math.random()*0.1+0.05) * ((Math.random()-0.5)>0 ? 1 : -1)
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_STYLES, newVal) : Math.max(window.sequenceEditor.MIN_STYLES, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedStyle(data)

    } else if (seq_edit_edit_select.value=="emotion") {
        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
        data = data.map((e, ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            const distFromMiddle = (e-window.sequenceEditor.MIN_EMOTIONS) - (window.sequenceEditor.MAX_EMOTIONS-window.sequenceEditor.MIN_EMOTIONS)/2
            const newVal = e + distFromMiddle*(Math.random()*0.1+0.05) * ((Math.random()-0.5)>0 ? 1 : -1)
            return newVal>0 ? Math.min(window.sequenceEditor.MAX_EMOTIONS, newVal) : Math.max(window.sequenceEditor.MIN_EMOTIONS, newVal)
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedEmotion(data)
    }
    kickOffAutoInferTimer()
})


increase_btn.addEventListener("click", () => {
    if (seq_edit_edit_select.value=="pitch") {
        window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p,pi) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
                return p
            }
            return p+0.1
        })
        window.sequenceEditor.grabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_edit_select.value=="energy") {
        window.sequenceEditor.energyNew = window.sequenceEditor.energyNew.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e-0.04
        })
        window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.energyNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }

    } else if (seq_edit_view_select.value.startsWith("style_")) {

        let [data, grabbers] = getSelectedStyleDataAndGrabbers()
        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e-0.04
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedStyle(data)


    } else if (seq_edit_edit_select.value=="emotion") {
        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e-0.04
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedEmotion(data)
    }
    kickOffAutoInferTimer()
})
decrease_btn.addEventListener("click", () => {
    if (seq_edit_edit_select.value=="pitch") {
        window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p,pi) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
                return p
            }
            return p-0.1
        })
        window.sequenceEditor.grabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }
    } else if (seq_edit_edit_select.value=="energy") {
        window.sequenceEditor.energyNew = window.sequenceEditor.energyNew.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e+0.04
        })
        window.sequenceEditor.energyGrabbers.forEach((slider, l) => {
            slider.setValueFromValue(window.sequenceEditor.energyNew[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEnergyNumb.value = parseInt(window.sequenceEditor.energyNew[window.sequenceEditor.letterFocus[0]]*100)/100
        }

    } else if (seq_edit_view_select.value.startsWith("style_")) {

        let [data, grabbers] = getSelectedStyleDataAndGrabbers()
        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e+0.04
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterStyleNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedStyle(data)


    } else if (seq_edit_edit_select.value=="emotion") {
        let [data, grabbers] = getSelectedEmotionDataAndGrabbers()
        data = data.map((e,ei) => {
            if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(ei)==-1) {
                return e
            }
            return e+0.04
        })
        grabbers.forEach((slider, l) => {
            slider.setValueFromValue(data[l])
        })
        if (window.sequenceEditor.letterFocus.length==1) {
            letterEmotionNumb.value = parseInt(data[window.sequenceEditor.letterFocus[0]]*100)/100
        }
        setNewDataToSelectedEmotion(data)
    }
    kickOffAutoInferTimer()
})

pace_slid.addEventListener("input", () => {
    paceNumbInput.value = pace_slid.value
})

pace_slid.addEventListener("change", () => {
    editorTooltip.style.display = "none"
    if (autoplay_ckbx.checked) {
        generateVoiceButton.click()
    }
    paceNumbInput.value = pace_slid.value
    window.sequenceEditor.pacing = parseFloat(pace_slid.value)
    window.sequenceEditor.init()
})

pace_slid.addEventListener("input", () => {
    window.sequenceEditor.pacing = parseFloat(pace_slid.value)
    window.sequenceEditor.sliderBoxes.forEach((box, i) => {
        box.setValueFromValue(window.sequenceEditor.dursNew[i])
    })
})
paceNumbInput.addEventListener("change", () => {
    pace_slid.value = paceNumbInput.value
    if (autoplay_ckbx.checked) {
        generateVoiceButton.click()
    }
    window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
    window.sequenceEditor.pacing = parseFloat(pace_slid.value)
    window.sequenceEditor.init()
})
paceNumbInput.addEventListener("keyup", () => {
    pace_slid.value = paceNumbInput.value
    window.sequenceEditor.sliderBoxes.forEach((box, i) => {box.setValueFromValue(window.sequenceEditor.dursNew[i])})
    window.sequenceEditor.pacing = parseFloat(pace_slid.value)
    window.sequenceEditor.init()
})
autoplay_ckbx.addEventListener("change", () => {
    window.userSettings.autoplay = autoplay_ckbx.checked
    saveUserSettings()
})


// Populate the languages dropdown
window.supportedLanguages = {
    // "am": "Amharic",
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "ha": "Hausa",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "jp": "Japanese",
    "ko": "Korean",
    "la": "Latin",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    // "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "yo": "Yoruba",
    "zh": "Chinese"
}
window.populateLanguagesDropdownsFromModel = (dropdown, modelJson=undefined) => {
    dropdown.innerHTML = ""

    Object.keys(window.supportedLanguages).sort((a,b)=>window.supportedLanguages[a]<window.supportedLanguages[b]?-1:1).forEach(key => {
        if (!modelJson || !modelJson.lang_capabilities || modelJson.lang_capabilities.includes(key)) {
            const opt = createElem("option", window.supportedLanguages[key])
            opt.value = key
            dropdown.appendChild(opt)
        }
    })
}
window.populateLanguagesDropdownsFromModel(base_lang_select)
window.populateLanguagesDropdownsFromModel(voiceWorkbenchLanguageDropdown)
base_lang_select.value = "en"
voiceWorkbenchLanguageDropdown.value = "en"



// For copying the generated ARPAbet sequence to the clipboard
editorContainer.addEventListener("contextmenu", event => {
    event.preventDefault()
    ipcRenderer.send('show-context-menu-editor')
})
ipcRenderer.on('context-menu-command', (e, command) => {

    if (command=="context-copy-editor") {
        if (window.sequenceEditor && window.sequenceEditor.sequence && window.sequenceEditor.sequence.length && window.currentModel && window.currentModel.modelType=="xVAPitch") {

            let seqARPAbet = window.sequenceEditor.sequence
            if (seqARPAbet[0]=="_") {
                seqARPAbet = seqARPAbet.slice(1, seqARPAbet.length)
            }
            if (seqARPAbet[seqARPAbet.length-1]=="_") {
                seqARPAbet = seqARPAbet.slice(0, seqARPAbet.length-1)
            }

            seqARPAbet = seqARPAbet.filter(val => val!="<PAD>")
            seqARPAbet = seqARPAbet.map(v => {
                if (v=="_") {
                    return "} {"
                }
                return v
            })

            clipboard.writeText("{"+seqARPAbet.join(" ")+"}")
        }
    }
})


// Audio player
window.initWaveSurfer = (src) => {
    if (window.wavesurfer) {
        window.wavesurfer.stop()
        wavesurferContainer.innerHTML = ""
    } else {
        window.wavesurfer = WaveSurfer.create({
            container: '#wavesurferContainer',
            backend: 'MediaElement',
            waveColor: `#${window.currentGame.themeColourPrimary}`,
            height: 100,
            progressColor: 'white',
            responsive: true,
        })
    }
    try {
        window.wavesurfer.setSinkId(window.userSettings.base_speaker)
    } catch (e) {
        console.log("Can't set sinkId")
    }
    if (src) {
        window.wavesurfer.load(src)
    }
    window.wavesurfer.on("finish", () => {
        samplePlayPause.innerHTML = window.i18n.PLAY
    })
    window.wavesurfer.on("seek", event => {
        if (event!=0) {
            window.wavesurfer.play()
            samplePlayPause.innerHTML = window.i18n.PAUSE
        }
    })
}
window.samplePlayPauseHandler = event => {
    if (window.wavesurfer) {
        if (event.ctrlKey) {
            if (window.wavesurfer.sink_id!=window.userSettings.alt_speaker) {
                window.wavesurfer.setSinkId(window.userSettings.alt_speaker)
            }
        } else {
            if (window.wavesurfer.sink_id!=window.userSettings.base_speaker) {
                window.wavesurfer.setSinkId(window.userSettings.base_speaker)
            }
        }

        if (window.wavesurfer.isPlaying()) {
            samplePlayPause.innerHTML = window.i18n.PLAY
            window.wavesurfer.playPause()
        } else {
            samplePlayPause.innerHTML = window.i18n.PAUSE
            window.wavesurfer.playPause()
        }
    }
}
samplePlayPause.addEventListener("click", window.samplePlayPauseHandler)


exports.Editor = Editor

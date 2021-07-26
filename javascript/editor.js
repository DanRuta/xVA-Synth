"use strict"


class Editor {

    constructor () {

        this.isCreated = false
        this.sliderBoxes = []
        this.grabbers = []

        // Data
        this.letters = []
        this.pitchNew = []
        this.dursNew = []
        this.pacing = 1
        this.ampFlatCounter = 0

        this.hasChanged = false
        this.autoInferTimer = null

        this.inputSequence = undefined
        this.currentVoice = undefined
        this.letterFocus = []
        this.letterClasses = []
        this.resetDurs = []
        this.resetPitch = []


        this.LEFT_RIGHT_SEQ_PADDING = 20
        this.EDITOR_HEIGHT = 150
        this.LETTERS_Y_OFFSET = 40
        this.SLIDER_GRABBER_H = 15
        this.MIN_LETTER_LENGTH = 20
        this.MAX_LETTER_LENGTH = 200
        this.SPACE_BETWEEN_LETTERS = 5


        this.multiLetterPitchDelta = undefined
        this.multiLetterStartPitchVals = []

        this.multiLetterLengthDelta = undefined
        this.multiLetterStartLengthVals = []

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
            const mouseX = parseInt(event.layerX)
            const mouseY = parseInt(event.layerY)
            this.canvas.style.cursor = "default"

            // Check grabber hover
            const isOnGrabber = this.grabbers.find(grabber => {
                const grabberX = grabber.getXLeft()
                return (mouseX>grabberX && mouseX<grabberX+grabber.width) && (mouseY>grabber.topLeftY && mouseY<grabber.topLeftY+grabber.height)
            })
            if (isOnGrabber!=undefined) {
                this.canvas.style.cursor = "n-resize"
                return
            }
            // Check letter hover
            const isOnLetter = this.letterClasses.find((letter, l) => {
                return (mouseY<this.LETTERS_Y_OFFSET) && (mouseX>this.sliderBoxes[l].getXLeft() && mouseX<this.sliderBoxes[l].getXLeft()+this.sliderBoxes[l].width)
            })
            if (isOnLetter!=undefined) {
                this.canvas.style.cursor = "pointer"
                return
            }
            // Check box length dragger
            const isBetweenBoxes = this.sliderBoxes.find(box => {
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
            const mouseX = parseInt(event.layerX)
            const mouseY = parseInt(event.layerY)
            mouseDownStart.x = mouseX
            mouseDownStart.y = mouseY

            // Check up-down pitch dragging
            const isOnGrabber = this.grabbers.find(grabber => {
                const grabberX = grabber.getXLeft()
                return (mouseX>grabberX && mouseX<grabberX+grabber.width) && (mouseY>grabber.topLeftY && mouseY<grabber.topLeftY+grabber.height)
            })
            if (isOnGrabber) {

                const slider = isOnGrabber

                if (this.letterFocus.length <= 1 || (!event.ctrlKey && !this.letterFocus.includes(slider.index))) {
                    this.setLetterFocus(this.grabbers.indexOf(slider))
                }
                this.multiLetterPitchDelta = slider.topLeftY
                this.multiLetterStartPitchVals = this.grabbers.map(slider => slider.topLeftY)

                elemDragged = isOnGrabber
                return
            }
            // Check sideways dragging
            const isBetweenBoxes = this.sliderBoxes.find(box => {
                const boxX = box.getXLeft()
                return (mouseY>box.topY && mouseY<box.topY+box.height) && (mouseX>(boxX+box.width-10) && mouseX<(boxX+box.width+10)+5)
            })
            if (isBetweenBoxes) {
                isBetweenBoxes.dragStart.width = isBetweenBoxes.width
                elemDragged = isBetweenBoxes
                return
            }

            // Check clicking on the top letters
            const isOnLetter = this.letterClasses.find((letter, l) => {
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

                if (elemDragged && (parseInt(event.layerX)-mouseDownStart.x || parseInt(event.layerY)-mouseDownStart.y)) {
                    this.hasChanged = true
                }

                if (elemDragged) {
                    if (elemDragged.type=="slider") {

                        elemDragged.setValueFromCoords(parseInt(event.layerY)-elemDragged.height/2)

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



                    } else if (elemDragged.type=="box") {

                        let newWidth = elemDragged.dragStart.width + parseInt(event.layerX)-mouseDownStart.x
                        newWidth = Math.max(20, newWidth)
                        newWidth = Math.min(newWidth, 200)
                        elemDragged.width = newWidth

                        elemDragged.percentAcross = (elemDragged.width-20) / (200-20)
                        this.dursNew[elemDragged.index] = Math.max(0.1, elemDragged.percentAcross*20)

                        elemDragged.grabber.width = elemDragged.width-2
                        elemDragged.letter.centerX = elemDragged.leftX + elemDragged.width/2

                        letterLengthNumb.value = parseInt(this.dursNew[elemDragged.index]*100)/100
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
            this.letterClasses.forEach(letter => {letter.context=this.context;letter.render()})
            this.sliderBoxes.forEach(sliderBox => {sliderBox.context=this.context;sliderBox.render()})
            this.grabbers.forEach(grabber => {grabber.context=this.context;grabber.render()})
        }
        requestAnimationFrame(() => {this.render()})
    }




    update () {

        this.letterClasses = []
        this.sliderBoxes = []
        this.grabbers = []

        let xCounter = 0
        let lastBox = undefined
        this.letters.forEach((letter, li) => {

            const dur = this.dursNew[li]
            const width = Math.max(25, dur*10)


            // Slider box
            const sliderBox = new SliderBox(this.context, li, lastBox, this.LETTERS_Y_OFFSET, this.EDITOR_HEIGHT, this.MIN_LETTER_LENGTH, this.MAX_LETTER_LENGTH)
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
            const pitchPercent = 1-(this.pitchNew[li]+3)/6
            const grabber = new SliderGrabber(this.context, li, sliderBox, (this.LETTERS_Y_OFFSET+1)+(this.SLIDER_GRABBER_H/2)+((this.EDITOR_HEIGHT-2)-this.SLIDER_GRABBER_H)*pitchPercent-this.SLIDER_GRABBER_H/2, width-2, this.SLIDER_GRABBER_H)
            grabber.render()
            this.grabbers.push(grabber)

            sliderBox.letter = letterClass
            sliderBox.grabber = grabber

            sliderBox.setValueFromValue(dur)

            xCounter += width + 5

        })
    }

    setLetterFocus (l, ctrlKey, altKey, shiftKey) {
        // On alt key modifier, make a selection on the whole whole word
        if (altKey) {
            this.letterFocus.push(l)
            let l2 = l
            // Looking backwards
            while (l2>=0) {
                let prevLetter = this.letters[l2]
                if (prevLetter!="_") {
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

        } else {

            if (shiftKey && this.letterFocus.length) {
                const lastSelected = this.letterFocus[this.letterFocus.length-1]

                if (l>lastSelected) {
                    for (let i=lastSelected; i<=l; i++) {
                        this.letterFocus.push(i)
                    }
                } else {
                    for (let i=l; i<=this.letterFocus[0]; i++) {
                        this.letterFocus.push(i)
                    }
                }
            } else {
                if (this.letterFocus.length && !ctrlKey) {
                    this.letterFocus.forEach(li => {
                        this.letterClasses[li].colour = "black"
                    })


                    this.letterFocus = []
                }
                this.letterFocus.push(l)
            }
        }

        this.letterFocus = Array.from(new Set(this.letterFocus.sort()))
        this.letterFocus.forEach(li => {
            this.letterClasses[li].colour = "red"
        })


        if (this.letterFocus.length==1) {
            letterPitchNumb.value = parseInt(this.pitchNew[this.letterFocus[0]]*100)/100
            letterLengthNumb.value = parseInt(parseFloat(this.dursNew[this.letterFocus[0]])*100)/100

            letterPitchNumb.disabled = false
            letterLengthNumb.disabled = false
        } else {
            letterPitchNumb.disabled = true
            letterPitchNumb.value = ""
            letterLengthNumb.disabled = true
            letterLengthNumb.value = ""
        }
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

    constructor (context, index, sliderBox, topLeftY, width, height) {
        this.type = "slider"
        this.context = context
        this.sliderBox = sliderBox
        this.topLeftY = topLeftY
        this.width = width
        this.height = height
        this.index = index

        this.isBeingDragged = false
        this.dragStart = {x: undefined, y: undefined}

        this.fillStyle = `#${window.currentGame[1]}`
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
        window.sequenceEditor.pitchNew[this.index] = (1-this.percentUp)*6-3
    }

    setValueFromValue (value) {
        value = Math.max(-3, value)
        value = Math.min(value, 3)
        this.percentUp = (value+3)/6

        this.topLeftY = (1-this.percentUp) * (window.sequenceEditor.EDITOR_HEIGHT-this.height) + window.sequenceEditor.LETTERS_Y_OFFSET
    }

}

class SliderBox {
    constructor (context, index, leftBox, topY, height, minLetterLength, maxLetterLength) {
        this.type = "box"
        this.context = context
        this.leftBox = leftBox
        this.topY = topY
        this.height = height
        this.index = index

        this.LEFT_RIGHT_SEQ_PADDING = 20
        this.MIN_LETTER_LENGTH = minLetterLength
        this.MAX_LETTER_LENGTH = maxLetterLength

        this.isBeingDragged = false
        this.dragStart = {width: undefined, y: undefined}
    }

    render () {
        this.context.globalAlpha = 0.3
        this.context.fillStyle = this.index%2==0 ? "white" : "black"
        this.context.fillRect(this.LEFT_RIGHT_SEQ_PADDING+ (this.leftBox?this.leftBox.getX():0), this.topY, this.width, this.height)

        this.context.beginPath()
        this.context.rect(this.LEFT_RIGHT_SEQ_PADDING+ (this.leftBox?this.leftBox.getX():0), this.topY, this.width, this.height)
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


    getX () {
        if (this.leftBox) {
            return this.leftBox.getX() + this.width + 5
        }
        return 0 + this.width + 5
    }
    getXLeft () {
        if (this.leftBox) {
            return this.LEFT_RIGHT_SEQ_PADDING+this.leftBox.getX()
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

    letterPitchNumb.disabled = true
    letterPitchNumb.value = ""
    letterLengthNumb.disabled = true
    letterLengthNumb.value = ""
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
}
letterLengthNumb.addEventListener("input", () => {
    updateLetterLengthFromInput()
})
letterLengthNumb.addEventListener("change", () => {
    updateLetterLengthFromInput()
})

// Reset button
reset_btn.addEventListener("click", () => {
    window.sequenceEditor.dursNew = window.sequenceEditor.resetDurs.map(v => v)
    window.sequenceEditor.pitchNew = window.sequenceEditor.resetPitch.map(p=>p)

    // Update the editor pitch values
    window.sequenceEditor.grabbers.forEach((slider, i) => {
        slider.setValueFromValue(window.sequenceEditor.pitchNew[i])
    })

    // Update the editor lengths
    window.sequenceEditor.sliderBoxes.forEach((box,i) => {
        box.setValueFromValue(window.sequenceEditor.dursNew[i])
    })

    if (window.sequenceEditor.letterFocus.length==1) {
        letterLengthNumb.value = parseFloat(window.sequenceEditor.dursNew[window.sequenceEditor.letterFocus[0]])
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
    pace_slid.value = 1
})
amplify_btn.addEventListener("click", () => {
    window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p, pi) => {
        if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        const newVal = p*1.025
        return newVal>0 ? Math.min(3, newVal) : Math.max(-3, newVal)
    })
    window.sequenceEditor.grabbers.forEach((slider, l) => {
        slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
    kickOffAutoInferTimer()
})
flatten_btn.addEventListener("click", () => {
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
    kickOffAutoInferTimer()
})
increase_btn.addEventListener("click", () => {
    window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p,pi) => {
        if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        return p+0.025
    })
    window.sequenceEditor.grabbers.forEach((slider, l) => {
        slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
    kickOffAutoInferTimer()
})
decrease_btn.addEventListener("click", () => {
    window.sequenceEditor.pitchNew = window.sequenceEditor.pitchNew.map((p,pi) => {
        if (window.sequenceEditor.letterFocus.length>1 && window.sequenceEditor.letterFocus.indexOf(pi)==-1) {
            return p
        }
        return p-0.025
    })
    window.sequenceEditor.grabbers.forEach((slider, l) => {
        slider.setValueFromValue(window.sequenceEditor.pitchNew[l])
    })
    if (window.sequenceEditor.letterFocus.length==1) {
        letterPitchNumb.value = parseInt(window.sequenceEditor.pitchNew[window.sequenceEditor.letterFocus[0]]*100)/100
    }
    kickOffAutoInferTimer()
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



exports.Editor = Editor
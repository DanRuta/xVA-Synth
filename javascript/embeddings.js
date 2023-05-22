"use strict"


window.openEmbeddingsWindow = () => {
    closeModal(undefined, embeddingsContainer).then(() => {
        embeddingsContainer.style.opacity = 0
        embeddingsContainer.style.display = "flex"
        requestAnimationFrame(() => requestAnimationFrame(() => embeddingsContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
    })
}

window.embeddingsState = {
    data: {},
    allData: {},
    clickedObject: undefined,
    spritesOn: true,
    gendersOn: false,
    voiceCheckboxes: [],
    isReady: false,
    isOpen: false,
    sceneData: {},
    mouseIsDown: false,
    rightMouseIsDown: false,
    mousePos: {x: 0, y: 0}
}



function componentToHex(c) {
    c = Math.min(255, c)
    var hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

function rgbToHex(r, g, b) {
    return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}
function hexToRgb(hex, normalize) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    const colour = result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
    if (normalize && colour) {
        colour.r /= 255
        colour.g /= 255
        colour.b /= 255
    }
    return colour
}


window.populateGamesList = () => {
    embeddingsGamesListContainer.innerHTML = ""
    Object.keys(window.gameAssets).sort((a,b)=>a<b?-1:1).forEach(gameId => {
        const gameSelectContainer = createElem("div")
        const gameCheckbox = createElem(`input#embs_${gameId}`, {type: "checkbox"})
        gameCheckbox.checked = true
        const gameButton = createElem("button.fixedColour")
        gameButton.style.setProperty("background-color", `#${window.embeddingsState.gameColours[gameId]}`, "important")
        gameButton.style.display = "flex"
        gameButton.style.alignItems = "center"
        gameButton.style.margin = "auto"
        gameButton.style.marginTop = "8px"
        const buttonLabel = createElem("span", window.embeddingsState.gameTitles[gameId])

        gameButton.addEventListener("click", e => {
            if (e.target==gameButton || e.target==buttonLabel) {
                gameCheckbox.click()
                window.populateVoicesList()
                window.computeEmbsAndDimReduction()
            }
        })
        gameButton.addEventListener("contextmenu", e => {
            if (e.target==gameButton || e.target==buttonLabel) {
                Array.from(embeddingsGamesListContainer.querySelectorAll("input")).forEach(ckbx => ckbx.checked = false)
                gameCheckbox.click()
                window.populateVoicesList()
                window.computeEmbsAndDimReduction()
            }
        })
        gameCheckbox.addEventListener("click", () => {
            window.populateVoicesList()
            window.computeEmbsAndDimReduction()
        })


        gameButton.appendChild(gameCheckbox)
        gameButton.appendChild(buttonLabel)
        gameSelectContainer.appendChild(gameButton)
        embeddingsGamesListContainer.appendChild(gameSelectContainer)
    })
}



window.populateVoicesList = () => {
    const enabledGames = Array.from(embeddingsGamesListContainer.querySelectorAll("input"))
        .map(elem => [elem.checked, elem.id.replace("embs_", "")])
        .filter(checkedId => checkedId[0])
        .map(checkedId => checkedId[1].replace("embs_", ""))

    const checkboxes = []

    embeddingsRecordsContainer.innerHTML = ""
    Object.keys(window.games).forEach(gameId => {
        if (!enabledGames.includes(gameId)) {
            return
        }

        window.games[gameId].models.forEach(model => {

            model.variants.forEach(variant => {
                const voiceRowElem = createElem("div")

                const voiceCkbx = createElem(`input#embsVoice_${variant.voiceId}`, {type: "checkbox"})
                voiceCkbx.checked = true
                voiceCkbx.addEventListener("click", () => {
                    window.computeEmbsAndDimReduction()
                })
                checkboxes.push(voiceCkbx)
                const showVoiceBtn = createElem("button.smallButton.fixedColour", "Show")
                showVoiceBtn.style.background = `#${window.embeddingsState.gameColours[model.gameId]}`
                showVoiceBtn.addEventListener("click", () => {
                    if (!voiceCkbx.checked) {
                        return window.errorModal(window.i18n.VEMB_VOICE_NOT_ENABLED)
                    }
                    const point = window.embeddingsState.sceneData.points.find(point => point.data.voiceId==variant.voiceId)
                    window.embeddingsState.sceneData.controls.target.set(point.position.x, point.position.y, point.position.z)

                    const cameraPos = window.embeddingsState.sceneData.camera.position
                    const deltaX = (point.position.x - cameraPos.x)
                    const deltaY = (point.position.y - cameraPos.y)
                    const deltaZ = (point.position.z - cameraPos.z)

                    window.embeddingsState.sceneData.camera.position.set(cameraPos.x+deltaX/2, cameraPos.y+deltaY/2, cameraPos.z+deltaZ/2)

                })

                const nameElem = createElem("div", model.voiceName+(model.variants.length>1?` (${variant.variantName})`:""))
                nameElem.title = model.voiceName
                const gameElem = createElem("div", window.embeddingsState.gameTitles[model.gameId])
                gameElem.title = window.embeddingsState.gameTitles[model.gameId]
                const voiceGender = createElem("div", variant.gender)
                voiceGender.title = variant.gender

                voiceRowElem.appendChild(createElem("div", voiceCkbx))
                voiceRowElem.appendChild(createElem("div", showVoiceBtn))
                voiceRowElem.appendChild(nameElem)
                voiceRowElem.appendChild(gameElem)
                voiceRowElem.appendChild(voiceGender)

                if (embeddingsSearchBar.value.length && !model.voiceName.toLowerCase().trim().includes(embeddingsSearchBar.value.toLowerCase().trim())) {
                    return
                }
                embeddingsRecordsContainer.appendChild(voiceRowElem)
            })
        })
    })

    window.embeddingsState.voiceCheckboxes = checkboxes

}
embeddingsSearchBar.addEventListener("keyup", () => window.populateVoicesList())



window.initDataMappings = () => {

    window.embeddingsState.voiceIdToModel = {}
    window.embeddingsState.gameShortIdToGameId = {}
    window.embeddingsState.gameColours = {}
    window.embeddingsState.gameTitles = {}


    const idToGame = {}
    Object.keys(window.gameAssets).forEach(gameId => {
        const id = window.gameAssets[gameId].gameCode
        idToGame[id] = gameId
    })

    Object.keys(window.gameAssets).forEach(gameId => {
        // Short game ID to full game ID
        const gameShortId = window.gameAssets[gameId].gameCode.toLowerCase()
        window.embeddingsState.gameShortIdToGameId[gameShortId] = gameId.toLowerCase()

        // Game title
        const title = window.gameAssets[gameId].gameName
        window.embeddingsState.gameTitles[gameId] = title

        // Game colour
        let colour = window.gameAssets[gameId].themeColourPrimary
        colour = colour.length==3 ? `${colour[0]}${colour[0]}${colour[1]}${colour[1]}${colour[2]}${colour[2]}` : colour
        window.embeddingsState.gameColours[gameId] = colour
    })

    Object.keys(window.games).forEach(gameId => {
        // Voice Id to model data
        window.games[gameId].models.forEach(model => {
            model.variants.forEach(variant => {
                window.embeddingsState.voiceIdToModel[variant.voiceId] = JSON.parse(JSON.stringify(model))
                if (model.variants.length>1) {
                    let voiceName = window.embeddingsState.voiceIdToModel[variant.voiceId].voiceName
                    voiceName = `${voiceName} (${variant.variantName})`
                    window.embeddingsState.voiceIdToModel[variant.voiceId].voiceName = voiceName
                }
            })
        })
    })
}




window.initEmbeddingsScene = () => {
    window.initDataMappings()
    window.populateGamesList()
    window.populateVoicesList()

    embeddingsSceneContainer.addEventListener("mousedown", (event) => {
        window.embeddingsState.mousePos.x = parseInt(event.layerX)
        window.embeddingsState.mousePos.y = parseInt(event.layerY)
    })
    embeddingsSceneContainer.addEventListener("mouseup", (event) => {
        const mouseX = parseInt(event.layerX)
        const mouseY = parseInt(event.layerY)

        if (event.button==0) {
            if (Math.abs(mouseX-window.embeddingsState.mousePos.x)<10 && Math.abs(mouseY-window.embeddingsState.mousePos.y)<10) {
                window.embeddingsState.mouseIsDown = true
                setTimeout(() => {window.embeddingsState.mouseIsDown = false}, 100)
            }
        } else if (event.button==2) {
            if (Math.abs(mouseX-window.embeddingsState.mousePos.x)<10 && Math.abs(mouseY-window.embeddingsState.mousePos.y)<10) {
                window.embeddingsState.rightMouseIsDown = true
                setTimeout(() => {window.embeddingsState.rightMouseIsDown = false}, 100)
            }
        }
    })
    window.embeddingsState.isReady = false

    const SPHERE_RADIUS = 3
    const SPHERE_V_COUNT = 50

    // Renderer
    window.embeddingsState.renderer = new THREE.WebGLRenderer({alpha: true, antialias: true})
    window.embeddingsState.renderer.setPixelRatio( window.devicePixelRatio )
    window.embeddingsState.renderer.setSize(embeddingsSceneContainer.offsetWidth, embeddingsSceneContainer.offsetHeight)
    embeddingsSceneContainer.appendChild(window.embeddingsState.renderer.domElement)

    // Scene and camera
    const scene = new THREE.Scene()
    window.embeddingsState.sceneData.camera = new THREE.PerspectiveCamera(60, embeddingsSceneContainer.offsetWidth/embeddingsSceneContainer.offsetHeight, 0.001, 100000)
    window.embeddingsState.sceneData.camera.position.set( -100, 0, 0 )

    // Controls
    window.embeddingsState.sceneData.controls = new THREE.OrbitControls(window.embeddingsState.sceneData.camera, window.embeddingsState.renderer.domElement)
    window.embeddingsState.sceneData.controls2 = new THREE.TrackballControls(window.embeddingsState.sceneData.camera, window.embeddingsState.renderer.domElement)
    window.embeddingsState.sceneData.controls.target.set(window.embeddingsState.sceneData.camera.position.x+0.15, window.embeddingsState.sceneData.camera.position.y, window.embeddingsState.sceneData.camera.position.z)


    window.embeddingsState.sceneData.controls.enableDamping = true
    window.embeddingsState.sceneData.controls.dampingFactor = 0.025
    window.embeddingsState.sceneData.controls.screenSpacePanning = true
    window.embeddingsState.sceneData.controls.rotateSpeed = 1/6
    window.embeddingsState.sceneData.controls.panSpeed = 1
    window.embeddingsState.sceneData.controls.minDistance = 50
    window.embeddingsState.sceneData.controls.maxDistance = 500

    window.embeddingsState.sceneData.controls2.noRotate = true
    window.embeddingsState.sceneData.controls2.noPan = true
    window.embeddingsState.sceneData.controls2.noZoom = true
    window.embeddingsState.sceneData.controls2.zoomSpeed = 1/2// 1.5
    window.embeddingsState.sceneData.controls2.dynamicDampingFactor = 0.2



    const light = new THREE.DirectionalLight( 0xffffff, 0.5 )
    light.position.set( -1, 1, 1 ).normalize()
    scene.add(light)
    scene.add(new THREE.AmbientLight( 0xffffff, 0.5 ))

    // Mouse event ray caster
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()
    window.embeddingsState.renderer.domElement.addEventListener("mousemove", event => {
        const sizeY = event.target.height
        const sizeX = event.target.width
        mouse.x = event.offsetX / sizeX * 2 - 1
        mouse.y = -event.offsetY / sizeY * 2 + 1
    }, false)


    window.embeddingsState.sceneData.sprites = []
    window.embeddingsState.sceneData.points = []

    window.refreshEmbeddingsScenePoints = () => {

        const enabledGames = Array.from(embeddingsGamesListContainer.querySelectorAll("input"))
            .map(elem => [elem.checked, elem.id.replace("embs_", "")])
            .filter(checkedId => checkedId[0])
            .map(checkedId => checkedId[1].replace("embs_", ""))


        const data = window.embeddingsState.data

        const newDataNames = Object.keys(data)
        const oldDataKept = []

        const newSprites = []
        const newPoints = []

        // Remove any existing data
        ;[window.embeddingsState.sceneData.points, window.embeddingsState.sceneData.sprites].forEach(dataList => {
            dataList.forEach(object => {
                const objectName = object.data.voiceId
                if (newDataNames.includes(objectName)) {
                    oldDataKept.push(objectName)
                    const coords = {
                        x: parseFloat(data[objectName][0]),
                        y: parseFloat(data[objectName][1])-(object.data.type=="text"?SPHERE_RADIUS*1.5:0),
                        z: parseFloat(data[objectName][2])
                    }
                    object.data.isMoving = true
                    object.data.newPos = coords
                    if (object.data.type=="text") {
                        newSprites.push(object)
                    } else {
                        newPoints.push(object)
                    }
                } else {
                    scene.remove(object)
                }
            })
        })

        // Add the new data
        window.embeddingsState.sceneData.sprites = newSprites
        window.embeddingsState.sceneData.points = newPoints
        Object.keys(data).forEach(voiceId => {

            if (oldDataKept.includes(voiceId)) {
                return
            }
            const game = Object.keys(window.embeddingsState.gameShortIdToGameId).includes(voiceId.split("_")[0]) ? window.embeddingsState.gameShortIdToGameId[voiceId.split("_")[0]] : "other"
            let gender
            if (Object.keys(window.embeddingsState.voiceIdToModel).includes(voiceId)) {
                gender = window.embeddingsState.voiceIdToModel[voiceId].gender || window.embeddingsState.voiceIdToModel[voiceId].variants[0].gender
            } else {
                gender = window.embeddingsState.allData[voiceId].voiceGender
            }
            gender = gender ? gender.toLowerCase() : "other"

            // if (!enabledGames.includes(game)) {
            //     return
            // }

            // Filter out data by gender
            // if (gender=="male" && !embeddingsMalesCkbx.checked) {
            //     return
            // }
            // if (gender=="female" && !embeddingsFemalesCkbx.checked) {
            //     return
            // }
            // if (gender=="other" && !embeddingsOtherGendersCkbx.checked) {
            //     return
            // }


            // Colour dict
            const colour = hexToRgb("#"+window.embeddingsState.gameColours[game])
            const genderColours = {
                "f": {r: 200, g: 0, b: 0},
                "m": {r: 0, g: 0, b: 200},
                "o": {r: 85, g: 85, b: 85},
            }
            const coords = {
                x: parseFloat(data[voiceId][0]),
                y: parseFloat(data[voiceId][1]),
                z: parseFloat(data[voiceId][2])
            }

            const genderColour = gender=="female" ? genderColours["f"] : (gender=="male" ? genderColours["m"] : genderColours["o"])


            const pointGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, SPHERE_V_COUNT, SPHERE_V_COUNT)
            const pointMaterial = new THREE.MeshLambertMaterial({
                color: window.embeddingsState.gendersOn ? rgbToHex(genderColour.r, genderColour.g, genderColour.b) : "#"+window.embeddingsState.gameColours[game],
                transparent: true
            })
            pointMaterial.emissive.emissiveIntensity = 1


            // Point sphere
            const point = new THREE.Mesh(pointGeometry, pointMaterial)
            point.position.x = coords.x
            point.position.y = coords.y
            point.position.z = coords.z
            point.name = `point|${voiceId}`
            point.data = {
                type: "point",
                voiceId: voiceId,
                game: game,
                gameColour: {r: colour.r, g: colour.g, b: colour.b},
                genderColour: genderColour
            }
            window.embeddingsState.sceneData.points.push(point)
            scene.add(point)

            let voiceName
            if (Object.keys(window.embeddingsState.voiceIdToModel).includes(voiceId)) {
                voiceName = window.embeddingsState.voiceIdToModel[voiceId].voiceName
            } else {
                voiceName = window.embeddingsState.allData[voiceId].voiceName
            }

            // Text sprite
            const sprite = new THREE.TextSprite({
                text: voiceName,
                fontFamily: 'Helvetica, sans-serif',
                fontSize: 2,
                strokeColor: '#ffffff',
                strokeWidth: 0,
                color: '#24ff00',
                material: {color: "white"}
            })
            sprite.position.x = coords.x
            sprite.position.y = coords.y-SPHERE_RADIUS*1.5
            sprite.position.z = coords.z
            sprite.name = `sprite|${voiceId}`
            sprite.data = {type: "text", voiceId: voiceId, game: game}
            window.embeddingsState.sceneData.sprites.push(sprite)
            scene.add(sprite)
        })
    }
    window.refreshEmbeddingsScenePoints()

    let hoveredObject = undefined
    let clickedObject = undefined

    window.embeddings_render = () => {
        if (!window.embeddingsState.isReady) {
            return
        }
        requestAnimationFrame(window.embeddings_render)

        const target = window.embeddingsState.sceneData.controls.target
        window.embeddingsState.sceneData.controls.update()
        window.embeddingsState.sceneData.controls2.target.set(target.x, target.y, target.z)
        window.embeddingsState.sceneData.controls2.update()

        window.embeddingsState.sceneData.camera.updateMatrixWorld()

        raycaster.setFromCamera( mouse, window.embeddingsState.sceneData.camera );

        // Move objects
        [window.embeddingsState.sceneData.points, window.embeddingsState.sceneData.sprites].forEach(dataList => {
            dataList.forEach(object => {
                if (object.data.isMoving) {
                    if (Math.abs(object.position.x-object.data.newPos.x)>0.005) {
                        object.position.x += (object.data.newPos.x - object.position.x) / 20
                        object.position.y += (object.data.newPos.y - object.position.y) / 20
                        object.position.z += (object.data.newPos.z - object.position.z) / 20

                    } else {
                        object.data.isMoving = false
                        object.data.newPos = undefined
                    }
                }
            })
        })


        // Handle mouse events
        let intersects = raycaster.intersectObjects(scene.children, true)
        if (intersects.length) {

            if (intersects.length>2) {
                intersects = [intersects.find(it => it.object.data.type=="point")]
            }
            if (intersects.length==0 || intersects[0]==undefined || intersects[0].object==undefined || intersects[0].object.data.type=="text") {
                window.embeddingsState.renderer.render(scene, window.embeddingsState.sceneData.camera)
                return
            }

            window.embeddingsState.renderer.domElement.style.cursor = "pointer"

            if (hoveredObject != undefined && hoveredObject.object.data.voiceId!=intersects[0].object.data.voiceId) {
                const colour = window.embeddingsState.gendersOn ? hoveredObject.object.data.genderColour : hoveredObject.object.data.gameColour
                hoveredObject.object.material.color.r = Math.min(1, colour.r/255)
                hoveredObject.object.material.color.g = Math.min(1, colour.g/255)
                hoveredObject.object.material.color.b = Math.min(1, colour.b/255)
            }
            hoveredObject = intersects[0]

            const colour = window.embeddingsState.gendersOn ? hoveredObject.object.data.genderColour : hoveredObject.object.data.gameColour
            hoveredObject.object.material.color.r = Math.min(1, colour.r/255*1.5)
            hoveredObject.object.material.color.g = Math.min(1, colour.g/255*1.5)
            hoveredObject.object.material.color.b = Math.min(1, colour.b/255*1.5)


            // Right click does voice audio preview
            if (window.embeddingsState.rightMouseIsDown) {
                window.embeddingsState.rightMouseIsDown = false
                const voiceId = hoveredObject.object.data.voiceId
                const gameId = hoveredObject.object.data.game
                const modelsPathForGame = window.userSettings[`modelspath_${gameId}`]
                const audioPreviewPath = `${modelsPathForGame}/${voiceId}.wav`

                if (fs.existsSync(audioPreviewPath)) {
                    const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                        src: audioPreviewPath
                    }))
                    audioPreview.setSinkId(window.userSettings.base_speaker)
                } else {
                    window.errorModal(window.i18n.VEMB_NO_PREVIEW)
                }
            }

            // Left click does voice click
            if (window.embeddingsState.mouseIsDown) {
                if (window.embeddingsState.clickedObject==undefined || window.embeddingsState.clickedObject.voiceId!=hoveredObject.object.data.voiceId) {
                    if (window.embeddingsState.clickedObject!=undefined) {
                        window.embeddingsState.clickedObject.object.material.emissive.setRGB(0,0,0)
                    }
                    window.embeddingsState.clickedObject = {
                        voiceId: hoveredObject.object.data.voiceId,
                        game: hoveredObject.object.data.game,
                        object: hoveredObject.object
                    }
                    hoveredObject.object.material.emissive.setRGB(0, 1, 0)

                    const voiceId = window.embeddingsState.clickedObject.object.data.voiceId

                    if (Object.keys(window.embeddingsState.voiceIdToModel).includes(voiceId)) {
                        embeddingsVoiceGameDisplay.innerHTML = window.embeddingsState.gameTitles[window.embeddingsState.clickedObject.object.data.game]
                        embeddingsVoiceNameDisplay.innerHTML = window.embeddingsState.voiceIdToModel[voiceId].voiceName
                        embeddingsVoiceGenderDisplay.innerHTML = window.embeddingsState.voiceIdToModel[voiceId].gender || window.embeddingsState.voiceIdToModel[voiceId].variants[0].gender
                    } else {
                        embeddingsVoiceGameDisplay.innerHTML = window.embeddingsState.gameTitles[window.embeddingsState.clickedObject.object.data.game]
                        embeddingsVoiceNameDisplay.innerHTML = window.embeddingsState.allData[voiceId].voiceName
                        embeddingsVoiceGenderDisplay.innerHTML = window.embeddingsState.allData[voiceId].voiceGender
                    }
                }
            }


        } else {
            window.embeddingsState.renderer.domElement.style.cursor = "default"
            if (hoveredObject != undefined) {
                const colour = window.embeddingsState.gendersOn ? hoveredObject.object.data.genderColour : hoveredObject.object.data.gameColour
                hoveredObject.object.material.color.r = Math.min(1, colour.r/255)
                hoveredObject.object.material.color.g = Math.min(1, colour.g/255)
                hoveredObject.object.material.color.b = Math.min(1, colour.b/255)
            }
            if (window.embeddingsState.mouseIsDown && window.embeddingsState.clickedObject!=undefined) {
                window.embeddingsState.clickedObject.object.material.emissive.setRGB(0,0,0)
                window.embeddingsState.clickedObject = undefined

                embeddingsVoiceGameDisplay.innerHTML = ""
                embeddingsVoiceNameDisplay.innerHTML = ""
                embeddingsVoiceGenderDisplay.innerHTML = ""
            }
        }

        window.embeddingsState.renderer.render(scene, window.embeddingsState.sceneData.camera)
    }
    window.embeddingsState.isReady = true
    window.embeddings_render()


    window.toggleSprites = () => {
        window.embeddingsState.spritesOn = !window.embeddingsState.spritesOn
        window.embeddingsState.sceneData.sprites.forEach(sprite => {
            sprite.material.visible = window.embeddingsState.spritesOn
        })
    }

    window.toggleGenders = () => {
        window.embeddingsState.gendersOn = !window.embeddingsState.gendersOn
        window.embeddingsState.sceneData.points.forEach(point => {
            const colour = window.embeddingsState.gendersOn ? point.data.genderColour : point.data.gameColour
            point.material.color.r = Math.min(1, colour.r/255)
            point.material.color.g = Math.min(1, colour.g/255)
            point.material.color.b = Math.min(1, colour.b/255)
        })
    }
    window.addEventListener("resize", () => {
        if (window.embeddingsState.isOpen) {
            window.embeddings_updateSize()
        }
    })
}
window.embeddings_updateSize = () => {
    if (window.embeddingsState.isReady) {
        window.embeddingsState.sceneData.camera.aspect = embeddingsSceneContainer.offsetWidth/embeddingsSceneContainer.offsetHeight
        window.embeddingsState.sceneData.camera.updateProjectionMatrix()
        window.embeddingsState.renderer.setSize(embeddingsSceneContainer.offsetWidth, embeddingsSceneContainer.offsetHeight)
    }
}


embeddingsPreviewButton.addEventListener("click", () => {
    if (window.embeddingsState.clickedObject) {

        const voiceId = window.embeddingsState.clickedObject.object.data.voiceId
        const gameId = window.embeddingsState.clickedObject.object.data.game
        const modelsPathForGame = window.userSettings[`modelspath_${gameId}`]
        const audioPreviewPath = `${modelsPathForGame}/${voiceId}.wav`

        if (fs.existsSync(audioPreviewPath)) {
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: audioPreviewPath
            }))
            audioPreview.setSinkId(window.userSettings.base_speaker)
        } else {
            window.errorModal(window.i18n.VEMB_NO_PREVIEW)
        }
    } else {
        window.errorModal(window.i18n.VEMB_SELECT_VOICE_FIRST)
    }
})
embeddingsLoadButton.addEventListener("click", () => {
    if (window.embeddingsState.clickedObject) {

        const voiceId = window.embeddingsState.clickedObject.object.data.voiceId
        const gameId = window.embeddingsState.clickedObject.object.data.game
        const modelsPathForGame = window.userSettings[`modelspath_${gameId}`]
        const modelPath = `${modelsPathForGame}/${voiceId}.pt`

        if (fs.existsSync(modelPath)) {

            window.changeGame(window.gameAssets[gameId])

            // Simulate voice loading through the UI
            let voiceName
            window.games[gameId].models.forEach(model => {
                model.variants.forEach(variant => {
                    if (variant.voiceId==voiceId) {
                        voiceName = model.voiceName
                    }
                })
            })
            const voiceButton = Array.from(voiceTypeContainer.children).find(button => button.innerHTML==voiceName)
            voiceButton.click()
            closeModal().then(() => {
                generateVoiceButton.click()
            })

        } else {
            window.errorModal(window.i18n.VEMB_NO_MODEL)
        }
    } else {
        window.errorModal(window.i18n.VEMB_SELECT_VOICE_FIRST)
    }
})

embeddingsKey.addEventListener("change", () => {
    if (embeddingsKey.value=="embsKey_game" && window.embeddingsState.gendersOn) {
        window.toggleGenders()
    } else if (embeddingsKey.value=="embsKey_gender" && !window.embeddingsState.gendersOn) {
        window.toggleGenders()
    }
})
embeddingsMalesCkbx.addEventListener("change", () => window.computeEmbsAndDimReduction())
embeddingsFemalesCkbx.addEventListener("change", () => window.computeEmbsAndDimReduction())
embeddingsOtherGendersCkbx.addEventListener("change", () => window.computeEmbsAndDimReduction())
embeddingsOnlyInstalledCkbx.addEventListener("change", () => window.computeEmbsAndDimReduction())
embeddingsAlgorithm.addEventListener("change", () => window.computeEmbsAndDimReduction())


window.computeEmbsAndDimReduction = (includeAllVoices=false) => {

    const enabledGames = Array.from(embeddingsGamesListContainer.querySelectorAll("input"))
        .map(elem => [elem.checked, elem.id.replace("embs_", "")])
        .filter(checkedId => checkedId[0])
        .map(checkedId => checkedId[1].replace("embs_", ""))

    const enabledVoices = window.embeddingsState.voiceCheckboxes
        .map(elem => [elem.checked, elem.id.replace("embsVoice_", "")])
        .filter(checkedId => checkedId[0])
        .map(checkedId => checkedId[1].replace("embsVoice_", ""))

    if (enabledVoices.length<=2) {
        return window.errorModal(window.i18n.EMBEDDINGS_NEED_AT_LEAST_3)
    }


    // Get together a list of voiceId->.wav path mappings
    const mappings = []

    if (includeAllVoices) {

        Object.keys(window.games).forEach(gameId => {
            const modelsPathForGame = window.userSettings[`modelspath_${gameId}`]

            window.games[gameId].models.forEach(model => {
                model.variants.forEach(variant => {
                    const audioPreviewPath = `${modelsPathForGame}/${variant.voiceId}.wav`
                    if (fs.existsSync(audioPreviewPath)) {
                        mappings.push(`${variant.voiceId}=${audioPreviewPath}=${model.voiceName}=${variant.gender}=${gameId}`)
                    }
                })
            })
        })

    } else {
        Object.keys(window.embeddingsState.allData).forEach(voiceId => {
            try {
                const voiceMeta = window.embeddingsState.allData[voiceId]
                const gender = voiceMeta.voiceGender.toLowerCase()

                // Filter game-level voices
                if (!enabledGames.includes(voiceMeta.gameId)) {
                    return
                }
                // Filter out by voice
                if (!enabledVoices.includes(voiceId)) {
                    return
                }
                // Filter out data by gender
                if (gender=="male" && !embeddingsMalesCkbx.checked) {
                    return
                }
                if (gender=="female" && !embeddingsFemalesCkbx.checked) {
                    return
                }
                if (gender=="other" && !embeddingsOtherGendersCkbx.checked) {
                    return
                }

                const modelsPathForGame = window.userSettings[`modelspath_${voiceMeta.gameId}`]
                let audioPreviewPath = `${modelsPathForGame}/${voiceId}.wav`

                if (!fs.existsSync(audioPreviewPath)) {
                    audioPreviewPath = ""
                }

                mappings.push(`${voiceId}=${audioPreviewPath}=${voiceMeta.voiceName}=${voiceMeta.voiceGender.toLowerCase()}=${voiceMeta.gameId}`)

            } catch (e) {console.log(e)}
        })
    }

    if (mappings.length<=2) {
        return window.errorModal(window.i18n.EMBEDDINGS_NEED_AT_LEAST_3)
    }


    window.spinnerModal(window.i18n.VEMB_RECOMPUTING)

    doFetch(`http://localhost:8008/computeEmbsAndDimReduction`, {
        method: "Post",
        body: JSON.stringify({
            mappings: mappings.join("\n"),
            onlyInstalled: embeddingsOnlyInstalledCkbx.checked,
            algorithm: embeddingsAlgorithm.value.split("_")[1],
            includeAllVoices
        })
    }).then(r=>r.text()).then(res => {
        window.embeddingsState.data = {}
        res.split("\n").forEach(voiceMetaAndCoords => {
            const voiceId = voiceMetaAndCoords.split("=")[0]
            const voiceName = voiceMetaAndCoords.split("=")[1]
            const voiceGender = voiceMetaAndCoords.split("=")[2]
            const gameId = voiceMetaAndCoords.split("=")[3]
            const coords = voiceMetaAndCoords.split("=")[4].split(",").map(v => parseFloat(v))
            window.embeddingsState.data[voiceId] = coords
            if (includeAllVoices) {
                window.embeddingsState.allData[voiceId] = {
                    voiceName,
                    voiceGender,
                    coords,
                    gameId
                }
            }
        })
        window.refreshEmbeddingsScenePoints()
        closeModal(undefined, embeddingsContainer)

    }).catch(e => {
        console.log(e)
        if (e.code =="ENOENT") {
            closeModal(null, modalContainer).then(() => {
                window.errorModal(window.i18n.ERR_SERVER)
            })
        }
    })
}

embeddingsCloseHelpUI.addEventListener("click", () => {
    embeddingsHelpUI.style.display = "none"
})
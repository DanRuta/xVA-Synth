"use strict"

window.dragDropModelInstallation = (eType, event) => {
    if (["dragenter", "dragover"].includes(eType)) {
        left.style.background = "#5b5b5b"
        left.style.color = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        left.style.background = "rgba(0,0,0,0)"
        left.style.color = "white"
    }

    event.preventDefault()
    event.stopPropagation()

    const dataLines = []

    if (eType=="drop") {
        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)

        const modelGroups = {} // Group up all files by their base name (for if loose files are given)
        files.forEach(file => {
            const baseName = file.name.split(".")[0]
            if (!modelGroups[baseName]) {
                modelGroups[baseName] = []
            }
            modelGroups[baseName].push(file.path)
        })

        const modelGroupsComplete = []
        const modelGroupsNotComplete = []
        Object.keys(modelGroups).forEach(key => {
            if (modelGroups[key][0].split(".").at(-1)!="zip") {
                const fileExtensions = modelGroups[key].map(filePath => filePath.split(".").at(-1))
                if (fileExtensions.includes("json") && fileExtensions.includes("pt")) {
                    modelGroupsComplete.push(key)
                } else {
                    modelGroupsNotComplete.push(key)
                }
            } else {
                modelGroupsComplete.push(key)
            }
        })

        if (modelGroupsNotComplete.length) {
            return window.errorModal(window.i18n.MODEL_INSTALL_DRAGDROP_INCOMPLETE.replace("_1", modelGroupsNotComplete.join(", ")))
        }

        const modelsInstalledSuccessfully = []
        const modelsFailedInstallation = []
        let lastGameInstalledOkFor = undefined


        const handleZip = (files, key) => {
            return new Promise(resolve => {
                let installedOk = false
                let game = undefined

                try {
                    if (fs.existsSync(`${window.path}/downloads`)) {
                        fs.readdirSync(`${window.path}/downloads`).forEach(fileName => {
                            fs.unlinkSync(`${window.path}/downloads/${fileName}`)
                        })
                    } else {
                        fs.mkdirSync(`${window.path}/downloads`)
                    }

                    window.unzipFileTo(files[0], `${window.path}/downloads`).then(() => {
                        const allFiles = fs.readdirSync(`${window.path}/downloads`)
                        const jsonFiles = allFiles.filter(fname => fname.endsWith(".json"))

                        jsonFiles.forEach(jsonFile => {
                            const jsonData = JSON.parse(fs.readFileSync(`${window.path}/downloads/${jsonFile}`))

                            game = jsonData.games[0].gameId
                            const voiceId = jsonData.games[0].voiceId
                            const modelsFolder = window.userSettings[`modelspath_${game}`]

                            if (!fs.existsSync(modelsFolder)) {
                                fs.mkdirSync(modelsFolder)
                            }

                            const allFilesForThisModel = allFiles.filter(fname => fname.includes(voiceId))
                            allFilesForThisModel.forEach(fname => {
                                fs.renameSync(`${window.path}/downloads/${fname}`, `${modelsFolder}/${fname}`)
                            })

                            installedOk = true
                        })

                        resolve([game, key, installedOk])
                    })

                } catch (e) {
                    resolve([game, key, false])
                }
            })
        }

        const handleLoose = (files, key) => {
            let game = undefined
            return new Promise(resolve => {
                try {
                    const jsonFile = files.find(fname => fname.endsWith(".json"))
                    const parentFolder = jsonFile.replaceAll("\\", "/").split("/").reverse().slice(1, 100000).reverse().join("/")
                    const jsonData = JSON.parse(fs.readFileSync(`${jsonFile}`))

                    const game = jsonData.games[0].gameId
                    const voiceId = jsonData.games[0].voiceId
                    const modelsFolder = window.userSettings[`modelspath_${game}`]

                    if (!fs.existsSync(modelsFolder)) {
                        fs.mkdirSync(modelsFolder)
                    }

                    const allFilesForThisModel = fs.readdirSync(parentFolder).filter(fname => fname.includes(voiceId))
                    allFilesForThisModel.forEach(fname => {
                        fs.renameSync(`${parentFolder}/${fname}`, `${modelsFolder}/${fname}`)
                    })

                    resolve([game, key, true])
                } catch (e) {
                    resolve([game, key, false])
                }
            })
        }

        const installPromises = []

        modelGroupsComplete.forEach(key => {
            try {
                const files = modelGroups[key]
                if (files[0].endsWith(".zip")) {
                    installPromises.push(handleZip(files, key))
                } else {
                    installPromises.push(handleLoose(files, key))
                }

            } catch (e) {
                console.log(e)
                window.appLogger.log(e)
                modelsFailedInstallation.push(key)
            }
        })


        Promise.all(installPromises).then(responses => {

            responses.forEach(([game, key, installedOk]) => {
                if (installedOk) {
                    lastGameInstalledOkFor = game
                    modelsInstalledSuccessfully.push(key)
                } else {
                    modelsFailedInstallation.push(key)
                }
            })

            let outputMessage = ""

            if (modelsInstalledSuccessfully.length) {
                outputMessage += window.i18n.MODEL_INSTALL_DRAGDROP_SUCCESS.replace("_1", modelsInstalledSuccessfully.length)
            }
            if (modelsFailedInstallation.length) {
                outputMessage += window.i18n.MODEL_INSTALL_DRAGDROP_FAILED.replace("_1", modelsFailedInstallation.length).replace("_2", modelsFailedInstallation.join(", "))
            }
            window.infoModal(outputMessage)

            if (lastGameInstalledOkFor) {
                window.changeGame(window.gameAssets[lastGameInstalledOkFor])
            }
            window.displayAllModels(true)
            window.loadAllModels(true).then(() => {
                changeGame(window.currentGame)
            })
        })


    }
}
left.addEventListener("dragenter", event => window.dragDropModelInstallation("dragenter", event), false)
left.addEventListener("dragleave", event => window.dragDropModelInstallation("dragleave", event), false)
left.addEventListener("dragover", event => window.dragDropModelInstallation("dragover", event), false)
left.addEventListener("drop", event => window.dragDropModelInstallation("drop", event), false)

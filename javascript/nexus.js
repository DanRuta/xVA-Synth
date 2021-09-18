"use strict"

window.nexusModelsList = []
window.endorsedRepos = new Set()
window.nexusState = {
    key: null,
    applicationSlug: "xvasynth",
    socket: null,
    uuid: null,
    token: null,
    downloadQueue: [],
    installQueue: [],
    finished: 0,
    repoLinks: []
}

// TEMP, maybe move to utils
// ==========================
const http = require("http")
const https = require("https")

// Utility for printing out in the dev console all the numerical game IDs on the Nexus
window.nexusGameIdToGameName = {}
window.getAllNexusGameIDs = (gameName) => {
    return new Promise((resolve) => {
        getData("", undefined, "GET").then(results => {
            results = gameName ? results.filter(x=>x.name.toLowerCase().includes(gameName)) : results
            resolve(results)
        })
    })
}

window.mod_search_nexus = (game_id, query) => {
    return new Promise(resolve => {
        doFetch(`https://search.nexusmods.com/mods/?game_id=${game_id}&terms=${encodeURI(query.split(' ').toString())}&include_adult=true`)
        .then(r => r.json())
        .then(r => {

            const data = r.results.map(res => {
                return {
                    downloads: res.downloads,
                    endorsements: res.endorsements,
                    game_id: res.game_id,
                    name: res.name,
                    author: res.username,
                    url: `https://www.nexusmods.com/${res.game_name}/mods/${res.mod_id}`
                }
            })

            resolve([r.total, data])
        })
    })
}

window.nexusDownload = (url, dest) => {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest)

        const request = https.get(url.replace("http:", "https:"), (response) => {
            // check if response is success
            if (response.statusCode !== 200) {
                console.log("url", url)
                console.log("Response status was " + response.statusCode, response)
                resolve()
                return
            }

            response.pipe(file)
        })

        file.on("finish", () => {
            file.close()
            resolve()
        })

        // check for request error too
        request.on("error", (err) => {
            fs.unlink(dest)
            return reject(err.message)
        })

        file.on("error", (err) => { // Handle errors
            fs.unlink(dest) // Delete the file async. (But we don't check the result)
            return reject(err.message)
        })
    })
}

window.initNexus = () => {
    return new Promise((resolve) => {

        window.nexusState.key = localStorage.getItem("nexus_API_key")

        if (window.nexusState.key) {
            window.showUserName()
            nexusLogInButton.innerHTML = window.i18n.LOG_OUT
            resolve()
        } else {
            try {
                window.nexusState.socket = new WebSocket("wss://sso.nexusmods.com")
            } catch (e) {
                console.log(e)
            }

            window.nexusState.socket.onclose = event => {
                console.log("socket closed")
                if (!window.nexusState.key) {
                    setTimeout(() => {window.initNexus()}, 5000)
                }
            }

            window.nexusState.socket.onmessage = event => {
                const response = JSON.parse(event.data)

                if (response && response.success) {
                    if (response.data.hasOwnProperty('connection_token')) {
                        localStorage.setItem('connection_token', response.data.connection_token)
                    } else if (response.data.hasOwnProperty('api_key')) {
                        console.log("API Key Received: " + response.data.api_key)
                        window.nexusState.key = response.data.api_key
                        localStorage.setItem('uuid', window.nexusState.uuid)
                        localStorage.setItem('nexus_API_key', window.nexusState.key)
                        window.showUserName()
                        window.pluginsManager.updateUI()
                        closeModal(undefined, nexusContainer)
                        nexusLogInButton.innerHTML = window.i18n.LOG_OUT
                        resolve()
                    }
                } else {
                    window.errorModal(`${window.i18n.ERR_LOGGING_INTO_NEXUS}: ${response.error}`)
                    reject()
                }
            }
        }
    })
}
nexusLogInButton.addEventListener("click", () => {

    if (nexusLogInButton.innerHTML==window.i18n.LOG_IN) {
        window.spinnerModal(window.i18n.LOGGING_INTO_NEXUS)

        window.nexusState.uuid = localStorage.getItem("uuid")
        window.nexusState.token = localStorage.getItem("connection_token")

        if (window.nexusState.uuid==null) {
            window.nexusState.uuid = uuidv4()
        }

        const data = {
            id: window.nexusState.uuid,
            token: window.nexusState.token,
            protocol: 2
        }
        window.nexusState.socket.send(JSON.stringify(data))

        shell.openExternal(`https://www.nexusmods.com/sso?id=${window.nexusState.uuid}&application=${window.nexusState.applicationSlug}`)

    } else {
        nexusNameDisplayContainer.style.opacity = 0
        localStorage.removeItem("nexus_API_key")
        localStorage.removeItem("uuid")
        localStorage.removeItem("connection_token")
        nexusAvatar.innerHTML = ""
        nexusUserName.innerHTML = ""
        nexusLogInButton.innerHTML = window.i18n.LOG_IN
        window.nexusState.uuid = null
        window.nexusState.key = null
        window.pluginsManager.updateUI()
    }
})



window.downloadFile = ([nexusGameId, nexusRepoId, outputFileName, fileId]) => {
    nexusDownloadLog.appendChild(createElem("div", `Downloading: ${outputFileName}`))
    return new Promise(async (resolve, reject) => {
        if (!fs.existsSync(`${window.path}/downloads`)) {
            fs.mkdirSync(`${window.path}/downloads`)
        }

        const downloadLink = await getData(`${nexusGameId}/mods/${nexusRepoId}/files/${fileId}/download_link.json`)
        if (!downloadLink.length && downloadLink.code==403) {

            window.errorModal(`${window.i18n.NEXUS_PREMIUM}<br><br>${window.i18n.NEXUS_ORIG_ERR}:<br>${downloadLink.message}`).then(() => {
                const queueIndex = window.nexusState.downloadQueue.findIndex(it => it[1]==fileId)
                window.nexusState.downloadQueue.splice(queueIndex, 1)
                nexusDownloadingCount.innerHTML = window.nexusState.downloadQueue.length

                nexusDownloadLog.appendChild(createElem("div", `${window.i18n.FAILED_DOWNLOAD}: ${outputFileName}`))

                reject()
            })

        } else {
            await window.nexusDownload(downloadLink[0].URI.replace("https", "http"), `${window.path}/downloads/${outputFileName}.zip`)

            const queueIndex = window.nexusState.downloadQueue.findIndex(it => it[1]==fileId)
            window.nexusState.downloadQueue.splice(queueIndex, 1)
            nexusDownloadingCount.innerHTML = window.nexusState.downloadQueue.length

            resolve()
        }
    })

}
window.installDownloadedModel = ([game, zipName]) => {
    nexusDownloadLog.appendChild(createElem("div", `${window.i18n.INSTALLING} ${zipName}`))
    return new Promise(resolve => {
        try {
            const modelsFolder = window.userSettings[`modelspath_${game}`]

            const unzipper = require('unzipper')
            const zipPath = `${window.path}/downloads/${zipName}.zip`

            if (!fs.existsSync(`${window.path}/downloads`)) {
                fs.mkdirSync(`${window.path}/downloads`)
            }


            fs.createReadStream(zipPath).pipe(unzipper.Parse()).on("entry", entry => {
                const fileName = entry.path
                const dirOrFile = entry.type

                if (/\/$/.test(fileName)) { // It's a directory
                    return
                }

                let fileContainerFolderPath = fileName.split("/").reverse()
                const justFileName = fileContainerFolderPath[0]

                entry.pipe(fs.createWriteStream(`${modelsFolder}/${justFileName}`))
            })
            .promise()
            .then(() => {
                window.appLogger.log(`${window.i18n.DONE_INSTALLING} ${zipName}`)

                const queueIndex = window.nexusState.installQueue.findIndex(it => it[1]==zipName)
                window.nexusState.installQueue.splice(queueIndex, 1)
                nexusInstallingCount.innerHTML = window.nexusState.installQueue.length

                nexusDownloadLog.appendChild(createElem("div", `${window.i18n.FINISHED} ${zipName}`))
                resolve()
            }, e => {
                console.log(e)
                window.appLogger.log(e)
                window.errorModal(e.message)
            })
        } catch (e) {
            console.log(e)
            window.appLogger.log(e)
            window.errorModal(e.message)
            resolve()
        }
    })
}

nexusDownloadAllBtn.addEventListener("click", async () => {

    for (let mi=0; mi<window.nexusState.filteredDownloadableModels.length; mi++) {
        const modelMeta = window.nexusState.filteredDownloadableModels[mi]
        window.nexusState.downloadQueue.push([modelMeta.voiceId, modelMeta.nexus_file_id])
        nexusDownloadingCount.innerHTML = window.nexusState.downloadQueue.length
    }

    for (let mi=0; mi<window.nexusState.filteredDownloadableModels.length; mi++) {

        const modelMeta = window.nexusState.filteredDownloadableModels[mi]
        await window.downloadFile([modelMeta.nexusGameId, modelMeta.nexusRepoId, modelMeta.voiceId, modelMeta.nexus_file_id])

        // Install the downloaded voice
        window.nexusState.installQueue.push([modelMeta.game, modelMeta.voiceId])
        nexusInstallingCount.innerHTML = window.nexusState.installQueue.length
        await window.installDownloadedModel([modelMeta.game, modelMeta.voiceId])

        fs.unlinkSync(`${window.path}/downloads/${modelMeta.voiceId}.zip`)

        window.nexusState.finished += 1
        nexusFinishedCount.innerHTML = window.nexusState.finished
        window.displayAllModels(true)
        window.loadAllModels(true)
    }
})


const getJSONData = (url) => {
    return new Promise(resolve => {
        doFetch(url).then(r=>r.json())
    })
}

window.showUserName = async () => {
    const data = await getuserData("validate.json")
    const img = createElem("img")
    img.src = data.profile_url
    img.style.height = "40px"
    nexusAvatar.innerHTML = ""
    img.addEventListener("load", () => {
        nexusAvatar.appendChild(img)
        nexusUserName.innerHTML = data.name
        nexusNameDisplayContainer.style.opacity = 1
    })
}

const getuserData = (url, data) => {
    return new Promise(resolve => {

        doFetch(`https://api.nexusmods.com/v1/users/${url}`, {
            method: "GET",
            headers: {
                apikey: window.nexusState.key
            }
        })
        .then(r=>r.json())
        .then(data => {
            resolve(data)
        })
        .catch(err => {
            console.log("err", err)
            resolve()
        })
    })
}
const getData = (url, data, type="GET") => {
    return new Promise(resolve => {
        const payload = {
            method: type,
            headers: {
                apikey: window.nexusState.key
            }
        }
        if (type=="POST") {
            const params = new URLSearchParams()
            Object.keys(data).forEach(key => {
                params.append(key, data[key])
            })
            payload.body = params
        }

        doFetch(`https://api.nexusmods.com/v1/games/${url}`, payload)
        .then(r=>r.json())
        .then(data => {
            resolve(data)
        })
        .catch(err => {
            console.log("err", err)
            resolve()
        })
    })
}
window.nexus_getData = getData
// ==========================




let hasPopulatedNexusGameListDropdown = false
window.openNexusWindow = () => {
    closeModal(undefined, nexusContainer).then(() => {
        nexusContainer.style.opacity = 0
        nexusContainer.style.display = "flex"
        requestAnimationFrame(() => requestAnimationFrame(() => nexusContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
    })

    const gameColours = {}
    Object.keys(window.gameAssets).forEach(gameId => {
        const colour = window.gameAssets[gameId].themeColourPrimary
        gameColours[gameId] = colour
    })

    nexusGamesList.innerHTML = ""
    Object.keys(window.gameAssets).sort((a,b)=>a<b?-1:1).forEach(gameId => {
        const gameSelectContainer = createElem("div")
        const gameCheckbox = createElem(`input#ngl_${gameId}`, {type: "checkbox"})
        gameCheckbox.checked = true
        const gameButton = createElem("button.fixedColour")
        gameButton.style.setProperty("background-color", `#${gameColours[gameId]}`, "important")
        gameButton.style.display = "flex"
        gameButton.style.alignItems = "center"
        gameButton.style.margin = "auto"
        gameButton.style.marginTop = "8px"
        const buttonLabel = createElem("span", window.gameAssets[gameId].gameName)

        gameButton.addEventListener("contextmenu", e => {
            if (e.target==gameButton || e.target==buttonLabel) {
                Array.from(nexusGamesList.querySelectorAll("input")).forEach(ckbx => ckbx.checked = false)
                gameCheckbox.click()
                window.displayAllModels()
            }
        })

        gameButton.addEventListener("click", e => {
            if (e.target==gameButton || e.target==buttonLabel) {
                gameCheckbox.click()
                window.displayAllModels()
            }
        })
        gameCheckbox.addEventListener("change", () => {
            window.displayAllModels()
        })

        gameButton.appendChild(gameCheckbox)
        gameButton.appendChild(buttonLabel)
        gameSelectContainer.appendChild(gameButton)
        nexusGamesList.appendChild(gameSelectContainer)
    })

    // Populate the game IDs for the Nexus repo searching
    if (!hasPopulatedNexusGameListDropdown) {
        window.getAllNexusGameIDs().then(results => {
            results = results.sort((a,b)=>a.name.toLowerCase()<b.name.toLowerCase()?-1:1)
            results.forEach(res => {

                window.nexusGameIdToGameName[res.id] = res.name

                const opt = createElem("option", {value: res.id})
                opt.innerHTML = res.name
                nexusAllGamesSelect.appendChild(opt)
            })
            if (fs.existsSync(`${window.path}/repositories.json`)) {
                const data = fs.readFileSync(`${window.path}/repositories.json`, "utf8")
                window.nexusReposList = JSON.parse(data)
                window.nexusUpdateModsUsedPanel()
            }
            hasPopulatedNexusGameListDropdown = true
        })
    }

}
window.setupModal(nexusMenuButton, nexusContainer, window.openNexusWindow)




nexusSearchBar.addEventListener("keyup", () => {
    window.displayAllModels()
})


window.getLatestModelsList = async () => {
    try {
        window.spinnerModal(window.i18n.CHECKING_NEXUS)
        window.nexusModelsList = []

        const idToGame = {}
        Object.keys(window.gameAssets).forEach(gameId => {
            const id = window.gameAssets[gameId].gameCode.toLowerCase()
            idToGame[id] = gameId
        })

        const repoLinks = window.nexusReposList.repos.filter(r=>r.enabled).map(r=>r.url)

        for (let li=0; li<repoLinks.length; li++) {

            const link = repoLinks[li].replace("\r","")
            const repoInfo = await getData(`${link.split(".com/")[1]}.json`)
            const author = repoInfo.author
            const nexusRepoId = repoInfo.mod_id
            const nexusRepoVersion = repoInfo.version
            const nexusGameId = repoInfo.domain_name

            const files = await getData(`${link.split(".com/")[1]}/files.json`)
            files["files"].forEach(file => {

                if (file.category_name=="OPTIONAL" || file.category_name=="OPTIONAL") {

                    if (!file.description.includes("Voice model")) {
                        return
                    }

                    const description = file.description
                    const parts = description.split("<br />")
                    let voiceId = parts.filter(line => line.startsWith("Voice ID:") || line.startsWith("VoiceID:"))[0]
                    voiceId = voiceId.includes("Voice ID: ") ? voiceId.split("Voice ID: ")[1].split(" ")[0] : voiceId.split("VoiceID: ")[1].split(" ")[0]
                    const game = idToGame[voiceId.split("_")[0]]
                    const name = parts.filter(line => line.startsWith("Voice model"))[0].split(" - ")[1]
                    const date = file.uploaded_time
                    const nexus_file_id = file.file_id

                    if (repoInfo.endorsement.endorse_status=="Endorsed") {
                        window.endorsedRepos.add(game)
                    }

                    const hasT2 = description.includes("Tacotron2")
                    const hasHiFi = description.includes("HiFi-GAN")
                    const version = file.version

                    let type
                    if (description.includes("Model:")) {
                        type = parts.filter(line => line.startsWith("Model: "))[0].split("Model: ")[1]
                    } else {
                        type = "FastPitch"
                        if (type=="FastPitch") {
                            if (hasT2) {
                                type = "T2+"+type
                            }
                        }
                        if (hasHiFi) {
                            type += "+HiFi"
                        }
                    }

                    const notes = description.includes("Notes:") ? parts.filter(line => line.startsWith("Notes: "))[0].split("Notes: ")[1] : ""
                    const meta = {author, description, version, voiceId, game, name, type, notes, date, nexusRepoId, nexusRepoVersion, nexusGameId, nexus_file_id, repoLink: link}
                    window.nexusModelsList.push(meta)
                }
            })
        }

        window.closeModal(undefined, nexusContainer)
        window.displayAllModels()

    } catch (e) {
        console.log(e)
        window.appLogger.log(e)
        window.errorModal(e.message)
    }
}

window.displayAllModels = (forceUpdate=false) => {

    if (!forceUpdate && window.nexusState.installQueue.length) {
        return
    }

    const enabledGames = Array.from(nexusGamesList.querySelectorAll("input"))
        .map(elem => [elem.checked, elem.id.replace("ngl_", "")])
        .filter(checkedId => checkedId[0])
        .map(checkedId => checkedId[1])

    const gameColours = {}
    Object.keys(window.gameAssets).forEach(gameId => {
        const colour = window.gameAssets[gameId].themeColourPrimary
        gameColours[gameId] = colour
    })
    const gameTitles = {}
    Object.keys(window.gameAssets).forEach(gameId => {
        const title = window.gameAssets[gameId].gameName
        gameTitles[gameId] = title
    })

    nexusRecordsContainer.innerHTML = ""

    window.nexusState.filteredDownloadableModels = []

    window.nexusModelsList.forEach(modelMeta => {
        if (!enabledGames.includes(modelMeta.game)) {
            return
        }
        if (nexusSearchBar.value.toLowerCase().trim().length && !modelMeta.name.toLowerCase().includes(nexusSearchBar.value.toLowerCase().trim())) {
            return
        }
        const existingModel = Object.keys(window.games).includes(modelMeta.game) ? window.games[modelMeta.game].models.find(model => model.voiceId==modelMeta.voiceId) : undefined
        if (existingModel && nexusOnlyNewUpdatedCkbx.checked && (window.checkVersionRequirements(modelMeta.version, String(existingModel.modelVersion)) || (modelMeta.version.replace(".0","")==String(existingModel.modelVersion))) ){
            return
        }



        const recordRow = createElem("div")
        const actionsElem = createElem("div")

        // Open link to the repo in the browser
        const openButton = createElem("button.smallButton.fixedColour", window.i18n.OPEN)
        openButton.style.setProperty("background-color", `#${gameColours[modelMeta.game]}`, "important")
        openButton.addEventListener("click", () => {
            shell.openExternal(modelMeta.repoLink)
        })
        actionsElem.appendChild(openButton)



        // Download
        const downloadButton = createElem("button.smallButton.fixedColour", window.i18n.DOWNLOAD)
        downloadButton.style.setProperty("background-color", `#${gameColours[modelMeta.game]}`, "important")
        downloadButton.addEventListener("click", async () => {
            // Download the voice
            window.nexusState.downloadQueue.push([modelMeta.voiceId, modelMeta.nexus_file_id])
            nexusDownloadingCount.innerHTML = window.nexusState.downloadQueue.length
            try {
                await window.downloadFile([modelMeta.nexusGameId, modelMeta.nexusRepoId, modelMeta.voiceId, modelMeta.nexus_file_id])

                // Install the downloaded voice
                window.nexusState.installQueue.push([modelMeta.game, modelMeta.voiceId])
                nexusInstallingCount.innerHTML = window.nexusState.installQueue.length
                await window.installDownloadedModel([modelMeta.game, modelMeta.voiceId])

                window.nexusState.finished += 1
                nexusFinishedCount.innerHTML = window.nexusState.finished
                window.displayAllModels()

            } catch (e) {}

        })
        if (existingModel && (modelMeta.version.replace(".0","")==String(existingModel.modelVersion) || window.checkVersionRequirements(modelMeta.version, String(existingModel.modelVersion)) ) ) {
        } else {
            window.nexusState.filteredDownloadableModels.push(modelMeta)
            actionsElem.appendChild(downloadButton)
        }


        // Endorse
        const endorsed = window.endorsedRepos.has(modelMeta.game)
        const endorseButton = createElem("button.smallButton.fixedColour", endorsed?"Unendorse":"Endorse")
        if (endorsed) {
            endorseButton.style.background = "none"
            endorseButton.style.border = `2px solid #${gameColours[modelMeta.game]}`
        } else {
            endorseButton.style.setProperty("background-color", `#${gameColours[modelMeta.game]}`, "important")
        }
        endorseButton.addEventListener("click", async () => {
            let response
            if (endorsed) {
                response = await getData(`${modelMeta.nexusGameId}/mods/${modelMeta.nexusRepoId}/abstain.json`, {
                    game_domain_name: modelMeta.nexusGameId,
                    id: modelMeta.nexusRepoId,
                    version: modelMeta.nexusRepoVersion
                }, "POST")
            } else {
                response = await getData(`${modelMeta.nexusGameId}/mods/${modelMeta.nexusRepoId}/endorse.json`, {
                    game_domain_name: modelMeta.nexusGameId,
                    id: modelMeta.nexusRepoId,
                    version: modelMeta.nexusRepoVersion
                }, "POST")
            }
            if (response && response.message && response.status=="Error") {
                if (response.message=="NOT_DOWNLOADED_MOD") {
                    response.message = window.i18n.NEXUS_NOT_DOWNLOADED_MOD
                } else if (response.message=="TOO_SOON_AFTER_DOWNLOAD") {
                    response.message = window.i18n.NEXUS_TOO_SOON_AFTER_DOWNLOAD
                } else if (response.message=="IS_OWN_MOD") {
                    response.message = window.i18n.NEXUS_IS_OWN_MOD
                }

                window.errorModal(response.message)
            } else {
                if (endorsed) {
                    window.endorsedRepos.delete(modelMeta.game)
                } else {
                    window.endorsedRepos.add(modelMeta.game)
                }
                window.displayAllModels()
            }
        })
        actionsElem.appendChild(endorseButton)



        const gameElem = createElem("div", gameTitles[modelMeta.game])
        gameElem.title = gameTitles[modelMeta.game]
        const nameElem = createElem("div", modelMeta.name)
        nameElem.title = modelMeta.name
        const authorElem = createElem("div", modelMeta.author)
        authorElem.title = modelMeta.author
        let versionElemText
        if (existingModel) {
            const yoursVersion = String(existingModel.modelVersion).includes(".") ? existingModel.modelVersion : existingModel.modelVersion+".0"
            versionElemText = `${modelMeta.version} (${window.i18n.YOURS}: ${yoursVersion})`
        } else {
            versionElemText = modelMeta.version
        }
        const versionElem = createElem("div", versionElemText)
        versionElem.title = versionElemText

        const date = new Date(modelMeta.date)
        const dateString = `${date.getDate()}/${date.getMonth()+1}/${date.getYear()+1900}`
        const dateElem = createElem("div", dateString)
        dateElem.title = dateString

        const typeElem = createElem("div", modelMeta.type)
        typeElem.title = modelMeta.type
        const notesElem = createElem("div", modelMeta.notes)
        notesElem.title = modelMeta.notes


        recordRow.appendChild(actionsElem)
        recordRow.appendChild(gameElem)
        recordRow.appendChild(nameElem)
        recordRow.appendChild(authorElem)
        recordRow.appendChild(versionElem)
        recordRow.appendChild(dateElem)
        recordRow.appendChild(typeElem)
        recordRow.appendChild(notesElem)

        nexusRecordsContainer.appendChild(recordRow)
    })
}


nexusCheckNow.addEventListener("click", () => window.getLatestModelsList())

window.setupModal(nexusManageReposButton, nexusReposContainer)

window.nexusUpdateModsUsedPanel = () => {
    nexusReposUsedContainer.innerHTML = ""

    window.nexusReposList.repos.forEach((repo, ri) => {
        const row = createElem("div")

        const enabledCkbx = createElem("input", {type: "checkbox"})
        enabledCkbx.checked = repo.enabled
        enabledCkbx.addEventListener("click", () => {
            window.nexusReposList.repos[ri].enabled = enabledCkbx.checked
            fs.writeFileSync(`${window.path}/repositories.json`, JSON.stringify(window.nexusReposList, null, 4), "utf8")
        })
        const enabledCkbxElem = createElem("div", enabledCkbx)

        const removeButton = createElem("button.smallButton", window.i18n.REMOVE)
        removeButton.style.background = `#${window.currentGame.themeColourPrimary}`
        const removeButtonElem = createElem("div", removeButton)
        const linkButton = createElem("button.smallButton", window.i18n.OPEN)
        linkButton.style.background = `#${window.currentGame.themeColourPrimary}`
        linkButton.addEventListener("click", () => {
            shell.openExternal(repo.url)
        })
        const linkButtonElem = createElem("div", linkButton)
        const gameElem = createElem("div", window.nexusGameIdToGameName[repo.game_id])
        gameElem.title = window.nexusGameIdToGameName[repo.game_id]
        const nameElem = createElem("div", repo.name)
        nameElem.title = repo.name
        const authorElem = createElem("div", repo.author)
        authorElem.title = repo.author
        const endorsementsElem = createElem("div", String(repo.endorsements))
        const downloadsElem = createElem("div", String(repo.downloads))


        row.appendChild(enabledCkbxElem)
        row.appendChild(linkButtonElem)
        row.appendChild(gameElem)
        row.appendChild(nameElem)
        row.appendChild(authorElem)
        row.appendChild(endorsementsElem)
        row.appendChild(downloadsElem)
        row.appendChild(removeButtonElem)
        nexusReposUsedContainer.appendChild(row)
    })
}


window.addRepoToApp = (repo) => {
    repo.enabled = true
    window.nexusReposList.repos.push(repo)
    window.nexusReposList.repos = window.nexusReposList.repos.sort((a,b)=>a.endorsements<b.endorsements?1:-1)
    fs.writeFileSync(`${window.path}/repositories.json`, JSON.stringify(window.nexusReposList, null, 4), "utf8")
    window.nexusUpdateModsUsedPanel()
}




nexusReposSearchBar.addEventListener("keydown", e => {
    if (e.key.toLowerCase()=="enter" && nexusReposSearchBar.value.length) {
        searchNexusButton.click()
    }
})
searchNexusButton.addEventListener("click", () => {
    const gameId = nexusAllGamesSelect.value ? parseInt(nexusAllGamesSelect.value) : undefined
    const query = nexusReposSearchBar.value
    nexusSearchContainer.innerHTML = ""
    window.mod_search_nexus(gameId, query).then(results => {

        const numResults = results[0]
        results = results[1]

        results.forEach(repo => {

            const row = createElem("div")
            const addButton = createElem("button.smallButton", window.i18n.ADD)
            const addButtonElem = createElem("div", addButton)
            addButton.style.background = `#${window.currentGame.themeColourPrimary}`
            if (window.nexusReposList.repos.find(r=>r.url==repo.url)) {
                addButton.disabled = true
            }
            addButton.addEventListener("click", () => {
                window.addRepoToApp(repo)
                addButton.disabled = true
            })

            const linkButton = createElem("button.smallButton", window.i18n.OPEN)
            linkButton.style.background = `#${window.currentGame.themeColourPrimary}`
            linkButton.addEventListener("click", () => {
                shell.openExternal(repo.url)
            })
            const linkButtonElem = createElem("div", linkButton)
            const gameElem = createElem("div", window.nexusGameIdToGameName[repo.game_id])
            gameElem.title = window.nexusGameIdToGameName[repo.game_id]
            const nameElem = createElem("div", repo.name)
            nameElem.title = repo.name
            const authorElem = createElem("div", repo.author)
            authorElem.title = repo.author
            const endorsementsElem = createElem("div", String(repo.endorsements))
            const downloadsElem = createElem("div", String(repo.downloads))


            row.appendChild(addButtonElem)
            row.appendChild(linkButtonElem)
            row.appendChild(gameElem)
            row.appendChild(nameElem)
            row.appendChild(authorElem)
            row.appendChild(endorsementsElem)
            row.appendChild(downloadsElem)
            nexusSearchContainer.appendChild(row)
        })

    })
})

nexusOnlyNewUpdatedCkbx.addEventListener("change", () => window.displayAllModels())
window.initNexus()
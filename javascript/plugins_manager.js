"use strict"

const fs = require("fs")

class PluginsManager {

    constructor (path, appLogger, appVersion) {

        this.path = `${__dirname.replace(/\\/g,"/").replace("/javascript", "")}/`.replace("/resources/app/resources/app", "/resources/app")
        this.appVersion = appVersion
        this.appLogger = appLogger
        this.plugins = []
        this.selectedPlugin = undefined
        this.hasRunPostStartPlugins = false
        this.changesToApply = {
            ticked: [],
            unticked: []
        }
        this.resetModules()


        this.scanPlugins()
        this.savePlugins()
        this.appLogger.log(`${this.path}/plugins`)
        if (fs.existsSync(`${this.path}/plugins`)) {
            fs.watch(`${this.path}/plugins`, {recursive: false, persistent: true}, (eventType, filename) => {
                this.scanPlugins()
                this.updateUI()
                this.savePlugins()
            })
        }

        plugins_moveUpBtn.addEventListener("click", () => {

            if (!this.selectedPlugin || this.selectedPlugin[1]==0) return

            const plugin = this.plugins.splice(this.selectedPlugin[1], 1)[0]
            this.plugins.splice(this.selectedPlugin[1]-1, 0, plugin)
            this.selectedPlugin[1] -= 1
            this.updateUI()
            plugins_applyBtn.disabled = false
        })

        plugins_moveDownBtn.addEventListener("click", () => {

            if (!this.selectedPlugin || this.selectedPlugin[1]==this.plugins.length-1) return

            const plugin = this.plugins.splice(this.selectedPlugin[1], 1)[0]
            this.plugins.splice(this.selectedPlugin[1]+1, 0, plugin)
            this.selectedPlugin[1] += 1
            this.updateUI()
            plugins_applyBtn.disabled = false
        })

        plugins_applyBtn.addEventListener("click", () => this.apply())
        plugins_main.addEventListener("click", (e) => {
            if (e.target == plugins_main) {
                this.selectedPlugin = undefined
                this.updateUI()
            }
        })

        window.pluginsManager = this
        this.loadModules()
    }

    resetModules () {
        this.setupModules = new Set()
        this.teardownModules = {}
        this.pluginsModules = {
            "start": {
                "pre": [],
                "post": []
            },
            "keep-sample": {
                "pre": [],
                "mid": [],
                "post": []
            },
            "batch-stop": {
                "post": []
            },
            "generate-voice": {
                "pre": []
            }
        }
        pluginsCSS.innerHTML = ""
    }

    scanPlugins () {

        const plugins = []

        try {
            const pluginIDs = fs.readdirSync(`${this.path}/plugins`)
            pluginIDs.forEach(pluginId => {
                try {
                    const pluginData = JSON.parse(fs.readFileSync(`${this.path}/plugins/${pluginId}/plugin.json`))

                    const minVersionOk = window.checkVersionRequirements(pluginData["min-app-version"], this.appVersion)
                    const maxVersionOk = window.checkVersionRequirements(pluginData["max-app-version"], this.appVersion, true)

                    plugins.push([pluginId, pluginData, false, minVersionOk, maxVersionOk])

                } catch (e) {
                    this.appLogger.log(`${window.i18n.ERR_LOADING_PLUGIN} ${pluginId}: ${e}`)
                }
            })


        } catch (e) {
            console.log(e)
        }

        const orderedPlugins = []

        // Order the found known plugins
        window.userSettings.plugins.loadOrder.split(",").forEach(pluginId => {
            for (let i=0; i<plugins.length; i++) {
                if (pluginId.replace("*", "")==plugins[i][0]) {
                    plugins[i][2] = pluginId.includes("*")
                    orderedPlugins.push(plugins[i])
                    plugins.splice(i,1)
                    break
                }
            }
        })

        // Add any remaining (new) plugins at the bottom of the list
        plugins.forEach(p => orderedPlugins.push(p))
        this.plugins = orderedPlugins
    }

    updateUI () {

        pluginsRecordsContainer.innerHTML = ""

        this.plugins.forEach(([pluginId, pluginData, isEnabled, minVersionOk, maxVersionOk], pi) => {
            const record = createElem("div")
            const enabledCkbx = createElem("input", {type: "checkbox"})
            enabledCkbx.checked = isEnabled
            record.appendChild(createElem("div", enabledCkbx))
            record.appendChild(createElem("div", `${pi}`))

            const pluginNameElem = createElem("div", pluginData["plugin-name"])
            pluginNameElem.title = pluginData["plugin-name"]
            record.appendChild(pluginNameElem)

            const pluginAuthorElem = createElem("div", pluginData["author"]||"")
            pluginAuthorElem.title = pluginData["author"]||""
            record.appendChild(pluginAuthorElem)

            const endorseButtonContainer = createElem("div")
            record.appendChild(endorseButtonContainer)
            if (pluginData["nexus-link"] && window.nexusState.key) {

                if (window.nexusState.key) {
                    window.nexus_getData(`${pluginData["nexus-link"].split(".com/")[1]}.json`).then(repoInfo => {
                        const endorseButton = createElem("button.smallButton", "Endorse")
                        const gameId = repoInfo.game_id
                        const nexusRepoId = repoInfo.mod_id

                        if (repoInfo.endorsement.endorse_status=="Endorsed") {
                            window.endorsedRepos.add(`plugin:${pluginId}`)
                            endorseButton.innerHTML = "Unendorse"
                            endorseButton.style.background = "none"
                            endorseButton.style.border = `2px solid #${window.currentGame ? currentGame.themeColourPrimary : "aaa"}`
                        } else {
                            endorseButton.style.setProperty("background-color", `#${window.currentGame ? currentGame.themeColourPrimary : "aaa"}`, "important")
                        }

                        endorseButtonContainer.appendChild(endorseButton)
                        endorseButton.addEventListener("click", async () => {
                            let response
                            if (window.endorsedRepos.has(`plugin:${pluginId}`)) {
                                response = await window.nexus_getData(`${gameId}/mods/${nexusRepoId}/abstain.json`, {
                                    game_domain_name: gameId,
                                    id: nexusRepoId,
                                    version: repoInfo.version
                                }, "POST")
                            } else {
                                response = await window.nexus_getData(`${gameId}/mods/${nexusRepoId}/endorse.json`, {
                                    game_domain_name: gameId,
                                    id: nexusRepoId,
                                    version: repoInfo.version
                                }, "POST")
                            }
                            if (response && response.message && response.status=="Error") {
                                if (response.message=="NOT_DOWNLOADED_MOD") {
                                    response.message = "You need to first download something from this repo to be able to endorse it."
                                } else if (response.message=="TOO_SOON_AFTER_DOWNLOAD") {
                                    response.message = "Nexus requires you to wait at least 15 mins (at the time of writing) before you can endorse."
                                } else if (response.message=="IS_OWN_MOD") {
                                    response.message = "Nexus does not allow you to rate your own content."
                                }

                                window.errorModal(response.message)
                            } else {

                                if (window.endorsedRepos.has(`plugin:${pluginId}`)) {
                                    window.endorsedRepos.delete(`plugin:${pluginId}`)
                                } else {
                                    window.endorsedRepos.add(`plugin:${pluginId}`)
                                }
                                this.updateUI()
                            }
                        })
                    })
                }
            }


            const hasBackendScript = !!Object.keys(pluginData["back-end-hooks"]).find(key => {
                return (key=="custom-event" && pluginData["back-end-hooks"]["custom-event"]["file"]) ||
                (pluginData["back-end-hooks"][key]["pre"] && pluginData["back-end-hooks"][key]["pre"]["file"]) ||
                (pluginData["back-end-hooks"][key]["mid"] && pluginData["back-end-hooks"][key]["mid"]["file"]) ||
                (pluginData["back-end-hooks"][key]["post"] && pluginData["back-end-hooks"][key]["post"]["file"])
            })
            const hasFrontendScript = !!pluginData["front-end-hooks"]
            const type =  hasFrontendScript && hasBackendScript ? "Both": (!hasFrontendScript && !hasBackendScript ? "None" : (hasFrontendScript ? "Front" : "Back"))

            record.appendChild(createElem("div", pluginData["plugin-version"]))
            record.appendChild(createElem("div", type))
            // Min app version requirement
            const minAppVersionElem = createElem("div", pluginData["min-app-version"])
            record.appendChild(minAppVersionElem)
            if (pluginData["min-app-version"] && !minVersionOk) {
                minAppVersionElem.style.color = "red"
                enabledCkbx.checked = false
                enabledCkbx.disabled = true
            }

            // Max app version requirement
            const maxAppVersionElem = createElem("div", pluginData["max-app-version"])
            record.appendChild(maxAppVersionElem)
            if (pluginData["max-app-version"] && !maxVersionOk) {
                maxAppVersionElem.style.color = "red"
                enabledCkbx.checked = false
                enabledCkbx.disabled = true
            }

            const shortDescriptionElem = createElem("div", pluginData["plugin-short-description"])
            shortDescriptionElem.title = pluginData["plugin-short-description"]
            record.appendChild(shortDescriptionElem)

            const pluginIdElem = createElem("div", pluginId)
            pluginIdElem.title = pluginId
            record.appendChild(pluginIdElem)

            pluginsRecordsContainer.appendChild(record)

            enabledCkbx.addEventListener("click", () => {
                this.plugins[pi][2] = enabledCkbx.checked
                plugins_applyBtn.disabled = false
            })

            record.addEventListener("click", (e) => {

                if (e.target==enabledCkbx || e.target.nodeName=="BUTTON") {
                    return
                }

                if (this.selectedPlugin) {
                    this.selectedPlugin[0].style.background = "none"
                    Array.from(this.selectedPlugin[0].children).forEach(child => child.style.color = "white")
                }

                this.selectedPlugin = [record, pi, pluginData]
                this.selectedPlugin[0].style.background = "white"
                Array.from(this.selectedPlugin[0].children).forEach(child => child.style.color = "black")

                plugins_moveUpBtn.disabled = false
                plugins_moveDownBtn.disabled = false

            })
            if (this.selectedPlugin && pi==this.selectedPlugin[1]) {
                this.selectedPlugin = [record, pi, pluginData]
                this.selectedPlugin[0].style.background = "white"
                Array.from(this.selectedPlugin[0].children).forEach(child => child.style.color = "black")
            }

        })
    }

    savePlugins () {
        window.userSettings.plugins.loadOrder = this.plugins.map(([pluginId, pluginData, isEnabled]) => `${isEnabled?"*":""}${pluginId}`).join(",")
        saveUserSettings()
        fs.writeFileSync(`./plugins.txt`, window.userSettings.plugins.loadOrder.replace(/,/g, "\n"))
    }

    apply () {

        const enabledPlugins = this.plugins.filter(([pluginId, pluginData, isEnabled]) => isEnabled).map(([pluginId, pluginData, isEnabled]) => pluginId)
        const newPlugins = enabledPlugins.filter(pluginId => !window.userSettings.plugins.loadOrder.includes(`*${pluginId}`))
        const removedPlugins = window.userSettings.plugins.loadOrder.split(",").filter(pluginId => pluginId.startsWith("*") && !enabledPlugins.includes(pluginId.slice(1, 100000)) ).map(pluginId => pluginId.slice(1, 100000))

        removedPlugins.forEach(pluginId => {
            if (this.teardownModules[pluginId]) {
                this.teardownModules[pluginId].forEach(func => func())
            }
        })

        const pluginLoadStatus = this.loadModules()
        if (pluginLoadStatus) {
            window.errorModal(`${window.i18n.FAILED_INIT_FOLLOWING} ${window.i18n.PLUGIN.toLowerCase()}: ${pluginLoadStatus}`)
            return
        }
        this.savePlugins()
        this.resetModules()
        plugins_applyBtn.disabled = true

        doFetch(`http://localhost:8008/refreshPlugins`, {
            method: "Post",
            body: "{}"
        }).then(r=>r.text()).then(status => {

            const plugins = status.split(",")
            const successful = plugins.filter(p => p=="OK")
            const failed = plugins.filter(p => p!="OK")

            let message = `${window.i18n.SUCCESSFULLY_INITIALIZED} ${successful.length} ${successful.length>1||successful.length==0?window.i18n.PLUGINS:window.i18n.PLUGIN}.`
            if (failed.length) {
                if (successful.length==0) {
                    message = ""
                }
                message += ` ${window.i18n.FAILED_INIT_FOLLOWING} ${failed.length>1?window.i18n.PLUGINS:window.i18n.PLUGIN}: <br>${failed.join("<br>")} <br><br>${window.i18n.CHECK_SERVERLOG}`
            }

            if (!status.length || successful.length==0 && failed.length==0) {
                message = window.i18n.SUCC_NO_ACTIVE_PLUGINS
            }

            const restartRequired = newPlugins.map(newPluginId => this.plugins.find(([pluginId, pluginData, isEnabled]) => pluginId==newPluginId))
                                              .filter(([pluginId, pluginData, isEnabled]) => !!pluginData["install-requires-restart"]).length +
                                    removedPlugins.map(removedPluginId => this.plugins.find(([pluginId, pluginData, isEnabled]) => pluginId==removedPluginId))
                                              .filter(([pluginId, pluginData, isEnabled]) => !!pluginData["uninstall-requires-restart"]).length
            if (restartRequired) {
                message += `<br><br> ${window.i18n.APP_RESTART_NEEDED}`
            }

            // Don't use window.errorModal, otherwise you get the error sound
            createModal("error", message)
        })
    }


    loadModules () {
        for (let pi=0; pi<this.plugins.length; pi++) {
            const [pluginId, pluginData, enabled] = this.plugins[pi]
            if (!enabled) continue
            let failed

            failed = this.loadModuleFns(pluginId, pluginData, "start", "pre")
            if (failed) return `${pluginId}->start->pre<br><br>${failed}`

            failed = this.loadModuleFns(pluginId, pluginData, "start", "post")
            if (failed) return `${pluginId}->start->post<br><br>${failed}`

            this.loadModuleFns(pluginId, pluginData, "keep-sample", "pre")
            if (failed) return `${pluginId}->keep-sample->pre<br><br>${failed}`

            this.loadModuleFns(pluginId, pluginData, "keep-sample", "mid")
            if (failed) return `${pluginId}->keep-sample->mid<br><br>${failed}`

            this.loadModuleFns(pluginId, pluginData, "keep-sample", "post")
            if (failed) return `${pluginId}->keep-sample->post<br><br>${failed}`

            this.loadModuleFns(pluginId, pluginData, "generate-voice", "pre")
            if (failed) return `${pluginId}->generate-voice->pre<br><br>${failed}`

            this.loadModuleFns(pluginId, pluginData, "batch-stop", "post")
            if (failed) return `${pluginId}->batch-stop->post<br><br>${failed}`


            if (Object.keys(pluginData).includes("front-end-style-files") && pluginData["front-end-style-files"].length) {
                pluginData["front-end-style-files"].forEach(styleFile => {
                    try {
                        if (styleFile.endsWith(".css")) {
                            const styleData = fs.readFileSync(`${this.path}/plugins/${pluginId}/${styleFile}`)
                            pluginsCSS.innerHTML += styleData
                        }
                    } catch (e) {
                        window.appLogger.log(`${window.i18n.ERR_LOADING_CSS} ${pluginId}: ${e}`)
                    }
                })
            }
        }
    }

    loadModuleFns (pluginId, pluginData, task, hookTime) {
        try {
            if (Object.keys(pluginData).includes("front-end-hooks") && Object.keys(pluginData["front-end-hooks"]).includes(task) && Object.keys(pluginData["front-end-hooks"][task]).includes(hookTime) ) {

                const file = pluginData["front-end-hooks"][task][hookTime]["file"]
                const functionName = pluginData["front-end-hooks"][task][hookTime]["function"]

                if (!file.endsWith(".js")) {
                    window.appLogger.log(`[${window.i18n.PLUGIN}: ${pluginId}]: ${window.i18n.CANT_IMPORT_FILE_FOR_HOOK_TASK_ENTRYPOINT.replace("_1", file).replace("_2", hookTime).replace("_3", task)}: ${window.i18n.ONLY_JS}`)
                    return
                }

                if (file && functionName) {
                    const module = require(`${this.path}/plugins/${pluginId}/${file}`)

                    if (module.teardown) {
                        if (!Object.keys(this.teardownModules).includes(pluginId)) {
                            this.teardownModules[pluginId] = []
                        }
                        this.teardownModules[pluginId].push(module.teardown)
                    }

                    if (module.setup && !this.setupModules.has(`${pluginId}/${file}`)) {
                        window.appLogger.setPrefix(pluginId)
                        module.setup(window)
                        window.appLogger.setPrefix("")
                        this.setupModules.add(`${pluginId}/${file}`)
                    }

                    this.pluginsModules[task][hookTime].push([pluginId, module[functionName]])
                }
            }
        } catch (e) {
            console.log(`${window.i18n.ERR_LOADING_PLUGIN} ${pluginId}->${task}->${hookTime}: ` + e.stack)
            window.appLogger.log(`${window.i18n.ERR_LOADING_PLUGIN} ${pluginId}->${task}->${hookTime}: ` + e)
            return e.stack
        }

    }

    runPlugins (pList, event, data) {
        if (pList.length) {
            console.log(`Running plugin for event: ${event}`)
        }
        pList.forEach(([pluginId, pluginFn]) => {
            try {
                window.appLogger.setPrefix(pluginId)
                pluginFn(window, data)
                window.appLogger.setPrefix("")

            } catch (e) {
                console.log(e, pluginFn)
                window.appLogger.log(`[${window.i18n.PLUGIN_RUN_ERROR} "${event}": ${pluginId}]: ${e}`)
            }
        })
    }


    _saveINIFile (IniSettings, settingsKey, pluginId, filePath) {
        const outputIni = []
        settingsOptionsContainer.querySelectorAll(`.${pluginId}_plugin_setting>div>input, .${pluginId}_plugin_setting>div>select`).forEach(input => {

            if (input.tagName=="SELECT") {
                const select = input
                const optionsList = Array.from(select.querySelectorAll("option")).map(option => {
                    return [option.innerHTML, option.value]
                })
                const optionsListString = `{${optionsList.map(kv => kv.join(":")).join(";")}}`

                outputIni.push(`${select.name.toLowerCase()}=${select.value} # ${optionsListString} ${select.getAttribute("comment")!="undefined" ? select.getAttribute("comment") : ""}`)
                IniSettings[select.name.toLowerCase()] = select.value

            } else {
                const value = input.type=="checkbox" ? (input.checked ? "true" : "false") : input.value
                outputIni.push(`${input.name.toLowerCase()}=${value}${input.getAttribute("comment")!="undefined" ? " # "+input.getAttribute("comment") : ""}`)
                IniSettings[input.name.toLowerCase()] = value
            }
        })

        fs.writeFileSync(filePath, outputIni.join("\n"), "utf8")
        window.pluginsContext[settingsKey] = IniSettings
    }

    registerINIFile (pluginId, settingsKey, filePath) {

        if (!pluginId || !settingsKey || !filePath) {
            return self.log(`You must provide the following to register an ini file: pluginId, settingsKey, filePath`)
        }

        if (fs.existsSync(filePath)) {

            if (document.querySelectorAll(`.${pluginId}_plugin_setting`).length) {
                return
            }

            const IniSettings = {}
            const iniFileData = fs.readFileSync(filePath, "utf8").split("\n")

            const hr = createElem(`hr.${pluginId}_plugin_setting`)
            settingsOptionsContainer.appendChild(hr)
            settingsOptionsContainer.appendChild(createElem("div.centeredSettingsSectionPlugins", createElem("div", window.i18n.SETTINGS_FOR_PLUGIN.replace("_1", pluginId)) ))

            iniFileData.forEach(keyVal => {
                if (!keyVal.trim().length) {
                    return
                }
                let comment = keyVal.includes("#") ? keyVal.split("#")[1].trim() : undefined
                keyVal = keyVal.split("#")[0].trim()
                const key = keyVal.split("=")[0].trim()
                const val = keyVal.split("=")[1].trim()
                IniSettings[key.toLowerCase()] = val

                const labelText = key[0].toUpperCase() + key.substring(1)
                let label, input
                const extraElems = []

                if (comment && (comment.includes("$filepicker") || comment.includes("$folderpicker"))) {

                    input = createElem("input", {name: key, comment: comment})
                    input.style.width = "80%"
                    input.value = val
                    const button = createElem("button.svgButton")
                    button.innerHTML = `<svg class="openFolderSVG" width="400" height="350" viewBox="0, 0, 400,350"><g id="svgg" ><path id="path0"  d="M39.960 53.003 C 36.442 53.516,35.992 53.635,30.800 55.422 C 15.784 60.591,3.913 74.835,0.636 91.617 C -0.372 96.776,-0.146 305.978,0.872 310.000 C 5.229 327.228,16.605 339.940,32.351 345.172 C 40.175 347.773,32.175 347.630,163.000 347.498 L 281.800 347.378 285.600 346.495 C 304.672 342.065,321.061 332.312,330.218 319.944 C 330.648 319.362,332.162 317.472,333.581 315.744 C 335.001 314.015,336.299 312.420,336.467 312.200 C 336.634 311.980,337.543 310.879,338.486 309.753 C 340.489 307.360,342.127 305.341,343.800 303.201 C 344.460 302.356,346.890 299.375,349.200 296.575 C 351.510 293.776,353.940 290.806,354.600 289.975 C 355.260 289.144,356.561 287.505,357.492 286.332 C 358.422 285.160,359.952 283.267,360.892 282.126 C 362.517 280.153,371.130 269.561,375.632 264.000 C 376.789 262.570,380.427 258.097,383.715 254.059 C 393.790 241.689,396.099 237.993,398.474 230.445 C 403.970 212.972,394.149 194.684,376.212 188.991 C 369.142 186.747,368.803 186.724,344.733 186.779 C 330.095 186.812,322.380 186.691,322.216 186.425 C 322.078 186.203,321.971 178.951,321.977 170.310 C 321.995 146.255,321.401 141.613,317.200 133.000 C 314.009 126.457,307.690 118.680,303.142 115.694 C 302.560 115.313,301.300 114.438,300.342 113.752 C 295.986 110.631,288.986 107.881,282.402 106.704 C 280.540 106.371,262.906 106.176,220.400 106.019 L 161.000 105.800 160.763 98.800 C 159.961 75.055,143.463 56.235,120.600 52.984 C 115.148 52.208,45.292 52.225,39.960 53.003 M120.348 80.330 C 130.472 83.988,133.993 90.369,133.998 105.071 C 134.003 120.968,137.334 127.726,147.110 131.675 L 149.400 132.600 213.800 132.807 C 272.726 132.996,278.392 133.071,280.453 133.690 C 286.872 135.615,292.306 141.010,294.261 147.400 C 294.928 149.578,294.996 151.483,294.998 168.000 L 295.000 186.200 292.800 186.449 C 291.590 186.585,254.330 186.725,210.000 186.759 C 163.866 186.795,128.374 186.977,127.000 187.186 C 115.800 188.887,104.936 192.929,96.705 198.458 C 95.442 199.306,94.302 200.000,94.171 200.000 C 93.815 200.000,89.287 203.526,87.000 205.583 C 84.269 208.039,80.083 212.649,76.488 217.159 C 72.902 221.657,72.598 222.031,70.800 224.169 C 70.030 225.084,68.770 226.620,68.000 227.582 C 67.230 228.544,66.054 229.977,65.387 230.766 C 64.720 231.554,62.727 234.000,60.957 236.200 C 59.188 238.400,56.346 241.910,54.642 244.000 C 52.938 246.090,50.163 249.510,48.476 251.600 C 44.000 257.146,36.689 266.126,36.212 266.665 C 35.985 266.921,34.900 268.252,33.800 269.623 C 32.700 270.994,30.947 273.125,29.904 274.358 C 28.861 275.591,28.006 276.735,28.004 276.900 C 28.002 277.065,27.728 277.200,27.395 277.200 C 26.428 277.200,26.700 96.271,27.670 93.553 C 30.020 86.972,35.122 81.823,40.800 80.300 C 44.238 79.378,47.793 79.296,81.800 79.351 L 117.800 79.410 120.348 80.330 M369.400 214.800 C 374.239 217.220,374.273 222.468,369.489 228.785 C 367.767 231.059,364.761 234.844,364.394 235.200 C 364.281 235.310,362.373 237.650,360.154 240.400 C 357.936 243.150,354.248 247.707,351.960 250.526 C 347.732 255.736,346.053 257.821,343.202 261.400 C 341.505 263.530,340.849 264.336,334.600 271.965 C 332.400 274.651,330.204 277.390,329.720 278.053 C 329.236 278.716,328.246 279.945,327.520 280.785 C 326.794 281.624,325.300 283.429,324.200 284.794 C 323.100 286.160,321.726 287.845,321.147 288.538 C 320.568 289.232,318.858 291.345,317.347 293.233 C 308.372 304.449,306.512 306.609,303.703 309.081 C 299.300 312.956,290.855 317.633,286.000 318.886 C 277.958 320.960,287.753 320.819,159.845 320.699 C 33.557 320.581,42.330 320.726,38.536 318.694 C 34.021 316.276,35.345 310.414,42.386 301.647 C 44.044 299.583,45.940 297.210,46.600 296.374 C 47.260 295.538,48.340 294.169,49.000 293.332 C 49.660 292.495,51.550 290.171,53.200 288.167 C 54.850 286.164,57.100 283.395,58.200 282.015 C 59.300 280.635,60.920 278.632,61.800 277.564 C 62.680 276.496,64.210 274.617,65.200 273.389 C 66.190 272.162,67.188 270.942,67.418 270.678 C 67.649 270.415,71.591 265.520,76.179 259.800 C 80.767 254.080,84.634 249.310,84.773 249.200 C 84.913 249.090,87.117 246.390,89.673 243.200 C 92.228 240.010,95.621 235.780,97.213 233.800 C 106.328 222.459,116.884 215.713,128.200 213.998 C 129.300 213.832,183.570 213.719,248.800 213.748 L 367.400 213.800 369.400 214.800 " stroke="none" fill="#fbfbfb" fill-rule="evenodd"></path><path id="path1" fill-opacity="0" d="M0.000 46.800 C 0.000 72.540,0.072 93.600,0.159 93.600 C 0.246 93.600,0.516 92.460,0.759 91.066 C 3.484 75.417,16.060 60.496,30.800 55.422 C 35.953 53.648,36.338 53.550,40.317 52.981 C 46.066 52.159,114.817 52.161,120.600 52.984 C 143.463 56.235,159.961 75.055,160.763 98.800 L 161.000 105.800 220.400 106.019 C 262.906 106.176,280.540 106.371,282.402 106.704 C 288.986 107.881,295.986 110.631,300.342 113.752 C 301.300 114.438,302.560 115.313,303.142 115.694 C 307.690 118.680,314.009 126.457,317.200 133.000 C 321.401 141.613,321.995 146.255,321.977 170.310 C 321.971 178.951,322.078 186.203,322.216 186.425 C 322.380 186.691,330.095 186.812,344.733 186.779 C 368.803 186.724,369.142 186.747,376.212 188.991 C 381.954 190.814,388.211 194.832,391.662 198.914 C 395.916 203.945,397.373 206.765,399.354 213.800 C 399.842 215.533,399.922 201.399,399.958 107.900 L 400.000 0.000 200.000 0.000 L 0.000 0.000 0.000 46.800 M44.000 79.609 C 35.903 81.030,30.492 85.651,27.670 93.553 C 26.700 96.271,26.428 277.200,27.395 277.200 C 27.728 277.200,28.002 277.065,28.004 276.900 C 28.006 276.735,28.861 275.591,29.904 274.358 C 30.947 273.125,32.700 270.994,33.800 269.623 C 34.900 268.252,35.985 266.921,36.212 266.665 C 36.689 266.126,44.000 257.146,48.476 251.600 C 50.163 249.510,52.938 246.090,54.642 244.000 C 56.346 241.910,59.188 238.400,60.957 236.200 C 62.727 234.000,64.720 231.554,65.387 230.766 C 66.054 229.977,67.230 228.544,68.000 227.582 C 68.770 226.620,70.030 225.084,70.800 224.169 C 72.598 222.031,72.902 221.657,76.488 217.159 C 80.083 212.649,84.269 208.039,87.000 205.583 C 89.287 203.526,93.815 200.000,94.171 200.000 C 94.302 200.000,95.442 199.306,96.705 198.458 C 104.936 192.929,115.800 188.887,127.000 187.186 C 128.374 186.977,163.866 186.795,210.000 186.759 C 254.330 186.725,291.590 186.585,292.800 186.449 L 295.000 186.200 294.998 168.000 C 294.996 151.483,294.928 149.578,294.261 147.400 C 292.306 141.010,286.872 135.615,280.453 133.690 C 278.392 133.071,272.726 132.996,213.800 132.807 L 149.400 132.600 147.110 131.675 C 137.334 127.726,134.003 120.968,133.998 105.071 C 133.993 90.369,130.472 83.988,120.348 80.330 L 117.800 79.410 81.800 79.351 C 62.000 79.319,44.990 79.435,44.000 79.609 M128.200 213.998 C 116.884 215.713,106.328 222.459,97.213 233.800 C 95.621 235.780,92.228 240.010,89.673 243.200 C 87.117 246.390,84.913 249.090,84.773 249.200 C 84.634 249.310,80.767 254.080,76.179 259.800 C 71.591 265.520,67.649 270.415,67.418 270.678 C 67.188 270.942,66.190 272.162,65.200 273.389 C 64.210 274.617,62.680 276.496,61.800 277.564 C 60.920 278.632,59.300 280.635,58.200 282.015 C 57.100 283.395,54.850 286.164,53.200 288.167 C 51.550 290.171,49.660 292.495,49.000 293.332 C 48.340 294.169,47.260 295.538,46.600 296.374 C 45.940 297.210,44.044 299.583,42.386 301.647 C 35.345 310.414,34.021 316.276,38.536 318.694 C 42.330 320.726,33.557 320.581,159.845 320.699 C 287.753 320.819,277.958 320.960,286.000 318.886 C 290.855 317.633,299.300 312.956,303.703 309.081 C 306.512 306.609,308.372 304.449,317.347 293.233 C 318.858 291.345,320.568 289.232,321.147 288.538 C 321.726 287.845,323.100 286.160,324.200 284.794 C 325.300 283.429,326.794 281.624,327.520 280.785 C 328.246 279.945,329.236 278.716,329.720 278.053 C 330.204 277.390,332.400 274.651,334.600 271.965 C 340.849 264.336,341.505 263.530,343.202 261.400 C 346.053 257.821,347.732 255.736,351.960 250.526 C 354.248 247.707,357.936 243.150,360.154 240.400 C 362.373 237.650,364.281 235.310,364.394 235.200 C 364.761 234.844,367.767 231.059,369.489 228.785 C 374.273 222.468,374.239 217.220,369.400 214.800 L 367.400 213.800 248.800 213.748 C 183.570 213.719,129.300 213.832,128.200 213.998 M399.600 225.751 C 399.600 231.796,394.623 240.665,383.715 254.059 C 380.427 258.097,376.789 262.570,375.632 264.000 C 371.130 269.561,362.517 280.153,360.892 282.126 C 359.952 283.267,358.422 285.160,357.492 286.332 C 356.561 287.505,355.260 289.144,354.600 289.975 C 353.940 290.806,351.510 293.776,349.200 296.575 C 346.890 299.375,344.460 302.356,343.800 303.201 C 342.127 305.341,340.489 307.360,338.486 309.753 C 337.543 310.879,336.634 311.980,336.467 312.200 C 336.299 312.420,335.001 314.015,333.581 315.744 C 332.162 317.472,330.648 319.362,330.218 319.944 C 321.061 332.312,304.672 342.065,285.600 346.495 L 281.800 347.378 163.000 347.498 C 32.175 347.630,40.175 347.773,32.351 345.172 C 16.471 339.895,3.810 325.502,0.820 309.326 C 0.591 308.085,0.312 306.979,0.202 306.868 C 0.091 306.757,-0.000 327.667,-0.000 353.333 L 0.000 400.000 200.000 400.000 L 400.000 400.000 400.000 312.400 C 400.000 264.220,399.910 224.800,399.800 224.800 C 399.690 224.800,399.600 225.228,399.600 225.751 " stroke="none" fill="#050505" fill-rule="evenodd"></path></g></svg>`

                    comment = comment.replace("$filepicker", "").replace("$folderpicker", "")

                    const openType = comment.includes("$filepicker") ? "openFile" : "openDirectory"
                    button.addEventListener("click", () => {
                        let filePathInput = electron.remote.dialog.showOpenDialog({ properties: [openType]})
                        if (filePathInput) {
                            filePathInput = filePathInput[0].replace(/\\/g, "/")
                            input.value = filePathInput.replace(/\\/g, "/")

                            this._saveINIFile(IniSettings, settingsKey, pluginId, filePath)
                        }
                    })
                    extraElems.push(button)



                } else if (comment && comment.includes("{") && comment.includes(":")) {

                    const optionsList = comment.split("{")[1].split("}")[0].split(";").map(kv => {
                        return [kv.split(":")[0], kv.split(":")[1]]
                    })
                    const optionElems = optionsList.map(data => {
                        const opt = createElem("option", {value: data[1]})
                        opt.innerHTML = data[0]
                        return opt
                    })

                    comment = comment.split("}").reverse()[0].trim()

                    input = createElem("select", {name: key, comment: comment})
                    optionElems.forEach(option => {
                        input.appendChild(option)
                    })
                    input.value = val

                } else {
                    const inputType = ["true","false"].includes(val.toLowerCase()) ? "checkbox" : "text"
                    input = createElem("input", {
                        type: inputType, name: key, comment: comment
                    })
                    if (inputType=="checkbox") {
                        input.checked = val.toLowerCase() == "true"
                    } else {
                        input.value = val
                    }
                }

                label = createElem("div", labelText.replace(/_/g, " ") + (comment ? `<br>(${comment})` : ""))


                input.addEventListener("change", () => {
                    this._saveINIFile(IniSettings, settingsKey, pluginId, filePath)
                })

                const rhd_elem = createElem("div")
                rhd_elem.appendChild(input)
                extraElems.forEach(elem => rhd_elem.appendChild(elem))
                if (extraElems.length) {
                    rhd_elem.style.flexDirection = "row"
                }

                settingsOptionsContainer.appendChild(createElem(`div.${pluginId}_plugin_setting`, [label, rhd_elem]))
            })

            window.pluginsContext[settingsKey] = IniSettings



        } else {
            this.log(`Ini file does not exists here: ${filePath}`)
        }

    }

}


exports.PluginsManager = PluginsManager
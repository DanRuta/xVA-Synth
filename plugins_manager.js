"use strict"

const fs = require("fs")

class PluginsManager {

    constructor (path, appLogger) {

        this.path = path
        this.appLogger = appLogger
        this.plugins = []
        this.selectedPlugin = undefined
        this.changesToApply = {
            ticked: [],
            unticked: []
        }
        this.resetModules()


        this.scanPlugins()
        this.updateUI()
        this.savePlugins()
        fs.watch(`${this.path}/plugins`, {recursive: false, persistent: true}, (eventType, filename) => {
            this.scanPlugins()
            this.updateUI()
            this.savePlugins()
        })

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

        this.loadModules()
    }

    resetModules () {
        this.setupModules = new Set()
        this.pluginsModules = {
            "start": {
                "pre": [],
                "post": []
            },
            "keep-sample": {
                "pre": [],
                "post": []
            },
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
                    plugins.push([pluginId, pluginData, false])

                } catch (e) {
                    this.appLogger.log(`Error loading plugin ${pluginId}: ${e}`)
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

        this.plugins.forEach(([pluginId, pluginData, isEnabled], pi) => {
            const record = createElem("div")
            const enabledCkbx = createElem("input", {type: "checkbox"})
            enabledCkbx.checked = isEnabled
            record.appendChild(createElem("div", enabledCkbx))
            record.appendChild(createElem("div", `${pi}`))
            record.appendChild(createElem("div", pluginData["plugin-name"]))

            const hasBackendScript = !!Object.keys(pluginData["back-end-entry-file"]).find(key => pluginData["back-end-entry-file"][key]["pre"]["file"] || pluginData["back-end-entry-file"][key]["post"]["file"])
            const hasFrontendScript = !!pluginData["front-end-entry-file"]
            const type =  hasFrontendScript && hasBackendScript ? "Both": (!hasFrontendScript && !hasBackendScript ? "None" : (hasFrontendScript ? "Front" : "Back"))

            // console.log(pluginId, pluginData)

            record.appendChild(createElem("div", pluginData["plugin-version"]))
            record.appendChild(createElem("div", type))
            record.appendChild(createElem("div", pluginData["plugin-short-description"]))
            record.appendChild(createElem("div", pluginId))
            pluginsRecordsContainer.appendChild(record)

            enabledCkbx.addEventListener("click", () => {
                this.plugins[pi][2] = enabledCkbx.checked
                plugins_applyBtn.disabled = false
            })

            record.addEventListener("click", (e) => {

                if (e.target==enabledCkbx) {
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
        this.savePlugins()
        this.resetModules()
        this.loadModules()
        plugins_applyBtn.disabled = true

        fetch(`http://localhost:8008/refreshPlugins`, {
            method: "Post",
            body: "{}"
        }).then(r=>r.text()).then(status => {

            const plugins = status.split(",")
            const successful = plugins.filter(p => p=="OK")
            const failed = plugins.filter(p => p!="OK")

            let message = `Successfully initialized ${successful.length} plugin${successful.length>1||successful.length==0?"s":""}.`
            if (failed.length) {
                if (successful.length==0) {
                    message = ""
                }
                message += ` Failed to initialize the following plugin${failed.length>1?"s":""}: <br>${failed.join("<br>")} <br><br>Check the server.log file for detailed error traces`
            }

            if (!status.length || successful.length==0 && failed.length==0) {
                message = "Sucess. No plugins active."
            }

            window.errorModal(message)
        })
    }


    loadModules () {
        this.plugins.forEach(([pluginId, pluginData, enabled]) => {
            if (!enabled) return
            this.loadModuleFns(pluginId, pluginData, "start", "pre")
            this.loadModuleFns(pluginId, pluginData, "start", "post")
            this.loadModuleFns(pluginId, pluginData, "keep-sample", "pre")
            this.loadModuleFns(pluginId, pluginData, "keep-sample", "post")

            if (Object.keys(pluginData).includes("front-end-style-files") && pluginData["front-end-style-files"].length) {
                pluginData["front-end-style-files"].forEach(styleFile => {
                    try {
                        if (styleFile.endsWith(".css")) {
                            const styleData = fs.readFileSync(`${this.path}/plugins/${pluginId}/${styleFile}`)
                            pluginsCSS.innerHTML += styleData
                        }
                    } catch (e) {
                        window.appLogger.log(`Error loading style file for plugin ${pluginId}: ${e}`)
                    }
                })

            }
        })
    }

    loadModuleFns (pluginId, pluginData, task, hookTime) {
        try {
            if (Object.keys(pluginData).includes("front-end-entry-file") && Object.keys(pluginData["front-end-entry-file"]).includes(task) && Object.keys(pluginData["front-end-entry-file"][task]).includes(hookTime) ) {

                const file = pluginData["front-end-entry-file"][task][hookTime]["file"]
                const functionName = pluginData["front-end-entry-file"][task][hookTime]["function"]

                if (!file.endsWith(".js")) {
                    window.appLogger.log(`[Plugin: ${pluginId}]: Cannot import ${file} file for ${hookTime} ${task} entry-point: Only JavaScript files are supported right now.`)
                    return
                }

                if (file && functionName) {
                    const module = require(`${self.path}/plugins/${pluginId}/${file}`)

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
            console.log(`Error loading plugin ${pluginId}->${task}->${hookTime}: ` + e)
            window.appLogger.log(`Error loading plugin ${pluginId}->${task}->${hookTime}: ` + e)
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
                window.appLogger.log(`[Plugin run error at event "${event}": ${pluginId}]: ${e}`)
            }
        })
    }

}



exports.PluginsManager = PluginsManager
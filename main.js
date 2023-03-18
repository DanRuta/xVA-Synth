
const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "resources/app" : "."

const remoteMain = require("@electron/remote/main")
remoteMain.initialize()

const fs = require("fs")
const {shell, app, BrowserWindow, ipcMain, Menu} = require("electron")
const {spawn} = require("child_process")

let pythonProcess

if (PRODUCTION) {
    // pythonProcess = spawn(`${path}/cpython/server.exe`, {stdio: "ignore"})
} else {
    pythonProcess = spawn("python", [`${path}/server.py`], {stdio: "ignore"})
}

let mainWindow
let discordClient
let discordClientStart = Date.now()

const createWindow = () => {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 1000,
        minHeight: 700,
        minWidth: 1200,
        frame: false,
        webPreferences: {
            nodeIntegration: true,
            enableRemoteModule: true,
            contextIsolation: false
        },
        icon: `${__dirname}/assets/x-icon.png`,
        // show: false,
    })
    remoteMain.enable(mainWindow.webContents)

    app.on('browser-window-created', (_, window) => {
        require("@electron/remote/main").enable(mainWindow.webContents)
    })

    mainWindow.loadFile("index.html")
    mainWindow.shell = shell

    mainWindow.on("ready-to-show", () => {
        mainWindow.show()
    })

    mainWindow.on("closed", () => mainWindow = null)
}

ipcMain.on("resize", (event, arg) => {
    mainWindow.setSize(arg.width, arg.height)
})
ipcMain.on("updatePosition", (event, arg) => {
    const bounds = mainWindow.getBounds()
    bounds.x = parseInt(arg.details[0])
    bounds.y = parseInt(arg.details[1])
    mainWindow.setBounds(bounds)
})
ipcMain.on("updateDiscord", (event, arg) => {

    // Disconnect if turned off
    if (!Object.keys(arg).includes("details")) {
        if (discordClient) {
            try {
                discordClient.disconnect()
            } catch (e) {}
        }
        discordClient = undefined
        return
    }

    if (!discordClient) {
        discordClient = require('discord-rich-presence')('885096702648938546')
    }

    discordClient.updatePresence({
        state: 'Generating AI voice acting',
        details: arg.details,
        startTimestamp: discordClientStart,
        largeImageKey: 'xvasynth_512_512',
        largeImageText: "xVASynth",
        smallImageKey: 'xvasynth_512_512',
        smallImageText: "xVASynth",
        instance: true,
    })
})
ipcMain.on("show-context-menu-editor", (event) => {
    const template = [
        {
            label: 'Copy ARPAbet [v3]',
            click: () => { event.sender.send('context-menu-command', 'context-copy-editor') }
        },
    ]
    const menu = Menu.buildFromTemplate(template)
    menu.popup(BrowserWindow.fromWebContents(event.sender))
})

ipcMain.on("show-context-menu", (event) => {
    const template = [
        {
            label: 'Copy',
            click: () => { event.sender.send('context-menu-command', 'context-copy') }
        },
        {
            label: 'Paste',
            click: () => { event.sender.send('context-menu-command', 'context-paste') }
        },
        // { type: 'separator' },
    ]
    const menu = Menu.buildFromTemplate(template)
    menu.popup(BrowserWindow.fromWebContents(event.sender))
})

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on("ready", createWindow)


// Quit when all windows are closed.
app.on("window-all-closed", () => {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== "darwin") {
        app.quit()
    }
})

app.on("activate", () => {
// On OS X it"s common to re-create a window in the app when the
// dock icon is clicked and there are no other windows open.
    if (mainWindow === null) {
        createWindow()
    }
})
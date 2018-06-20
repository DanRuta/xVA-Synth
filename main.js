const fs = require("fs")
const {shell, app, BrowserWindow} = require("electron")

let mainWindow

const createWindow = () => {

    mainWindow = new BrowserWindow({width: 800, height: 600, "minHeight": 510, "minWidth": 960, frame: false})

    // mainWindow.setMenu(null)
    mainWindow.loadFile("index.html")
    mainWindow.shell = shell

    // Emitted when the window is closed.
    mainWindow.on("closed", () => mainWindow = null)
}

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
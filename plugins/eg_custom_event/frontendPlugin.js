"use strict"


const postStartFn = (window, data) => {
    window.appLogger.log("postStartFn")

    const button = document.createElement("button")
    button.innerHTML = "Custom event"
    // Style the button with the current game's colour
    button.style.background = `#${window.currentGame[1]}`

    adv_opts.children[1].appendChild(button)

    button.addEventListener("click", () => {

        fetch(`http://localhost:8008/customEvent`, {
            method: "Post",
            body: JSON.stringify({
                pluginId: "eg_custom_event",
                data1: "some data",
                data2: "some more data",
                // ....
            })
        }).then(r=>r.text()).then(() => {
            window.appLogger.log("custom event finished")
            console.log("custom event finished")
        })
    })

}


exports.postStartFn = postStartFn

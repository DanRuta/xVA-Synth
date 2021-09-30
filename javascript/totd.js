"use strict"
// tip of the day

const tips = {
    "1": window.i18n.TOTD_1,
    "2": window.i18n.TOTD_2,
    "3": window.i18n.TOTD_3,
    "4": window.i18n.TOTD_4,
    "5": window.i18n.TOTD_5,
    "6": window.i18n.TOTD_6,
    "7": window.i18n.TOTD_7,
    "8": window.i18n.TOTD_8,
    "9": window.i18n.TOTD_9,
    "10": window.i18n.TOTD_10,
    "11": window.i18n.TOTD_11,
    "12": window.i18n.TOTD_12,
    "13": window.i18n.TOTD_13,
    "14": window.i18n.TOTD_14,
    "15": window.i18n.TOTD_15,
    "16": window.i18n.TOTD_16,
    "17": window.i18n.TOTD_17,
    "18": window.i18n.TOTD_18,
    "19": window.i18n.TOTD_19,
    "20": window.i18n.TOTD_20,
    "21": window.i18n.TOTD_21,
    "22": window.i18n.TOTD_22,
    "23": window.i18n.TOTD_23,
    "24": window.i18n.TOTD_24,
    "25": window.i18n.TOTD_25,
    "26": window.i18n.TOTD_26,
    "27": window.i18n.TOTD_27,
    "28": window.i18n.TOTD_28,
    "29": window.i18n.TOTD_29,
    "30": window.i18n.TOTD_30
}

window.totd_state = {
    startupChecked: false,
    filteredIDs: [],
    tipPageIndex: 0
}

const initTipOfTheDayMenu = (now, tipIDs) => {

    window.totd_state.filteredIDs = tipIDs

    totdContainer.style.opacity = 1
    totdContainer.style.display = "flex"
    chromeBar.style.opacity = 1
    requestAnimationFrame(() => requestAnimationFrame(() => totdContainer.style.opacity = 1))

    return new Promise(resolve => {
        // Close button
        totd_close.addEventListener("click", () => {
            closeModal(totdContainer)
            resolve()
        })

        localStorage.setItem("totd_lastDate", now.toJSON(now))
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[0]]

        totd_counter.innerHTML = `1/${window.totd_state.filteredIDs.length}`

        saveSeenTip(window.totd_state.filteredIDs[0])
    })
}


const saveSeenTip = ID => {
    let seenTipIDs = localStorage.getItem("totd_seenIDs")
    seenTipIDs = seenTipIDs ? seenTipIDs.split(",") : []
    seenTipIDs = new Set(seenTipIDs)
    seenTipIDs.add(ID)
    localStorage.setItem("totd_seenIDs", Array.from(seenTipIDs).join(","))
}

setting_btnShowTOTD.addEventListener("click", () => {
    window.showTipIfEnabledAndNewDay(true)
})


totdPrevTipBtn.addEventListener("click", () => {
    const newIndex = Math.max(0, window.totd_state.tipPageIndex-1)
    if (newIndex!=window.totd_state.tipPageIndex) {
        window.totd_state.tipPageIndex = newIndex
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[window.totd_state.tipPageIndex]]
        saveSeenTip(window.totd_state.filteredIDs[window.totd_state.tipPageIndex])
        totd_counter.innerHTML = `${window.totd_state.tipPageIndex+1}/${window.totd_state.filteredIDs.length}`
    }
})

totdNextTipBtn.addEventListener("click", () => {
    const newIndex = Math.min(window.totd_state.filteredIDs.length-1, window.totd_state.tipPageIndex+1)
    if (newIndex!=window.totd_state.tipPageIndex) {
        window.totd_state.tipPageIndex = newIndex
        tipMessage.innerHTML = tips[window.totd_state.filteredIDs[window.totd_state.tipPageIndex]]
        saveSeenTip(window.totd_state.filteredIDs[window.totd_state.tipPageIndex])
        totd_counter.innerHTML = `${window.totd_state.tipPageIndex+1}/${window.totd_state.filteredIDs.length}`
    }
})


window.showTipIfEnabledAndNewDay = (justShowIt) => {

    window.totd_state.startupChecked = true

    return new Promise(async resolve => {
        const lastDate = localStorage.getItem("totd_lastDate")
        const now = new Date()

        // If this has never happened before, or the last date is not today, then show the tip menu
        if (justShowIt || !lastDate || lastDate.split("T")[0]!=now.toJSON().split("T")[0]) {

            // If the tips of the day are enabled
            if (justShowIt || window.userSettings.showTipOfTheDay) {

                let shuffledTipIDs = window.shuffle(Object.keys(tips))

                // If only new/unseen tips are to be shown, get the seen list, and filter out the seen tips
                if (window.userSettings.showUnseenTipOfTheDay) {
                    let seenTipIDs = localStorage.getItem("totd_seenIDs")
                    if (seenTipIDs) {
                        seenTipIDs = seenTipIDs.split(",")
                        shuffledTipIDs = shuffledTipIDs.filter(id => !seenTipIDs.includes(id))
                    }
                }

                // If there are any tips remaining, after any filtering, then show the menu
                if (shuffledTipIDs && shuffledTipIDs.length) {
                    await initTipOfTheDayMenu(now, shuffledTipIDs)
                    resolve()
                } else if (justShowIt) {
                    window.errorModal(window.i18n.TOTD_NO_UNSEEN)
                }
            } else {
                resolve()
            }
        } else {
            resolve()
        }
    })
}


"use strict"

// https://stackoverflow.com/questions/18052762/remove-directory-which-is-not-empty
const path = require('path');
const deleteFolderRecursive = function (directoryPath, keepRoot=false) {
if (fs.existsSync(directoryPath)) {
    fs.readdirSync(directoryPath).forEach((file, index) => {
      const curPath = path.join(directoryPath, file);
      if (fs.lstatSync(curPath).isDirectory()) {
       // recurse
        deleteFolderRecursive(curPath);
      } else {
        // delete file
        fs.unlinkSync(curPath);
      }
    });
    if (!keepRoot) {
        fs.rmdirSync(directoryPath);
    }
  }
};


window.batch_state = {
    lines: [],
    lastModel: undefined,
    lastVocoder: undefined,
    lineIndex: 0,
    status: false,
    outPathsChecked: [],
    skippedExisting: 0
}

// https://stackoverflow.com/questions/1293147/example-javascript-code-to-parse-csv-data
function CSVToArray( strData, strDelimiter ){
    // Check to see if the delimiter is defined. If not,
    // then default to comma.
    strDelimiter = (strDelimiter || ",");

    // Create a regular expression to parse the CSV values.
    var objPattern = new RegExp(
        (
            // Delimiters.
            "(\\" + strDelimiter + "|\\r?\\n|\\r|^)" +

            // Quoted fields.
            "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +

            // Standard fields.
            "([^\"\\" + strDelimiter + "\\r\\n]*))"
        ),
        "gi"
        );

    // Create an array to hold our data. Give the array
    // a default empty first row.
    var arrData = [[]];

    // Create an array to hold our individual pattern
    // matching groups.
    var arrMatches = null;

    // Keep looping over the regular expression matches
    // until we can no longer find a match.
    while (arrMatches = objPattern.exec( strData )){

        // Get the delimiter that was found.
        var strMatchedDelimiter = arrMatches[ 1 ];

        // Check to see if the given delimiter has a length
        // (is not the start of string) and if it matches
        // field delimiter. If id does not, then we know
        // that this delimiter is a row delimiter.
        if (
            strMatchedDelimiter.length &&
            strMatchedDelimiter !== strDelimiter
            ){
            // Since we have reached a new row of data,
            // add an empty row to our data array.
            arrData.push( [] );
        }

        var strMatchedValue;

        // Now that we have our delimiter out of the way,
        // let's check to see which kind of value we
        // captured (quoted or unquoted).
        if (arrMatches[ 2 ]){
            // We found a quoted value. When we capture
            // this value, unescape any double quotes.
            strMatchedValue = arrMatches[ 2 ].replace(
                new RegExp( "\"\"", "g" ),
                "\""
                );
        } else {
            // We found a non-quoted value.
            strMatchedValue = arrMatches[ 3 ];

        }

        // Now that we have our value string, let's add
        // it to the data array.
        arrData[ arrData.length - 1 ].push( strMatchedValue );
    }

    // Return the parsed data.
    return( arrData );
}


batch_generateSample.addEventListener("click", () => {
    // const lines = []
    const csv = ["game_id,voice_id,text,vocoder,out_path,pacing"] // TODO: ffmpeg options
    const games = Object.keys(window.games)

    if (games.length==0) {
        window.errorModal(window.i18n.BATCH_ERR_NO_VOICES)
        return
    }

    const sampleText = [
        "Include as many lines of text you wish with one line of data per line of voice to be read out.",
        "Make sure that the required columns (game_id, voice_id, and text) are filled out",
        "The others can be left blank, and the app will figure out some sensible defaults",
        "The valid options for vocoder are one of: quickanddirty, waveglow, waveglowBIG, hifi",
        "If your specified model does not have a bespoke hifi model, it will use the waveglow model, also the default if you leave this blank.",
        "For all the other options, you can leave them blank.",
        "If no output path is specified for a specific voice, the default batch output directory will be used",
    ]

    sampleText.forEach(line => {
        const game = games[parseInt(Math.random()*games.length)]
        const gameModels = window.games[game].models
        const model = gameModels[parseInt(Math.random()*gameModels.length)]

        console.log("game", game, "model", model)

        const record = {
            game_id: game,
            voice_id: model.voiceId,
            text: line,
            vocoder: ["quickanddirty","waveglow","waveglowBIG","hifi"][parseInt(Math.random()*4)],
            out_path: "",
            pacing: 1
        }
        // lines.push(record)
        csv.push(Object.values(record).map(v => typeof v =="string" ? `"${v}"` : v).join(","))
    })

    const out_directory = `${__dirname.replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app")
    if (!fs.existsSync(out_directory)){
        fs.mkdirSync(out_directory)
    }
    fs.writeFileSync(`${out_directory}/sample.csv`, csv.join("\n"))
    shell.showItemInFolder(`${out_directory}/sample.csv`)
})

const readFileTxt = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
            lines.forEach(line => {
                if (line.trim().length) {
                    const record = {}
                    record.game_id = window.currentModel.games[0].gameId
                    record.voice_id = window.currentModel.games[0].voiceId
                    record.text = line
                    if (window.currentModel.hifi) {
                        record.vocoder = "hifi"
                    }
                    dataLines.push(record)
                }
            })
            resolve(dataLines)
        }
    })
}

const readFile = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
            const header = lines.shift().split(",").map(head => head.replace(/\r/, ""))
            lines.forEach(line => {
                const record = {}
                if (line.trim().length) {
                    const parts = CSVToArray(line)[0]
                    parts.forEach((val, vi) => {
                        record[header[vi].replace(/^"/, "").replace(/"$/, "")] = val
                    })
                    dataLines.push(record)
                }
            })
            resolve(dataLines)
        }
    })
}

const uploadBatchCSVs = async (eType, event) => {

    if (["dragenter", "dragover"].includes(eType)) {
        batch_main.style.background = "#5b5b5b"
        batch_main.style.color = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        batch_main.style.background = "#4b4b4b"
        batch_main.style.color = "gray"
    }

    event.preventDefault()
    event.stopPropagation()

    const dataLines = []

    if (eType=="drop") {

        batchDropZoneNote.innerHTML = window.i18n.PROCESSING_DATA
        window.batch_state.skippedExisting = 0

        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)
        for (let fi=0; fi<files.length; fi++) {
            const file = files[fi]
            if (!file.name.endsWith(".csv")) {
                if (file.name.toLowerCase().endsWith(".txt")) {
                    if (window.currentModel) {
                        const records = await readFileTxt(file)
                        records.forEach(item => {
                            if (window.userSettings.batch_skipExisting) {
                                let outPath

                                if (item.out_path && item.out_path.split("/").reverse()[0].includes(".")) {
                                    outPath = item.out_path
                                } else {
                                    if (item.out_path) {
                                        outPath = item.out_path
                                    } else {
                                        outPath = window.userSettings.batchOutFolder
                                    }
                                    outPath = `${outPath}/${item.voice_id}_${item.vocoder}_${item.text.replace(/[\/\\:\*?<>"|]*/g, "").slice(0, 75).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                                }

                                outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath

                                if (!fs.existsSync(outPath)) {
                                    dataLines.push(item)
                                } else {
                                    window.batch_state.skippedExisting++
                                }

                            } else {
                                dataLines.push(item)
                            }
                        })
                    }
                    continue
                } else {
                    continue
                }
            }

            const records = await readFile(file)
            records.forEach(item => {

                if (window.userSettings.batch_skipExisting) {
                    let outPath

                    if (item.out_path && item.out_path.split("/").reverse()[0].includes(".")) {
                        outPath = item.out_path
                    } else {
                        if (item.out_path) {
                            outPath = item.out_path
                        } else {
                            outPath = window.userSettings.batchOutFolder
                        }
                        outPath = `${outPath}/${item.voice_id}_${item.vocoder}_${item.text.replace(/[\/\\:\*?<>"|]*/g, "").slice(0, 75).replace(/\.$/, "")}.${window.userSettings.audio.format}`
                    }

                    outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath

                    if (!fs.existsSync(outPath)) {
                        dataLines.push(item)
                    } else {
                        window.batch_state.skippedExisting++
                    }
                } else {
                    dataLines.push(item)
                }
            })
        }

        if (dataLines.length==0 && window.batch_state.skippedExisting) {
            batchDropZoneNote.innerHTML = window.i18n.BATCH_DROPZONE
            return window.errorModal(window.i18n.BATCH_ERR_SKIPPEDALL.replace("_1", window.batch_state.skippedExisting))
        }

        const cleanedData = preProcessCSVData(dataLines)
        if (cleanedData.length) {
            populateRecordsList(cleanedData)
            refreshRecordsList()
        } else {
            batch_clearBtn.click()
        }

        batchDropZoneNote.innerHTML = window.i18n.BATCH_DROPZONE
    }
}

const preProcessCSVData = data => {

    batch_main.style.display = "block"
    batchDropZoneNote.style.display = "none"
    batchRecordsHeader.style.display = "flex"
    batch_clearBtn.style.display = "inline-block"
    Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${window.currentGame[1]}`)

    const availableGames = Object.keys(window.games)
    for (let di=0; di<data.length; di++) {
        try {
            const record = data[di]

            // Validate the records first
            // ==================
            if (!record.game_id) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} game_id`)
                return []
            }
            if (!record.voice_id) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} voice_id`)
                return []
            }
            if (!record.text || record.text.length==0) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.MISSING} text`)
                return []
            }

            // Check that the game_id exists
            if (!availableGames.includes(record.game_id)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: game_id "${record.game_id}" ${window.i18n.BATCH_ERR_GAMEID} (${availableGames.join(',')})`)
                return []
            }
            // Check that the voice_id exists
            const gameVoices = window.games[record.game_id].models.map(item => item.voiceId)
            if (!gameVoices.includes(record.voice_id)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: voice_id "${record.voice_id}" ${window.i18n.BATCH_ERR_VOICEID}: ${record.game_id}`)
                return []
            }
            // Check that the vocoder exists
            if (!["quickanddirty", "waveglow", "waveglowBIG", "hifi", undefined].includes(record.vocoder)) {
                window.errorModal(`[${window.i18n.LINE}: ${di+2}] ${window.i18n.ERROR}: ${window.i18n.BATCHH_VOCODER} "${record.vocoder}" ${window.i18n.BATCH_ERR_VOCODER1}: quickanddirty, waveglow, waveglowBIG, hifi ${window.i18n.BATCH_ERR_VOCODER2}`)
                return []
            }

            // Fill with defaults
            // ==================
            if (!record.out_path) {
                record.out_path = window.userSettings.batchOutFolder
            }
            if (!record.pacing) {
                record.pacing = 1
            }
            record.pacing = parseFloat(record.pacing)
            if (!record.vocoder || (record.vocoder=="hifi" && !window.games[record.game_id].models.find(rec => rec.voiceId==record.voice_id).hifi)) {
                record.vocoder = "waveglow"
            }
        } catch (e) {
            console.log(e)
            window.appLogger.log(e)
            console.log(data[di])
            console.log(window.games[data[di].game_id])
            console.log(window.games[data[di].game_id].models.find(rec => rec.voiceId==data[di].voice_id))
        }
    }

    return data
}

const populateRecordsList = records => {
    batch_synthesizeBtn.style.display = "inline-block"
    batchDropZoneNote.style.display = "none"

    records.forEach((record, ri) => {
        const row = createElem("div")

        const rNumElem = createElem("div", batchRecordsContainer.children.length.toString())
        const rStatusElem = createElem("div", "Ready")
        const rActionsElem = createElem("div")
        const rVoiceElem = createElem("div", record.voice_id)
        const rTextElem = createElem("div", record.text)
        rTextElem.title = record.text
        const rGameElem = createElem("div", record.game_id)
        const rVocoderElem = createElem("div", record.vocoder)
        const rOutPathElem = createElem("div", "&lrm;"+record.out_path+"&lrm;")
        rOutPathElem.title = record.out_path
        const rPacingElem = createElem("div", (record.pacing||" ").toString())


        row.appendChild(rNumElem)
        row.appendChild(rStatusElem)
        row.appendChild(rActionsElem)
        row.appendChild(rVoiceElem)
        row.appendChild(rTextElem)
        row.appendChild(rGameElem)
        row.appendChild(rVocoderElem)
        row.appendChild(rOutPathElem)
        row.appendChild(rPacingElem)

        window.batch_state.lines.push([record, row, ri])
    })
}

const refreshRecordsList = () => {
    batchRecordsContainer.innerHTML = ""
    const finalOrder = groupLines()
    finalOrder.forEach(recordAndElem => {
        recordAndElem[1].children[0].innerHTML = batchRecordsContainer.children.length.toString()
        batchRecordsContainer.appendChild(recordAndElem[1])
    })
    window.batch_state.lines = finalOrder
}

// Sort the lines by voice_id, and then by vocoder used
const groupLines = () => {
    const voices_order = []

    const lines = window.batch_state.lines.sort((a,b) => {
        return a.voice_id - b.voice_id
    })

    const voices_groups = {}

    // Get the order of the voice_id, and group them up
    window.batch_state.lines.forEach(record => {
        if (!voices_order.includes(record[0].voice_id)) {
            voices_order.push(record[0].voice_id)
            voices_groups[record[0].voice_id] = []
        }
        voices_groups[record[0].voice_id].push(record)
    })

    // Go through the voice groups and sort them by vocoder
    voices_order.forEach(voice_id => {
        voices_groups[voice_id] = voices_groups[voice_id].sort((a,b) => a[0].vocoder<b[0].vocoder?1:-1)
    })

    // Collate everything back into the final order
    const finalOrder = []
    voices_order.forEach(voice_id => {
        voices_groups[voice_id].forEach(record => finalOrder.push(record))
    })

    return finalOrder
}


batch_clearBtn.addEventListener("click", () => {

    window.batch_state.lines = []
    batch_main.style.display = "flex"
    batchDropZoneNote.style.display = "block"
    batchRecordsHeader.style.display = "none"
    batch_clearBtn.style.display = "none"
    batch_outputFolderInput.style.display = "inline-block"
    batch_clearDirOpts.style.display = "flex"
    batch_skipExistingOpts.style.display = "flex"
    batch_progressItems.style.display = "none"
    batch_progressBar.style.display = "none"

    batch_pauseBtn.style.display = "none"
    batch_stopBtn.style.display = "none"
    batch_synthesizeBtn.style.display = "none"

    batchRecordsContainer.innerHTML = ""
})


const startBatch = () => {

    // Output directory
    if (!fs.existsSync(window.userSettings.batchOutFolder)) {
        window.userSettings.batchOutFolder.split("/")
         .reduce((prevPath, folder) => {
           const currentPath = path.join(prevPath, folder, path.sep);
           if (!fs.existsSync(currentPath)){
             fs.mkdirSync(currentPath);
           }
           return currentPath;
         }, '');
    }
    if (batch_clearDirFirstCkbx.checked) {
        deleteFolderRecursive(window.userSettings.batchOutFolder, true)
    }

    batch_synthesizeBtn.style.display = "none"
    batch_clearBtn.style.display = "none"
    batch_outputFolderInput.style.display = "none"
    batch_clearDirOpts.style.display = "none"
    batch_skipExistingOpts.style.display = "none"
    batch_progressItems.style.display = "flex"
    batch_progressBar.style.display = "flex"
    batch_pauseBtn.style.display = "inline-block"
    batch_stopBtn.style.display = "inline-block"
    batch_openDirBtn.style.display = "none"

    window.batch_state.lines.forEach(record => {
        record[1].children[1].innerHTML = window.i18n.READY
        record[1].children[1].style.background = "none"
    })

    window.batch_state.lineIndex = 0
    window.batch_state.state = true
    window.batch_state.outPathsChecked = []
    performSynthesis()
}

const batchChangeVoice = (game, voice) => {
    return new Promise((resolve) => {

        // Update the main app with any changes, if a voice has already been selected
        if (window.currentModel) {
            generateVoiceButton.innerHTML = window.i18n.LOAD_MODEL
            keepSampleButton.style.display = "none"
            samplePlay.style.display = "none"

            const modelGameFolder = window.currentModel.audioPreviewPath.split("/")[0]
            const modelFileName = window.currentModel.audioPreviewPath.split("/")[1].split(".wav")[0]
            generateVoiceButton.dataset.modelQuery = JSON.stringify({
                outputs: parseInt(window.currentModel.outputs),
                model: `${window.path}/models/${modelGameFolder}/${modelFileName}`,
                model_speakers: window.currentModel.emb_size,
                cmudict: window.currentModel.cmudict
            })
        }


        if (window.batch_state.state) {
            batch_progressNotes.innerHTML = `${window.i18n.BATCH_CHANGING_MODEL_TO}: ${voice}`
        }

        const model = window.games[game].models.find(model => model.voiceId==voice).model

        fetch(`http://localhost:8008/loadModel`, {
            method: "Post",
            body: JSON.stringify({"outputs": null, "model": `${window.userSettings[`modelspath_${game}`]}/${voice}`, "model_speakers": model.emb_size})
        }).then(r=>r.text()).then(res => {
            resolve()
        }).catch(e => {
            console.log(e)
            window.appLogger.log(e)
            batch_pauseBtn.click()
            if (e.code =="ENOENT") {
                closeModal().then(() => {
                    createModal("error", window.i18n.ERR_SERVER)
                })
            }
        })
    })
}
const batchChangeVocoder = (vocoder, game, voice) => {
    return new Promise((resolve) => {
        console.log("Changing vocoder: ", vocoder)
        if (window.batch_state.state) {
            batch_progressNotes.innerHTML = `${window.i18n.BATCH_CHANGING_VOCODER_TO}: ${vocoder}`
        }

        const vocoderMappings = [["waveglow", "256_waveglow"], ["waveglowBIG", "big_waveglow"], ["quickanddirty", "qnd"], ["hifi", `${game}/${voice}.hg.pt`]]

        fetch(`http://localhost:8008/setVocoder`, {
            method: "Post",
            body: JSON.stringify({vocoder: vocoderMappings.find(record => record[0]==vocoder)[1]})
        }).then(() => {
            resolve()
        }).catch(e => {
            console.log(e)
            window.appLogger.log(e)
            window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}:<br><br>`+e)
            batch_pauseBtn.click()
        })
    })
}


const prepareLinesBatchForSynth = () => {

    const linesBatch = []
    const records = []
    let firstItemVoiceId = undefined
    let firstItemVocoder = undefined
    let speaker_i = undefined

    for (let i=0; i<Math.min(window.userSettings.batch_batchSize, window.batch_state.lines.length-window.batch_state.lineIndex); i++) {

        const record = window.batch_state.lines[window.batch_state.lineIndex+i]

        const vocoderMappings = [["waveglow", "256_waveglow"], ["waveglowBIG", "big_waveglow"], ["quickanddirty", "qnd"], ["hifi", `${record[0].game_id}/${record[0].voice_id}.hg.pt`]]
        const vocoder = vocoderMappings.find(voc => voc[0]==record[0].vocoder)[1]

        if (firstItemVoiceId==undefined) firstItemVoiceId = record[0].voice_id
        if (firstItemVocoder==undefined) firstItemVocoder = vocoder

        if (record[0].voice_id!=firstItemVoiceId || vocoder!=firstItemVocoder) {
            break
        }

        const model = window.games[record[0].game_id].models.find(model => model.voiceId==record[0].voice_id).model

        const sequence = record[0].text
        const pitch = undefined // maybe later
        const duration = undefined // maybe later
        speaker_i = model.games[0].emb_i
        const pace = record[0].pacing

        const tempFileNum = `${Math.random().toString().split(".")[1]}`
        const tempFileLocation = `${window.path}/output/temp-${tempFileNum}.wav`

        let outPath
        let outFolder

        if (record[0].out_path.split("/").reverse()[0].includes(".")) {
            outPath = record[0].out_path
            outFolder = String(record[0].out_path).split("/").reverse().slice(1,10000).reverse().join("/")
        } else {
            outPath = `${record[0].out_path}/${record[0].voice_id}_${record[0].vocoder}_${sequence.replace(/[\/\\:\*?<>"|]*/g, "")}.${window.userSettings.audio.format}`
            outFolder = record[0].out_path
        }
        outFolder = outFolder.length ? outFolder : window.userSettings.batchOutFolder

        linesBatch.push([sequence, pitch, duration, pace, tempFileLocation, outPath, outFolder])
        records.push(record)
    }

    return [speaker_i, firstItemVoiceId, firstItemVocoder, linesBatch, records]
}


const addActionButtons = (records, ri) => {

    let audioPreview
    const playButton = createElem("button.smallButton", window.i18n.PLAY)
    playButton.style.background = `#${window.currentGame[1]}`
    playButton.addEventListener("click", () => {

        let audioPreviewPath = records[ri][0].fileOutputPath
        if (audioPreviewPath.startsWith("./")) {
            audioPreviewPath = window.userSettings.batchOutFolder + audioPreviewPath.replace("./", "/")
        }

        if (audioPreview==undefined) {
            const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {
                src: audioPreviewPath
            }))
            audioPreview.addEventListener("play", () => {
                if (window.ctrlKeyIsPressed) {
                    audioPreview.setSinkId(window.userSettings.alt_speaker)
                } else {
                    audioPreview.setSinkId(window.userSettings.base_speaker)
                }
            })
            audioPreview.setSinkId(window.userSettings.base_speaker)
        }
    })
    const editButton = createElem("button.smallButton", window.i18n.EDIT)
    editButton.style.background = `#${window.currentGame[1]}`
    editButton.addEventListener("click", () => {
        audioPreview = undefined


        if (window.batch_state.state) {
            window.errorModal(window.i18n.BATCH_ERR_EDIT)
            return
        }

        // Change app theme to the voice's game
        window.changeGame(window.gameAssets[records[ri][0].game_id])

        // Simulate voice loading through the UI
        const voiceName = window.games[records[ri][0].game_id].models.find(model => model.voiceId==records[ri][0].voice_id).voiceName
        const voiceButton = Array.from(voiceTypeContainer.children).find(button => button.innerHTML==voiceName)
        voiceButton.click()
        generateVoiceButton.click()
        dialogueInput.value = records[ri][0].text

        let audioPreviewPath = records[ri][0].fileOutputPath
        if (audioPreviewPath.startsWith("./")) {
            audioPreviewPath = window.userSettings.batchOutFolder + audioPreviewPath.replace("./", "/")
        }
        keepSampleButton.dataset.newFileLocation = "BATCH_EDIT"+audioPreviewPath
    })
    records[ri][1].children[2].appendChild(playButton)
    records[ri][1].children[2].appendChild(editButton)

}


const batchKickOffMPffmpegOutput = (records, tempPaths, outPaths, options) => {
    return new Promise((resolve, reject) => {
        fetch(`http://localhost:8008/batchOutputAudio`, {
            method: "Post",
            body: JSON.stringify({
                input_paths: tempPaths,
                output_paths: outPaths,
                processes: window.userSettings.batch_MPCount,
                options: JSON.stringify(options)
            })
        }).then(r=>r.text()).then(res => {
            res = res.split("\n")
            res.forEach((resItem, ri) => {
                if (resItem.length && resItem!="-") {
                    console.log("resItem", resItem, resItem.length, resItem.length!="-")
                    window.appLogger.log("resItem", resItem)
                    if (window.batch_state.state) {
                        batch_pauseBtn.click()
                    }

                    records[ri][1].children[1].innerHTML = window.i18n.FAILED
                    records[ri][1].children[1].style.background = "red"

                } else {

                    records[ri][1].children[1].innerHTML = window.i18n.DONE
                    records[ri][1].children[1].style.background = "green"
                    fs.unlinkSync(tempPaths[ri])
                    addActionButtons(records, ri)
                }

                window.batch_state.lineIndex += 1
            })
            resolve()

        }).catch(e => {
            console.log("e")
            console.log(e)
        })
    })
}




const batchKickOffFfmpegOutput = (ri, linesBatch, records, tempFileLocation, body) => {
    return new Promise((resolve, reject) => {
        fetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body
        }).then(r=>r.text()).then(res => {
            if (res.length && res!="-") {
                window.appLogger.log("res", res)
                if (window.batch_state.state) {
                    batch_pauseBtn.click()
                }

                for (let ri2=ri; ri2<linesBatch.length; ri2++) {
                    records[ri][1].children[1].innerHTML = window.i18n.FAILED
                    records[ri][1].children[1].style.background = "red"
                }

                reject(res)
            } else {
                records[ri][1].children[1].innerHTML = window.i18n.DONE
                records[ri][1].children[1].style.background = "green"
                fs.unlinkSync(tempFileLocation)
                addActionButtons(records, ri)
                resolve()
            }
        }).catch(e => {
            if (e.code=="ECONNREFUSED") {
                batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, body)
            }
        })

    })

}

const batchKickOffGeneration = () => {
    return new Promise((resolve) => {

        const [speaker_i, voice_id, vocoder, linesBatch, records] = prepareLinesBatchForSynth()
        records.forEach((record, ri) => {
            record[1].children[1].innerHTML = window.i18n.RUNNING
            record[1].children[1].style.background = "goldenrod"
            record[0].fileOutputPath = linesBatch[ri][5]
        })

        const record = window.batch_state.lines[window.batch_state.lineIndex]

        if (window.batch_state.state) {
            if (linesBatch.length==1) {
                batch_progressNotes.innerHTML = `${window.i18n.SYNTHESIZING}: <i>${record[0].text}</i>`
            } else {
                batch_progressNotes.innerHTML = `${window.i18n.SYNTHESIZING} ${linesBatch.length} ${window.i18n.LINES}`
            }
        }
        const batchPostData = {
            batchSize: window.userSettings.batch_batchSize,
            defaultOutFolder: window.userSettings.batchOutFolder,
            speaker_i, vocoder, linesBatch
        }
        fetch(`http://localhost:8008/synthesize_batch`, {
            method: "Post",
            body: JSON.stringify(batchPostData)
        }).then(r=>r.text()).then(async (res) => {

            if (res) {
                if (res=="CUDA OOM") {
                    window.errorModal(window.i18n.BATCH_ERR_CUDA_OOM)
                } else {
                    window.errorModal(res)
                }
                if (window.batch_state.state) {
                    batch_pauseBtn.click()
                }
                return
            }


            // Create the output directory if it does not exist
            linesBatch.forEach(record => {
                let outFolder = record[6].startsWith("./") ? window.userSettings.batchOutFolder + record[6].slice(1,100000) : record[6]

                if (!window.batch_state.outPathsChecked.includes(outFolder)) {
                    window.batch_state.outPathsChecked.push(outFolder)
                    if (!fs.existsSync(outFolder)) {
                        fs.mkdirSync(outFolder)
                    }
                }
            })

            if (window.userSettings.audio.ffmpeg) {
                const options = {
                    hz: window.userSettings.audio.hz,
                    padStart: window.userSettings.audio.padStart,
                    padEnd: window.userSettings.audio.padEnd,
                    bit_depth: window.userSettings.audio.bitdepth,
                    amplitude: window.userSettings.audio.amplitude
                }

                if (window.batch_state.state) {
                    batch_progressNotes.innerHTML = window.i18n.BATCH_OUTPUTTING_FFMPEG
                }

                const tempPaths = linesBatch.map(line => line[4])
                const outPaths = linesBatch.map((line, li) => {
                    let outPath = linesBatch[li][5].includes(":") || linesBatch[li][5].includes("./") ? linesBatch[li][5] : `${linesBatch[li][6]}/${linesBatch[li][5]}`
                    if (outPath.startsWith("./")) {
                        outPath = window.userSettings.batchOutFolder + outPath.slice(1,100000)
                    }
                    return outPath
                })

                if (window.userSettings.batch_useMP) {
                    await batchKickOffMPffmpegOutput(records, tempPaths, outPaths, options)
                } else {
                    for (let ri=0; ri<linesBatch.length; ri++) {
                        let tempFileLocation = tempPaths[ri]
                        let outPath = outPaths[ri]
                        try {
                            if (window.batch_state.state) {
                                records[ri][1].children[1].innerHTML = window.i18n.OUTPUTTING
                                if (window.userSettings.batch_fastMode) {
                                    batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, JSON.stringify({
                                        input_path: tempFileLocation,
                                        output_path: outPath,
                                        options: JSON.stringify(options)
                                    }))
                                } else {
                                    await batchKickOffFfmpegOutput(ri, linesBatch, records, tempFileLocation, JSON.stringify({
                                        input_path: tempFileLocation,
                                        output_path: outPath,
                                        options: JSON.stringify(options)
                                    }))
                                }
                                window.batch_state.lineIndex += 1
                            }
                        } catch (e) {
                            console.log(e)
                            window.errorModal(`${window.i18n.SOMETHING_WENT_WRONG}:<br><br>`+e)
                            resolve()
                        }
                    }
                }
                resolve()
            } else {
                linesBatch.forEach((lineRecord, li) => {
                    let tempFileLocation = lineRecord[4]
                    let outPath = lineRecord[5]
                    try {
                        fs.copyFileSync(tempFileLocation, outPath)
                        records[li][1].children[1].innerHTML = window.i18n.DONE
                        records[li][1].children[1].style.background = "green"

                        window.batch_state.lineIndex += 1

                        addActionButtons(records, li)

                    } catch (err) {
                        console.log(err)
                        window.appLogger.log(err)
                        window.errorModal(err.message)
                        batch_pauseBtn.click()
                    }
                    resolve()
                })
            }
        }).catch(async e => {
            if (e.code=="ECONNREFUSED") {
                await batchKickOffGeneration()
                resolve()
            } else {
                console.log(e)
                window.appLogger.log(e)
                window.errorModal(e.message)
                batch_pauseBtn.click()
            }
        })
    })
}

const performSynthesis = async () => {

    if (!window.batch_state.state) {
        return
    }

    const percentDone = (window.batch_state.lineIndex) / window.batch_state.lines.length * 100
    batch_progressBar.style.background = `linear-gradient(90deg, green ${parseInt(percentDone)}%, rgba(255,255,255,0) ${parseInt(percentDone)}%)`
    batch_progressBar.innerHTML = `${parseInt(percentDone* 100)/100}%`
    window.electronBrowserWindow.setProgressBar(percentDone/100)

    const record = window.batch_state.lines[window.batch_state.lineIndex]

    // Change the voice model if the next line uses a different one
    if (window.batch_state.lastModel!=record[0].voice_id) {
        await batchChangeVoice(record[0].game_id, record[0].voice_id)
        window.batch_state.lastModel = record[0].voice_id
    }

    // Change the vocoder if the next line uses a different one
    if (window.batch_state.lastVocoder!=record[0].vocoder) {
        await batchChangeVocoder(record[0].vocoder, record[0].game_id, record[0].voice_id)
        window.batch_state.lastVocoder = record[0].vocoder
    }

    await batchKickOffGeneration()

    if (window.batch_state.lineIndex==window.batch_state.lines.length) {
        // The end
        batch_stopBtn.click()
        batch_openDirBtn.style.display = "inline-block"
    } else {
        performSynthesis()
    }
}

const pauseResumeBatch = () => {

    batch_progressNotes.innerHTML = window.i18n.PAUSED

    const isRunning = window.batch_state.state
    batch_pauseBtn.innerHTML = isRunning ? window.i18n.RESUME : window.i18n.PAUSE
    window.batch_state.state = !isRunning

    if (!isRunning) {
        performSynthesis()
    }
}

const stopBatch = () => {
    window.electronBrowserWindow.setProgressBar(0)
    window.batch_state.state = false
    window.batch_state.lineIndex = 0

    batch_synthesizeBtn.style.display = "inline-block"
    batch_clearBtn.style.display = "inline-block"
    batch_outputFolderInput.style.display = "inline-block"
    batch_clearDirOpts.style.display = "flex"
    batch_skipExistingOpts.style.display = "flex"
    batch_progressItems.style.display = "none"
    batch_progressBar.style.display = "none"
    batch_pauseBtn.style.display = "none"
    batch_stopBtn.style.display = "none"

    window.batch_state.lines.forEach(record => {
        if (record[1].children[1].innerHTML==window.i18n.READY || record[1].children[1].innerHTML==window.i18n.RUNNING) {
            record[1].children[1].innerHTML = window.i18n.STOPPED
            record[1].children[1].style.background = "none"
        }
    })
}


const openOutput = () => {
    shell.showItemInFolder(window.userSettings.batchOutFolder+"/dummy.txt")
}

batch_main.addEventListener("dragenter", event => uploadBatchCSVs("dragenter", event), false)
batch_main.addEventListener("dragleave", event => uploadBatchCSVs("dragleave", event), false)
batch_main.addEventListener("dragover", event => uploadBatchCSVs("dragover", event), false)
batch_main.addEventListener("drop", event => uploadBatchCSVs("drop", event), false)


batch_synthesizeBtn.addEventListener("click", startBatch)
batch_pauseBtn.addEventListener("click", pauseResumeBatch)
batch_stopBtn.addEventListener("click", stopBatch)
batch_openDirBtn.addEventListener("click", openOutput)

exports.uploadBatchCSVs = uploadBatchCSVs
exports.startBatch = startBatch

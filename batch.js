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
        window.errorModal("No voice models available in the app. Load at least one.")
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
            const lines = reader.result.split("\n")
            const header = lines.shift().split(",").map(head => head.replace(/\r/, ""))
            lines.forEach(line => {
                const record = {}
                if (line.trim().length) {
                    const parts = CSVToArray(line)[0]
                    parts.forEach((val, vi) => {
                        record[header[vi].replace(/^"/, "").replace(/"$/, "")] = val//?val.replace(/^"/, "").replace(/"$/, ""):val
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
        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)
        for (let fi=0; fi<files.length; fi++) {
            const file = files[fi]
            if (!file.name.endsWith(".csv")) {
                if (file.name.endsWith(".txt")) {
                    if (window.currentModel) {
                        const records = await readFileTxt(file)
                        records.forEach(item => dataLines.push(item))
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
                        outPath = `${outPath}/${item.voice_id}_${item.vocoder}_${item.text.replace(/[\/\\:\*?<>"|]*/g, "")}.${window.userSettings.audio.format}`
                    }

                    outPath = outPath.startsWith("./") ? window.userSettings.batchOutFolder + outPath.slice(1,100000) : outPath


                    if (!fs.existsSync(outPath)) {
                        dataLines.push(item)
                    }
                } else {
                    dataLines.push(item)
                }
            })
        }
        console.log("dataLines")
        console.log(dataLines)

        const cleanedData = preProcessCSVData(dataLines)
        if (cleanedData.length) {
            populateRecordsList(cleanedData)
            refreshRecordsList()
        } else {
            batch_clearBtn.click()
        }
    }
}

const preProcessCSVData = data => {

    batch_main.style.display = "block"
    batchDropZoneNote.style.display = "none"
    batchRecordsHeader.style.display = "flex"
    batch_clearBtn.style.display = "inline-block"
    // batchRecordsHeader.style.backgroundColor = `#${window.currentGame[1]}`
    Array.from(batchRecordsHeader.children).forEach(item => item.style.backgroundColor = `#${window.currentGame[1]}`)

    const availableGames = Object.keys(window.games)
    for (let di=0; di<data.length; di++) {
        try {
            const record = data[di]

            // Validate the records first
            // ==================
            if (!record.game_id) {
                // console.log(`[Line: ${di}] ERROR: Missing game_id value`)
                window.errorModal(`[Line: ${di+2}] ERROR: Missing game_id value`)
                return []
            }
            if (!record.voice_id) {
                window.errorModal(`[Line: ${di+2}] ERROR: Missing voice_id value`)
                return []
            }
            if (!record.text || record.text.length==0) {
                window.errorModal(`[Line: ${di+2}] ERROR: Missing text value`)
                return []
            }

            // Check that the game_id exists
            if (!availableGames.includes(record.game_id)) {
                window.errorModal(`[Line: ${di+2}] ERROR: game_id "${record.game_id}" does not match any available games (${availableGames.join(',')})`)
                return []
            }
            // Check that the voice_id exists
            const gameVoices = window.games[record.game_id].models.map(item => item.voiceId)
            if (!gameVoices.includes(record.voice_id)) {
                window.errorModal(`[Line: ${di+2}] ERROR: voice_id "${record.voice_id}" does not match any in the game: ${record.game_id}`)
                return []
            }
            // Check that the vocoder exists
            if (!["quickanddirty", "waveglow", "waveglowBIG", "hifi", undefined].includes(record.vocoder)) {
                window.errorModal(`[Line: ${di+2}] ERROR: Vocoder "${record.vocoder}" does not exist. Available options: quickanddirty, waveglow, waveglowBIG, hifi  (or leaving it blank)`)
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
            // if (!record.vocoder || !window.games[record.game_id].models[record.voice_id].hifi) {
            // console.log(record.vocoder, window.games[record.game_id].models, )
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

        // const rNumElem = createElem("div", ri.toString())
        const rNumElem = createElem("div", batchRecordsContainer.children.length.toString())
        const rStatusElem = createElem("div", "Ready")
        const rGameElem = createElem("div", record.game_id)
        const rVoiceElem = createElem("div", record.voice_id)
        const rTextElem = createElem("div", record.text)
        const rVocoderElem = createElem("div", record.vocoder)
        const rOutPathElem = createElem("div", "&lrm;"+record.out_path+"&lrm;")
        const rPacingElem = createElem("div", (record.pacing||" ").toString())


        row.appendChild(rNumElem)
        row.appendChild(rStatusElem)
        row.appendChild(rGameElem)
        row.appendChild(rVoiceElem)
        row.appendChild(rTextElem)
        row.appendChild(rVocoderElem)
        row.appendChild(rOutPathElem)
        row.appendChild(rPacingElem)

        window.batch_state.lines.push([record, row, ri])
    })
}

const refreshRecordsList = () => {
    batchRecordsContainer.innerHTML = ""
    const finalOrder = groupLines()
    // window.batch_lines.forEach(recordAndElem => {
    finalOrder.forEach(recordAndElem => {
        recordAndElem[1].children[0].innerHTML = batchRecordsContainer.children.length.toString()
        batchRecordsContainer.appendChild(recordAndElem[1])
    })
    window.batch_state.lines = finalOrder
}

// Sort the lines by voice_id, and then by vocoder used
const groupLines = () => {
    const voices_order = []

    // const lines = window.batch_lines.sort((a,b) => {
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
    // batch_openDirBtn.style.display = "none"

    batchRecordsContainer.innerHTML = ""
})

const startBatch = () => {

    // Output directory
    if (!fs.existsSync(window.userSettings.batchOutFolder)) {
        fs.mkdirSync(window.userSettings.batchOutFolder)
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
    // batch_openDirBtn.style.display = "inline-block"
    batch_openDirBtn.style.display = "none"

    window.batch_state.lines.forEach(record => {
        record[1].children[1].innerHTML = "Ready"
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
            generateVoiceButton.innerHTML = "Load model"
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
            batch_progressNotes.innerHTML = `Changing voice model to: ${voice}`
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
                    createModal("error", "There was an issue connecting to the python server.<br><br>Try again in a few seconds. If the issue persists, make sure localhost port 8008 is free, or send the server.log file to me on GitHub or Nexus.")
                })
            }
        })
    })
}
const batchChangeVocoder = (vocoder, game, voice) => {
    return new Promise((resolve) => {
        console.log("Changing vocoder: ", vocoder)
        if (window.batch_state.state) {
            batch_progressNotes.innerHTML = `Changing vocoder to: ${vocoder}`
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
            window.errorModal("Something went wrong:<br><br>"+e)
            batch_pauseBtn.click()
        })
    })
}

const clearOldTempFiles = () => {
    const oldTempFiles = fs.readdirSync(`${window.path}/output`).filter(fileName => fileName.includes("temp-"))
    oldTempFiles.forEach(file => {
        try {
            fs.unlinkSync(`${window.path}/output/${file}`)
        } catch (e) {console.log(e)}
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

        clearOldTempFiles()

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

const batchKickOffFfmpegOutput = (ri, linesBatch, records, body) => {
    return new Promise((resolve, reject) => {
        fetch(`http://localhost:8008/outputAudio`, {
            method: "Post",
            body
        }).then(r=>r.text()).then(res => {
            if (res.length) {
                window.appLogger.log("res", res)
                if (window.batch_state.state) {
                    batch_pauseBtn.click()
                }

                for (let ri2=ri; ri2<linesBatch.length; ri2++) {
                    records[ri][1].children[1].innerHTML = "Ready"
                    records[ri][1].children[1].style.background = "none"
                }

                reject(res)
            } else {
                records[ri][1].children[1].innerHTML = "Done"
                records[ri][1].children[1].style.background = "green"
                window.batch_state.lineIndex += 1
                resolve()
            }
        })
    })

}

const batchKickOffGeneration = () => {
    return new Promise((resolve) => {

        const [speaker_i, voice_id, vocoder, linesBatch, records] = prepareLinesBatchForSynth()
        records.forEach(record => {
            record[1].children[1].innerHTML = "Running"
            record[1].children[1].style.background = "goldenrod"
        })

        const record = window.batch_state.lines[window.batch_state.lineIndex]

        if (window.batch_state.state) {
            if (linesBatch.length==1) {
                batch_progressNotes.innerHTML = `Synthesizing line: <i>${record[0].text}</i>`
            } else {
                batch_progressNotes.innerHTML = `Synthesizing ${linesBatch.length} lines`
            }
        }
        fetch(`http://localhost:8008/synthesize_batch`, {
            method: "Post",
            body: JSON.stringify({speaker_i, vocoder, linesBatch})
        }).then(r=>r.text()).then(async (res) => {

            if (res) {
                if (res=="CUDA OOM") {
                    window.errorModal("CUDA OOM: There is not enough VRAM to run this. Try lowering the batch size, or shortening very long sentences.")
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
                    batch_progressNotes.innerHTML = `Outputting audio via ffmpeg...`
                }

                for (let ri=0; ri<linesBatch.length; ri++) {
                    let tempFileLocation = linesBatch[ri][4]
                    let outPath = linesBatch[ri][5].includes(":") || linesBatch[ri][5].includes("./") ? linesBatch[ri][5] : `${linesBatch[ri][6]}/${linesBatch[ri][5]}`
                    if (outPath.startsWith("./")) {
                        outPath = window.userSettings.batchOutFolder + outPath.slice(1,100000)
                    }
                    try {
                        if (window.batch_state.state) {
                            await batchKickOffFfmpegOutput(ri, linesBatch, records, JSON.stringify({
                                input_path: tempFileLocation,
                                output_path: outPath,
                                options: JSON.stringify(options)
                            }))
                        }
                    } catch (e) {
                        console.log(e)
                        window.errorModal("Something went wrong:<br><br>"+e)
                        resolve()
                    }
                }
                resolve()
            } else {
                linesBatch.forEach((lineRecord, li) => {
                    let tempFileLocation = lineRecord[4]
                    let outPath = lineRecord[5]
                    try {
                        fs.copyFileSync(tempFileLocation, outPath)
                        records[li][1].children[1].innerHTML = "Done"
                        records[li][1].children[1].style.background = "green"
                        window.batch_state.lineIndex += 1

                    } catch (err) {
                        console.log(err)
                        window.appLogger.log(err)
                        window.appLogger.log(err)
                        batch_pauseBtn.click()
                    }
                    resolve()
                })
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

    batch_progressNotes.innerHTML = `Paused`

    const isRunning = window.batch_state.state
    batch_pauseBtn.innerHTML = isRunning ? "Resume" : "Pause"
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
        if (record[1].children[1].innerHTML=="Ready" || record[1].children[1].innerHTML=="Running") {
            record[1].children[1].innerHTML = "Stopped"
            record[1].children[1].style.background = "none"
        }
    })

    clearOldTempFiles()
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

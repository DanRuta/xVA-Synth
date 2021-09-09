"use strict"

const saveUserSettings = () => localStorage.setItem("userSettings", JSON.stringify(window.userSettings))
// const saveUserSettings = () => {}

const deleteFolderRecursive = function (directoryPath) {
    if (fs.existsSync(directoryPath)) {
        fs.readdirSync(directoryPath).forEach((file, index) => {
          const curPath = `${directoryPath}/${file}`;
          if (fs.lstatSync(curPath).isDirectory()) {
           // recurse
            deleteFolderRecursive(curPath);
          } else {
            // delete file
            fs.unlinkSync(curPath);
          }
        });
        fs.rmdirSync(directoryPath);
      }
    };

// Load user settings
window.userSettings = localStorage.getItem("userSettings") ||
    {
        useGPU: false,
        customWindowSize:`${window.innerHeight},${window.innerWidth}`,
        autoplay: false,
        autoPlayGen: false,
        audio: {
            format: "wav"
        },
        plugins: {

        }
    }
if ((typeof window.userSettings)=="string") {
    window.userSettings = JSON.parse(window.userSettings)
}
if (!Object.keys(window.userSettings).includes("installation")) { // For backwards compatibility
    window.userSettings.installation = "cpu"
}
if (!Object.keys(window.userSettings).includes("audio")) { // For backwards compatibility
    window.userSettings.audio = {format: "wav", hz: 22050, padStart: 0, padEnd: 0}
}
if (!Object.keys(window.userSettings).includes("sliderTooltip")) { // For backwards compatibility
    window.userSettings.sliderTooltip = true
}
if (!Object.keys(window.userSettings).includes("darkPrompt")) { // For backwards compatibility
    window.userSettings.darkPrompt = false
}
if (!Object.keys(window.userSettings).includes("showDiscordStatus")) { // For backwards compatibility
    window.userSettings.showDiscordStatus = true
}
if (!Object.keys(window.userSettings).includes("prompt_fontSize")) { // For backwards compatibility
    window.userSettings.prompt_fontSize = 13
}
if (!Object.keys(window.userSettings).includes("bg_gradient_opacity")) { // For backwards compatibility
    window.userSettings.bg_gradient_opacity = 13
}
if (!Object.keys(window.userSettings).includes("autoReloadVoices")) { // For backwards compatibility
    window.userSettings.autoReloadVoices = false
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("hz")) { // For backwards compatibility
    window.userSettings.audio.hz = 22050
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("padStart")) { // For backwards compatibility
    window.userSettings.audio.padStart = 0
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("padEnd")) { // For backwards compatibility
    window.userSettings.audio.padEnd = 0
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("ffmpeg")) { // For backwards compatibility
    window.userSettings.audio.ffmpeg = true
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("ffmpeg_preview")) { // For backwards compatibility
    window.userSettings.audio.ffmpeg_preview = true
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("bitdepth")) { // For backwards compatibility
    window.userSettings.audio.bitdepth = "pcm_s32le"
}
if (!Object.keys(window.userSettings).includes("showEditorFFMPEGAmplitude")) { // For backwards compatibility
    window.userSettings.showEditorFFMPEGAmplitude = false
}
if (!Object.keys(window.userSettings).includes("vocoder")) { // For backwards compatibility
    window.userSettings.vocoder = "256_waveglow"
}
if (!Object.keys(window.userSettings).includes("audio") || !Object.keys(window.userSettings.audio).includes("amplitude")) { // For backwards compatibility
    window.userSettings.audio.amplitude = 1
}
if (!Object.keys(window.userSettings).includes("keepPaceOnNew")) { // For backwards compatibility
    window.userSettings.keepPaceOnNew = true
}
if (!Object.keys(window.userSettings).includes("batchOutFolder")) { // For backwards compatibility
    window.userSettings.batchOutFolder = `${__dirname.replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
}
if (!Object.keys(window.userSettings).includes("batch_clearDirFirst")) { // For backwards compatibility
    window.userSettings.batch_clearDirFirst = false
}
if (!Object.keys(window.userSettings).includes("batch_fastMode")) { // For backwards compatibility
    window.userSettings.batch_fastMode = false
}
if (!Object.keys(window.userSettings).includes("batch_useMP")) { // For backwards compatibility
    window.userSettings.batch_useMP = false
}
if (!Object.keys(window.userSettings).includes("batch_MPCount")) { // For backwards compatibility
    window.userSettings.batch_MPCount = 0
}
if (!Object.keys(window.userSettings).includes("batch_skipExisting")) { // For backwards compatibility
    window.userSettings.batch_skipExisting = true
}
if (!Object.keys(window.userSettings).includes("batch_doGrouping")) { // For backwards compatibility
    window.userSettings.batch_doGrouping = true
}
if (!Object.keys(window.userSettings).includes("batch_doVocoderGrouping")) { // For backwards compatibility
    window.userSettings.batch_doVocoderGrouping = false
}
if (!Object.keys(window.userSettings).includes("batch_delimiter")) { // For backwards compatibility
    window.userSettings.batch_delimiter = ","
}
if (!Object.keys(window.userSettings).includes("batch_paginationSize")) { // For backwards compatibility
    window.userSettings.batch_paginationSize = 100
}
if (!Object.keys(window.userSettings).includes("defaultToHiFi")) { // For backwards compatibility
    window.userSettings.defaultToHiFi = true
}
if (!Object.keys(window.userSettings).includes("batch_batchSize")) { // For backwards compatibility
    window.userSettings.batch_batchSize = 1
}
if (!Object.keys(window.userSettings).includes("autoPlayGen")) { // For backwards compatibility
    window.userSettings.autoPlayGen = true
}
if (!Object.keys(window.userSettings).includes("outputJSON")) { // For backwards compatibility
    window.userSettings.outputJSON = true
}
if (!Object.keys(window.userSettings).includes("keepEditorOnVoiceChange")) { // For backwards compatibility
    window.userSettings.keepEditorOnVoiceChange = false
}
if (!Object.keys(window.userSettings).includes("filenameNumericalSeq")) { // For backwards compatibility
    window.userSettings.filenameNumericalSeq = false
}
if (!Object.keys(window.userSettings).includes("useErrorSound")) { // For backwards compatibility
    window.userSettings.useErrorSound = false
}
if (!Object.keys(window.userSettings).includes("showTipOfTheDay")) { // For backwards compatibility
    window.userSettings.showTipOfTheDay = true
}
if (!Object.keys(window.userSettings).includes("showUnseenTipOfTheDay")) { // For backwards compatibility
    window.userSettings.showUnseenTipOfTheDay = false
}
if (!Object.keys(window.userSettings).includes("playChangedAudio")) { // For backwards compatibility
    window.userSettings.playChangedAudio = false
}

if (!Object.keys(window.userSettings).includes("errorSoundFile")) { // For backwards compatibility
    window.userSettings.errorSoundFile = `${__dirname.replace(/\\/g,"/")}/lib/xp_error.mp3`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
}
if (!Object.keys(window.userSettings).includes("plugins")) { // For backwards compatibility
    window.userSettings.plugins = {}
}
if (!Object.keys(window.userSettings.plugins).includes("loadOrder")) { // For backwards compatibility
    window.userSettings.plugins.loadOrder = ""
}
if (!Object.keys(window.userSettings).includes("externalAudioEditor")) { // For backwards compatibility
    window.userSettings.externalAudioEditor = ""
}
if (!Object.keys(window.userSettings).includes("s2s_autogenerate")) { // For backwards compatibility
    window.userSettings.s2s_autogenerate = true
}
if (!Object.keys(window.userSettings).includes("s2s_prePitchShift")) { // For backwards compatibility
    window.userSettings.s2s_prePitchShift = false
}
if (!Object.keys(window.userSettings).includes("s2s_removeNoise")) { // For backwards compatibility
    window.userSettings.s2s_removeNoise = false
}
if (!Object.keys(window.userSettings).includes("s2s_noiseRemStrength")) { // For backwards compatibility
    window.userSettings.s2s_noiseRemStrength = 0.25
}
if (!Object.keys(window.userSettings).includes("waveglow_path")) { // For backwards compatibility
    window.userSettings.waveglow_path = `${__dirname.replace(/\\/g,"/")}/models/waveglow_256channels_universal_v4.pt`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
}
if (!Object.keys(window.userSettings).includes("bigwaveglow_path")) { // For backwards compatibility
    window.userSettings.bigwaveglow_path = `${__dirname.replace(/\\/g,"/")}/models/nvidia_waveglowpyt_fp32_20190427.pt`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
}

const updateUIWithSettings = () => {
    useGPUCbx.checked = window.userSettings.useGPU
    autoplay_ckbx.checked = window.userSettings.autoplay
    // setting_slidersTooltip.checked = window.userSettings.sliderTooltip
    setting_defaultToHiFi.checked = window.userSettings.defaultToHiFi
    setting_keepPaceOnNew.checked = window.userSettings.keepPaceOnNew
    setting_autoplaygenCbx.checked = window.userSettings.autoPlayGen
    setting_darkprompt.checked = window.userSettings.darkPrompt
    setting_show_discord_status.checked = window.userSettings.showDiscordStatus
    setting_prompt_fontSize.value = window.userSettings.prompt_fontSize
    setting_bg_gradient_opacity.value = window.userSettings.bg_gradient_opacity
    setting_areload_voices.checked = window.userSettings.autoReloadVoices
    setting_output_json.checked = window.userSettings.outputJSON
    setting_output_num_seq.checked = window.userSettings.filenameNumericalSeq
    setting_keepEditorOnVoiceChange.checked = window.userSettings.keepEditorOnVoiceChange
    setting_use_error_sound.checked = window.userSettings.useErrorSound
    setting_error_sound_file.value = window.userSettings.errorSoundFile

    setting_showTipOfTheDay.checked = window.userSettings.showTipOfTheDay
    totdShowTips.checked = window.userSettings.showTipOfTheDay
    setting_showUnseenTipOfTheDay.checked = window.userSettings.showUnseenTipOfTheDay
    totdShowOnlyUnseenTips.checked = window.userSettings.showUnseenTipOfTheDay

    setting_playChangedAudio.checked = window.userSettings.playChangedAudio

    setting_external_audio_editor.value = window.userSettings.externalAudioEditor
    setting_audio_ffmpeg.checked = window.userSettings.audio.ffmpeg
    setting_audio_ffmpeg_preview.checked = window.userSettings.audio.ffmpeg && window.userSettings.audio.ffmpeg_preview
    setting_audio_format.value = window.userSettings.audio.format
    setting_audio_hz.value = window.userSettings.audio.hz
    setting_audio_pad_start.value = window.userSettings.audio.padStart
    setting_audio_pad_end.value = window.userSettings.audio.padEnd
    setting_audio_bitdepth.value = window.userSettings.audio.bitdepth
    setting_audio_amplitude.value = window.userSettings.audio.amplitude
    setting_editor_audio_amplitude.value = window.userSettings.audio.amplitude
    setting_show_editor_ffmpegamplitude.checked = window.userSettings.showEditorFFMPEGAmplitude
    editor_amplitude_options.style.display = window.userSettings.showEditorFFMPEGAmplitude ? "flex" : "none"

    setting_s2s_autogenerate.checked = window.userSettings.s2s_autogenerate
    // setting_s2s_prePitchShift.checked = window.userSettings.s2s_prePitchShift
    setting_s2s_removeNoise.checked = window.userSettings.s2s_removeNoise
    setting_s2s_noiseRemStrength.value = window.userSettings.s2s_noiseRemStrength

    setting_batch_fastmode.checked = window.userSettings.batch_fastMode
    setting_batch_multip.checked = window.userSettings.batch_useMP
    setting_batch_multip_count.value = window.userSettings.batch_MPCount
    setting_batch_delimiter.value = window.userSettings.batch_delimiter
    setting_batch_paginationSize.value = window.userSettings.batch_paginationSize
    setting_batch_doGrouping.checked = window.userSettings.batch_doGrouping
    setting_batch_doVocoderGrouping.checked = window.userSettings.batch_doVocoderGrouping

    batch_batchSizeInput.value = parseInt(window.userSettings.batch_batchSize)
    batch_skipExisting.checked = window.userSettings.batch_skipExisting
    batch_clearDirFirstCkbx.checked = window.userSettings.batch_clearDirFirst

    setting_256waveglow_path.value = window.userSettings.waveglow_path
    setting_bigwaveglow_path.value = window.userSettings.bigwaveglow_path

    const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
    ipcRenderer.send("resize", {height, width})
}
updateUIWithSettings()
saveUserSettings()


// Add the SVG code this way, because otherwise the index.html file will be spammed with way too much svg code
Array.from(window.document.querySelectorAll(".svgButton")).forEach(svgButton => {
    svgButton.innerHTML = `<svg class="openFolderSVG" width="400" height="350" viewBox="0, 0, 400,350"><g id="svgg" ><path id="path0"  d="M39.960 53.003 C 36.442 53.516,35.992 53.635,30.800 55.422 C 15.784 60.591,3.913 74.835,0.636 91.617 C -0.372 96.776,-0.146 305.978,0.872 310.000 C 5.229 327.228,16.605 339.940,32.351 345.172 C 40.175 347.773,32.175 347.630,163.000 347.498 L 281.800 347.378 285.600 346.495 C 304.672 342.065,321.061 332.312,330.218 319.944 C 330.648 319.362,332.162 317.472,333.581 315.744 C 335.001 314.015,336.299 312.420,336.467 312.200 C 336.634 311.980,337.543 310.879,338.486 309.753 C 340.489 307.360,342.127 305.341,343.800 303.201 C 344.460 302.356,346.890 299.375,349.200 296.575 C 351.510 293.776,353.940 290.806,354.600 289.975 C 355.260 289.144,356.561 287.505,357.492 286.332 C 358.422 285.160,359.952 283.267,360.892 282.126 C 362.517 280.153,371.130 269.561,375.632 264.000 C 376.789 262.570,380.427 258.097,383.715 254.059 C 393.790 241.689,396.099 237.993,398.474 230.445 C 403.970 212.972,394.149 194.684,376.212 188.991 C 369.142 186.747,368.803 186.724,344.733 186.779 C 330.095 186.812,322.380 186.691,322.216 186.425 C 322.078 186.203,321.971 178.951,321.977 170.310 C 321.995 146.255,321.401 141.613,317.200 133.000 C 314.009 126.457,307.690 118.680,303.142 115.694 C 302.560 115.313,301.300 114.438,300.342 113.752 C 295.986 110.631,288.986 107.881,282.402 106.704 C 280.540 106.371,262.906 106.176,220.400 106.019 L 161.000 105.800 160.763 98.800 C 159.961 75.055,143.463 56.235,120.600 52.984 C 115.148 52.208,45.292 52.225,39.960 53.003 M120.348 80.330 C 130.472 83.988,133.993 90.369,133.998 105.071 C 134.003 120.968,137.334 127.726,147.110 131.675 L 149.400 132.600 213.800 132.807 C 272.726 132.996,278.392 133.071,280.453 133.690 C 286.872 135.615,292.306 141.010,294.261 147.400 C 294.928 149.578,294.996 151.483,294.998 168.000 L 295.000 186.200 292.800 186.449 C 291.590 186.585,254.330 186.725,210.000 186.759 C 163.866 186.795,128.374 186.977,127.000 187.186 C 115.800 188.887,104.936 192.929,96.705 198.458 C 95.442 199.306,94.302 200.000,94.171 200.000 C 93.815 200.000,89.287 203.526,87.000 205.583 C 84.269 208.039,80.083 212.649,76.488 217.159 C 72.902 221.657,72.598 222.031,70.800 224.169 C 70.030 225.084,68.770 226.620,68.000 227.582 C 67.230 228.544,66.054 229.977,65.387 230.766 C 64.720 231.554,62.727 234.000,60.957 236.200 C 59.188 238.400,56.346 241.910,54.642 244.000 C 52.938 246.090,50.163 249.510,48.476 251.600 C 44.000 257.146,36.689 266.126,36.212 266.665 C 35.985 266.921,34.900 268.252,33.800 269.623 C 32.700 270.994,30.947 273.125,29.904 274.358 C 28.861 275.591,28.006 276.735,28.004 276.900 C 28.002 277.065,27.728 277.200,27.395 277.200 C 26.428 277.200,26.700 96.271,27.670 93.553 C 30.020 86.972,35.122 81.823,40.800 80.300 C 44.238 79.378,47.793 79.296,81.800 79.351 L 117.800 79.410 120.348 80.330 M369.400 214.800 C 374.239 217.220,374.273 222.468,369.489 228.785 C 367.767 231.059,364.761 234.844,364.394 235.200 C 364.281 235.310,362.373 237.650,360.154 240.400 C 357.936 243.150,354.248 247.707,351.960 250.526 C 347.732 255.736,346.053 257.821,343.202 261.400 C 341.505 263.530,340.849 264.336,334.600 271.965 C 332.400 274.651,330.204 277.390,329.720 278.053 C 329.236 278.716,328.246 279.945,327.520 280.785 C 326.794 281.624,325.300 283.429,324.200 284.794 C 323.100 286.160,321.726 287.845,321.147 288.538 C 320.568 289.232,318.858 291.345,317.347 293.233 C 308.372 304.449,306.512 306.609,303.703 309.081 C 299.300 312.956,290.855 317.633,286.000 318.886 C 277.958 320.960,287.753 320.819,159.845 320.699 C 33.557 320.581,42.330 320.726,38.536 318.694 C 34.021 316.276,35.345 310.414,42.386 301.647 C 44.044 299.583,45.940 297.210,46.600 296.374 C 47.260 295.538,48.340 294.169,49.000 293.332 C 49.660 292.495,51.550 290.171,53.200 288.167 C 54.850 286.164,57.100 283.395,58.200 282.015 C 59.300 280.635,60.920 278.632,61.800 277.564 C 62.680 276.496,64.210 274.617,65.200 273.389 C 66.190 272.162,67.188 270.942,67.418 270.678 C 67.649 270.415,71.591 265.520,76.179 259.800 C 80.767 254.080,84.634 249.310,84.773 249.200 C 84.913 249.090,87.117 246.390,89.673 243.200 C 92.228 240.010,95.621 235.780,97.213 233.800 C 106.328 222.459,116.884 215.713,128.200 213.998 C 129.300 213.832,183.570 213.719,248.800 213.748 L 367.400 213.800 369.400 214.800 " stroke="none" fill="#fbfbfb" fill-rule="evenodd"></path><path id="path1" fill-opacity="0" d="M0.000 46.800 C 0.000 72.540,0.072 93.600,0.159 93.600 C 0.246 93.600,0.516 92.460,0.759 91.066 C 3.484 75.417,16.060 60.496,30.800 55.422 C 35.953 53.648,36.338 53.550,40.317 52.981 C 46.066 52.159,114.817 52.161,120.600 52.984 C 143.463 56.235,159.961 75.055,160.763 98.800 L 161.000 105.800 220.400 106.019 C 262.906 106.176,280.540 106.371,282.402 106.704 C 288.986 107.881,295.986 110.631,300.342 113.752 C 301.300 114.438,302.560 115.313,303.142 115.694 C 307.690 118.680,314.009 126.457,317.200 133.000 C 321.401 141.613,321.995 146.255,321.977 170.310 C 321.971 178.951,322.078 186.203,322.216 186.425 C 322.380 186.691,330.095 186.812,344.733 186.779 C 368.803 186.724,369.142 186.747,376.212 188.991 C 381.954 190.814,388.211 194.832,391.662 198.914 C 395.916 203.945,397.373 206.765,399.354 213.800 C 399.842 215.533,399.922 201.399,399.958 107.900 L 400.000 0.000 200.000 0.000 L 0.000 0.000 0.000 46.800 M44.000 79.609 C 35.903 81.030,30.492 85.651,27.670 93.553 C 26.700 96.271,26.428 277.200,27.395 277.200 C 27.728 277.200,28.002 277.065,28.004 276.900 C 28.006 276.735,28.861 275.591,29.904 274.358 C 30.947 273.125,32.700 270.994,33.800 269.623 C 34.900 268.252,35.985 266.921,36.212 266.665 C 36.689 266.126,44.000 257.146,48.476 251.600 C 50.163 249.510,52.938 246.090,54.642 244.000 C 56.346 241.910,59.188 238.400,60.957 236.200 C 62.727 234.000,64.720 231.554,65.387 230.766 C 66.054 229.977,67.230 228.544,68.000 227.582 C 68.770 226.620,70.030 225.084,70.800 224.169 C 72.598 222.031,72.902 221.657,76.488 217.159 C 80.083 212.649,84.269 208.039,87.000 205.583 C 89.287 203.526,93.815 200.000,94.171 200.000 C 94.302 200.000,95.442 199.306,96.705 198.458 C 104.936 192.929,115.800 188.887,127.000 187.186 C 128.374 186.977,163.866 186.795,210.000 186.759 C 254.330 186.725,291.590 186.585,292.800 186.449 L 295.000 186.200 294.998 168.000 C 294.996 151.483,294.928 149.578,294.261 147.400 C 292.306 141.010,286.872 135.615,280.453 133.690 C 278.392 133.071,272.726 132.996,213.800 132.807 L 149.400 132.600 147.110 131.675 C 137.334 127.726,134.003 120.968,133.998 105.071 C 133.993 90.369,130.472 83.988,120.348 80.330 L 117.800 79.410 81.800 79.351 C 62.000 79.319,44.990 79.435,44.000 79.609 M128.200 213.998 C 116.884 215.713,106.328 222.459,97.213 233.800 C 95.621 235.780,92.228 240.010,89.673 243.200 C 87.117 246.390,84.913 249.090,84.773 249.200 C 84.634 249.310,80.767 254.080,76.179 259.800 C 71.591 265.520,67.649 270.415,67.418 270.678 C 67.188 270.942,66.190 272.162,65.200 273.389 C 64.210 274.617,62.680 276.496,61.800 277.564 C 60.920 278.632,59.300 280.635,58.200 282.015 C 57.100 283.395,54.850 286.164,53.200 288.167 C 51.550 290.171,49.660 292.495,49.000 293.332 C 48.340 294.169,47.260 295.538,46.600 296.374 C 45.940 297.210,44.044 299.583,42.386 301.647 C 35.345 310.414,34.021 316.276,38.536 318.694 C 42.330 320.726,33.557 320.581,159.845 320.699 C 287.753 320.819,277.958 320.960,286.000 318.886 C 290.855 317.633,299.300 312.956,303.703 309.081 C 306.512 306.609,308.372 304.449,317.347 293.233 C 318.858 291.345,320.568 289.232,321.147 288.538 C 321.726 287.845,323.100 286.160,324.200 284.794 C 325.300 283.429,326.794 281.624,327.520 280.785 C 328.246 279.945,329.236 278.716,329.720 278.053 C 330.204 277.390,332.400 274.651,334.600 271.965 C 340.849 264.336,341.505 263.530,343.202 261.400 C 346.053 257.821,347.732 255.736,351.960 250.526 C 354.248 247.707,357.936 243.150,360.154 240.400 C 362.373 237.650,364.281 235.310,364.394 235.200 C 364.761 234.844,367.767 231.059,369.489 228.785 C 374.273 222.468,374.239 217.220,369.400 214.800 L 367.400 213.800 248.800 213.748 C 183.570 213.719,129.300 213.832,128.200 213.998 M399.600 225.751 C 399.600 231.796,394.623 240.665,383.715 254.059 C 380.427 258.097,376.789 262.570,375.632 264.000 C 371.130 269.561,362.517 280.153,360.892 282.126 C 359.952 283.267,358.422 285.160,357.492 286.332 C 356.561 287.505,355.260 289.144,354.600 289.975 C 353.940 290.806,351.510 293.776,349.200 296.575 C 346.890 299.375,344.460 302.356,343.800 303.201 C 342.127 305.341,340.489 307.360,338.486 309.753 C 337.543 310.879,336.634 311.980,336.467 312.200 C 336.299 312.420,335.001 314.015,333.581 315.744 C 332.162 317.472,330.648 319.362,330.218 319.944 C 321.061 332.312,304.672 342.065,285.600 346.495 L 281.800 347.378 163.000 347.498 C 32.175 347.630,40.175 347.773,32.351 345.172 C 16.471 339.895,3.810 325.502,0.820 309.326 C 0.591 308.085,0.312 306.979,0.202 306.868 C 0.091 306.757,-0.000 327.667,-0.000 353.333 L 0.000 400.000 200.000 400.000 L 400.000 400.000 400.000 312.400 C 400.000 264.220,399.910 224.800,399.800 224.800 C 399.690 224.800,399.600 225.228,399.600 225.751 " stroke="none" fill="#050505" fill-rule="evenodd"></path></g></svg>`
})


// Installation sever handling
// =========================
settings_installation.innerHTML = window.userSettings.installation=="cpu" ? `CPU` : "CPU+GPU"
setting_change_installation.innerHTML = window.userSettings.installation=="cpu" ? `Change to CPU+GPU` : `Change to CPU`


setting_change_installation.addEventListener("click", () => {
    spinnerModal("Changing installation sever...")
    doFetch(`http://localhost:8008/stopServer`, {
        method: "Post",
        body: JSON.stringify({})
    }).then(r=>r.text()).then(console.log) // The server stopping should mean this never runs
    .catch(() => {

        if (window.userSettings.installation=="cpu") {
            window.userSettings.installation = "gpu"
            useGPUCbx.disabled = false
            settings_installation.innerHTML = `GPU`
            setting_change_installation.innerHTML = `Change to CPU`
        } else {
            doFetch(`http://localhost:8008/setDevice`, {
                method: "Post",
                body: JSON.stringify({device: "cpu"})
            })

            window.userSettings.installation = "cpu"
            useGPUCbx.checked = false
            useGPUCbx.disabled = true
            window.userSettings.useGPU = false
            settings_installation.innerHTML = `CPU`
            setting_change_installation.innerHTML = `Change to CPU+GPU`
        }
        saveUserSettings()

        // Start the new server
        if (window.PRODUCTION) {
            window.appLogger.log(window.userSettings.installation)
            window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
        } else {
            window.pythonProcess = spawn("python", [`${path}/server.py`], {stdio: "ignore"})
        }

        window.currentModel = undefined
        title.innerHTML = window.i18n.SELECT_VOICE_TYPE
        keepSampleButton.style.display = "none"
        wavesurferContainer.innerHTML = ""
        generateVoiceButton.dataset.modelQuery = "null"
        generateVoiceButton.dataset.modelIDLoaded = undefined
        generateVoiceButton.innerHTML = window.i18n.LOAD_MODEL
        generateVoiceButton.disabled = true
        window.serverIsUp = false
        window.doWeirdServerStartupCheck(`${window.i18n.LOADING}...<br>${window.i18n.MAY_TAKE_A_MINUTE}<br><br>${window.i18n.STARTING_PYTHON}...`)
    })
})

// =========================




// Audio hardware
// ==============
navigator.mediaDevices.enumerateDevices().then(devices => {
    devices = devices.filter(device => device.kind=="audiooutput" && device.deviceId!="communications")

    // Base device
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_base_speaker.appendChild(option)
    })
    setting_base_speaker.addEventListener("change", () => {
        window.userSettings.base_speaker = setting_base_speaker.value
        window.saveUserSettings()

        window.document.querySelectorAll("audio").forEach(audioElem => {
            audioElem.setSinkId(window.userSettings.base_speaker)
        })
    })
    if (Object.keys(window.userSettings).includes("base_speaker")) {
        setting_base_speaker.value = window.userSettings.base_speaker
    } else {
        window.userSettings.base_speaker = setting_base_speaker.value
        window.saveUserSettings()
    }

    // Alternate device
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_alt_speaker.appendChild(option)
    })
    setting_alt_speaker.addEventListener("change", () => {
        window.userSettings.alt_speaker = setting_alt_speaker.value
        window.saveUserSettings()
    })
    if (Object.keys(window.userSettings).includes("alt_speaker")) {
        setting_alt_speaker.value = window.userSettings.alt_speaker
    } else {
        window.userSettings.alt_speaker = setting_alt_speaker.value
        window.saveUserSettings()
    }
})



// Settings Menu
// =============
useGPUCbx.addEventListener("change", () => {
    spinnerModal(window.i18n.CHANGING_DEVICE)
    doFetch(`http://localhost:8008/setDevice`, {
        method: "Post",
        body: JSON.stringify({device: useGPUCbx.checked ? "gpu" : "cpu"})
    }).then(r=>r.text()).then(res => {
        window.closeModal(undefined, settingsContainer)
        window.userSettings.useGPU = useGPUCbx.checked
        saveUserSettings()
    }).catch(e => {
        console.log(e)
        if (e.code =="ENOENT") {
            window.closeModal(undefined, settingsContainer).then(() => {
                window.errorModal(window.i18n.THERE_WAS_A_PROBLEM)
            })
        }
    })
})


const initMenuSetting = (elem, setting, type, callback=undefined, valFn=undefined) => {

    valFn = valFn ? valFn : x=>x

    if (type=="checkbox") {
        elem.addEventListener("click", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.checked)
            } else {
                window.userSettings[setting] = valFn(elem.checked)
            }
            saveUserSettings()
            if (callback) callback()
        })
    } else {
        elem.addEventListener("change", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.value)
            } else {
                window.userSettings[setting] = valFn(elem.value)
            }
            saveUserSettings()
            if (callback) callback()
        })
    }
}
const initFilePickerButton = (button, input, setting, properties, filters=undefined, defaultPath=undefined, callback=undefined) => {
    button.addEventListener("click", () => {
        const defaultPath = input.value.replace(/\//g, "\\")
        let filePath = electron.remote.dialog.showOpenDialog({ properties, filters, defaultPath})
        if (filePath) {
            filePath = filePath[0].replace(/\\/g, "/")
            input.value = filePath.replace(/\\/g, "/")
            setting = typeof(setting)=="function" ? setting() : setting
            window.userSettings[setting] = filePath
            saveUserSettings()
            if (callback) {
                callback()
            }
        }
    })
}

const setPromptTheme = () => {
    if (window.userSettings.darkPrompt) {
        dialogueInput.style.backgroundColor = "rgba(25,25,25,0.9)"
        dialogueInput.style.color = "white"
    } else {
        dialogueInput.style.backgroundColor = "rgba(255,255,255,0.9)"
        dialogueInput.style.color = "black"
    }
}
const updateDiscord = () => {
    let gameName = undefined
    if (window.userSettings.showDiscordStatus && window.currentGame) {
        gameName = (window.currentGame.length==5 ? window.currentGame[4] : window.currentGame[3]).split(".")[0]
    }
    ipcRenderer.send('updateDiscord', {details: gameName})
}
const setPromptFontSize = () => {
    dialogueInput.style.fontSize = `${window.userSettings.prompt_fontSize}pt`
}
const updateBackground = () => {
    const background = `linear-gradient(0deg, rgba(128,128,128,${window.userSettings.bg_gradient_opacity}) 0px, rgba(0,0,0,0)), url("assets/${window.currentGame.join("-")}")`
    // Fade the background image transition
    rightBG1.style.background = background
    rightBG2.style.opacity = 0
    setTimeout(() => {
        rightBG2.style.background = rightBG1.style.background
        rightBG2.style.opacity = 1
    }, 1000)
}

initMenuSetting(setting_autoplaygenCbx, "autoPlayGen", "checkbox")
// initMenuSetting(setting_slidersTooltip, "sliderTooltip", "checkbox")
initMenuSetting(setting_defaultToHiFi, "defaultToHiFi", "checkbox")
initMenuSetting(setting_keepPaceOnNew, "keepPaceOnNew", "checkbox")
initMenuSetting(setting_areload_voices, "autoReloadVoices", "checkbox")
initMenuSetting(setting_output_json, "outputJSON", "checkbox")
initMenuSetting(setting_keepEditorOnVoiceChange, "keepEditorOnVoiceChange", "checkbox")
initMenuSetting(setting_output_num_seq, "filenameNumericalSeq", "checkbox")
initMenuSetting(setting_darkprompt, "darkPrompt", "checkbox", setPromptTheme)
initMenuSetting(setting_show_discord_status, "showDiscordStatus", "checkbox", updateDiscord)
initMenuSetting(setting_prompt_fontSize, "prompt_fontSize", "number", setPromptFontSize)
initMenuSetting(setting_bg_gradient_opacity, "bg_gradient_opacity", "number", updateBackground)
initMenuSetting(setting_use_error_sound, "useErrorSound", "checkbox")
initMenuSetting(setting_error_sound_file, "errorSoundFile", "text")
initFilePickerButton(setting_errorSoundFileBtn, setting_error_sound_file, "errorSoundFile", ["openFile"], [{name: "Audio", extensions: ["wav", "mp3", "ogg"]}])

initMenuSetting(setting_showTipOfTheDay, "showTipOfTheDay", "checkbox", () => {
    totdShowTips.checked = setting_showTipOfTheDay.checked
})
initMenuSetting(totdShowTips, "showTipOfTheDay", "checkbox", () => {
    setting_showTipOfTheDay.checked = totdShowTips.checked
})
initMenuSetting(setting_showUnseenTipOfTheDay, "showUnseenTipOfTheDay", "checkbox", () => {
    totdShowOnlyUnseenTips.checked = setting_showUnseenTipOfTheDay.checked
})
initMenuSetting(totdShowOnlyUnseenTips, "showUnseenTipOfTheDay", "checkbox", () => {
    setting_showUnseenTipOfTheDay.checked = totdShowOnlyUnseenTips.checked
})

initMenuSetting(setting_playChangedAudio, "playChangedAudio", "checkbox")


initMenuSetting(setting_external_audio_editor, "externalAudioEditor", "text")
initFilePickerButton(setting_externalEditorButton, setting_external_audio_editor, "externalAudioEditor", ["openFile"])

initMenuSetting(setting_audio_ffmpeg, "audio.ffmpeg", "checkbox", () => {
    setting_audio_ffmpeg_preview.checked = window.userSettings.audio.ffmpeg && window.userSettings.audio.ffmpeg_preview
    setting_audio_ffmpeg_preview.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
    setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
    setting_editor_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
})
initMenuSetting(setting_audio_ffmpeg_preview, "audio.ffmpeg_preview", "checkbox")
initMenuSetting(setting_audio_format, "audio.format", "text")
initMenuSetting(setting_audio_hz, "audio.hz", "text", undefined, parseInt)
initMenuSetting(setting_audio_pad_start, "audio.padStart", "text", undefined, parseInt)
initMenuSetting(setting_audio_pad_end, "audio.padEnd", "text", undefined, parseInt)
initMenuSetting(setting_audio_bitdepth, "audio.bitdepth", "select")
initMenuSetting(setting_audio_amplitude, "audio.amplitude", "number", () => {
    setting_editor_audio_amplitude.value = setting_audio_amplitude.value
}, parseFloat)
initMenuSetting(setting_editor_audio_amplitude, "audio.amplitude", "number", () => {
    setting_audio_amplitude.value = setting_editor_audio_amplitude.value
}, parseFloat)
initMenuSetting(setting_show_editor_ffmpegamplitude, "showEditorFFMPEGAmplitude", "checkbox", () => {
    editor_amplitude_options.style.display = window.userSettings.showEditorFFMPEGAmplitude ? "flex" : "none"
})

initMenuSetting(setting_batch_fastmode, "batch_fastMode", "checkbox")
initMenuSetting(setting_batch_multip, "batch_useMP", "checkbox")
initMenuSetting(setting_batch_multip_count, "batch_MPCount", "number", undefined, parseInt)
initMenuSetting(setting_batch_doGrouping, "batch_doGrouping", "checkbox")
initMenuSetting(setting_batch_doVocoderGrouping, "batch_doVocoderGrouping", "checkbox")
initMenuSetting(batch_clearDirFirstCkbx, "batch_clearDirFirst", "checkbox")
initMenuSetting(batch_skipExisting, "batch_skipExisting", "checkbox")
initMenuSetting(batch_batchSizeInput, "batch_batchSize", "text", undefined, parseInt)
initMenuSetting(setting_batch_delimiter, "batch_delimiter")
initMenuSetting(setting_batch_paginationSize, "batch_paginationSize", "number", undefined, parseInt)


initMenuSetting(setting_s2s_autogenerate, "s2s_autogenerate", "checkbox")
// initMenuSetting(setting_s2s_prePitchShift, "s2s_prePitchShift", "checkbox")
initMenuSetting(setting_s2s_removeNoise, "s2s_removeNoise", "checkbox")
initMenuSetting(setting_s2s_noiseRemStrength, "s2s_noiseRemStrength", "number", undefined, parseFloat)



initMenuSetting(setting_256waveglow_path, "waveglow_path", "text")
initFilePickerButton(setting_waveglowPathButton, setting_256waveglow_path, "waveglow_path", ["openFile"], [{name: "Pytorch checkpoint", extensions: ["pt"]}])
initMenuSetting(setting_bigwaveglow_path, "bigwaveglow_path", "text")
initFilePickerButton(setting_bigwaveglowPathButton, setting_bigwaveglow_path, "bigwaveglow_path", ["openFile"], [{name: "Pytorch checkpoint", extensions: ["pt"]}])

initFilePickerButton(setting_modelsPathButton, setting_models_path_input, ()=>`modelspath_${window.currentGame[0]}`, ["openDirectory"], undefined, undefined, ()=>window.updateGameList())
initFilePickerButton(setting_outPathButton, setting_out_path_input, ()=>`outpath_${window.currentGame[0]}`, ["openDirectory"], undefined, undefined, ()=>{
    if (window.currentModelButton) {
        window.currentModelButton.click()
    }
})


setPromptTheme()
setPromptFontSize()

setting_audio_format.disabled = !window.userSettings.audio.ffmpeg
setting_audio_hz.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_start.disabled = !window.userSettings.audio.ffmpeg
setting_audio_pad_end.disabled = !window.userSettings.audio.ffmpeg
setting_audio_bitdepth.disabled = !window.userSettings.audio.ffmpeg
setting_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg
setting_editor_audio_amplitude.disabled = !window.userSettings.audio.ffmpeg

// Output path
fs.readdir(`${window.path}/models`, (err, gameDirs) => {
    gameDirs.filter(name => !name.includes(".")).forEach(gameFolder => {
        // Initialize the default output directory setting for this game
        if (!Object.keys(window.userSettings).includes(`outpath_${gameFolder}`)) {
            window.userSettings[`outpath_${gameFolder}`] = `${__dirname.replace(/\\/g,"/")}/output/${gameFolder}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
            saveUserSettings()
        }
    })
})
setting_out_path_input.addEventListener("change", () => {
    const gameFolder = window.currentGame[0]

    setting_out_path_input.value = setting_out_path_input.value.replace(/\/\//g, "/").replace(/\\/g,"/")
    window.userSettings[`outpath_${gameFolder}`] = setting_out_path_input.value
    saveUserSettings()
    if (window.currentModelButton) {
        window.currentModelButton.click()
    }
})
// Models path
fs.readdir(`${window.path}/assets`, (err, assetFiles) => {
    assetFiles.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png"))&&fn.split("-").length==4).forEach(assetFileName => {
        const gameId = assetFileName.split("-")[0]
        const gameName = assetFileName.split("-").reverse()[0].split(".")[0]
        // Initialize the default models directory setting for this game
        if (!Object.keys(window.userSettings).includes(`modelspath_${gameId}`)) {
            window.userSettings[`modelspath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/models/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
            window.userSettings[`outpath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/output/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
            saveUserSettings()
        }
    })
})


// Batch stuff
// Output folder
batch_outputFolderInput.addEventListener("change", () => {
    if (batch_outputFolderInput.value.length==0) {
        window.errorModal(window.i18n.ENTER_DIR_PATH)
        batch_outputFolderInput.value = window.userSettings.batchOutFolder
    } else {
        window.userSettings.batchOutFolder = batch_outputFolderInput.value
        saveUserSettings()
    }
})
batch_outputFolderInput.value = window.userSettings.batchOutFolder
// ======



reset_settings_btn.addEventListener("click", () => {
    window.confirmModal(window.i18n.SURE_RESET_SETTINGS).then(confirmation => {
        if (confirmation) {
            window.userSettings.audio.format = "wav"
            window.userSettings.audio.hz = 22050
            window.userSettings.audio.padStart = 0
            window.userSettings.audio.padEnd = 0
            window.userSettings.audio.ffmpeg = false
            window.userSettings.autoPlayGen = false
            window.userSettings.autoReloadVoices = false
            window.userSettings.autoplay = true
            window.userSettings.darkPrompt = false
            window.userSettings.defaultToHiFi = true
            window.userSettings.keepPaceOnNew = true
            window.userSettings.sliderTooltip = true
            window.userSettings.audio.bitdepth = "pcm_s32le"

            window.userSettings.batch_batchSize = 1
            window.userSettings.batch_clearDirFirst= false
            window.userSettings.batch_fastMode = false
            window.userSettings.batch_skipExisting = true
            updateUIWithSettings()
            saveUserSettings()
        }
    })
})
reset_paths_btn.addEventListener("click", () => {
    window.confirmModal(window.i18n.SURE_RESET_PATHS).then(confirmation => {
        if (confirmation) {

            const pathKeys = Object.keys(window.userSettings).filter(key => key.includes("modelspath_"))
            pathKeys.forEach(key => {
                delete window.userSettings[key]
            })

            const currGame = window.currentGame ? window.currentGame[0] : undefined

            // Models paths
            const assetFiles = fs.readdirSync(`${path}/assets`)
            assetFiles.filter(fn=>(fn.endsWith(".jpg")||fn.endsWith(".png"))&&fn.split("-").length==4).forEach(assetFileName => {
                const gameId = assetFileName.split("-")[0]
                window.userSettings[`modelspath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/models/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
                if (gameId==currGame) {
                    setting_models_path_input.value = window.userSettings[`modelspath_${gameId}`]
                }
            })

            // Output paths
            const gameDirs = fs.readdirSync(`${path}/models`)
            gameDirs.filter(name => !name.includes(".")).forEach(gameId => {
                window.userSettings[`outpath_${gameId}`] = `${__dirname.replace(/\\/g,"/")}/output/${gameId}`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
                if (gameId==currGame) {
                    setting_out_path_input.value = window.userSettings[`outpath_${gameId}`]
                }
            })

            if (window.currentModelButton) {
                window.currentModelButton.click()
            }

            window.userSettings.batchOutFolder = `${__dirname.replace(/\\/g,"/")}/batch`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")
            batch_outputFolderInput.value = window.userSettings.batchOutFolder

            window.loadAllModels().then(() => {
                if (currGame) {
                    window.changeGame(window.currentGame.join("-"))
                }
            })
            saveUserSettings()

            // Gather the model paths to send to the server
            const modelsPaths = {}
            Object.keys(window.userSettings).filter(key => key.includes("modelspath_")).forEach(key => {
                modelsPaths[key.split("_")[1]] = window.userSettings[key]
            })
            doFetch(`http://localhost:8008/setAvailableVoices`, {
                method: "Post",
                body: JSON.stringify({
                    modelsPaths: JSON.stringify(modelsPaths)
                })
            })
        }
    })
})

// Search settings
const settingItems = Array.from(settingsOptionsContainer.children)
searchSettingsInput.addEventListener("keyup", () => {

    const query = searchSettingsInput.value.trim().toLowerCase()

    const filteredItems = settingItems.map(el => {
        if (el.tagName=="HR") {return [el, true]}
        if (el.tagName=="DIV") {
            if (!query.length || el.children[0].innerHTML.toLowerCase().includes(query)) {
                return [el, true]
            }
        }
        return [el, false]
    })

    let lastIsHR = false
    filteredItems.forEach(elem => {
        const [el, showIt] = elem
        if (el.tagName=="HR") {
            if (lastIsHR) {
                el.style.display = "none"
                return
            }
            lastIsHR = true
            el.style.display = "flex"
        } else {
            if (showIt) {
                el.style.display = "flex"
                lastIsHR = false
            } else {
                el.style.display = "none"
            }
        }
    })

})

window.saveUserSettings = saveUserSettings
exports.saveUserSettings = saveUserSettings
exports.deleteFolderRecursive = deleteFolderRecursive

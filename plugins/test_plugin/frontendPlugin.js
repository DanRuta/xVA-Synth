"use strict"

const preStartFn = (window, data) => {
    window.appLogger.log("preStartFn")
    console.log("preKeepSample")
}

const postStartFn = (window, data) => {
    window.appLogger.log("postStartFn")
    console.log("postStartFn")
}

const preKeepSample = (window, data) => {
    window.appLogger.log("preKeepSample", data)
    console.log("preKeepSample data", data)
}

const postKeepSample = (window, data) => {
    window.appLogger.log("postKeepSample")
    console.log("postKeepSample")
}

// Optional
// =======
const setup = () => {
    window.appLogger.log("setup")
    console.log("setup")
}
// =======

exports.preStartFn = preStartFn
exports.postStartFn = postStartFn
exports.preKeepSample = preKeepSample
exports.postKeepSample = postKeepSample
exports.setup = setup
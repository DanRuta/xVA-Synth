! function(t, e) {
    "object" == typeof exports && "undefined" != typeof module ? module.exports = e(require("three")) : "function" == typeof define && define.amd ? define(["three"], e) : ((t = "undefined" != typeof globalThis ? globalThis : t || self).THREE = t.THREE || {}, t.THREE.TextTexture = e(t.THREE))
}(this, (function(t) {
    "use strict";
    let e = class extends t.Texture {
        constructor() {
            super(document.createElement("canvas"));
            let e = null,
                i = () => e || (e = this.createDrawable()),
                n = () => i().width,
                o = () => i().height,
                r = !0,
                l = 1,
                a = () => t.MathUtils.ceilPowerOfTwo(n() * l),
                s = () => t.MathUtils.ceilPowerOfTwo(o() * l),
                h = t => {
                    if (l !== t) {
                        let e = a(),
                            i = s();
                        l = t;
                        let n = a(),
                            o = s();
                        n === e && o === i || (r = !0)
                    }
                },
                c = (() => {
                    let e = new t.Vector3,
                        i = new t.Vector2,
                        r = new t.Vector3,
                        l = new t.Vector3,
                        a = new t.Vector2;
                    return (s, h, c) => {
                        if (a.set(n(), o()), a.x && a.y) {
                            s.getWorldPosition(r), c.getWorldPosition(e);
                            let n = r.distanceTo(e);
                            if (c.isPerspectiveCamera && (n *= 2 * Math.tan(t.MathUtils.degToRad(c.fov) / 2)), (c.isPerspectiveCamera || c.isOrthographicCamera) && (n /= c.zoom), n) {
                                var f, d;
                                s.getWorldScale(l);
                                let t = null !== (f = null === (d = h.capabilities) || void 0 === d ? void 0 : d.maxTextureSize) && void 0 !== f ? f : 1 / 0;
                                return h.getDrawingBufferSize(i), Math.min(Math.max(l.x / n * (i.x / a.x), l.y / n * (i.y / a.y)), t / a.x, t / a.y)
                            }
                        }
                        return 0
                    }
                })();
            Object.defineProperties(this, {
                width: {
                    get: n
                },
                height: {
                    get: o
                },
                pixelRatio: {
                    get: () => l,
                    set: h
                },
                needsRedraw: {
                    set(t) {
                        t && (r = !0, e = null)
                    }
                }
            }), Object.assign(this, {
                redraw() {
                    if (r) {
                        let t = this.image,
                            e = t.getContext("2d");
                        e.clearRect(0, 0, t.width, t.height), t.width = a(), t.height = s(), t.width && t.height ? (e.save(), e.scale(t.width / n(), t.height / o()), ((...t) => {
                            i().draw(...t)
                        })(e), e.restore()) : t.width = t.height = 1, r = !1, this.needsUpdate = !0
                    }
                },
                setOptimalPixelRatio(...t) {
                    h(c(...t))
                }
            })
        }
    };
    e.prototype.isDynamicTexture = !0;
    let i = class extends e {
        constructor({
            alignment: t = "center",
            backgroundColor: e = "rgba(0,0,0,0)",
            color: i = "#fff",
            fontFamily: n = "sans-serif",
            fontSize: o = 16,
            fontStyle: r = "normal",
            fontVariant: l = "normal",
            fontWeight: a = "normal",
            lineGap: s = 1 / 4,
            padding: h = .5,
            strokeColor: c = "#fff",
            strokeWidth: f = 0,
            text: d = ""
        } = {}) {
            super(), Object.entries({
                alignment: t,
                backgroundColor: e,
                color: i,
                fontFamily: n,
                fontSize: o,
                fontStyle: r,
                fontVariant: l,
                fontWeight: a,
                lineGap: s,
                padding: h,
                strokeColor: c,
                strokeWidth: f,
                text: d
            }).forEach((([t, e]) => {
                Object.defineProperty(this, t, {
                    get: () => e,
                    set(t) {
                        e !== t && (e = t, this.needsRedraw = !0)
                    }
                })
            }))
        }
        get lines() {
            let {
                text: t
            } = this;
            return t ? t.split("\n") : []
        }
        get font() {
            return function(t, e, i, n, o) {
                let r = document.createElement("span");
                return r.style.font = "1px serif", r.style.fontFamily = t, r.style.fontSize = "".concat(e, "px"), r.style.fontStyle = i, r.style.fontVariant = n, r.style.fontWeight = o, r.style.font
            }(this.fontFamily, this.fontSize, this.fontStyle, this.fontVariant, this.fontWeight)
        }
        checkFontFace() {
            try {
                let {
                    font: t
                } = this;
                return document.fonts.check(t)
            } catch (e) {}
            return !0
        }
        async loadFontFace() {
            try {
                let {
                    font: t
                } = this;
                await document.fonts.load(t)
            } catch (e) {}
        }
        createDrawable() {
            let {
                alignment: t,
                backgroundColor: e,
                color: i,
                font: n,
                fontSize: o,
                lineGap: r,
                lines: l,
                padding: a,
                strokeColor: s,
                strokeWidth: h
            } = this;
            a *= o, r *= o, h *= o;
            let c = l.length,
                f = o + r,
                d = c ? (() => {
                    let t = document.createElement("canvas").getContext("2d");
                    return t.font = n, Math.max(...l.map((e => t.measureText(e).width)))
                })() : 0,
                g = a + h / 2,
                u = d + 2 * g;
            return {
                width: u,
                height: (c ? o + f * (c - 1) : 0) + 2 * g,
                draw(r) {
                    let a;
                    r.fillStyle = e, r.fillRect(0, 0, r.canvas.width, r.canvas.height);
                    let c = g + o / 2;
                    Object.assign(r, {
                        fillStyle: i,
                        font: n,
                        lineWidth: h,
                        miterLimit: 1,
                        strokeStyle: s,
                        textAlign: (() => {
                            switch (t) {
                                case "left":
                                    return a = g, "left";
                                case "right":
                                    return a = u - g, "right"
                            }
                            return a = u / 2, "center"
                        })(),
                        textBaseline: "middle"
                    }), l.forEach((t => {

                        // r.lineWidth=60
                        // r.shadowColor="white"
                        // r.shadowBlur=2
                        // r.fillStyle = "white"
                        // r.fillText(t, a, c);

                        r.lineWidth=5
                        r.shadowColor="black"
                        r.shadowBlur=0
                        r.fillStyle = "black"
                        r.fillText(t, a, c);

                        h && r.strokeText(t, a, c), c += f
                    }))
                }
            }
        }
    };
    return i.prototype.isTextTexture = !0, i
}));
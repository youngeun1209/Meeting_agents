// Renders the app icon to a 1024x1024 PNG with CoreGraphics.
//
// Why not an SVG: the only SVG rasteriser on this machine is ImageMagick's built-in MSVG
// renderer (librsvg is not installed), and it silently drops linearGradient and filter
// elements -- the first attempt came out with a black background and no microphone arc.
// CoreGraphics is already present, exact, and needs no new dependency.
//
// Build + run:  swiftc -O MakeIcon.swift -o makeicon && ./makeicon out.png
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let S: CGFloat = 1024
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon-1024.png"

let cs = CGColorSpaceCreateDeviceRGB()
guard let ctx = CGContext(data: nil, width: Int(S), height: Int(S), bitsPerComponent: 8,
                          bytesPerRow: 0, space: cs,
                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
    fatalError("could not create context")
}
// Flip to top-left origin so the geometry below reads like screen coordinates.
ctx.translateBy(x: 0, y: S)
ctx.scaleBy(x: 1, y: -1)

func rgb(_ r: Int, _ g: Int, _ b: Int, _ a: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: cs, components: [CGFloat(r)/255, CGFloat(g)/255, CGFloat(b)/255, a])!
}

// --- background: indigo gradient, full bleed (iOS applies the corner mask itself) ---
let bg = CGGradient(colorsSpace: cs, colors: [rgb(124,131,247), rgb(99,102,241), rgb(67,56,202)] as CFArray,
                    locations: [0, 0.55, 1])!
ctx.drawLinearGradient(bg, start: .zero, end: CGPoint(x: S, y: S), options: [])

func roundedRect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ r: CGFloat) -> CGPath {
    CGPath(roundedRect: CGRect(x: x, y: y, width: w, height: h), cornerWidth: r, cornerHeight: r, transform: nil)
}

func shadow(_ on: Bool) {
    if on { ctx.setShadow(offset: CGSize(width: 0, height: -18), blur: 40, color: rgb(30, 27, 75, 0.32)) }
    else { ctx.setShadow(offset: .zero, blur: 0, color: nil) }
}

// --- notepad ---
shadow(true)
ctx.addPath(roundedRect(252, 180, 520, 664, 46)); ctx.setFillColor(rgb(255,255,255)); ctx.fillPath()
shadow(false)

// top binding strip
ctx.saveGState()
ctx.addPath(roundedRect(252, 180, 520, 664, 46)); ctx.clip()
ctx.addPath(roundedRect(252, 180, 520, 100, 0)); ctx.setFillColor(rgb(226,229,255)); ctx.fillPath()
ctx.restoreGState()

// spiral rings straddling the top edge
ctx.setFillColor(rgb(129,140,248))
for x in stride(from: CGFloat(332), through: 632, by: 100) {
    ctx.addPath(roundedRect(x, 146, 28, 92, 14)); ctx.fillPath()
}

// ruled lines
ctx.setFillColor(rgb(199,203,227))
for (y, w) in [(CGFloat(360), CGFloat(392)), (432, 392), (504, 236)] {
    ctx.addPath(roundedRect(316, y, w, 22, 11)); ctx.fillPath()
}

// --- microphone ---
let micTop = rgb(79,70,229), micBot = rgb(55,48,163)
let micGrad = CGGradient(colorsSpace: cs, colors: [micTop, micBot] as CFArray, locations: [0, 1])!

shadow(true)
// capsule body -- filled via a clipped gradient so it keeps the depth of the original design
ctx.saveGState()
ctx.addPath(roundedRect(452, 512, 120, 216, 60))
ctx.setFillColor(micTop); ctx.fillPath()          // opaque base so the shadow is cast
ctx.restoreGState()
shadow(false)

ctx.saveGState()
ctx.addPath(roundedRect(452, 512, 120, 216, 60)); ctx.clip()
ctx.drawLinearGradient(micGrad, start: CGPoint(x: 0, y: 512), end: CGPoint(x: 0, y: 728), options: [])
ctx.restoreGState()

// the U-shaped pickup arc under the capsule
ctx.setStrokeColor(micBot); ctx.setLineWidth(34); ctx.setLineCap(.round)
ctx.addArc(center: CGPoint(x: 512, y: 654), radius: 118, startAngle: 0, endAngle: .pi, clockwise: false)
ctx.strokePath()

// stem + base
ctx.setFillColor(micBot)
ctx.addPath(roundedRect(497, 754, 30, 68, 15)); ctx.fillPath()
ctx.addPath(roundedRect(432, 810, 160, 32, 16)); ctx.fillPath()

guard let image = ctx.makeImage() else { fatalError("no image") }
let url = URL(fileURLWithPath: out)
guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    fatalError("no destination")
}
CGImageDestinationAddImage(dest, image, nil)
guard CGImageDestinationFinalize(dest) else { fatalError("write failed") }
print("wrote \(out)")

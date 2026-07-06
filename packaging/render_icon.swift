// Renders the app icon (microphone + note) to a 1024x1024 PNG using AppKit.
// Usage: swiftc render_icon.swift -o render_icon && ./render_icon out.png
import AppKit

let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon_1024.png"
let S: CGFloat = 1024

let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(S), pixelsHigh: Int(S),
                           bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                           isPlanar: false, colorSpaceName: .deviceRGB,
                           bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let ctx = NSGraphicsContext.current!.cgContext

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: r/255, green: g/255, blue: b/255, alpha: a)
}

// --- Background squircle with vertical gradient -----------------------------
let bgRect = NSRect(x: 48, y: 48, width: S-96, height: S-96)
let bgPath = NSBezierPath(roundedRect: bgRect, xRadius: 210, yRadius: 210)
bgPath.addClip()
let grad = NSGradient(starting: color(90, 132, 255), ending: color(108, 74, 240))!
grad.draw(in: bgRect, angle: -90)

// --- Note card (tilted, behind the mic) -------------------------------------
ctx.saveGState()
let cardCenter = NSPoint(x: 430, y: 560)
ctx.translateBy(x: cardCenter.x, y: cardCenter.y)
ctx.rotate(by: -8 * .pi / 180)
// soft shadow
let shadow = NSShadow()
shadow.shadowColor = color(20, 20, 60, 0.35)
shadow.shadowBlurRadius = 34
shadow.shadowOffset = NSSize(width: 0, height: -18)
shadow.set()
let cardRect = NSRect(x: -210, y: -260, width: 420, height: 520)
let card = NSBezierPath(roundedRect: cardRect, xRadius: 44, yRadius: 44)
color(255, 255, 255).setFill()
card.fill()
// clear shadow for the text lines
NSShadow().set()
// gray "text" lines near the top
color(176, 186, 214).setFill()
let lineX: CGFloat = -150
let lineW: [CGFloat] = [300, 300, 240, 300, 190]
var lineY: CGFloat = 150
for w in lineW {
    let l = NSBezierPath(roundedRect: NSRect(x: lineX, y: lineY, width: w, height: 26),
                         xRadius: 13, yRadius: 13)
    l.fill()
    lineY -= 66
}
ctx.restoreGState()

// --- Microphone (front, bottom-right) ---------------------------------------
ctx.saveGState()
let micShadow = NSShadow()
micShadow.shadowColor = color(20, 20, 60, 0.30)
micShadow.shadowBlurRadius = 30
micShadow.shadowOffset = NSSize(width: 0, height: -16)
micShadow.set()

let micCX: CGFloat = 660
// capsule body
let bodyW: CGFloat = 168
let bodyRect = NSRect(x: micCX - bodyW/2, y: 470, width: bodyW, height: 330)
let body = NSBezierPath(roundedRect: bodyRect, xRadius: bodyW/2, yRadius: bodyW/2)
let micGrad = NSGradient(starting: color(255, 255, 255), ending: color(226, 232, 250))!
micGrad.draw(in: body, angle: -90)
NSShadow().set()

// U-shaped cradle (stroke arc)
let cradle = NSBezierPath()
let cr = NSRect(x: micCX - 150, y: 330, width: 300, height: 300)
cradle.appendArc(withCenter: NSPoint(x: cr.midX, y: cr.midY),
                 radius: 150, startAngle: 200, endAngle: 340, clockwise: false)
color(255, 255, 255).setStroke()
cradle.lineWidth = 34
cradle.lineCapStyle = .round
cradle.stroke()

// stem + base
let stem = NSBezierPath(rect: NSRect(x: micCX - 17, y: 250, width: 34, height: 120))
color(255, 255, 255).setFill()
stem.fill()
let base = NSBezierPath(roundedRect: NSRect(x: micCX - 95, y: 226, width: 190, height: 34),
                        xRadius: 17, yRadius: 17)
base.fill()

// grille lines on the mic body
color(176, 186, 214, 0.9).setStroke()
for i in 0..<3 {
    let gy = 690 - CGFloat(i) * 52
    let g = NSBezierPath()
    g.move(to: NSPoint(x: micCX - 52, y: gy))
    g.line(to: NSPoint(x: micCX + 52, y: gy))
    g.lineWidth = 12
    g.lineCapStyle = .round
    g.stroke()
}
ctx.restoreGState()

NSGraphicsContext.restoreGraphicsState()

let png = rep.representation(using: .png, properties: [:])!
try! png.write(to: URL(fileURLWithPath: outPath))
print("wrote \(outPath)")

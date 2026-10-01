// Figure 1 -- HoloQPI overview, fully editable (native shapes and text).
// Real-data panels are inserted as images; network outputs are placeholder
// boxes named after the PNG that make_fig1_panels.py writes.
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const PANELS = process.argv[2];              // folder with p1_*/p2_* PNGs
const OUT = process.argv[3];

const pres = new pptxgen();
pres.defineLayout({ name: "FIG1", width: 13.333, height: 7.2 });
pres.layout = "FIG1";
pres.title = "HoloQPI Figure 1 overview";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// palette (same slots as the manuscript's generated figures)
const C = {
  ink: "1A1A1A", ink2: "555555", line: "9A9A9A", faint: "F4F6F8",
  blue: "0072B2", blueL: "DCEAF4", orange: "D55E00", orangeL: "FBE6DA",
  purple: "882255", purpleL: "F2E1EA", green: "009E73", greenL: "D9F0E8",
  grey: "6B6B6B", greyL: "EDEDED", gold: "B8860B",
};
const F = "Arial";

// ---------- helpers ----------
function txt(t, x, y, w, h, o = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: F, fontSize: 9, color: C.ink,
    margin: 0, valign: "top", isTextBox: true }, o));
}
function box(x, y, w, h, fill, line, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h,
    fill: { color: fill }, line: { color: line, width: 1 }, rectRadius: 0.06 }, o));
}
function rect(x, y, w, h, fill, line, o = {}) {
  s.addShape(pres.shapes.RECTANGLE, Object.assign({ x, y, w, h,
    fill: { color: fill }, line: line ? { color: line, width: 0.75 } : { type: "none" } }, o));
}
function arrow(x1, y1, x2, y2, color = C.ink2, o = {}) {
  s.addShape(pres.shapes.LINE, Object.assign({ x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: 1.5, endArrowType: "triangle" } }, o));
}
function line(x1, y1, x2, y2, color = C.line, o = {}) {
  s.addShape(pres.shapes.LINE, Object.assign({ x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1 } }, o));
}
function img(file, x, y, w, h, o = {}) {
  const p = path.join(PANELS, file);
  if (fs.existsSync(p)) {
    s.addImage(Object.assign({ path: p, x, y, w, h, altText: file }, o));
    rect(x, y, w, h, "FFFFFF", C.line, { fill: { type: "none" } });
    return true;
  }
  return placeholder(file, x, y, w, h);
}
function placeholder(file, x, y, w, h, note) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "EFEFEF" },
    line: { color: "8C8C8C", width: 0.75, dashType: "dash" } });
  if (w < 0.5) return false;                 // too narrow for a label (colour bars)
  txt([{ text: "PLACEHOLDER", options: { bold: true, fontSize: 6.5, color: "7A7A7A", breakLine: true } },
       { text: file, options: { fontSize: 6.3, color: "4A4A4A", breakLine: !!note } },
       ...(note ? [{ text: note, options: { fontSize: 6.5, color: "7A7A7A", italic: true } }] : [])],
      x + 0.04, y + 0.04, w - 0.08, h - 0.08, { align: "center", valign: "middle" });
  return false;
}
function header(letter, title, x, y, w, color) {
  txt([{ text: letter + "  ", options: { bold: true, color } },
       { text: title, options: { bold: true, color: C.ink } }], x, y, w, 0.3,
      { fontSize: 12.5, valign: "middle" });
}
function chip(t, x, y, w, h, fill, line_, color, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h,
    fill: { color: fill }, line: { color: line_, width: 0.75 }, rectRadius: 0.05 }, o));
  txt(t, x, y, w, h, Object.assign({ align: "center", valign: "middle", fontSize: 8, color }, o.text || {}));
}

// ---------- layout ----------
const TOP = 0.18, ROW_H = 5.0;               // upper row (a)-(d)
const BOT = TOP + ROW_H + 0.17, BOT_H = 1.72; // lower row (e)-(f)
const colA = { x: 0.18, w: 2.62 }, colB = { x: 3.02, w: 3.86 },
      colC = { x: 7.10, w: 2.46 }, colD = { x: 9.78, w: 3.37 };

// panel frames
[[colA, C.blue], [colB, C.blue], [colC, C.blue], [colD, C.blue]].forEach(([c]) =>
  box(c.x, TOP, c.w, ROW_H, "FFFFFF", "C9CED6"));

// ======================= (a) input =======================
header("a", "Raw hologram input", colA.x + 0.12, TOP + 0.08, colA.w - 0.2, C.blue);
txt("Off-axis hologram (one channel, 8-bit), cropped 1024² → 900² onto the phase grid (origin moved by (−6, −2) px)",
    colA.x + 0.12, TOP + 0.4, colA.w - 0.24, 0.36, { fontSize: 7.5, color: C.ink2 });
const aImg = { x: colA.x + 0.2, y: TOP + 0.8, w: 2.1 };
img("p1_offaxis_hologram.png", aImg.x, aImg.y, aImg.w, aImg.w);
// fringe detail and spectrum
const smallW = 1.0, sy = aImg.y + aImg.w + 0.08;
img("p1_offaxis_zoom.png", aImg.x, sy, smallW, smallW);
img("p1_offaxis_fft.png", aImg.x + aImg.w - smallW, sy, smallW, smallW);
txt("fringe detail", aImg.x, sy + smallW + 0.02, smallW, 0.16, { fontSize: 7, color: C.ink2, align: "center" });
txt("|FFT|: DC + sidebands", aImg.x + aImg.w - smallW - 0.1, sy + smallW + 0.02, smallW + 0.2, 0.16,
    { fontSize: 7, color: C.ink2, align: "center" });
// in-line inset
const iy = sy + smallW + 0.22;
img("p1_inline_hologram.png", aImg.x, iy, 0.58, 0.58);
txt([{ text: "In-line (Gabor) hologram", options: { bold: true, breakLine: true } },
     { text: "same field; same architecture trained separately (In-Line Neural Configuration)" }],
    aImg.x + 0.66, iy - 0.02, aImg.w - 0.6, 0.62, { fontSize: 6.8, color: C.ink2, valign: "middle" });

// ======================= (b) network =======================
header("b", "End-to-end network", colB.x + 0.12, TOP + 0.08, colB.w - 0.2, C.blue);
txt("One shared encoder, two decoders; no classical reconstruction in the inference path",
    colB.x + 0.12, TOP + 0.4, colB.w - 0.24, 0.3, { fontSize: 7.5, color: C.ink2 });

// encoder: 5 feature maps at strides 2..32 (tallest -> shortest), centred vertically
const encX0 = colB.x + 0.22, encMid = TOP + 2.25;
const encH = [1.75, 1.42, 1.1, 0.8, 0.55], encW = 0.15, encGap = 0.09;
encH.forEach((h, i) => {
  const x = encX0 + i * (encW + encGap);
  s.addShape(pres.shapes.RECTANGLE, { x, y: encMid - h / 2, w: encW, h,
    fill: { color: C.blue, transparency: 10 + i * 8 }, line: { color: "FFFFFF", width: 0.5 } });
});
txt([{ text: "Shared encoder", options: { bold: true, breakLine: true } },
     { text: "MobileNetV2 (ImageNet init.), 1 input channel; features at strides 2–32" }],
    encX0 - 0.05, encMid + 0.95, 1.75, 0.5, { fontSize: 6.8, color: C.ink2 });

// two decoders
const decX0 = encX0 + 5 * (encW + encGap) + 0.4;
const decH = [0.6, 0.78, 0.96, 1.14];
function decoder(yMid, color, label, heads) {
  decH.forEach((h, i) => {
    const x = decX0 + i * (encW + 0.06);
    s.addShape(pres.shapes.RECTANGLE, { x, y: yMid - h / 2 * 0.55, w: encW, h: h * 0.55,
      fill: { color, transparency: 35 - i * 8 }, line: { color: "FFFFFF", width: 0.5 } });
  });
  txt(label, decX0 - 0.05, yMid + 0.36, 1.1, 0.3, { fontSize: 6.8, color: C.ink2 });
  // heads
  const hx = decX0 + 4 * (encW + 0.06) + 0.1;
  heads.forEach(([t, dashed, col], j) => {
    const hy = yMid - 0.2 + (j - (heads.length - 1) / 2) * 0.34 - (heads.length === 1 ? -0.07 : 0);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: hx, y: hy, w: 0.98, h: 0.27,
      fill: { color: "FFFFFF" }, line: { color: col, width: 1, dashType: dashed ? "dash" : "solid" },
      rectRadius: 0.05 });
    txt(t, hx, hy, 0.98, 0.27, { fontSize: 6.8, align: "center", valign: "middle", color: C.ink });
  });
  return hx;
}
const phaseY = encMid - 0.85, segY = encMid + 0.7;
const headX = decoder(phaseY, C.blue, "Phase decoder\n256/128/64/32",
  [["phase head (rad)", false, C.blue], ["amplitude head*", true, C.purple]]);
decoder(segY, C.blue, "Segmentation decoder\n256/128/64/32", [["2-class head", false, C.blue]]);
// encoder -> decoders
const encR = encX0 + 5 * (encW + encGap) - encGap;
arrow(encR + 0.04, encMid - 0.1, decX0 - 0.04, phaseY, C.ink2);
arrow(encR + 0.04, encMid + 0.1, decX0 - 0.04, segY, C.ink2);
// skip connections hint
txt("U-Net skip connections from every encoder stage; outputs at stride 2, bilinearly upsampled to 900²",
    colB.x + 0.12, TOP + 3.78, colB.w - 0.24, 0.32, { fontSize: 6.8, color: C.ink2, italic: true });
// size chips
chip([{ text: "9.60 M parameters · 45.85 GMAC", options: { bold: true } }],
     colB.x + 0.14, TOP + 4.33, 1.82, 0.28, C.blueL, C.blue, C.ink, { text: { fontSize: 7 } });
chip([{ text: "Compact: 256-ch projection, 3.36 M", options: { bold: true } }],
     colB.x + 2.02, TOP + 4.33, 1.72, 0.28, C.greenL, C.green, C.ink, { text: { fontSize: 7 } });
txt("* amplitude head only in +Amplitude and +Fwd configurations",
    colB.x + 0.14, TOP + 4.66, colB.w - 0.28, 0.3, { fontSize: 6.3, color: C.ink2 });

// ======================= (c) outputs =======================
header("c", "Predicted fields", colC.x + 0.12, TOP + 0.08, colC.w - 0.2, C.blue);
const oW = 1.12, oX = colC.x + 0.2, cbW = 0.27, cbH = 0.96;
const outs = [
  ["p3_phase_pred.png", "Quantitative phase (rad)", "p3_phase_colorbar.png", TOP + 0.48],
  ["p3_amplitude_pred.png", "Transmitted amplitude*", "p3_amplitude_colorbar.png", TOP + 2.0],
  ["p3_segmentation_pred.png", "Cell segmentation (instances)", null, TOP + 3.52],
];
// Colour-bar tick labels are editable text. Phase range = reference-phase
// display range read from fig1_panels.json (shared by the
// reference and predicted maps). Amplitude range: fill in from the json.
const INFO = JSON.parse(fs.readFileSync(path.join(PANELS, "fig1_panels.json"), "utf8"));
const [PLO, PHI] = INFO.phase_display_range_rad;
const PV = [4, 2, 0, -2].filter(v => v >= PLO && v <= PHI);
const ticks = {
  "p3_phase_colorbar.png": { lo: PLO, hi: PHI, marks: PV.map(v => (v < 0 ? "−" : "") + Math.abs(v)), vals: PV },
  "p3_amplitude_colorbar.png": { marks: ["max", "min"], pos: [0, 1] },
};
outs.forEach(([f, t1, cb, y]) => {
  img(f, oX, y, oW, oW);
  if (cb) {
    const cy0 = y + (oW - cbH) / 2, cbw = 0.12;
    img(cb, oX + oW + 0.06, cy0, cbw, cbH);
    const t = ticks[cb];
    const fr = t.vals ? t.vals.map(v => (t.hi - v) / (t.hi - t.lo)) : t.pos;
    t.marks.forEach((m, k) => txt(m, oX + oW + 0.06 + cbw + 0.04, cy0 + fr[k] * cbH - 0.07, 0.35, 0.14,
      { fontSize: 6.3, color: C.ink2, valign: "middle" }));
  }
  txt(t1, oX - 0.06, y + oW + 0.03, oW + 0.5, 0.2, { fontSize: 7.2, bold: true, color: C.ink });
});
txt("rad", oX + oW + 0.02, TOP + 0.48 + (oW - cbH) / 2 - 0.17, 0.3, 0.14, { fontSize: 6.3, color: C.ink2 });
txt("* +Amplitude / +Fwd only; agreement is with a reconstruction-derived amplitude, not a measured one",
    oX + oW + 0.5, TOP + 2.02, colC.w - oW - 0.62, 1.1, { fontSize: 6, color: C.ink2, italic: true });
txt("2-class map →\nwatershed\ninstances;\ncolour =\nidentity only", oX + oW + 0.1, TOP + 3.6, colC.w - oW - 0.3, 0.8,
    { fontSize: 6.5, color: C.ink2 });

// ======================= (d) measurement =======================
header("d", "Per-cell measurement", colD.x + 0.12, TOP + 0.08, colD.w - 0.2, C.blue);
const mW = 1.6;
img("p4_measurement_overlay.png", colD.x + 0.16, TOP + 0.5, mW, mW);
txt([{ text: "Cell k, predicted phase φ over its domain Ωₖ:", options: { bold: true, breakLine: true } },
     { text: "Aₖ = Nₖ dx dy", options: { breakLine: true } },
     { text: "Cₖ = min(1, 4π Nₖ / Pₖ²)", options: { breakLine: true } },
     { text: "Sₖ = Σ φ dx dy", options: { breakLine: true } },
     { text: "mₖ = λ / (2πα) · Sₖ", options: { bold: true } }],
    colD.x + mW + 0.24, TOP + 0.52, colD.w - mW - 0.32, 1.25, { fontSize: 7.2, color: C.ink, paraSpaceAfter: 3 });
txt("λ = 0.666 µm\nα = 0.2 mL/g\ndx = dy = 0.285 µm", colD.x + mW + 0.24, TOP + 1.62, colD.w - mW - 0.32, 0.5,
    { fontSize: 6.5, color: C.ink2 });

// measurement list cards
const cards = [
  ["Dry mass", "pg per cell (primary quantity)"],
  ["Projected area", "µm² per cell"],
  ["Circularity", "shape descriptor, 0–1"],
  ["Integrated phase Sₖ", "rad·µm² per cell (30–6000 µm² cells)"],
];
cards.forEach(([t1, t2], i) => {
  const y = TOP + 2.42 + i * 0.47;
  box(colD.x + 0.16, y, colD.w - 0.32, 0.4, i === 0 ? C.blueL : C.faint, i === 0 ? C.blue : "D5DAE0");
  txt([{ text: t1 + "  ", options: { bold: true } }, { text: t2, options: { color: C.ink2 } }],
      colD.x + 0.28, y, colD.w - 0.5, 0.4, { fontSize: 7.8, valign: "middle" });
});
txt("Counts: detection recall and precision reported alongside, because per-cell errors are computed only for detected and matched cells",
    colD.x + 0.16, TOP + 4.35, colD.w - 0.32, 0.55, { fontSize: 6.6, color: C.ink2, italic: true });

// flow arrows between upper panels
const ay = TOP + 2.2;
arrow(colA.x + colA.w + 0.02, ay, colB.x - 0.02, ay, C.blue);
arrow(colB.x + colB.w + 0.02, ay, colC.x - 0.02, ay, C.blue);
arrow(colC.x + colC.w + 0.02, ay, colD.x - 0.02, ay, C.blue);

// ======================= (e) training =======================
const eX = colA.x, eW = colB.x + colB.w - colA.x;
box(eX, BOT, eW, BOT_H, "FFFFFF", "C9CED6");
header("e", "Supervision and training objective", eX + 0.12, BOT + 0.06, 4.2, C.blue);
const rW = 0.78, rY = BOT + 0.42;
img("p2_reference_phase.png", eX + 0.2, rY, rW, rW);
img("p2_reference_labels.png", eX + 0.2 + rW + 0.08, rY, rW, rW);
txt("reference phase", eX + 0.2, rY + rW + 0.02, rW, 0.15, { fontSize: 6.5, color: C.ink2, align: "center" });
txt("Otsu labels", eX + 0.28 + rW, rY + rW + 0.02, rW, 0.15, { fontSize: 6.5, color: C.ink2, align: "center" });
txt("labels are derived from the reference phase (no manual annotation)", eX + 0.16, rY + rW + 0.2, 1.8, 0.28,
    { fontSize: 6.2, color: C.ink2, italic: true });
// loss boxes
const lx = eX + 2.05, lw = eW - 2.2;
txt([{ text: "Every configuration:  ", options: { bold: true } },
     { text: "ℒphase = L1 + 0.5·|∇| + 0.2·(1−SSIM)    ℒseg = Dice + CE" }],
    lx, BOT + 0.43, lw, 0.24, { fontSize: 7.5 });
txt("Added one term at a time (ablations):", lx, BOT + 0.72, lw, 0.2, { fontSize: 7.2, bold: true, color: C.ink2 });
txt("measurement-aware", lx, BOT + 0.98, 1.2, 0.27, { fontSize: 6.8, color: C.orange, bold: true, valign: "middle" });
let cx = lx + 1.2;
[["+IPP (per-cell)", 0.95], ["+IPP (image)", 0.85], ["+Area", 0.55], ["+BGA", 0.55]].forEach(([t, w]) => {
  chip(t, cx, BOT + 0.98, w, 0.27, C.orangeL, C.orange, C.ink, { text: { fontSize: 7 } }); cx += w + 0.07; });
txt("output / forward model", lx, BOT + 1.3, 1.2, 0.27, { fontSize: 6.8, color: C.purple, bold: true, valign: "middle" });
cx = lx + 1.2;
chip("+Amplitude", cx, BOT + 1.3, 0.8, 0.27, C.purpleL, C.purple, C.ink, { text: { fontSize: 7 } });
arrow(cx + 0.82, BOT + 1.435, cx + 0.98, BOT + 1.435, C.purple);
chip("+Fwd (fixed / free z)", cx + 1.0, BOT + 1.3, 1.3, 0.27, C.purpleL, C.purple, C.ink, { text: { fontSize: 7 } });
txt("hologram forward model on Â·exp(iφ)", cx + 2.38, BOT + 1.3, lw - 3.6, 0.27,
    { fontSize: 6.2, color: C.ink2, italic: true, valign: "middle" });

// ======================= (f) evaluation =======================
const fX = colC.x, fW = colD.x + colD.w - colC.x;
box(fX, BOT, fW, BOT_H, "FFFFFF", "C9CED6");
header("f", "Evaluation on the measurand", fX + 0.12, BOT + 0.06, 4, C.blue);
const evW = (fW - 0.44) / 3;
const ev = [
  ["vs reference", "per-cell MAPE over matched cells (IoU ≥ 0.5), with recall, coverage-adjusted MAPE and per-field Bland–Altman", C.blueL, C.blue],
  ["vs Classical Pipeline", "angular-spectrum reconstruction + the same segmentation labelling and measurement chain; off-axis and in-line", C.greyL, C.grey],
  ["Model-free error floor", "known boundary shifts propagated to area and mass; chain checked on analytic fields", C.faint, "D5DAE0"],
];
ev.forEach(([t1, t2, fill, ln], i) => {
  const x = fX + 0.16 + i * (evW + 0.06);
  box(x, BOT + 0.42, evW, 0.98, fill, ln);
  txt([{ text: t1, options: { bold: true, breakLine: true } }, { text: t2, options: { color: C.ink2 } }],
      x + 0.08, BOT + 0.48, evW - 0.16, 1.05, { fontSize: 7, paraSpaceAfter: 2 });
});

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f));

'use strict';
const fs = require('node:fs');
const PptxGenJS = require('pptxgenjs');

async function generate(project, output) {
  const pptx = new PptxGenJS();
  const scale = project.settings.inches_per_pixel;
  const width = project.source.width * scale;
  const height = project.source.height * scale;
  pptx.defineLayout({ name: 'SOURCE', width, height });
  pptx.layout = 'SOURCE';
  pptx.author = 'Smart Overlay OCR';
  pptx.subject = 'Editable OCR overlays; original raster preserved';
  pptx.lang = 'en-US';
  const slide = pptx.addSlide();
  const nativePdf = project.source.mode === 'PDF';
  const originalPdf = nativePdf && project.native_pdf_background_mode === 'full_original';
  const sourceExact = originalPdf && project.native_pdf_overlay_mode === 'source_exact';
  const fragmentById = new Map((project.fragments || []).map(f => [f.id, f]));
  let maskCount = 0;

  // Full native PDF is the untouched page raster. Replace ONLY the ink of
  // original PDF glyphs with small OCR-like white masks, never whole table
  // cells or the entire page. OCR and legacy-cleaned PDFs are unchanged.
  function maskOriginalGlyphs(ownerId, ids, fillColor, clipPx = null) {
    if (!originalPdf || sourceExact) return;
    for (const id of ids || []) {
      const fragment = fragmentById.get(id);
      if (!fragment || !fragment.text.trim()) continue;
      const b = fragment.bbox;
      const pad = Math.min(0.8, Math.max(0.15, b.h * 0.025));
      const inset = 0.6;
      const x0 = Math.max(0, b.x - pad, clipPx ? clipPx.x + inset : 0);
      const y0 = Math.max(0, b.y - pad, clipPx ? clipPx.y + inset : 0);
      const x1 = Math.min(project.source.width, b.x + b.w + pad,
                          clipPx ? clipPx.x + clipPx.w - inset : project.source.width);
      const y1 = Math.min(project.source.height, b.y + b.h + pad,
                          clipPx ? clipPx.y + clipPx.h - inset : project.source.height);
      if (x1 <= x0 || y1 <= y0) continue;
      slide.addShape(pptx.ShapeType.rect, {
        x: x0*scale, y: y0*scale, w: (x1-x0)*scale, h: (y1-y0)*scale,
        fill: { color: hex(fillColor) },
        line: { color: hex(fillColor), transparency: 100 },
        objectName: `${ownerId}_mask_${maskCount++}`
      });
    }
  }
  slide.addImage({ data: `${project.source.raster_mime};base64,${project.source.raster_b64}`,
                   x: 0, y: 0, w: width, h: height, altText: project.source.filename });
  const hex = value => value.replace('#', '');
  const style = s => ({ fontFace: s.font, fontSize: s.font_size, color: hex(s.color),
                        transparency: sourceExact ? 100 : 0,
                        fill: { color: hex(s.fill), transparency: nativePdf ? 100 : 0 }, align: s.align,
                        margin: 0, breakLine: false, valign: 'mid' });
  for (const obj of project.objects) {
    if (obj.excluded) continue;
    // Empty review regions carry no replacement text and must not erase the source.
    if (obj.type === 'text' && !obj.text.trim()) continue;
    if (obj.needs_review && project.allow_unreviewed !== true) throw new Error(`Review required: ${obj.id}`);
    const b = obj.type === 'text' && obj.frame ? obj.frame : obj.bbox;
    const position = { x: b.x*scale, y: b.y*scale, w: b.w*scale, h: b.h*scale };
    if (obj.type === 'text') {
      if (originalPdf) maskOriginalGlyphs(obj.id, obj.ocr_ids, obj.style.fill);
      if (!nativePdf && !project.inline_text_fill) for (const mask of obj.masks || []) {
        slide.addShape(pptx.ShapeType.rect, {
          x: mask.x*scale, y: mask.y*scale, w: mask.w*scale, h: mask.h*scale,
          fill: { color: hex(obj.style.fill) },
          line: { transparency: 100, color: hex(obj.style.fill) },
          objectName: `${obj.id}_mask`
        });
      }
      slide.addText(obj.text, { ...position, ...style(obj.style), rotate: (obj.rotation+360)%360,
                     fill: { color: hex(obj.style.fill),
                             transparency: nativePdf ? 100 : (project.inline_text_fill ? 0 : 100) },
                     line: { color: hex(obj.style.fill), transparency: 100 },
                     paraSpaceAfterPt: 0, charSpacing: 0, wrap: true,
                     objectName: obj.id, isTextBox: true });
    } else if (obj.type === 'table') {
      if (originalPdf) {
        // Mask only PDF characters within their native cell footprint.
        // The original ruled table stays completely visible beneath the
        // transparent PPTX table, including every merged-cell separator.
        for (const cell of obj.cells) {
          const left = obj.bbox.x + obj.col_widths.slice(0,cell.col).reduce((x,y) => x+y,0);
          const top = obj.bbox.y + obj.row_heights.slice(0,cell.row).reduce((x,y) => x+y,0);
          const widthPx = obj.col_widths.slice(cell.col,cell.col+cell.colspan).reduce((x,y) => x+y,0);
          const heightPx = obj.row_heights.slice(cell.row,cell.row+cell.rowspan).reduce((x,y) => x+y,0);
          maskOriginalGlyphs(obj.id, cell.ocr_ids, cell.style.fill,
                             { x:left, y:top, w:widthPx, h:heightPx });
        }
      }
      // Omit covered positions. PptxGenJS inserts native hMerge/vMerge cells.
      const rows = Array.from({ length: obj.rows }, (_, r) => obj.cells
        .filter(c => c.row === r).sort((a,b) => a.col-b.col)
        .map(c => ({ text: c.text, options: { ...style(c.style),
          rowspan: c.rowspan, colspan: c.colspan,
          border: { type: nativePdf ? 'none' : (c.style.border_pt ? 'solid' : 'none'),
                    pt: nativePdf ? 0 : c.style.border_pt, color: hex(c.style.border_color) } } })));
      slide.addTable(rows, { ...position, margin: 0, autoPage: false,
                       colW: obj.col_widths.map(v => v*scale),
                       rowH: obj.row_heights.map(v => v*scale),
                       fontFace: obj.style.font, fontSize: obj.style.font_size,
                       objectName: obj.id, verbose: false });
    }
  }
  slide.addNotes(`Original ${project.source.width} × ${project.source.height} pixels. ` +
    (sourceExact ? 'Born-digital PDF: source page looks exact; editable text/cells have transparent ink until manually revealed.'
                 : originalPdf ? 'Born-digital PDF: ENTIRE original page (including all ink) is the base image; extracted labels and tables are editable OCR-style overlays with localized glyph masks.'
                 : nativePdf ? 'Legacy native PDF: older text-cleaned base image with editable overlays.'
                 : `For raster export use ${1/scale} DPI. OCR and table cells are editable overlays.`));
  await pptx.writeFile({ fileName: output, compression: true });
}

if (require.main === module) {
  generate(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), process.argv[3])
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
module.exports = { generate };

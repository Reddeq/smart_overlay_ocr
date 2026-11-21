'use strict';
const fs = require('node:fs');
const PptxGenJS = require('pptxgenjs');

async function generate(project, output) {
  const pages = project.slides || [project];
  if (!Array.isArray(pages) || !pages.length) throw new Error('No pages to export');
  const pptx = new PptxGenJS();
  const scale = pages[0].settings.inches_per_pixel;
  const width = pages[0].source.width * scale;
  const height = pages[0].source.height * scale;
  pptx.defineLayout({ name: 'SOURCE', width, height });
  pptx.layout = 'SOURCE';
  pptx.author = 'Smart Overlay OCR';
  pptx.subject = 'Editable OCR overlays; original raster preserved';
  pptx.lang = 'en-US';
  for (const page of pages) addProjectSlide(pptx, page);
  await pptx.writeFile({ fileName: output, compression: true });
}

function addProjectSlide(pptx, project) {
  const scale = project.settings.inches_per_pixel;
  const width = project.source.width * scale;
  const height = project.source.height * scale;
  const slide = pptx.addSlide();
  const nativePdf = project.source.mode === 'PDF';
  const originalPdf = nativePdf && project.native_pdf_background_mode === 'full_original';
  const sourceExact = originalPdf && project.native_pdf_overlay_mode === 'source_exact';
  const editablePdf = originalPdf && project.native_pdf_overlay_mode === 'editable';
  const fontScale = editablePdf ? 72 * project.native_pdf_zoom * scale : 1;
  const fragmentById = new Map((project.fragments || []).map(f => [f.id, f]));
  let maskCount = 0;

  // Backward compatibility ONLY: saved OCR-style PDF projects used separate
  // glyph masks. New editable projects use their frames/cells' own fills.
  function maskOriginalGlyphs(ownerId, ids, fillColor, clipPx = null) {
    if (!originalPdf || sourceExact || editablePdf) return;
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
  const fontFace = name => editablePdf ? ({
    Helvetica: 'Arial', 'Helvetica-Bold': 'Arial', 'Helvetica-Oblique': 'Arial', 'Helvetica-BoldOblique': 'Arial',
    'Times-Roman': 'Times New Roman', 'Times-Bold': 'Times New Roman', 'Times-Italic': 'Times New Roman', 'Times-BoldItalic': 'Times New Roman',
    Courier: 'Courier New', 'Courier-Bold': 'Courier New', 'Courier-Oblique': 'Courier New', 'Courier-BoldOblique': 'Courier New'
  }[name] || name) : name;
  const style = s => ({ fontFace: fontFace(s.font), fontSize: s.font_size * fontScale, color: hex(s.color),
                        bold: s.bold, italic: s.italic,
                        transparency: sourceExact ? 100 : 0,
                        fill: { color: hex(s.fill), transparency: nativePdf && !editablePdf ? 100 : 0 }, align: s.align,
                        margin: 0, breakLine: false, valign: 'middle' });
  function nativeText(obj) {
    const lines = obj.native_layout?.paragraphs ?? obj.native_layout?.lines;
    if (!editablePdf || !lines || lines.map(l => l.runs.map(r => r.text).join('')).join('\n') !== obj.text) {
      return obj.text; // Never resurrect original PDF text after an edit/CAT translation.
    }
    const result = [];
    const reference = obj.native_layout.reference_style || obj.style;
    const edited = (key, original) => obj.style[key] !== reference[key] ? obj.style[key] : original;
    const sizeRatio = obj.style.font_size / reference.font_size;
    for (const [lineIndex, line] of lines.entries()) {

      for (const [runIndex, run] of line.runs.entries()) {
        result.push({ text: run.text, options: {
          fontFace: fontFace(edited('font', run.font)), fontSize: run.font_size * fontScale * sizeRatio,
          color: hex(edited('color', run.color)), bold: edited('bold', run.bold), italic: edited('italic', run.italic),
          transparency: 0,

          paraSpaceBefore: 0, paraSpaceAfter: 0,
          breakLine: lineIndex < lines.length-1 && runIndex === line.runs.length-1
        } });
      }
    }
    return result;
  }
  function cellLayout(cell) {
    if (!editablePdf || !cell.native_layout) return {};
    const rotation = cell.native_layout.rotation || 0;
    const local = cell.native_layout.margin.map(v => v * fontScale);
    // PDF anchors are text-local; DrawingML cell margins are slide-physical.
    const margin = rotation === -90 ? [local[1], local[2], local[3], local[0]]
                 : rotation === 90 ? [local[3], local[0], local[1], local[2]] : local;
    // In PptxGenJS 4.0.1 ALL margins use inches when top < 1, otherwise points.
    return { valign: cell.native_layout.valign || 'top', margin: margin[0] >= 1 ? margin : margin.map(v => v/72),
             textDirection: rotation === -90 ? 'vert270' : rotation === 90 ? 'vert' : 'horz' };
  }
  function textContent(obj) {
    const text = nativeText(obj);
    // PptxGenJS inherits falsy run flags from shape/cell defaults. Keep the
    // defaults neutral so a normal run after a bold/italic run stays normal.
    const options = { ...style(obj.style), ...(Array.isArray(text) ? { bold: false, italic: false } : {}) };
    return { text, options };
  }
  for (const obj of project.objects) {
    if (obj.excluded) continue;
    // Empty review regions must not erase the source. A deliberately cleared
    // native label still owns its source glyphs and must retain its filled frame.
    if (obj.type === 'text' && !obj.text.trim() && !(editablePdf && obj.ocr_ids?.length)) continue;
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
      const content = textContent(obj);
      slide.addText(content.text, { ...position, ...content.options, rotate: (obj.rotation+360)%360,
                     fill: { color: hex(obj.style.fill),
                             transparency: editablePdf ? 0 : nativePdf ? 100 : (project.inline_text_fill ? 0 : 100) },
                     line: { color: hex(obj.style.fill), transparency: 100 },
                     paraSpaceAfterPt: 0, charSpacing: 0, wrap: true,
                     ...(editablePdf ? { valign: 'top', fit: 'resize', paraSpaceBefore: 0, paraSpaceAfter: 0 } : {}),
                     objectName: obj.id, isTextBox: true });
    } else if (obj.type === 'table') {
      if (originalPdf && !editablePdf && !sourceExact) {
        // Preserve saved legacy transparent-table/glyph-mask projects.
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
        .map(c => { const content = textContent(c); return { text: content.text, options: { ...content.options, ...cellLayout(c),
          rowspan: c.rowspan, colspan: c.colspan,
          border: { type: editablePdf ? 'solid' : nativePdf ? 'none' : (c.style.border_pt ? 'solid' : 'none'),
                    pt: editablePdf ? (c.style.border_pt || 0.5) * fontScale : nativePdf ? 0 : c.style.border_pt,
                    color: hex(c.style.border_color) } } }; }));
      slide.addTable(rows, { ...position, margin: 0, autoPage: false,
                       colW: obj.col_widths.map(v => v*scale),
                       rowH: obj.row_heights.map(v => v*scale),
                       fontFace: obj.style.font, fontSize: obj.style.font_size,
                       objectName: obj.id, verbose: false });
    }
  }
  slide.addNotes(`Original ${project.source.width} × ${project.source.height} pixels. ` +
    (project.native_pdf_page ? `Source PDF page: ${project.native_pdf_page}. ` : '') +
    (sourceExact ? 'Born-digital PDF: source page looks exact; editable text/cells have transparent ink until manually revealed.'
                 : editablePdf ? 'Born-digital PDF: untouched full-page PNG with visible editable text and tables. Only text frames and cells have opaque fills; no separate masks.'
                 : originalPdf ? 'Legacy born-digital PDF: full original page with localized glyph masks.'
                 : nativePdf ? 'Legacy native PDF: older text-cleaned base image with editable overlays.'
                 : `For raster export use ${1/scale} DPI. OCR and table cells are editable overlays.`));
}

if (require.main === module) {
  generate(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), process.argv[3])
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
module.exports = { generate };

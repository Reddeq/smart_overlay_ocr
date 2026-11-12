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
  slide.addImage({ data: `${project.source.raster_mime};base64,${project.source.raster_b64}`,
                   x: 0, y: 0, w: width, h: height, altText: project.source.filename });
  const hex = value => value.replace('#', '');
  const style = s => ({ fontFace: s.font, fontSize: s.font_size, color: hex(s.color),
                        fill: { color: hex(s.fill), transparency: 0 }, align: s.align,
                        margin: 0, breakLine: false, valign: 'mid' });
  for (const obj of project.objects) {
    if (obj.excluded) continue;
    if (obj.needs_review && project.allow_unreviewed !== true) throw new Error(`Review required: ${obj.id}`);
    const b = obj.type === 'text' && obj.frame ? obj.frame : obj.bbox;
    const position = { x: b.x*scale, y: b.y*scale, w: b.w*scale, h: b.h*scale };
    if (obj.type === 'text') {
      slide.addText(obj.text, { ...position, ...style(obj.style), rotate: (obj.rotation+360)%360,
                     line: { color: hex(obj.style.fill), transparency: 100 },
                     paraSpaceAfterPt: 0, charSpacing: 0, wrap: true,
                     objectName: obj.id, isTextBox: true });
    } else if (obj.type === 'table') {
      // Omit covered positions. PptxGenJS inserts native hMerge/vMerge cells.
      const rows = Array.from({ length: obj.rows }, (_, r) => obj.cells
        .filter(c => c.row === r).sort((a,b) => a.col-b.col)
        .map(c => ({ text: c.text, options: { ...style(c.style),
          rowspan: c.rowspan, colspan: c.colspan,
          border: { type: c.style.border_pt ? 'solid' : 'none',
                    pt: c.style.border_pt, color: hex(c.style.border_color) } } })));
      slide.addTable(rows, { ...position, margin: 0, autoPage: false,
                       colW: obj.col_widths.map(v => v*scale),
                       rowH: obj.row_heights.map(v => v*scale),
                       fontFace: obj.style.font, fontSize: obj.style.font_size,
                       objectName: obj.id, verbose: false });
    }
  }
  slide.addNotes(`Original ${project.source.width} × ${project.source.height} pixels. ` +
    `For raster export use ${1/scale} DPI. OCR and table cells are editable overlays.`);
  await pptx.writeFile({ fileName: output, compression: true });
}

if (require.main === module) {
  generate(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), process.argv[3])
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
module.exports = { generate };

"""Shared real-Office measurement for structure-first PDF export tests."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def measure_in_powerpoint(output, project, directory):
    if not os.environ.get('SMART_OVERLAY_POWERPOINT_PDF'):
        pytest.skip('Set SMART_OVERLAY_POWERPOINT_PDF for real Office verification')
    if os.name!='nt' or not shutil.which('powershell.exe'):
        pytest.skip('Real Office verification requires Windows PowerShell')
    expected={obj.id:{'cells':[{'row':c.row+1,'col':c.col+1} for c in obj.cells]}
              for obj in project.objects if obj.type=='table' and not obj.excluded}
    expected_path=directory/'office-expected.json'
    expected_path.write_text(json.dumps(expected),encoding='utf8')
    script=directory/'measure.ps1'
    script.write_text(r'''
param([string]$InputFile,[string]$ExpectedFile,[string]$Directory)
$ErrorActionPreference='Stop'
$expected=Get-Content -LiteralPath $ExpectedFile -Raw -Encoding UTF8 | ConvertFrom-Json
$app=New-Object -ComObject PowerPoint.Application
$initial=$app.Presentations.Count
$presentation=$null
try {
 $presentation=$app.Presentations.Open($InputFile,-1,0,0)
 $slide=$presentation.Slides.Item(1)
 $slide.Export((Join-Path $Directory 'powerpoint.png'),'PNG',2400,[int][math]::Round(2400*$presentation.PageSetup.SlideHeight/$presentation.PageSetup.SlideWidth))
 $tables=@();$labels=@()
 foreach($shape in $slide.Shapes) {
  if($shape.HasTable -eq -1) {
   $heights=@();$cells=@()
   for($r=1;$r -le $shape.Table.Rows.Count;$r++) {$heights+=$shape.Table.Rows.Item($r).Height}
   $spec=$expected.PSObject.Properties[$shape.Name].Value
   foreach($cell in $spec.cells) {
    $s=$shape.Table.Cell($cell.row,$cell.col).Shape
    $cells+=[pscustomobject]@{row=$cell.row;col=$cell.col;text=$s.TextFrame.TextRange.Text}
   }
   $tables+=[pscustomobject]@{name=$shape.Name;rows=$shape.Table.Rows.Count;columns=$shape.Table.Columns.Count;heights=$heights;left=$shape.Left;top=$shape.Top;width=$shape.Width;height=$shape.Height;cells=$cells}
  } elseif($shape.HasTextFrame -eq -1) {
   $labels+=[pscustomobject]@{name=$shape.Name;left=$shape.Left;top=$shape.Top;width=$shape.Width;height=$shape.Height;text=$shape.TextFrame.TextRange.Text;font_size=$shape.TextFrame.TextRange.Font.Size}
  }
 }
 [pscustomobject]@{tables=$tables;labels=$labels} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Directory 'powerpoint.json') -Encoding UTF8
 $presentation.SaveAs((Join-Path $Directory 'powerpoint.pdf'),32)
} finally {
 if($null -ne $presentation) {$presentation.Close()}
 if($initial -eq 0 -and $app.Presentations.Count -eq 0) {$app.Quit()}
}
''',encoding='utf-8-sig')
    process=subprocess.run(['powershell.exe','-NoProfile','-File',str(script),str(output),
                            str(expected_path),str(directory)],capture_output=True,text=True,timeout=180)
    assert process.returncode==0,process.stderr
    return json.loads((directory/'powerpoint.json').read_text(encoding='utf-8-sig'))

"""Truth-based OCR quality evaluation; never equate model agreement with accuracy.

Ground truth is HUMAN supplied. Input report and truth CSV remain local and
may contain confidential translation-client text. Do not commit either.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def edit_distance(a,b):
    if len(a)<len(b):
        a,b=b,a
    prev=list(range(len(b)+1))
    for i,char in enumerate(a,1):
        curr=[i]
        for j,other in enumerate(b,1):
            curr.append(min(prev[j]+1,curr[j-1]+1,prev[j-1]+(char!=other)))
        prev=curr
    return prev[-1]


def score(rows,truth, field):
    compared=[]
    for row in rows:
        expected=truth.get(row["id"])
        candidate=row.get(field)
        if expected is None or candidate is None:
            continue
        compared.append((row["id"],str(candidate),str(expected)))
    edits=sum(edit_distance(x[1],x[2]) for x in compared)
    chars=sum(max(1,len(x[2])) for x in compared)
    exact=sum(1 for x in compared if x[1]==x[2])
    return {"measured":len(compared),"exact_matches":exact,
            "exact_accuracy":round(exact/len(compared),4) if compared else None,
            "character_error_rate":round(edits/chars,4) if chars else None,
            "errors":[{"id":id,"actual":got,"expected":correct,
                       "edit_distance":edit_distance(got,correct)}
                      for id,got,correct in compared if got!=correct]}


def evaluate(report,truth):
    if not isinstance(report.get("records"),list):
        raise ValueError("Нет OCR ensemble records")
    rows=report["records"]
    labels={k:v for k,v in truth.items() if v is not None and v!=""}
    baseline=score(rows,labels,"original")
    alternate=score(rows,labels,"consensus")
    final=score(rows,labels,"final")
    return {"total_fragments":len(rows),"human_labeled":len(labels),
            "baseline_ppocrv5":baseline,
            "alternate_consensus":alternate,"final_pptx":final,
            "note":"CER is measured only on labeled OCR IDs. Candidate accuracy has "
                   "different coverage; compare using identical IDs before concluding improvement."}


def truth_template(report_path,output):
    report=json.loads(Path(report_path).read_text("utf-8"))
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream)
        writer.writerow(["id","original","truth"])
        for row in report["records"]:
            writer.writerow([row["id"],row["original"],""])
    return output


def evaluate_csv(report_path,truth_path,output):
    report=json.loads(Path(report_path).read_text("utf-8"))
    truth={}
    with Path(truth_path).open("r",newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        if not {"id","truth"}.issubset(reader.fieldnames or []):
            raise ValueError("Эталон CSV должен содержать столбцы id,truth")
        for row in reader:
            if row["id"] and row["truth"]:
                if row["id"] in truth:
                    raise ValueError("Повторяющийся OCR ID в эталоне")
                truth[row["id"]]=row["truth"]
    result=evaluate(report,truth)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

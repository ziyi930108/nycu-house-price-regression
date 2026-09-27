"""Check speaking-content coverage without training models or assigning fictional rehearsal times."""
from pathlib import Path
import argparse, hashlib, json, re

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    data=json.loads(args.input.read_text(encoding="utf-8-sig"))
    rows=[]
    for position,slide in enumerate(data["slides"],1):
        role=slide.get("role","main" if slide.get("seconds_estimate",slide.get("seconds",0))>0 else "appendix")
        script=slide.get("script","")
        han=len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]",script))
        seconds=slide.get("seconds_estimate",slide.get("seconds",0))
        rows.append({"slide":position,"title":slide["title"],"role":role,
                     "han_characters":han,"planned_seconds":seconds})
    main_rows=[r for r in rows if r["role"]=="main"]
    han_total=sum(r["han_characters"] for r in main_rows)
    planned=sum(r["planned_seconds"] for r in main_rows)
    receipt={"source_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
             "main_slides":len(main_rows),"appendix_slides":len(rows)-len(main_rows),
             "main_script_han_characters":han_total,"planned_seconds":planned,
             "han_only_minutes_at_300_per_minute":han_total/300,
             "han_only_minutes_at_250_per_minute":han_total/250,
             "at_least_12_minutes_at_300_han_per_minute":han_total>=3600,
             "method":"Count only CJK ideographs in main-slide scripts. Exclude titles, sources, key points, appendix and Q&A. English, digits, punctuation and pauses add no time in this conservative calculation.",
             "actual_rehearsal_performed":False,
             "limitation":"Assumed speaking rates only. Individual speaking speed and oral expansion require a timed rehearsal; the assigned seconds are a pacing plan, not evidence of elapsed duration.",
             "slides":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="slides"},ensure_ascii=False))
    if han_total<3600:raise SystemExit("Main script needs at least 3600 Han characters for this 300-character/minute planning check.")
if __name__=="__main__":main()

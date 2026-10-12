"""Score canonical reports with maximum-cardinality matching.
Inputs are independent annotations and one canonical record list per completed system run.
Use partial predictions for failed/unresolved runs: never drop these runs from evaluation.
"""
from pathlib import Path
import argparse,json
from fsr_metrics import score,point_relation,box_relation

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    annotations={x['sample_id']:x for x in (json.loads(s) for s in a.annotations.read_text().splitlines() if s.strip())}
    rows=[]
    for line in a.reports.read_text().splitlines():
        if not line.strip(): continue
        report=json.loads(line); gt=annotations[report['sample_id']]
        if gt.get('geometry','point')=='box':
            relation=box_relation(gt['iou_threshold'],True,gt.get('require_text',False))
        else: relation=point_relation(gt['tau'])
        result=score(gt['targets'],report['predictions'],relation)
        result.update({k:report.get(k) for k in ('sample_id','source_id','model_key','condition','repeat','usage')})
        result['unresolved_regions']=report.get('unresolved_regions',[])
        result['format_failure']=bool(report.get('format_failure',False))
        result['exact_report_success']=result['exact_set'] and not result['unresolved_regions'] and not result['format_failure']
        rows.append(result)
    a.out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    print(f'Scored {len(rows)} completed reports; no model calls.')
if __name__=='__main__': main()

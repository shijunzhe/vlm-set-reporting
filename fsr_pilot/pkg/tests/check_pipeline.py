"""End-to-end ORACLE software fixture. Never treat this as model performance.
Creates responses directly from smoke annotations in a temporary directory, ingests,
scores and checks the aggregation interface. No API calls or saved model results.
"""
from pathlib import Path
import json,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[1]
reqs=[json.loads(s) for s in (root/'examples/e1_requests.jsonl').read_text().splitlines() if s.strip()]
gt={x['sample_id']:x for x in (json.loads(s) for s in (root/'examples/annotations_SMOKE_ONLY.jsonl').read_text().splitlines() if s.strip())}
responses=[]
for q in reqs:
    rows=[]
    registry={x['region_id']:x for x in q['task_payload']['ordered_registry']}
    for rid in q['requested']:
        reg=registry[rid]
        items=[{'u':(t['point'][0]-reg['x0'])/reg['width'],'v':(t['point'][1]-reg['y0'])/reg['height']} for t in gt[q['sample_id']]['targets'] if t['owner']==rid]
        rows.append({'region_id':rid,'status':'found' if items else 'empty','objects':items})
    responses.append({'request_id':q['request_id'],'status':'completed',
                      'provider_model_id':'ORACLE_SOFTWARE_FIXTURE_NOT_A_MODEL',
                      'raw_text':'\n'.join(json.dumps(x) for x in rows),
                      'finish_reason':'stop','usage':None})
with tempfile.TemporaryDirectory() as d:
    d=Path(d); rp=d/'responses.jsonl'; cp=d/'reports.jsonl'; sp=d/'scores.jsonl'
    rp.write_text(''.join(json.dumps(x)+'\n' for x in responses))
    subprocess.run([sys.executable,str(root/'src/ingest_e1.py'),'--requests',str(root/'examples/e1_requests.jsonl'),'--responses',str(rp),'--out',str(cp)],check=True,capture_output=True)
    subprocess.run([sys.executable,str(root/'src/score_reports.py'),'--annotations',str(root/'examples/annotations_SMOKE_ONLY.jsonl'),'--reports',str(cp),'--out',str(sp)],check=True,capture_output=True)
    scores=[json.loads(s) for s in sp.read_text().splitlines() if s.strip()]
    assert len(scores)==24
    assert all(x['set_error']==0 and x['B']==0 and x['chi']==0 and x['exact_report_success'] for x in scores)
    sys.path.insert(0,str(root/'src'))
    from paired_analysis import paired
    subset=[x for x in scores if x['model_key']=='model_A']
    result=paired(subset,'S4','J16','B',draws=200)
    assert result['delta']==0 and result['interval']==[0,0]
    # No inferential claim: this is a deterministic oracle fixture.
    broken=[x for x in subset if not (x['condition']=='S4' and x['sample_id']=='smoke_00')]
    try: paired(broken,'S4','J16','B',draws=10);raise AssertionError('missing pair not detected')
    except ValueError: pass
out={'fixture':'oracle software test, not empirical model evidence','requests':len(reqs),
     'complete_system_reports':24,'pipelines_passed':True,'no_model_calls':True,
     'temporary_oracle_predictions_deleted':True}
(root/'checks/pipeline_fixture_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

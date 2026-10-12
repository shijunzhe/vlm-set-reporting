"""Convert ALL completed/failed E1 call responses into S1/S4/J16 canonical reports.
Missing responses cause an error: unexecuted requests are not silently labeled failures.
Expected response JSONL fields: request_id, status=completed|failed, raw_text,
finish_reason, usage (optional), provider_model_id (log it in real experiments).
"""
from pathlib import Path
from collections import defaultdict
import argparse,json
from fsr_ledger import Region,Ledger

def main():
    p=argparse.ArgumentParser(); p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--responses',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    reqs=[json.loads(s) for s in a.requests.read_text().splitlines() if s.strip()]
    responses={}
    for line in a.responses.read_text().splitlines():
        if not line.strip(): continue
        x=json.loads(line)
        if x['request_id'] in responses: raise ValueError('Duplicate response IDs; reconcile transport retries in log first')
        if x['status'] not in ('completed','failed'): raise ValueError('Only executed requests can be ingested')
        responses[x['request_id']]=x
    expected={q['request_id'] for q in reqs}
    if expected!=set(responses):
        raise ValueError(f'Request/response mismatch: {len(expected-set(responses))} missing, {len(set(responses)-expected)} extra')
    groups=defaultdict(list)
    for q in reqs: groups[(q['sample_id'],q['model_key'],q['repeat'],q['batch_size'])].append(q)
    output=[]
    for (sample,model,rep,b),qs in sorted(groups.items()):
        first=qs[0]; registry={r['region_id']:Region(**r) for r in first['task_payload']['ordered_registry']}
        ledger=Ledger(registry,first['evidence_sha256']); failures=0; audit=[]
        usage={'calls':len(qs),'input_tokens_known_sum':0,'output_tokens_known_sum':0,
               'missing_usage_calls':0,'all_provider_usage':[]}
        for q in sorted(qs,key=lambda x:x['request_id']):
            r=responses[q['request_id']]
            failed=r['status']=='failed'
            if failed: failures+=1
            raw='' if failed else r.get('raw_text','')
            info=ledger.apply(q['request_id'],q['requested'],raw,q['evidence_sha256'])
            audit.append({'request_id':q['request_id'],'status':r['status'],
                          'finish_reason':r.get('finish_reason'),'decoder_audit':info})
            u=r.get('usage')
            usage['all_provider_usage'].append(u)
            if u is None: usage['missing_usage_calls']+=1
            else:
                usage['input_tokens_known_sum']+=u.get('input_tokens') or 0
                usage['output_tokens_known_sum']+=u.get('output_tokens') or 0
        final=ledger.report()
        format_failure=failures>0 or any(any(x['decoder_audit'].get(k,0) for k in
                            ('malformed_lines','invalid_objects','duplicate_region_rows','contradictory_rows','unrequested_rows')) for x in audit)
        final.update({'sample_id':sample,'source_id':first['source_id'],'model_key':model,
                      'condition':'J16' if b==16 else f'S{b}','repeat':rep,
                      'format_failure':format_failure,'failed_calls':failures,
                      'usage':usage,'audit':audit,'evidence_sha256':first['evidence_sha256']})
        output.append(final)
    a.out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    print(f'Created {len(output)} completed system reports; no model calls made.')
if __name__=='__main__': main()

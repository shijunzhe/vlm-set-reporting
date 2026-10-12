"""Build provider-neutral E1 requests. Images are reused byte-for-byte.
Usage: python src/build_manifest.py --samples examples/samples.jsonl --out examples/e1_requests.jsonl
All ground truth stays in a separate evaluator file and is never serialized here.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def build(samples:list[dict],models:list[str],repeats:int,seed:int,root:Path)->list[dict]:
    requests=[]
    for sample in samples:
        m=len(sample['regions'])
        if m!=16: raise ValueError('Frozen E1 core requires a 16-region registry')
        image=(root/sample['image_path']).resolve()
        ref=(root/sample['reference_path']).resolve()
        evidence_hash=hashlib.sha256((sha(image)+sha(ref)).encode()).hexdigest()
        for rep in range(repeats):
            local_seed=int(hashlib.sha256(f"{seed}:{sample['sample_id']}:{rep}".encode()).hexdigest()[:12],16)
            order=[r['region_id'] for r in sample['regions']]
            random.Random(local_seed).shuffle(order)
            roster={r['region_id']:r for r in sample['regions']}
            for model in models:
                for b in (1,4,16):
                    for offset in range(0,m,b):
                        active=order[offset:offset+b]
                        active_set=set(active)
                        # The complete roster/order is identical across loads; only a bit mask changes.
                        mask=''.join('1' if rid in active_set else '0' for rid in order)
                        payload={'task':'reference-conditioned point detection',
                                 'image_width':sample['width'],'image_height':sample['height'],
                                 'coordinate_contract':'region_local_uv_half_open_v1',
                                 'ordered_registry':[roster[rid] for rid in order],
                                 'active_mask':mask}
                        requests.append({'request_id':f"E1:{sample['sample_id']}:{model}:{rep}:b{b}:g{offset//b}",
                                         'experiment':'E1-Q','sample_id':sample['sample_id'],
                                         'source_id':sample['source_id'],'model_key':model,
                                         'repeat':rep,'batch_size':b,'requested':active,
                                         'image_path':sample['image_path'],'reference_path':sample['reference_path'],
                                         'evidence_sha256':evidence_hash,'task_payload':payload,
                                         'status':'not_run'})
    random.Random(seed).shuffle(requests)
    return requests

def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--models',nargs='+',default=['model_A','model_B'])
    p.add_argument('--repeats',type=int,default=1); p.add_argument('--seed',type=int,default=20261008)
    a=p.parse_args()
    samples=[json.loads(s) for s in a.samples.read_text().splitlines() if s.strip()]
    rows=build(samples,a.models,a.repeats,a.seed,a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    print(json.dumps({'requests':len(rows),'status':'manifest only; no model calls made'},ensure_ascii=False))
if __name__=='__main__': main()

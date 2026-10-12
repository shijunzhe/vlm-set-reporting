"""Build the FSR pilot manifest (no model calls).
Arms per scene, model and repeat (one shared seeded region order per scene/repeat):
  E1  regional contract, region-local (u,v), loads b in {1,4,16}: 21 calls (package E1-Q, unchanged)
  G   same masks and order, but scene-normalized (x,y); code checks the declared region: 21 calls
  M   masked evidence: only the requested region's pixels kept (same canvas, same overlay), b=1,
      region-local contract: 16 calls (separates visible content L from report load A)
  F   one free global list, scene-normalized (x,y): 1 call
"""
import argparse, hashlib, json, random, sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pkg' / 'src'))
from build_manifest import build as build_e1, sha  # package E1 builder, unchanged

def evid(img, ref):
    return hashlib.sha256((sha(img) + sha(ref)).encode()).hexdigest()

def order_for(seed, sample, rep):
    local = int(hashlib.sha256(f"{seed}:{sample['sample_id']}:{rep}".encode()).hexdigest()[:12], 16)
    o = [r['region_id'] for r in sample['regions']]; random.Random(local).shuffle(o); return o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', default='runs/pilot_requests.jsonl'); ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--samples', default='data/samples.jsonl'); a = ap.parse_args()
    samples = [json.loads(s) for s in (ROOT / a.samples).read_text().splitlines() if s.strip()]
    rows = []
    for q in build_e1(samples, a.models, 1, a.seed, ROOT):
        q['arm'] = 'E1'; q['prompt'] = 'regional_local'; rows.append(q)
    for s in samples:
        roster = {r['region_id']: r for r in s['regions']}
        ref = ROOT / s['reference_path']; img = ROOT / s['image_path']
        o = order_for(a.seed, s, 0)
        reg = [roster[r] for r in o]
        for m in a.models:
            base = dict(sample_id=s['sample_id'], source_id=s['source_id'], model_key=m, repeat=0,
                        reference_path=s['reference_path'], status='not_run')
            for b in (1, 4, 16):
                for off in range(0, 16, b):
                    act = o[off:off + b]; mask = ''.join('1' if r in act else '0' for r in o)
                    rows.append(dict(base, request_id=f"G:{s['sample_id']}:{m}:0:b{b}:g{off//b}", experiment='pilot',
                                     arm='G', prompt='regional_global', batch_size=b, requested=act,
                                     image_path=s['image_path'], evidence_sha256=evid(img, ref),
                                     task_payload={'task': 'reference-conditioned point detection', 'image_width': s['width'],
                                                   'image_height': s['height'], 'coordinate_contract': 'scene_normalized_xy_half_open_v1',
                                                   'ordered_registry': reg, 'active_mask': mask}))
            for rid in o:
                mask = ''.join('1' if r == rid else '0' for r in o)
                mp = f"data/masked/{s['sample_id']}_{rid}.png"
                rows.append(dict(base, request_id=f"M:{s['sample_id']}:{m}:0:{rid}", experiment='pilot', arm='M',
                                 prompt='regional_local', batch_size=1, requested=[rid], image_path=mp,
                                 evidence_sha256=evid(ROOT / mp, ref),
                                 task_payload={'task': 'reference-conditioned point detection', 'image_width': s['width'],
                                               'image_height': s['height'], 'coordinate_contract': 'region_local_uv_half_open_v1',
                                               'ordered_registry': reg, 'active_mask': mask}))
            rows.append(dict(base, request_id=f"F:{s['sample_id']}:{m}:0", experiment='pilot', arm='F', prompt='free',
                             batch_size=16, requested=o, image_path=s['image_path'], evidence_sha256=evid(img, ref),
                             task_payload={'image_width': s['width'], 'image_height': s['height'], 'ordered_registry': reg}))
    random.Random(a.seed).shuffle(rows)
    (ROOT / a.out).parent.mkdir(exist_ok=True)
    (ROOT / a.out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    from collections import Counter
    print(len(rows), Counter(r['arm'] for r in rows), Counter(r['model_key'] for r in rows))

if __name__ == '__main__':
    main()

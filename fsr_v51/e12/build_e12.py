"""Build the v5-E12 request manifest (E1 task transformations + E2 representation/mechanism).
No model calls. Prompts are assembled from shared fragments so that conditions differ only
in the factor they manipulate. Usage: python build_e12.py [--models gpt56,gpt61,opus55] [--reps 3]
"""
import argparse, hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).parent
PRE = 'The first image is the reference symbol; the second image is the full scene.\n\n'

HEAD = ("You are performing reference-conditioned visual set reporting. The full scene, reference image, coordinate "
        "registry and output contract are given. A responsibility region is an area to inspect, not a pre-detected object. "
        "There may be zero, one, or several matching instances in a region. Use any visible context needed to identify the requested target.")
REQ = ("The ordered_registry is the complete region roster. requested_regions lists the region IDs to report in this call; "
       "do not report any other region. The same full image may remain visible regardless of which regions are requested. "
       "Answer the requested regions in the order listed.")
FMT = {
    'ndjson': ("Return exactly one JSON object per requested region, one object per line (NDJSON), with no surrounding list, "
               "prose, code fence, global count, subtotal, or explanation."),
    'array': ("Return exactly one JSON array that contains one JSON object per requested region, in the order listed, with no "
              "prose, code fence, global count, subtotal, or explanation outside the array."),
}
STATUS = ("Each region object must have region_id, status and objects. status is empty, found, or unknown. Use empty only when "
          "you judge that no matching target center is in the responsibility region. Use found when you have completed your best "
          "inspection and report all matching centers in that region. Use unknown when you cannot complete the judgment; partial "
          "recovered objects may be included. Do not invent objects to reach an expected number. A region is not limited to one object.")
FRAME = {
    'local': ("Each object has u and v, finite normalized coordinates measured within that responsibility region: "
              "native_x = region.x0 + region.width*u, native_y = region.y0 + region.height*v. Both must be at least 0 and strictly "
              "below 1. Give four decimal places."),
    'global': ("Each object has x and y, finite coordinates normalized to the FULL SCENE: native_x = image_width*x, "
               "native_y = image_height*y. Both must be at least 0 and strictly below 1. Give four decimal places."),
}
TAIL = ("Report an instance only in the region containing its center. Keep distinct nearby instances distinct. "
        "Do not report the reference image as a target in the scene.")
EX = {
    ('ndjson', 'local'): ('{"region_id":"R001","status":"empty","objects":[]}\n'
                          '{"region_id":"R002","status":"found","objects":[{"u":0.3125,"v":0.5840},{"u":0.7510,"v":0.2266}]}\n'
                          '{"region_id":"R003","status":"unknown","objects":[]}'),
    ('ndjson', 'global'): ('{"region_id":"R001","status":"empty","objects":[]}\n'
                           '{"region_id":"R002","status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565}]}\n'
                           '{"region_id":"R003","status":"unknown","objects":[]}'),
    ('array', 'local'): ('[{"region_id":"R001","status":"empty","objects":[]},'
                         '{"region_id":"R002","status":"found","objects":[{"u":0.3125,"v":0.5840},{"u":0.7510,"v":0.2266}]},'
                         '{"region_id":"R003","status":"unknown","objects":[]}]'),
    ('array', 'global'): ('[{"region_id":"R001","status":"empty","objects":[]},'
                          '{"region_id":"R002","status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565}]},'
                          '{"region_id":"R003","status":"unknown","objects":[]}]'),
}
USER = ("Inspect the attached scene for every instance of the attached reference. Use the responsibility registry and "
        "requested_regions in the following payload. Do not alter the coordinate contract. Report only the requested regions, "
        "including an explicit empty or unknown row when appropriate.\n\n{payload}")
FREE_SYS = "You are performing reference-conditioned visual set reporting. Follow the output contract exactly."
FREE = ("Find every instance of the attached reference in the attached full scene. Return one JSON object with status and objects. "
        "status is found, empty, or unknown. objects is a list of centers, with x and y normalized to the FULL SCENE "
        "(0 <= x < 1, 0 <= y < 1), given with four decimal places. Each entry is {{\"x\":...,\"y\":...}}. {order} Return no total count "
        "and no narrative. Include all recovered points even if the result is incomplete, and set status to unknown when you cannot "
        "complete it. Do not count the reference image itself. Do not invent objects to meet a number.\n\n"
        "The identical complete region registry is provided as public coordinate metadata; it does not indicate true objects. "
        "This arm reports a single global list rather than separate region rows.\n\n{registry}")
ORDER = {'down': 'List the objects from the top of the scene to the bottom, and from left to right within a row.',
         'up': 'List the objects from the bottom of the scene to the top, and from left to right within a row.'}


def system_rows(fmt, frame):
    return '\n\n'.join([HEAD, REQ, FMT[fmt] + ' ' + STATUS, FRAME[frame] + ' ' + TAIL,
                        'Examples of schema only, not answers for the current image:\n' + EX[(fmt, frame)]])


def conditions(s):
    """13 calls per scene and repetition."""
    nat = [r['region_id'] for r in s['regions']]            # row-major R001..R016
    c = [dict(cond='F-down', kind='free', order='down'), dict(cond='F-up', kind='free', order='up'),
         dict(cond='J-local', kind='rows', fmt='ndjson', frame='local', requested=nat),
         dict(cond='J-global', kind='rows', fmt='ndjson', frame='global', requested=nat),
         dict(cond='J-global-rev', kind='rows', fmt='ndjson', frame='global', requested=nat[::-1]),
         dict(cond='A-local', kind='rows', fmt='array', frame='local', requested=nat),
         dict(cond='A-global', kind='rows', fmt='array', frame='global', requested=nat)]
    for r in s['focal_regions']:
        c += [dict(cond='D-local', kind='rows', fmt='ndjson', frame='local', requested=[r], focal=r),
              dict(cond='D-global', kind='rows', fmt='ndjson', frame='global', requested=[r], focal=r),
              dict(cond='M-local', kind='rows', fmt='ndjson', frame='local', requested=[r], focal=r, masked=True)]
    return c


def build(models, reps, seed=20261010):
    samples = [json.loads(l) for l in (ROOT / 'data_e12/samples.jsonl').read_text().splitlines() if l.strip()]
    rows = []
    for s in samples:
        reg = s['regions']
        for m in models:
            for rep in range(reps):
                for c in conditions(s):
                    rid = f"E12:{s['sample_id']}:{m}:{c['cond']}" + (f":{c['focal']}" if 'focal' in c else '') + f":r{rep}"
                    img = f"data_e12/masked/{s['sample_id']}_{c['focal']}.png" if c.get('masked') else s['image_path']
                    if c['kind'] == 'free':
                        system = FREE_SYS
                        user = PRE + FREE.format(order=ORDER[c['order']], registry=json.dumps(reg))
                        requested = [r['region_id'] for r in reg]
                    else:
                        payload = {'task': 'reference-conditioned point detection', 'image_width': s['width'], 'image_height': s['height'],
                                   'coordinate_contract': 'region_local_uv_half_open_v1' if c['frame'] == 'local' else 'scene_normalized_xy_half_open_v1',
                                   'ordered_registry': reg, 'requested_regions': c['requested']}
                        system = system_rows(c['fmt'], c['frame']); user = PRE + USER.format(payload=json.dumps(payload))
                        requested = c['requested']
                    rows.append({'request_id': rid, 'sample_id': s['sample_id'], 'dose': s['dose'], 'model_key': m, 'rep': rep,
                                 'condition': c['cond'], 'kind': c['kind'], 'format': c.get('fmt', 'free'), 'frame': c.get('frame', 'global'),
                                 'order': c.get('order', 'natural' if c['cond'] != 'J-global-rev' else 'reverse'),
                                 'focal': c.get('focal'), 'requested': requested, 'image_path': img, 'reference_path': s['reference_path'],
                                 'system': system, 'user': user,
                                 'prompt_sha256': hashlib.sha256((system + '\n---\n' + user).encode()).hexdigest()})
    random.Random(seed).shuffle(rows)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='gpt56,gpt61,opus55,sonnet55'); ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args(); rows = build(a.models.split(','), a.reps)
    (ROOT / 'runs_e12').mkdir(exist_ok=True)
    (ROOT / 'runs_e12/requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    from collections import Counter
    print(len(rows), 'requests'); print(Counter(r['condition'] for r in rows)); print(Counter(r['model_key'] for r in rows))

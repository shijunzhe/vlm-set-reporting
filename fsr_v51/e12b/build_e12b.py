"""Build the v5.2-E12b request manifest (blocks A, Adef, B1, B2). No model calls.
Block C (repair) is built later from the base responses by build_e12b_repair.py.

Arms (all global frame, same shared instruction text; they differ only in the output contract and its example):
  A1 sparse rows      one NDJSON row only for regions where targets were found
  A2 full rows        one NDJSON row for EVERY registered region (found / empty / unknown)
  A3 engineered list  one JSON object with a single list of centres (no region rows)
  A3rev               A3 with the registry listed in reverse order (R016 first); used only to separate scan order from
                      image position in the end-of-scan omission test (A-P2)
Usage: python build_e12b.py [--models gpt56,gpt61,opus55,sonnet55,gem38]
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PRE = 'The first image is the reference symbol; the second image is the full scene.\n\n'

COMMON = ("You are performing reference-conditioned visual set reporting. Find every instance of the reference symbol in the "
          "full scene. A coordinate registry of regions is given as public metadata: a region is an area to inspect, not a "
          "pre-detected object, and it may contain zero, one, or several matching instances. Use any visible context needed to "
          "identify the target. Scan the scene systematically, region by region in registry order, and check every region "
          "before you finish; do not stop early. Do not invent objects to reach an expected number. Keep distinct nearby "
          "instances distinct. Do not report the reference image as a target in the scene.")
FRAME = ("Each object has x and y, finite coordinates normalized to the FULL SCENE: native_x = image_width*x, "
         "native_y = image_height*y. Both must be at least 0 and strictly below 1. Give four decimal places.")
CONTRACT = {
    'A1': ("Output contract: return one JSON object per line (NDJSON) ONLY for regions in which you found at least one "
           "matching target, in registry order. Do not output any line for a region without a matching target. Each object "
           "has region_id, status and objects. status is found, or unknown when you could not complete that region (partial "
           "recovered objects may be included). Report an instance only in the region containing its center. No surrounding "
           "list, prose, code fence, total count or explanation."),
    'A2': ("Output contract: return exactly one JSON object per registered region, one object per line (NDJSON), in registry "
           "order, covering EVERY region in the registry. Each object has region_id, status and objects. status is empty, "
           "found, or unknown. Use empty only when you judge that no matching target center is in the region. Use found when "
           "you have completed your best inspection and report all matching centers in that region. Use unknown when you "
           "cannot complete the judgment; partial recovered objects may be included. Report an instance only in the region "
           "containing its center. No surrounding list, prose, code fence, total count or explanation."),
    'A3': ("Output contract: return exactly one JSON object with status and objects. objects is a single list of the centers "
           "of all matching instances in the whole scene, listed in registry scan order. status is found when you have "
           "completed your inspection of every region, empty when the scene contains no matching target, and unknown when "
           "you could not complete it (include all recovered points). No region rows, total count, prose, code fence or "
           "explanation."),
}
EX = {
    'A1': ('{"region_id":"R002","status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565}]}\n'
           '{"region_id":"R005","status":"found","objects":[{"x":0.0912,"y":0.3301}]}'),
    'A2': ('{"region_id":"R001","status":"empty","objects":[]}\n'
           '{"region_id":"R002","status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565}]}\n'
           '{"region_id":"R003","status":"unknown","objects":[]}'),
    'A3': '{"status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565},{"x":0.0912,"y":0.3301}]}',
}
USER = ("Inspect the attached scene for every instance of the attached reference, using the region registry in the "
        "following payload. Follow the output contract exactly.\n\n{payload}")


def system(arm):
    a = 'A3' if arm == 'A3rev' else arm
    return '\n\n'.join([COMMON, FRAME, CONTRACT[a], 'Example of the schema only, not an answer for the current image:\n' + EX[a]])


def registry(k, W=1024):
    c = W // k
    return [{'region_id': f'R{r * k + q + 1:03d}', 'x0': q * c, 'y0': r * c, 'width': c, 'height': c} for r in range(k) for q in range(k)]


def request(s, m, arm, grid, rep, block, setting='min', k=4):
    reg = registry(k)
    if arm == 'A3rev': reg = reg[::-1]
    payload = {'task': 'reference-conditioned point detection', 'image_width': s['width'], 'image_height': s['height'],
               'coordinate_contract': 'scene_normalized_xy_half_open_v1', 'ordered_registry': reg}
    sysm = system(arm); user = PRE + USER.format(payload=json.dumps(payload))
    tag = f"{block}:{s['sample_id']}:{m}:{arm}:grid{grid}:k{k}:{setting}:r{rep}"
    return {'request_id': 'E12b:' + tag, 'block': block, 'sample_id': s['sample_id'], 'cell': s['cell'], 'dose': s['dose'],
            'glyph': s['glyph'], 'model_key': m, 'setting': setting, 'rep': rep, 'arm': arm, 'grid': grid, 'granularity': k,
            'kind': 'free' if arm in ('A3', 'A3rev') else 'rows', 'format': 'ndjson', 'frame': 'global', 'condition': f'{arm}-grid{grid}-k{k}',
            'requested': [r['region_id'] for r in reg], 'registry': reg, 'focal': None,
            'image_path': s['image_paths'][grid], 'reference_path': s['reference_path'], 'system': sysm, 'user': user,
            'prompt_sha256': hashlib.sha256((sysm + '\n---\n' + user).encode()).hexdigest()}


def build(models, seed=20261011):
    S = [json.loads(l) for l in (ROOT / 'data_e12b/samples.jsonl').read_text().splitlines() if l.strip()]
    A = [s for s in S if s['block'] == 'A']; B2 = [s for s in S if s['block'] == 'B2']
    b1 = [s for s in A if int(s['sample_id'][-2:]) < 6]          # 6 per dose, fixed before any call
    rows = []
    for m in models:
        for s in A:
            for arm in ('A1', 'A2', 'A3'):
                for grid in ('on', 'off'):
                    for rep in (0, 1): rows.append(request(s, m, arm, grid, rep, 'A'))
            for arm in ('A2', 'A3'): rows.append(request(s, m, arm, 'on', 0, 'Adef', setting='default'))
            rows.append(request(s, m, 'A3rev', 'on', 0, 'Arev'))
        for s in b1:
            for k in (2, 8):
                for rep in (0, 1): rows.append(request(s, m, 'A2', 'off', rep, 'B1', k=k))
        for s in B2:
            for arm in ('A2', 'A3'): rows.append(request(s, m, arm, 'on', 0, 'B2'))
    assert len({r['request_id'] for r in rows}) == len(rows)
    random.Random(seed).shuffle(rows)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='gpt56,gpt61,opus55,sonnet55,gem38')
    a = ap.parse_args(); rows = build(a.models.split(','))
    (ROOT / 'runs_e12b').mkdir(exist_ok=True)
    out = ROOT / 'runs_e12b/requests.jsonl'; out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(len(rows), 'requests', hashlib.sha256(out.read_bytes()).hexdigest())
    print(Counter(r['block'] for r in rows)); print(Counter(r['model_key'] for r in rows))

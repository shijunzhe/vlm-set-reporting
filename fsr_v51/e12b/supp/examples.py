"""Registry-consistent schema examples (v5.4 fix). The E12b A2 example was written for the 4x4 registry and was reused
unchanged for the 2x2 and 8x8 registries of block B1, where its points fall in other regions. Here the example is
generated from the registry and validated before any request is sent.

Rule: row 1 = first region, status empty; row 2 = second region, status found, two points at region-local positions
(0.3125, 0.5840) and (0.7510, 0.2266) (the E12 local example); row 3 = third region, status unknown. For the 4x4
registry the original E12b text is kept verbatim (it is consistent there), so block-A requests are unchanged.
"""
import json, re

ORIG_4x4 = ('{"region_id":"R001","status":"empty","objects":[]}\n'
            '{"region_id":"R002","status":"found","objects":[{"x":0.3280,"y":0.1460},{"x":0.4378,"y":0.0565}]}\n'
            '{"region_id":"R003","status":"unknown","objects":[]}')
LOCAL = [(0.3125, 0.5840), (0.7510, 0.2266)]


def a2_example(reg, W=1024, H=1024):
    if len(reg) == 16 and reg[1]['width'] == 256: return ORIG_4x4
    r1, r2, r3 = reg[0], reg[1], reg[2]
    pts = [((r2['x0'] + r2['width'] * u) / W, (r2['y0'] + r2['height'] * v) / H) for u, v in LOCAL]
    objs = ','.join('{"x":%.4f,"y":%.4f}' % p for p in pts)
    return (f'{{"region_id":"{r1["region_id"]}","status":"empty","objects":[]}}\n'
            f'{{"region_id":"{r2["region_id"]}","status":"found","objects":[{objs}]}}\n'
            f'{{"region_id":"{r3["region_id"]}","status":"unknown","objects":[]}}')


def validate(system_text, reg, W=1024, H=1024):
    """Every example point must lie in the region its row declares, and every declared region must exist."""
    m = re.search(r'Example of the schema only, not an answer for the current image:\n(.*)$', system_text, re.S)
    if not m: return True, 'no example'
    by = {r['region_id']: r for r in reg}
    for line in m.group(1).strip().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue                                     # list-type (A3) example has no region rows
        if 'region_id' not in row: continue
        r = by.get(row['region_id'])
        if r is None: return False, f'unknown region {row["region_id"]}'
        for o in row.get('objects', []):
            x, y = o['x'] * W, o['y'] * H
            if not (r['x0'] <= x < r['x0'] + r['width'] and r['y0'] <= y < r['y0'] + r['height']):
                return False, f'point {o} not in {row["region_id"]}'
    return True, 'ok'

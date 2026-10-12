"""E3 shared library: task text, registry, prompts for the four candidate interfaces, registry-consistent examples with
validation, label-free decoding with observable flags, and scoring (points: Euclidean max matching; boxes: Jaccard max
matching with labels and ignore regions; box centres: Euclidean with the per-image tolerance).

Candidates: structure L (one list) or R (one row per registered region) x setting min (lowest reasoning) or def (API
default). All share the same task text, scan instruction and coordinate contract; they differ only in the output
contract (and the reasoning setting, which is a runtime parameter).
"""
import json, math, re
import numpy as np
from scipy.optimize import linear_sum_assignment

K = 4                                          # 4x4 registry (E12b: best or tied for every model)
DEC = json.JSONDecoder()


# ------------------------------------------------------------------ registry and task text
def registry(W, H, k=K):
    xs = [round(W * i / k) for i in range(k + 1)]; ys = [round(H * i / k) for i in range(k + 1)]
    return [{'region_id': f'R{r * k + q + 1:03d}', 'x0': xs[q], 'y0': ys[r], 'width': xs[q + 1] - xs[q], 'height': ys[r + 1] - ys[r]}
            for r in range(k) for q in range(k)]


def task_text(item):
    ds = item['dataset']
    if ds == 'fsc147':
        return ('The first image is a reference crop showing one instance of the target object. Find every instance of this object in the '
                'second image (the scene), including the instance the reference was cropped from.')
    if ds == 'sku110k':
        return 'Find every individual product on the shelves in the image: each separately visible package or item, including partly visible ones.'
    if ds == 'dota':
        return ('This is an aerial image. Find every object of the following categories: ' + ', '.join(item['labels']) +
                '. Give each object its category exactly as written.')
    return ('This is a document page. Find every layout element and label it with exactly one of: ' + ', '.join(item['labels']) +
            '. Each text block, list item, section header, page header, page footer, footnote, caption, table, picture, formula and title is one element.')


def multi_label(item): return item['task'] == 'box' and len(item['labels']) > 1


COMMON = ('You are performing visual set reporting. {task} A coordinate registry of regions is given as public metadata: a region is an '
          'area to inspect, not a pre-detected object, and it may contain zero, one, or several matching instances. Scan the image '
          'systematically, region by region in registry order, and check every region before you finish; do not stop early. Do not '
          'invent objects to reach an expected number. Keep distinct nearby instances distinct.')
FRAME_POINT = ('Each object has x and y, the coordinates of its center normalized to the FULL IMAGE: native_x = image_width*x, '
               'native_y = image_height*y, both at least 0 and strictly below 1. Give four decimal places.')
FRAME_BOX = ('Each object has x1, y1, x2, y2: the left, top, right and bottom edges of its axis-aligned bounding box, normalized to the '
             'FULL IMAGE (native = image size times value), with 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1. Give four decimal places.')
LABEL = ' Each object also has label, one of the allowed labels exactly as written.'
CONTRACT = {
    'R': ('Output contract: return exactly one JSON object per registered region, one object per line (NDJSON), in registry order, '
          'covering EVERY region in the registry. Each object has region_id, status and objects. status is empty, found, or unknown. '
          'Use empty only when you judge that no matching object center is in the region. Use found when you have completed your best '
          'inspection and report all matching objects whose center is in that region. Use unknown when you cannot complete the judgment; '
          'partial recovered objects may be included. Report an object only in the region containing its center. No surrounding list, '
          'prose, code fence, total count or explanation.'),
    'L': ('Output contract: return exactly one JSON object with status and objects. objects is a single list of all matching objects in '
          'the whole image, listed in registry scan order. status is found when you have completed your inspection of every region, empty '
          'when the image contains no matching object, and unknown when you could not complete it (include all recovered objects). No region '
          'rows, total count, prose, code fence or explanation.'),
}


def obj_json(item, cx, cy, W, H, label=None):
    if item['task'] == 'point':
        return '{"x":%.4f,"y":%.4f}' % (cx / W, cy / H)
    w, h = 0.04 * W, 0.04 * H
    s = '{"x1":%.4f,"y1":%.4f,"x2":%.4f,"y2":%.4f' % ((cx - w / 2) / W, (cy - h / 2) / H, (cx + w / 2) / W, (cy + h / 2) / H)
    return s + (',"label":"%s"}' % label if multi_label(item) else '}')


def example(item, arm, reg):
    """Schema example generated from the registry: two objects in region 2 (local 0.3125,0.5840 and 0.7510,0.2266), one in region 5."""
    W, H = item['width'], item['height']; lab = item['labels'][0]
    def at(r, u, v): return r['x0'] + r['width'] * u, r['y0'] + r['height'] * v
    o2 = [obj_json(item, *at(reg[1], 0.3125, 0.5840), W, H, lab), obj_json(item, *at(reg[1], 0.7510, 0.2266), W, H, lab)]
    o5 = [obj_json(item, *at(reg[4], 0.4120, 0.3330), W, H, lab)]
    if arm == 'L':
        return '{"status":"found","objects":[' + ','.join(o2 + o5) + ']}'
    return '\n'.join(['{"region_id":"%s","status":"empty","objects":[]}' % reg[0]['region_id'],
                      '{"region_id":"%s","status":"found","objects":[%s]}' % (reg[1]['region_id'], ','.join(o2)),
                      '{"region_id":"%s","status":"unknown","objects":[]}' % reg[2]['region_id'],
                      '{"region_id":"%s","status":"found","objects":[%s]}' % (reg[4]['region_id'], ','.join(o5))])


def validate_example(item, arm, reg, text):
    """Every example object must have valid coordinates; for rows, its center must lie in the declared region."""
    by = {r['region_id']: r for r in reg}; W, H = item['width'], item['height']
    lines = [text] if arm == 'L' else text.splitlines()
    for ln in lines:
        row = json.loads(ln); objs = row['objects']
        for o in objs:
            rec = to_record(item, o)
            if rec is None: return False
            if arm == 'R':
                r = by[row['region_id']]
                if not (r['x0'] <= rec['cx'] < r['x0'] + r['width'] and r['y0'] <= rec['cy'] < r['y0'] + r['height']): return False
    return True


def build_prompt(item, arm):
    reg = registry(item['width'], item['height'])
    sysm = '\n\n'.join([COMMON.format(task=task_text(item)), (FRAME_POINT if item['task'] == 'point' else FRAME_BOX) + (LABEL if multi_label(item) else ''),
                        CONTRACT[arm], 'Example of the schema only, not an answer for the current image:\n' + example(item, arm, reg)])
    assert validate_example(item, arm, reg, example(item, arm, reg))
    payload = {'image_width': item['width'], 'image_height': item['height'], 'coordinate_contract': 'full_image_normalized_half_open_v1',
               'ordered_registry': reg}
    if multi_label(item): payload['allowed_labels'] = item['labels']
    pre = ('The first image is the reference crop; the second image is the scene.\n\n' if item['dataset'] == 'fsc147' else '')
    user = pre + 'Inspect the image using the region registry in the following payload. Follow the output contract exactly.\n\n' + json.dumps(payload)
    return sysm, user, reg


def subset_prompt(item, quadrant):
    """SKU split arm: the L contract restricted to objects whose center lies in one quadrant (the full image stays visible)."""
    sysm, user, reg = build_prompt(item, 'L')
    q = quadrant
    sysm += ('\n\nIn this call, report ONLY objects whose center lies inside the rectangle x in [%.4f, %.4f), y in [%.4f, %.4f) '
             '(normalized full-image coordinates). Ignore objects outside it.' % (q[0], q[2], q[1], q[3]))
    return sysm, user, reg


# ------------------------------------------------------------------ decoding (label-free)
def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def to_record(item, o):
    if not isinstance(o, dict): return None
    W, H = item['width'], item['height']
    if item['task'] == 'point':
        x, y = o.get('x'), o.get('y')
        if not (num(x) and num(y) and 0 <= x < 1 and 0 <= y < 1): return None
        return dict(label=item['labels'][0], cx=W * x, cy=H * y, box=None)
    v = [o.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if not all(num(t) for t in v): return None
    x1, y1, x2, y2 = v
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1): return None
    lab = o.get('label') if multi_label(item) else item['labels'][0]
    if lab not in item['labels']: return None
    return dict(label=lab, cx=W * (x1 + x2) / 2, cy=H * (y1 + y2) / 2, box=[W * x1, H * y1, W * x2, H * y2])


OBJ = re.compile(r'\{[^{}]*\}')
RID = re.compile(r'"region_id"\s*:\s*"(R\d{3})"')
STAT = re.compile(r'"status"\s*:\s*"(\w+)"')


def decode(item, arm, raw, requested=None):
    """Returns (records, audit). Strict JSON first; otherwise a lenient readout (objects matched by regex) flagged as salvaged."""
    raw = raw if isinstance(raw, str) else ''
    t = re.sub(r'^```[a-zA-Z]*\s*', '', raw.strip()); t = re.sub(r'\s*```$', '', t)
    au = dict(salvaged=0, invalid=0, unknown=0, missing_rows=0, status=None, n_rows=0)
    recs = []
    reg_ids = requested or [r['region_id'] for r in registry(item['width'], item['height'])]
    if arm == 'L':
        objs = None
        try:
            s = t.index('{'); v, _ = DEC.raw_decode(t, s)
            if isinstance(v, dict) and isinstance(v.get('objects'), list): objs = v['objects']; au['status'] = v.get('status')
        except ValueError:
            pass
        if objs is None:
            if t: au['salvaged'] = 1
            st = STAT.search(t); au['status'] = st.group(1) if st else None
            objs = []
            for m in OBJ.finditer(t):
                try: objs.append(json.loads(m.group(0)))
                except ValueError: au['invalid'] += 1
            objs = [o for o in objs if isinstance(o, dict) and ('x' in o or 'x1' in o)]
        for o in objs:
            r = to_record(item, o)
            if r is None: au['invalid'] += 1
            else: recs.append(r)
        au['unknown'] = int(au['status'] == 'unknown')
        return recs, au
    rows = None
    try:
        rows = [json.loads(l) for l in t.splitlines() if l.strip()]
        if not all(isinstance(r, dict) for r in rows): rows = None
    except ValueError:
        rows = None
    seen = set()
    if rows is None:
        if t: au['salvaged'] = 1
        ms = list(RID.finditer(t))
        for i, m in enumerate(ms):
            seg = t[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(t)]
            rid = m.group(1); seen.add(rid); st = STAT.search(seg)
            if st and st.group(1) == 'unknown': au['unknown'] += 1
            for mm in OBJ.finditer(seg):
                try: o = json.loads(mm.group(0))
                except ValueError: au['invalid'] += 1; continue
                if 'x' in o or 'x1' in o:
                    r = to_record(item, o)
                    if r is None: au['invalid'] += 1
                    else: recs.append(r)
    else:
        for row in rows:
            rid = row.get('region_id'); seen.add(rid)
            if row.get('status') == 'unknown': au['unknown'] += 1
            for o in row.get('objects') or []:
                r = to_record(item, o)
                if r is None: au['invalid'] += 1
                else: recs.append(r)
    au['n_rows'] = len(seen); au['missing_rows'] = len([r for r in reg_ids if r not in seen])
    return recs, au


def flags(item, arm, raw, finish, status_ok, au):
    trunc = str(finish) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS')
    f = dict(failed=int(not status_ok), empty=int(status_ok and not (raw or '').strip()), truncated=int(trunc), salvaged=int(au['salvaged'] > 0),
             unknown=int(au['unknown'] > 0), missing_rows=int(arm == 'R' and au['missing_rows'] > 0), invalid=int(au['invalid'] > 0))
    f['any'] = int(any(f.values())); return f


# ------------------------------------------------------------------ scoring
LABEL_OFFSET = 1e6


def _pts(recs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    return [(r['cx'] + LABEL_OFFSET * idx.get(r['label'], 0), r['cy']) for r in recs]


def match_points(a, b, tol):
    if not a or not b: return []
    A = np.array(a, float); B = np.array(b, float)
    ok = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1]) <= tol
    r, c = linear_sum_assignment((~ok).astype(np.int8))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


def iou_matrix(A, B):
    A = np.array(A, float); B = np.array(B, float)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    I = ix * iy; ua = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ub = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return I / (ua[:, None] + ub[None, :] - I + 1e-12)


def match_boxes(a_recs, b_boxes_labels, t):
    """a_recs: records with box/label; b: list of (box, label). Max matching with IoU >= t and equal labels."""
    if not a_recs or not b_boxes_labels: return []
    M = iou_matrix([r['box'] for r in a_recs], [b for b, _ in b_boxes_labels])
    lab = np.array([[r['label'] == l for _, l in b_boxes_labels] for r in a_recs])
    ok = (M >= t - 1e-12) & lab
    r, c = linear_sum_assignment((~ok).astype(np.int8))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


def in_ignore(r, ignore):
    return any(b[0] <= r['cx'] <= b[2] and b[1] <= r['cy'] <= b[3] for b in ignore)


def error(item, recs, metric):
    """metric: 'point' (FSC points / box centres, tolerance tau_point) or ('box', t). Unmatched predictions whose center lies in an
    ignore region are not counted. Returns (E, TP, n_pred_counted)."""
    Y = item['targets']; n = len(Y)
    if metric == 'point':
        P = _pts(recs, item['labels'])
        Yp = [(((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + LABEL_OFFSET * item['labels'].index(t['label'])),
               (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2)) for t in Y]
        m = match_points(P, Yp, item['tau_point'])
    else:
        m = match_boxes(recs, [(t['box'], t['label']) for t in Y], metric[1])
    matched = {i for i, _ in m}
    counted = [i for i in range(len(recs)) if i in matched or not in_ignore(recs[i], item['ignore'])]
    return len(counted) + n - 2 * len(m), len(m), len(counted)


def disagreement(item, ra, rb, metric):
    """U between two reports at the doubled tolerance (points: 2 tau; boxes: IoU >= 2t - 1)."""
    if metric == 'point':
        m = match_points(_pts(ra, item['labels']), _pts(rb, item['labels']), 2 * item['tau_point'])
    else:
        t = 2 * metric[1] - 1
        if t <= 0: return 0
        m = match_boxes(ra, [(r['box'], r['label']) for r in rb], t)
    return len(ra) + len(rb) - 2 * len(m)


def f1(item, recs, metric):
    E, tp, npred = error(item, recs, metric); n = len(item['targets'])
    if n == 0 and npred == 0: return 1.0
    return 2 * tp / (n + npred) if (n + npred) else 0.0


def region_split(item, recs, metric_tol=None):
    """Count/placement split on the 4x4 registry (geometric ownership by center), point metric."""
    reg = registry(item['width'], item['height'])
    def rid(x, y):
        for r in reg:
            if r['x0'] <= x < r['x0'] + r['width'] and r['y0'] <= y < r['y0'] + r['height']: return r['region_id']
        return None
    ys, ps = {}, {}
    for t in item['targets']:
        c = t['point'] if 'point' in t else [(t['box'][0] + t['box'][2]) / 2, (t['box'][1] + t['box'][3]) / 2]
        ys.setdefault(rid(*c), []).append((c[0] + LABEL_OFFSET * item['labels'].index(t['label']), c[1]))
    for p, r in zip(_pts(recs, item['labels']), recs): ps.setdefault(rid(r['cx'], r['cy']), []).append(p)
    cnt = plc = 0
    for g in set(ys) | set(ps):
        a, b = ps.get(g, []), ys.get(g, []); tp = len(match_points(a, b, item['tau_point']))
        cnt += abs(len(a) - len(b)); plc += 2 * (min(len(a), len(b)) - tp)
    return cnt, plc

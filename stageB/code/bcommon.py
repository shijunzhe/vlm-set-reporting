"""Shared helpers for FSR stage B: coordinate contracts, JSON decoding, matching. No model calls."""
import json, math, re, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
ROOT = '/home/claude/stageB'
MODELS = ['gpt61', 'sonnet55', 'gem38']
MODE = {'gpt61': 'grid', 'gem38': 'grid', 'sonnet55': 'pixel'}
DEC = json.JSONDecoder()


def fit_anthropic(im):
    r = min(1.0, 1568 / max(im.size), (1_150_000 / (im.width * im.height)) ** 0.5)
    return im if r >= 1 else im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def fit_max(im, side=2048):
    s = max(im.size)
    return im if s <= side else im.resize((max(1, round(im.width * side / s)), max(1, round(im.height * side / s))), Image.LANCZOS)


def prepare(im, model):
    """image actually sent: Sonnet pre-resized so the API does not resize again; others capped at 2048."""
    return fit_anthropic(im) if model == 'sonnet55' else fit_max(im)


def coord_sentence(mode, w, h):
    if mode == 'grid':
        return ('Report coordinates on a 0-1000 grid: x = 0 is the left edge and x = 1000 the right edge of the image, '
                'y = 0 is the top edge and y = 1000 the bottom edge. Use integers.')
    return (f'The image is {w}x{h} pixels. Report coordinates as integer pixel positions in this image, where (0,0) is the '
            'top-left corner.')


def first_json(raw):
    if not raw: return None
    s = raw.strip()
    s = re.sub(r'^```(?:json)?', '', s).strip(); s = re.sub(r'```$', '', s).strip()
    for i, ch in enumerate(s):
        if ch in '{[':
            try:
                v, _ = DEC.raw_decode(s[i:]); return v
            except Exception:
                continue
    return None


def salvage_points(raw):
    """recover point objects from truncated JSON."""
    out = []
    for m in re.finditer(r'\{[^{}]*\}', raw or ''):
        try: out.append(json.loads(m.group(0)))
        except Exception: pass
    return out


def _xy(o):
    if not isinstance(o, dict): return None
    for kx, ky in (('cx', 'cy'), ('x', 'y')):
        if kx in o:
            vx = o[kx]
            if isinstance(vx, (list, tuple)) and len(vx) == 2 and ky not in o: return float(vx[0]), float(vx[1])
            if ky in o:
                try: return float(vx), float(o[ky])
                except Exception: return None
    if 'point' in o and isinstance(o['point'], (list, tuple)) and len(o['point']) == 2:
        return float(o['point'][0]), float(o['point'][1])
    return None


def to_pixels(x, y, mode, w, h):
    """sent-frame -> original-image pixel frame (w, h = original size; for pixel mode the caller passes the scale)."""
    if mode == 'grid': return x / 1000 * w, y / 1000 * h
    return x, y


def decode_shapes(raw, mode, w, h, sent_w=None, sent_h=None, list_key='shapes', count_key='counts_by_type'):
    """returns (records, declared_counts or None, audit). records in original pixel coordinates."""
    v = first_json(raw); au = dict(json_ok=isinstance(v, dict), salvaged=0)
    items = None; dec = None
    if isinstance(v, dict):
        items = v.get(list_key)
        if items is None:
            for k in ('shapes', 'points', 'objects', 'elements'):
                if isinstance(v.get(k), list): items = v[k]; break
        dec = v.get(count_key)
        if dec is None and 'count' in v: dec = v.get('count')
    if items is None:
        items = salvage_points(raw); au['salvaged'] = 1
    if dec is None and raw:
        m_ = re.search(r'"' + count_key + r'"\s*:\s*(\{[^{}]*\})', raw)
        if m_:
            try: dec = json.loads(m_.group(1)); au['dec_regex'] = 1
            except Exception: dec = None
        if dec is None:
            m_ = re.search(r'"count"\s*:\s*(\d+)', raw)
            if m_: dec = int(m_.group(1)); au['dec_regex'] = 1
    sx = (w / sent_w) if (mode == 'pixel' and sent_w) else 1.0; sy = (h / sent_h) if (mode == 'pixel' and sent_h) else 1.0
    recs = []
    for o in items or []:
        p = _xy(o)
        if p is None: continue
        x, y = to_pixels(p[0], p[1], mode, w, h)
        if mode == 'pixel': x, y = x * sx, y * sy
        recs.append(dict(type=str(o.get('type', o.get('label', 'obj'))).lower().strip(), cx=x, cy=y))
    return recs, dec, au


def match_typed(P, G, tol, typed=True):
    if not P or not G: return []
    p = np.array([[s['cx'], s['cy']] for s in P], float); g = np.array([[s['cx'], s['cy']] for s in G], float)
    D = np.sqrt(((p[:, None] - g[None]) ** 2).sum(-1))
    if typed: D = np.where(np.array([[a['type'] != b['type'] for b in G] for a in P]), np.inf, D)
    r, c = linear_sum_assignment(np.where(D <= tol, D, 1e6))
    return [(int(i), int(j)) for i, j in zip(r, c) if D[i, j] <= tol]


def draw_markers(im, pts, arm=8, stroke=3, color=(255, 0, 255, 120)):
    im = im.convert('RGBA'); ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for x, y in pts:
        d.line([(x - arm, y), (x + arm, y)], fill=color, width=stroke); d.line([(x, y - arm), (x, y + arm)], fill=color, width=stroke)
    return Image.alpha_composite(im, ov).convert('RGB')


def write_manifest(path, reqs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for q in reqs: f.write(json.dumps(q) + '\n')


def load_responses(subset, stage):
    R = {}
    for m in MODELS + ['qwen3vl']:
        f = os.path.join(ROOT, f'runs/{subset}/responses_{stage}_{m}.jsonl')
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in R: R[r['request_id']] = r
    return R


def load_requests(subset, stage):
    return {json.loads(l)['request_id']: json.loads(l) for l in open(os.path.join(ROOT, f'runs/{subset}/requests_{stage}.jsonl'))}


PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75), 'qwen3vl': (0.0, 0.0)}
def cost(m, r): return (r['usage'].get('input_tokens', 0) * PRICE[m][0] + r['usage'].get('output_tokens', 0) * PRICE[m][1]) / 1e6 if r and r.get('usage') else 0.0


NUM = re.compile(r'-?\d+(?:\.\d+)?')
def boxes_positional(raw):
    """box objects read positionally (x1, y1, x2, y2 in written order). Tolerates key-name drift such as
    {"x": 0, "y": 204, "x": 67, "y2": 431}, which some models emit late in long lists. Returns (boxes, n_irregular)."""
    out = []; irregular = 0
    for m_ in re.finditer(r'\{[^{}]*\}', raw or ''):
        txt = m_.group(0)
        if '"elements"' in txt or '"crops"' in txt: continue
        keys = re.findall(r'"([A-Za-z_0-9]+)"\s*:', txt); nums = NUM.findall(re.sub(r'"[A-Za-z_0-9]+"\s*:', ':', txt))
        if len(nums) < 4: continue
        if keys[:4] != ['x1', 'y1', 'x2', 'y2']: irregular += 1
        x1, y1, x2, y2 = (float(v) for v in nums[:4])
        out.append([x1, y1, x2, y2])
    return out, irregular

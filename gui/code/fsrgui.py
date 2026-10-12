"""FSR GUI experiment library (no model calls here): image views and crops, overlays, prompts, decoders, label-free
correspondence, constructions (single / intersection / union / majority), FSR-Direct edit validation and commit (edit
ledger), and scoring (labels are used only in the scoring functions).

Coordinates: every model-facing coordinate is normalized to the image (or crop) it refers to, 0 <= x < 1. Internally all
records are stored as full-image normalized boxes [x1, y1, x2, y2] (parsing) or points [x, y] (click).
"""
import json, math, re, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = '/home/claude/gui'
MAX_SIDE = 2048
OVL = 0.06                                              # crop overlap (fraction of the image side)
CROPS = [(0.0, 0.0, 0.5 + OVL / 2, 0.5 + OVL / 2), (0.5 - OVL / 2, 0.0, 1.0, 0.5 + OVL / 2),
         (0.0, 0.5 - OVL / 2, 0.5 + OVL / 2, 1.0), (0.5 - OVL / 2, 0.5 - OVL / 2, 1.0, 1.0)]
DEC = json.JSONDecoder()
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except OSError:
    FONT = ImageFont.load_default()


# ------------------------------------------------------------------ images
def load(path):
    im = Image.open(path); im.load()
    return im.convert('RGB')


def fit(im, max_side=MAX_SIDE):
    s = max(im.size)
    if s <= max_side: return im
    r = max_side / s
    return im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)


def fit_anthropic(im):
    """Resize so that the Anthropic API does not resize again (long side <= 1568 and at most 1.15 megapixels)."""
    r = min(1.0, 1568 / max(im.size), (1_150_000 / (im.width * im.height)) ** 0.5)
    return im if r >= 1 else im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def crop_native(im, c):
    W, H = im.size
    return im.crop((round(c[0] * W), round(c[1] * H), round(c[2] * W), round(c[3] * H)))


def crop_norm(im, c):
    """c = (x1, y1, x2, y2) normalized; native-resolution crop, then capped at MAX_SIDE."""
    W, H = im.size
    return fit(im.crop((round(c[0] * W), round(c[1] * H), round(c[2] * W), round(c[3] * H))))


def save(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path): im.save(path, format='PNG', optimize=True)
    if os.path.getsize(path) > 4_500_000:                # stay below every provider's per-image limit
        jp = path[:-4] + '.jpg'
        im.save(jp, format='JPEG', quality=92); return jp
    return path


def draw_boxes(im, items, frame=(0, 0, 1, 1)):
    """items: list of (box_full_norm, tag, color). Boxes are drawn relative to frame (a crop of the full image)."""
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size; fx1, fy1, fx2, fy2 = frame
    lw = max(1, round(max(W, H) / 900))
    for b, tag, col in items:
        x1 = (b[0] - fx1) / (fx2 - fx1) * W; y1 = (b[1] - fy1) / (fy2 - fy1) * H
        x2 = (b[2] - fx1) / (fx2 - fx1) * W; y2 = (b[3] - fy1) / (fy2 - fy1) * H
        if x2 < 0 or y2 < 0 or x1 > W or y1 > H: continue
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        if tag:
            tw = d.textlength(tag, font=FONT); tx = max(0, min(W - tw - 2, x1)); ty = max(0, y1 - 16)
            d.rectangle([tx, ty, tx + tw + 2, ty + 15], fill=col); d.text((tx + 1, ty), tag, fill=(255, 255, 255), font=FONT)
    return im


def draw_marker(im, x, y, frame=(0, 0, 1, 1)):
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size; fx1, fy1, fx2, fy2 = frame
    px = (x - fx1) / (fx2 - fx1) * W; py = (y - fy1) / (fy2 - fy1) * H; r = max(8, round(max(W, H) / 80)); lw = max(2, r // 4)
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=lw)
    d.line([px - 2 * r, py, px - r // 2, py], fill=(255, 0, 0), width=lw); d.line([px + r // 2, py, px + 2 * r, py], fill=(255, 0, 0), width=lw)
    d.line([px, py - 2 * r, px, py - r // 2], fill=(255, 0, 0), width=lw); d.line([px, py + r // 2, px, py + 2 * r], fill=(255, 0, 0), width=lw)
    return im


def window(x, y, frac):
    """Square-ish window of size frac x frac (normalized) centered at (x, y), shifted to stay inside the image."""
    x1 = min(max(0.0, x - frac / 2), 1 - frac); y1 = min(max(0.0, y - frac / 2), 1 - frac)
    return (x1, y1, x1 + frac, y1 + frac)


# ------------------------------------------------------------------ coordinate contracts
# 'grid': integers on a 0-1000 grid of the referenced image (GPT-6.1, Gemini). 'pixel': pixel coordinates of the image exactly
# as sent, which is pre-resized so that the provider does not resize it again (Sonnet; the pilot showed it answers in pixels).
def coord_sentence(mode, what):
    if mode == 'grid':
        return (f'Coordinates are integers on a 0 to 1000 grid of the {what}: 0 is the left (or top) edge and 1000 the right (or bottom) '
                'edge, whatever the pixel size of the image. Never give pixel coordinates.')
    return (f'Coordinates are pixel coordinates of the {what} exactly as you see it: x from 0 at its left edge to its width at the right '
            'edge, y from 0 at its top edge to its height at the bottom edge. Pixel sizes are given in the request.')


def rng_text(mode):
    return '0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000' if mode == 'grid' else '0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height'


TASK_PARSE = ('Find every visible UI element in the screenshot: buttons, icons, menu items, tabs, links, text input fields, checkboxes, '
              'radio buttons, dropdowns, sliders, list or table items, labels, headings, text blocks and images. Report each distinct '
              'element once, with a tight box around that element only.')
CROP_INTRO = ('The four images are overlapping crops of ONE screenshot at native resolution: crop 1 is the top-left part, crop 2 the '
              'top-right, crop 3 the bottom-left and crop 4 the bottom-right. Neighbouring crops overlap by a narrow strip.')
EDIT_RULES = ('Return only atomic edits. Allowed operations: "add" (new elements: "new" is a list of one or more elements), "delete" '
              '("ids": draft elements that are not UI elements, are duplicates of another draft element, or do not exist), "replace" '
              '(one id, one new element: corrects the box or name of the same element), "split" (one id, two or more new elements: '
              'distinct elements wrongly combined), "merge" (two or more ids, one new element: reports of the same element). Every '
              'edit has "evidence": at most twelve words about what is visible in the image. Do not delete a real element whose box '
              'is only inaccurate; use replace. Overlap alone does not prove a duplicate. At most 32 edits; put the most important '
              'first. If the draft needs no change, return {"edits": []}.')


def frame_box(mode, what):
    return (f'Each element has x1, y1, x2, y2: the left, top, right and bottom edges of its tight axis-aligned bounding box, with '
            f'{rng_text(mode)}. ' + coord_sentence(mode, what) + ' Each element also has name: at most four words of its visible text, '
            'or a short description if it has no text.')


def prompts(mode):
    P = {}
    P['draft'] = ('You are performing visual set reporting on a screenshot. ' + TASK_PARSE + ' Scan the whole screenshot systematically, top '
                  'to bottom and left to right, and check every part before you finish; do not stop early. Do not invent elements.\n\n' +
                  frame_box(mode, 'FULL IMAGE') + '\n\nOutput contract: return exactly one JSON object {"status": ..., "elements": [...]}. '
                  'status is found when you have completed the inspection, empty when there is no UI element, unknown when you could not '
                  'complete it (include all recovered elements). No prose, no code fence.\n\nExample of the schema only:\n'
                  '{"status":"found","elements":[{"x1":12,"y1":20,"x2":61,"y2":48,"name":"File"},'
                  '{"x1":400,"y1":300,"x2":520,"y2":340,"name":"Search field"}]}')
    P['tiles'] = ('You are performing visual set reporting on a screenshot. ' + CROP_INTRO + ' ' + TASK_PARSE + ' Inspect each crop '
                  'completely; do not stop early. Do not invent elements. Report an element in every crop where its center is visible.\n\n' +
                  frame_box(mode, 'CROP it is reported in') + '\n\nOutput contract: return exactly one JSON object {"status": ..., "crops": '
                  '[{"crop": 1, "elements": [...]}, {"crop": 2, ...}, {"crop": 3, ...}, {"crop": 4, ...}]}. status is found, empty or unknown '
                  'as usual. No prose, no code fence.')
    P['audit_full'] = ('You are auditing a draft list of UI elements for a screenshot. ' + TASK_PARSE + '\n\nThe first image is the '
                       'screenshot. The second image is the same screenshot with the draft elements drawn as numbered red boxes. The draft '
                       'is a hypothesis and may contain wrong, duplicated or badly placed elements, and it may miss elements. Do two things: '
                       '(1) check the existing draft elements against the screenshot; (2) scan all four quadrants of the screenshot for '
                       'visible UI elements missing from the draft, including areas with no red box. No box in a region is not evidence '
                       'that the region is empty.\n\n' + frame_box(mode, 'FULL IMAGE') + ' Draft boxes in the text payload use the same '
                       'coordinates.\n\n' + EDIT_RULES + '\n\nOutput contract: one JSON object {"edits": [...]}, e.g. {"edits":[{"op":"add",'
                       '"new":[{"x1":100,"y1":100,"x2":150,"y2":130,"name":"Save"}],"evidence":"save icon in toolbar not in draft"},'
                       '{"op":"delete","ids":[7],"evidence":"empty background"}]}. No prose, no code fence.')
    P['audit_crop'] = ('You are auditing a draft list of UI elements for a screenshot. ' + TASK_PARSE + '\n\n' + CROP_INTRO + ' The draft '
                       'elements are drawn on the crops as numbered red boxes. The draft is a hypothesis and may contain wrong, duplicated '
                       'or badly placed elements, and it may miss elements, especially small ones. Do two things: (1) check the existing '
                       'draft elements against the crops; (2) scan every crop completely for visible UI elements missing from the draft, '
                       'including areas with no red box. No box in a region is not evidence that the region is empty.\n\nNew elements have '
                       f'"crop" (1 to 4) and x1, y1, x2, y2 with {rng_text(mode)} in THAT CROP. ' + coord_sentence(mode, 'crop') +
                       ' Each new element also has name (at most four words). Draft boxes in the text payload are given in the coordinates '
                       'of the FULL screenshot, as stated in the request.\n\n' + EDIT_RULES + '\n\nOutput contract: one JSON object '
                       '{"edits": [...]}, e.g. {"edits":[{"op":"add","new":[{"crop":2,"x1":100,"y1":200,"x2":160,"y2":240,"name":"Bold"}],'
                       '"evidence":"bold button not in draft"}]}. No prose, no code fence.')
    P['verify_full'] = ('You are checking proposed changes to a draft list of UI elements of a screenshot. ' + TASK_PARSE + '\n\nThe first '
                        'image is the screenshot. The second image shows each proposed change k: red boxes labelled k- are the draft elements '
                        'the change removes or corrects; green boxes labelled k+ are the elements it would add. For EACH proposed change '
                        'decide: "apply" if the change is visually justified and makes the list more faithful to the screenshot; "keep" to '
                        'retain the draft as it is; "uncertain" if the image does not let you decide. Check whether an added element really '
                        'exists and is not already in the draft, whether a deleted element really is not a UI element or is a duplicate, and '
                        'whether a corrected box fits the element better. Do not propose new changes.\n\nOutput contract: one JSON object '
                        '{"decisions": [{"change": 1, "decision": "apply"}, ...]} with one entry per change. No prose, no code fence.')
    P['verify_crop'] = P['verify_full'].replace('The first image is the screenshot. The second image shows each proposed change k:',
                                                CROP_INTRO + ' The crops show each proposed change k:')
    P['click'] = ('You are a computer-use agent. Given a screenshot and an instruction, return the point to click to carry out the '
                  'instruction. ' + coord_sentence(mode, 'FULL IMAGE') + '\n\nOutput contract: one JSON object {"x": ..., "y": ...}; if the '
                  'target is not visible, {"status": "not_found"}. No prose, no code fence.')
    P['zoom'] = ('You are a computer-use agent. The image is a zoomed crop of a larger screenshot, at native resolution. Given an instruction, '
                 'return the point to click inside THIS crop. ' + coord_sentence(mode, 'CROP') + ' If the target element is not inside this '
                 'crop, say so.\n\nOutput contract: one JSON object {"x": ..., "y": ...} or {"status": "not_in_crop"}. No prose, no code fence.')
    P['vclick'] = ('You are checking a click proposed by a computer-use agent. The image is a zoomed crop of a screenshot at native '
                   'resolution; the red circle with a cross marks the proposed click. Decide whether clicking at the center of the marker '
                   'would hit the UI element the instruction refers to. ' + coord_sentence(mode, 'CROP') + '\n\nOutput contract: one JSON '
                   'object {"decision": "correct"} or {"decision": "uncertain"} or {"decision": "wrong", "x": ..., "y": ...} where x, y is '
                   'the correct point in this crop if the right element is visible in the crop; if it is not visible, {"decision": "wrong"} '
                   'without coordinates. No prose, no code fence.')
    return P


def size_note(mode, sizes, names):
    if mode == 'grid': return ''
    return 'Pixel sizes: ' + '; '.join(f'{n} {w}x{h}' for n, (w, h) in zip(names, sizes)) + '.\n\n'


def enc_box(b, mode, scale):
    if mode == 'grid': return [round(v * 1000) for v in b]
    W, H = scale; return [round(b[0] * W), round(b[1] * H), round(b[2] * W), round(b[3] * H)]


def user_draft(mode, size):
    return size_note(mode, [size], ['image']) + 'Report every visible UI element of this screenshot. Follow the output contract exactly.'


def user_tiles(mode, sizes):
    return size_note(mode, sizes, ['crop 1', 'crop 2', 'crop 3', 'crop 4']) + 'Report every visible UI element in each of the four crops. Follow the output contract exactly.'


def user_audit(recs, mode, full_size, crop_sizes=None):
    pay = [dict(zip(('id', 'x1', 'y1', 'x2', 'y2', 'name'), [i + 1] + enc_box(b, mode, full_size) + [n])) for i, (b, n) in enumerate(recs)]
    note = size_note(mode, [full_size] + (crop_sizes or []), ['full screenshot (payload frame)'] + ([f'crop {k}' for k in (1, 2, 3, 4)] if crop_sizes else []))
    return (note + 'Draft elements (ids match the numbered red boxes; coordinates of the full screenshot):\n' + json.dumps(pay) +
            '\n\nReturn the edits. Follow the output contract exactly.')


def user_verify(changes, mode, full_size):
    out = [{'change': k, 'op': ch['op'], 'removes': [{'id': i, 'box': enc_box(b, mode, full_size)} for i, b in ch['old']],
            'adds': [{'box': enc_box(b, mode, full_size), 'name': n} for b, n in ch['new']]} for k, ch in enumerate(changes, 1)]
    return (size_note(mode, [full_size], ['full screenshot (payload frame)']) + 'Proposed changes (boxes in coordinates of the full '
            'screenshot; they match the k-/k+ labels in the image):\n' + json.dumps(out) + '\n\nReturn one decision per change. Follow the output contract exactly.')


def user_click(instr, plat, mode, size):
    return size_note(mode, [size], ['image']) + f'Platform: {plat}. Instruction: "{instr}". Return the click point.'


# ------------------------------------------------------------------ decoding (label-free)
OBJ = re.compile(r'\{[^{}]*\}')


def first_json(raw):
    t = (raw or '').strip(); t = re.sub(r'^```[a-zA-Z]*\s*', '', t); t = re.sub(r'\s*```$', '', t)
    try:
        s = t.index('{'); v, _ = DEC.raw_decode(t, s); return v
    except ValueError:
        return None


def salvage(raw):
    """Lenient readout when the envelope is not valid JSON: every flat {...} object that parses (flagged as salvaged)."""
    out = []
    for m in OBJ.finditer(raw or ''):
        try: out.append(json.loads(m.group(0)))
        except ValueError: pass
    return [o for o in out if isinstance(o, dict)]


def _num(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _coords(o, keys):
    """Accept flat keys, a nested 'box' dict or list, or Gemini-style box_2d [ymin, xmin, ymax, xmax]."""
    if keys == 'box':
        if isinstance(o.get('box_2d'), list) and len(o['box_2d']) == 4:
            y1, x1, y2, x2 = o['box_2d']; return [x1, y1, x2, y2]
        if isinstance(o.get('box'), list) and len(o['box']) == 4: return list(o['box'])
        if isinstance(o.get('box'), dict): o = o['box']
        return [o.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if isinstance(o.get('point'), list) and len(o['point']) == 2: return list(o['point'])
    return [o.get('x'), o.get('y')]


def _to_unit(v, flag, scale):
    """grid (scale None): values in [0, 1000] / 1000; pixel (scale = (W, H)): x / W, y / H, values up to 2% beyond the size clipped.
    A record whose values are all <= 1 with a fractional part is read as already normalized (flagged)."""
    if not all(_num(t) for t in v) or min(v) < 0: return None
    if max(v) <= 1.0 and any(abs(t - round(t)) > 1e-9 for t in v): flag[0] += 1; return [float(t) for t in v]
    if scale is None:
        return None if max(v) > 1000 else [t / 1000 for t in v]
    W, H = scale; den = [W, H] * (len(v) // 2)
    u = [t / d for t, d in zip(v, den)]
    return None if max(u) > 1.02 else u


def norm_box(o, flag, scale=None):
    v = _to_unit(_coords(o, 'box'), flag, scale)
    if v is None: return None
    x1, y1, x2, y2 = [min(max(t, 0.0), 1.0) for t in v]
    return [x1, y1, x2, y2] if (x1 < x2 and y1 < y2) else None


def norm_pt(o, flag, scale=None):
    v = _to_unit(_coords(o, 'pt'), flag, scale)
    return None if v is None else [min(max(v[0], 0.0), 1.0), min(max(v[1], 0.0), 1.0)]


def name_of(o):
    n = o.get('name', o.get('label'))
    return (n if isinstance(n, str) else '')[:40]


def decode_list(raw, scale=None):
    v = first_json(raw); au = dict(json_ok=int(isinstance(v, dict) and isinstance(v.get('elements'), list)), invalid=0, rescaled=0, status=None, salvaged=0)
    sf = [0]; recs = []
    if au['json_ok']: au['status'] = v.get('status'); els = v['elements']
    else: els = [o for o in salvage(raw) if any(k in o for k in ('x1', 'box', 'box_2d'))]; au['salvaged'] = int(bool(els))
    for o in els:
        b = norm_box(o, sf, scale) if isinstance(o, dict) else None
        if b is None: au['invalid'] += 1
        else: recs.append((b, name_of(o)))
    au['rescaled'] = sf[0]
    return recs, au


def crop_to_full(b, c):
    w, h = c[2] - c[0], c[3] - c[1]
    return [c[0] + b[0] * w, c[1] + b[1] * h, c[0] + b[2] * w, c[1] + b[3] * h]


def decode_tiles(raw, scales=None):
    v = first_json(raw); au = dict(json_ok=int(isinstance(v, dict) and isinstance(v.get('crops'), list)), invalid=0, rescaled=0, status=None, dedup=0)
    sf = [0]; recs = []
    if au['json_ok']:
        au['status'] = v.get('status')
        for row in v['crops']:
            if not isinstance(row, dict) or row.get('crop') not in (1, 2, 3, 4): continue
            k = row['crop'] - 1; c = CROPS[k]
            for o in row.get('elements') or []:
                b = norm_box(o, sf, scales[k] if scales else None) if isinstance(o, dict) else None
                if b is None: au['invalid'] += 1
                else: recs.append((crop_to_full(b, c), name_of(o)))
    au['rescaled'] = sf[0]
    keep = []                                            # duplicates from the overlap strips: IoU >= 0.5, keep the first
    for r in recs:
        if any(iou(r[0], k[0]) >= 0.5 for k in keep): au['dedup'] += 1; continue
        keep.append(r)
    return keep, au


OPS = ('add', 'delete', 'replace', 'split', 'merge')


def normalize_edit(e):
    """Lenient reading of one edit (synonym keys are flagged by the caller): op from 'op'/'action'/'edit'/'type' or from a key
    named after the operation; ids from 'ids'/'id'; new from 'new' (a dict is read as a one-element list; null as empty)."""
    e = dict(e); lenient = 0
    op = e.get('op')
    if op not in OPS:
        for k in ('action', 'edit', 'type', 'operation'):
            if e.get(k) in OPS: op = e[k]; lenient = 1; break
    if op not in OPS:
        for k in OPS:
            if k in e:
                v = e[k]; op = k; lenient = 1
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v): e['new'] = v
                elif isinstance(v, dict): e['new'] = [v]
                elif isinstance(v, int) and not isinstance(v, bool) and 'id' not in e and 'ids' not in e: e['ids'] = [v]
                break
    ids = e.get('ids') if 'ids' in e else ([e['id']] if 'id' in e else [])
    if isinstance(ids, int) and not isinstance(ids, bool): ids = [ids]
    ids = [i for i in ids if isinstance(i, int) and not isinstance(i, bool)] if isinstance(ids, list) else []
    nl = e.get('new'); nl = [nl] if isinstance(nl, dict) else (nl if isinstance(nl, list) else [])
    if op is None and ids and not nl and e.get('new', 0) is None: op = 'delete'; lenient = 1
    return op, ids, nl, lenient


def decode_edits(raw, n_draft, crop_mode=False, full_scale=None, crop_scales=None):
    v = first_json(raw); au = dict(json_ok=int(isinstance(v, dict)), proposed=0, invalid=0, deferred_cap=0, rescaled=0)
    sf = [0]; out = []
    eds = v.get('edits') if isinstance(v, dict) and isinstance(v.get('edits'), list) else []
    for e in eds:
        au['proposed'] += 1
        if not isinstance(e, dict): au['invalid'] += 1; continue
        op, ids, nl, len_ = normalize_edit(e); au['lenient'] = au.get('lenient', 0) + len_
        new = []; bad_new = 0
        for o in nl:
            if not isinstance(o, dict): bad_new += 1; continue
            if crop_mode:
                c = o.get('crop')
                if c not in (1, 2, 3, 4): bad_new += 1; continue
                b = norm_box(o, sf, crop_scales[c - 1] if crop_scales else None)
                if b is not None: b = crop_to_full(b, CROPS[c - 1])
            else:
                b = norm_box(o, sf, full_scale)
            if b is None: bad_new += 1; continue
            new.append((b, name_of(o)))
        ok = (op == 'add' and not ids and len(new) >= 1) or (op == 'delete' and len(ids) >= 1 and not new) or \
             (op == 'replace' and len(ids) == 1 and len(new) == 1) or (op == 'split' and len(ids) == 1 and len(new) >= 2) or \
             (op == 'merge' and len(ids) >= 2 and len(new) == 1)
        ok = ok and bad_new == 0 and all(1 <= i <= n_draft for i in ids) and len(set(ids)) == len(ids)
        if not ok: au['invalid'] += 1; continue
        if len(out) >= 32: au['deferred_cap'] += 1; continue
        out.append(dict(op=op, ids=ids, new=new, evidence=str(e.get('evidence', ''))[:120]))
    au['rescaled'] = sf[0]
    return out, au


def decode_decisions(raw, n):
    v = first_json(raw); dec = ['keep'] * n; au = dict(json_ok=int(isinstance(v, dict)), missing=0, repeated=0, malformed=0)
    seen = set()
    for d in (v.get('decisions') or []) if isinstance(v, dict) else []:
        if not isinstance(d, dict) or not isinstance(d.get('change'), int) or not 1 <= d['change'] <= n: au['malformed'] += 1; continue
        k = d['change'] - 1
        if k in seen: au['repeated'] += 1; dec[k] = 'keep'; continue          # repeated -> unchanged
        seen.add(k); x = d.get('decision'); dec[k] = x if x in ('apply', 'keep', 'uncertain') else 'keep'
    au['missing'] = n - len(seen)
    return dec, au


def decode_click(raw, scale=None):
    v = first_json(raw); sf = [0]
    if not isinstance(v, dict): return None, dict(json_ok=0, status=None, rescaled=0)
    return norm_pt(v, sf, scale), dict(json_ok=1, status=v.get('status'), rescaled=sf[0])


def decode_vclick(raw, scale=None):
    v = first_json(raw); sf = [0]
    if not isinstance(v, dict): return 'keep', None, dict(json_ok=0)
    d = v.get('decision'); p = norm_pt(v, sf, scale) if d == 'wrong' else None
    return (d if d in ('correct', 'wrong', 'uncertain') else 'keep'), p, dict(json_ok=1)


# ------------------------------------------------------------------ geometry, correspondence, constructions
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    U = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I
    return I / U if U > 0 else 0.0


def iou_mat(A, B):
    if not A or not B: return np.zeros((len(A), len(B)))
    A = np.array(A, float); B = np.array(B, float)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    I = ix * iy; ua = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ub = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return I / np.maximum(ua[:, None] + ub[None, :] - I, 1e-12)


def lex_match(A, B, thr=0.5):
    """Maximum-cardinality one-to-one matching with IoU >= thr, ties broken by total IoU (label-free correspondence)."""
    M = iou_mat(A, B)
    if M.size == 0: return []
    W = np.where(M >= thr, 1000.0 + M, 0.0); r, c = linear_sum_assignment(-W)
    return [(int(i), int(j)) for i, j in zip(r, c) if M[i, j] >= thr]


def clusters(reports, thr=0.5):
    """reports: list of record lists [(box, name)]. Sequential correspondence against cluster representatives (first member)."""
    cl = []
    for ri, P in enumerate(reports):
        pairs = lex_match([c['rep'] for c in cl], [p[0] for p in P], thr); inv = {j: i for i, j in pairs}
        for j, p in enumerate(P):
            if j in inv: cl[inv[j]]['members'].append(p[0]); cl[inv[j]]['sup'].add(ri)
            else: cl.append(dict(rep=p[0], members=[p[0]], sup={ri}, name=p[1]))
    for c in cl: c['fused'] = list(np.mean(np.array(c['members']), 0))
    return cl


def construct(reports, kind):
    if kind == 'single': return [r[0] for r in reports[0]]
    cl = clusters(reports); K = len(reports)
    need = {'inter': K, 'union': 1, 'majority': K // 2 + 1}[kind]
    return [c['fused'] for c in cl if len(c['sup']) >= need]


# ------------------------------------------------------------------ FSR-Direct commit (edit ledger)
def changes_for_verify(edits, draft):
    return [dict(op=e['op'], old=[(i, draft[i - 1][0]) for i in e['ids']], new=e['new']) for e in edits]


def commit(draft, edits, decisions):
    """Apply accepted, non-conflicting edits; untouched draft records are copied unchanged (Proposition D1). Returns the final list
    of (box, name), and the ledger."""
    used = set(); ledger = []; removed = set(); added = []
    for k, (e, d) in enumerate(zip(edits, decisions)):
        if d != 'apply': ledger.append(dict(change=k + 1, op=e['op'], status=d)); continue
        if used & set(e['ids']): ledger.append(dict(change=k + 1, op=e['op'], status='deferred_conflict')); continue
        used |= set(e['ids']); removed |= set(e['ids']); added += e['new']
        ledger.append(dict(change=k + 1, op=e['op'], status='committed', ids=e['ids'], n_new=len(e['new'])))
    final = [r for i, r in enumerate(draft, 1) if i not in removed] + list(added)
    return final, ledger


def apply_one(draft, e):
    return [r for i, r in enumerate(draft, 1) if i not in set(e['ids'])] + list(e['new'])


# ------------------------------------------------------------------ scoring (labels used here only)
def score_boxes(pred, gt, thr=0.5):
    """pred, gt: lists of boxes. Returns dict(E, tp, fp, fn, f1, n, cnt) at IoU >= thr, and the centre-hit variant (prediction
    centre inside a ground-truth box, one-to-one)."""
    n = len(gt); cnt = len(pred)
    tp = len(lex_match(pred, gt, thr)) if pred and gt else 0
    if pred and gt:
        P = np.array(pred, float); G = np.array(gt, float); cx = (P[:, 0] + P[:, 2]) / 2; cy = (P[:, 1] + P[:, 3]) / 2
        ok = (cx[:, None] >= G[None, :, 0]) & (cx[:, None] <= G[None, :, 2]) & (cy[:, None] >= G[None, :, 1]) & (cy[:, None] <= G[None, :, 3])
        r, c = linear_sum_assignment(-ok.astype(float)); tc = int(sum(ok[i, j] for i, j in zip(r, c)))
    else:
        tc = 0
    f1 = 1.0 if n + cnt == 0 else 2 * tp / (n + cnt)
    return dict(E=cnt + n - 2 * tp, tp=tp, fp=cnt - tp, fn=n - tp, f1=f1, n=n, cnt=cnt, tp_c=tc, E_c=cnt + n - 2 * tc,
                f1_c=(1.0 if n + cnt == 0 else 2 * tc / (n + cnt)))


def hit(p, bbox_norm):
    return p is not None and bbox_norm[0] <= p[0] <= bbox_norm[2] and bbox_norm[1] <= p[1] <= bbox_norm[3]

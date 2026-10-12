"""E3 Direct mini test (no model calls here): FSR-Direct audit and verification on E3 drafts.
Images: 20 per dataset (SKU-110K, DOTA, FSC-147), stratified by E3 density stratum; 3 models. Draft = the existing E3 L-min call
(rep 0, 'min' reasoning). Arms: A3 (full image + numbered overlay of the draft) and A4 (four native-resolution crops with the
draft drawn). The coordinate contract is the E3 one (normalized to the full image or to the crop, four decimals), which all three
models followed in E3.
  python -I code/e3direct.py --stage B|C
"""
import argparse, json, os, sys, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrgui as G
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L
import analyze_e3 as A0
from score_e3 import load_items
OUT = '/home/claude/gui/runs/e3direct'
MODELS = ['gpt61', 'sonnet55', 'gem38']
os.makedirs(OUT, exist_ok=True)


def pick():
    items = load_items(); rng = random.Random('e3direct-v1'); sel = []
    for ds in ('sku110k', 'dota', 'fsc147'):
        its = sorted([i for i in items.values() if i['dataset'] == ds], key=lambda x: x['id'])
        for s, k in ((0, 7), (1, 7), (2, 6)):
            sel += rng.sample([i for i in its if i['stratum'] == s], k)
    return items, sel


def drafts():
    rows = A0.load_rows('runs_main/scored.jsonl'); D = {}
    for x in rows:
        if x['cand'] == 'L-min' and x['block'] in ('MAIN', 'AGG'): D[(x['item'], x['model'], x['rep'])] = A0.recs_of(x)
    return D


def to_norm(it, r):
    W, H = it['width'], it['height']
    if it['task'] == 'point': return dict(x=r['cx'] / W, y=r['cy'] / H)
    b = r['box']; return dict(x1=b[0] / W, y1=b[1] / H, x2=b[2] / W, y2=b[3] / H, label=r['label'])


def nbox(it, r):
    """box used for drawing, in normalized full-image coordinates (points get a small square)."""
    W, H = it['width'], it['height']
    if it['task'] == 'point' or r.get('box') is None:
        s = 0.008; return [r['cx'] / W - s, r['cy'] / H - s, r['cx'] / W + s, r['cy'] / H + s]
    b = r['box']; return [b[0] / W, b[1] / H, b[2] / W, b[3] / H]


def frame(it, what):
    if it['task'] == 'point':
        return (f'Each object is given by x and y, the coordinates of its center normalized to the {what} (0 <= x < 1, 0 <= y < 1, four decimals).')
    s = (f'Each object is given by x1, y1, x2, y2, the edges of its axis-aligned bounding box normalized to the {what} '
         '(0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1, four decimals)')
    return s + ('; and label, one of: ' + ', '.join(it['labels']) + '.' if L.multi_label(it) else '.')


RULES = ('Return only atomic edits: "add" ("new": a list of one or more objects), "delete" ("ids": draft objects that are not real '
         'target objects, are duplicates, or do not exist), "replace" (one id, one new object: corrects the position or label of the '
         'same object), "split" (one id, two or more new objects), "merge" (two or more ids, one new object). Every edit has '
         '"evidence": at most twelve words about what is visible. Do not delete a real object whose position is only inaccurate; use '
         'replace. Overlap alone does not prove a duplicate. At most 32 edits, most important first. If the draft needs no change, '
         'return {"edits": []}.')


def example(it, crop):
    c = '"crop":2,' if crop else ''
    if it['task'] == 'point':
        o = '{%s"x":0.4120,"y":0.3330}' % c
    else:
        o = '{%s"x1":0.4120,"y1":0.3330,"x2":0.4610,"y2":0.3790%s}' % (c, (',"label":"%s"' % it['labels'][0]) if L.multi_label(it) else '')
    return ('{"edits":[{"op":"add","ids":[],"new":[%s],"evidence":"unmarked object on the left"},{"op":"delete","ids":[7],"new":[],'
            '"evidence":"mark on empty background"},{"op":"replace","ids":[12],"new":[%s],"evidence":"mark offset from the object"}]}' % (o, o))


def sys_audit(it, crop):
    pre = 'The first image is the reference crop of one target instance. ' if it['dataset'] == 'fsc147' else ''
    if not crop:
        view = (pre + 'The next image is the scene; the image after it is the same scene with the draft objects drawn as numbered red '
                'marks. ')
        coords = frame(it, 'FULL IMAGE')
    else:
        view = (pre + 'The next four images are overlapping crops of the scene at native resolution: crop 1 top-left, crop 2 top-right, '
                'crop 3 bottom-left, crop 4 bottom-right, with the draft objects drawn as numbered red marks. ')
        coords = 'New objects have "crop" (1 to 4) and coordinates normalized to THAT CROP. ' + frame(it, 'crop')
    return ('You are auditing a draft answer to a visual set reporting task. Task: ' + L.task_text(it) + '\n\n' + view +
            'The draft is a hypothesis: it may contain wrong, duplicated or badly placed objects and it may miss objects. Do two things: '
            '(1) check the existing draft objects against the image; (2) scan the whole image for target objects missing from the draft, '
            'including areas with no red mark. No mark in a region is not evidence that the region is empty.\n\n' + coords +
            ' Draft objects in the text payload are normalized to the full image.\n\n' + RULES + '\n\nOutput contract: one JSON object '
            '{"edits": [...]}; every edit has "op" (one of add, delete, replace, split, merge), "ids" (a list, empty for add), "new" (a list of '
            'objects, empty for delete) and "evidence". Example of the schema only: ' + example(it, crop) + ' No prose, no code fence.')


def sys_verify(it, crop):
    pre = 'The first image is the reference crop of one target instance. ' if it['dataset'] == 'fsc147' else ''
    view = ('The next image is the scene; the image after it shows each proposed change k: red marks labelled k- are draft objects the '
            'change removes or corrects; green marks labelled k+ are objects it would add. ') if not crop else \
           ('The next four images are overlapping crops of the scene (crop 1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right) showing '
            'each proposed change k: red marks labelled k- are draft objects the change removes or corrects; green marks labelled k+ are '
            'objects it would add. ')
    return ('You are checking proposed changes to a draft answer of a visual set reporting task. Task: ' + L.task_text(it) + '\n\n' + pre +
            view + 'For EACH change decide "apply" if it is visually justified and makes the answer more faithful to the image, "keep" to '
            'retain the draft, or "uncertain" if the image does not let you decide. Do not propose new changes.\n\nOutput contract: one '
            'JSON object {"decisions": [{"change": 1, "decision": "apply"}, ...]}. No prose, no code fence.')


def rec_from_obj(it, o, c=None):
    """model object (normalized to full image or crop c) -> E3 record dict, or None."""
    W, H = it['width'], it['height']; f = [0]
    if it['task'] == 'point':
        p = G.norm_pt(o, f) if all(G._num(o.get(k)) and 0 <= o.get(k) <= 1 for k in ('x', 'y')) else None
        if p is None: return None
        if c: p = [c[0] + p[0] * (c[2] - c[0]), c[1] + p[1] * (c[3] - c[1])]
        return dict(cx=p[0] * W, cy=p[1] * H, label=it['labels'][0], box=None)
    v = [o.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if not all(G._num(t) and 0 <= t <= 1 for t in v) or not (v[0] < v[2] and v[1] < v[3]): return None
    b = G.crop_to_full(v, c) if c else v
    lab = o.get('label') if L.multi_label(it) else it['labels'][0]
    if lab not in it['labels']: return None
    B = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
    return dict(cx=(B[0] + B[2]) / 2, cy=(B[1] + B[3]) / 2, label=lab, box=B)


def decode_edits(it, raw, n, crop):
    v = G.first_json(raw); au = dict(json_ok=int(isinstance(v, dict)), proposed=0, invalid=0, deferred_cap=0); out = []
    for e in (v.get('edits') or []) if isinstance(v, dict) and isinstance(v.get('edits'), list) else []:
        au['proposed'] += 1
        if not isinstance(e, dict): au['invalid'] += 1; continue
        op, ids, nl, len_ = G.normalize_edit(e); au['lenient'] = au.get('lenient', 0) + len_
        new = []; bad = 0
        for o in nl:
            if not isinstance(o, dict): bad += 1; continue
            c = None
            if crop:
                if o.get('crop') not in (1, 2, 3, 4): bad += 1; continue
                c = G.CROPS[o['crop'] - 1]
            r = rec_from_obj(it, o, c)
            if r is None: bad += 1
            else: new.append(r)
        ok = (op == 'add' and not ids and new) or (op == 'delete' and ids and not new) or (op == 'replace' and len(ids) == 1 and len(new) == 1) or \
             (op == 'split' and len(ids) == 1 and len(new) >= 2) or (op == 'merge' and len(ids) >= 2 and len(new) == 1)
        ok = bool(ok) and bad == 0 and all(1 <= i <= n for i in ids) and len(set(ids)) == len(ids)
        if not ok: au['invalid'] += 1; continue
        if len(out) >= 32: au['deferred_cap'] += 1; continue
        out.append(dict(op=op, ids=ids, new=new))
    return out, au


def payload(it, recs):
    return [dict(id=i + 1, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in to_norm(it, r).items()}) for i, r in enumerate(recs)]


def stage_b():
    items, sel = pick(); D = drafts(); R = []
    for m in MODELS:
        for it in sel:
            recs = D.get((it['id'], m, 0))
            if recs is None: continue
            im = G.load(os.path.join(E3, it['image_path'])); ref = [os.path.join(E3, it['reference_path'])] if it['dataset'] == 'fsc147' else []
            tags = [(nbox(it, r), str(i + 1), (220, 0, 0)) for i, r in enumerate(recs)]
            base = f'data/views_e3/{m}/{it["id"]}'
            ov = G.save(G.fit(G.draw_boxes(im, tags)), f'/home/claude/gui/{base}/overlay_full.png')
            user = ('Draft objects (ids match the numbered marks; normalized to the full image):\n' + json.dumps(payload(it, recs)) +
                    '\n\nReturn the edits. Follow the output contract exactly.')
            R.append(dict(request_id=f"{it['id']}|{m}|A3audit", model_key=m, setting='min', images=ref + [os.path.join(E3, it['image_path']), ov],
                          system=sys_audit(it, False), user=user, meta=dict(item=it['id'], arm='A3audit', n_draft=len(recs))))
            cr = [G.save(G.draw_boxes(G.crop_norm(im, c), tags, frame=c), f'/home/claude/gui/{base}/overlay_crop{k + 1}.png') for k, c in enumerate(G.CROPS)]
            R.append(dict(request_id=f"{it['id']}|{m}|A4audit", model_key=m, setting='min', images=ref + cr, system=sys_audit(it, True), user=user,
                          meta=dict(item=it['id'], arm='A4audit', n_draft=len(recs))))
    return R


def stage_c():
    items, sel = pick(); D = drafts(); R = []
    B = {}
    for m in MODELS:
        f = f'{OUT}/responses_B_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': B[r['request_id']] = r
    for m in MODELS:
        for it in sel:
            recs = D.get((it['id'], m, 0))
            if recs is None: continue
            im = G.load(os.path.join(E3, it['image_path'])); ref = [os.path.join(E3, it['reference_path'])] if it['dataset'] == 'fsc147' else []
            for arm, crop in (('A3', False), ('A4', True)):
                rb = B.get(f"{it['id']}|{m}|{arm}audit")
                if rb is None: continue
                edits, _ = decode_edits(it, rb['raw_text'], len(recs), crop)
                if not edits: continue
                tags = []
                for k, e in enumerate(edits, 1):
                    tags += [(nbox(it, recs[i - 1]), f'{k}-', (220, 0, 0)) for i in e['ids']] + [(nbox(it, r), f'{k}+', (0, 160, 0)) for r in e['new']]
                base = f'data/views_e3/{m}/{it["id"]}'
                if not crop:
                    ims = ref + [os.path.join(E3, it['image_path']), G.save(G.fit(G.draw_boxes(im, tags)), f'/home/claude/gui/{base}/changes_full.png')]
                else:
                    ims = ref + [G.save(G.draw_boxes(G.crop_norm(im, c), tags, frame=c), f'/home/claude/gui/{base}/changes_crop{k + 1}.png') for k, c in enumerate(G.CROPS)]
                ch = [dict(change=k, op=e['op'], removes=[dict(id=i, **{a: (round(b, 4) if isinstance(b, float) else b) for a, b in to_norm(it, recs[i - 1]).items()}) for i in e['ids']],
                           adds=[{a: (round(b, 4) if isinstance(b, float) else b) for a, b in to_norm(it, r).items()} for r in e['new']]) for k, e in enumerate(edits, 1)]
                user = 'Proposed changes (normalized to the full image; they match the k-/k+ marks):\n' + json.dumps(ch) + '\n\nReturn one decision per change.'
                R.append(dict(request_id=f"{it['id']}|{m}|{arm}verify", model_key=m, setting='min', images=ims, system=sys_verify(it, crop), user=user,
                              meta=dict(item=it['id'], arm=f'{arm}verify', n_changes=len(edits))))
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); a = ap.parse_args()
    os.chdir(E3)
    R = stage_b() if a.stage == 'B' else stage_c()
    with open(f'{OUT}/requests_{a.stage}.jsonl', 'w') as f:
        for r in R: f.write(json.dumps(r) + '\n')
    print(a.stage, len(R), collections.Counter((r['model_key'], r['meta']['arm']) for r in R))

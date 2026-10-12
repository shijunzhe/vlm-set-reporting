"""Build request manifests for the GUI experiment (no model calls).
  python -I code/build_gui.py --stage A --subset pilot|main
Stage A: parsing drafts D1, D2, D3 (full view) and tiles T1, T2 (four crops); click C1, C2 (full view).
Stage B (needs A): parsing audits — A3 (full view + overlay of D1), A4 (four crops with D1 drawn); click B3 verification
         (crop around C1 with a marker) and B4 zoom (crop around C1).
Stage C (needs B): parsing verification of the proposed changes (A3: full view + change overlay; A4: four crops + change overlay).
All requests use the 'min' reasoning setting. Coordinate contract per model: 'grid' (0-1000) for gpt61 and gem38, 'pixel' for
sonnet55 with images pre-resized so that the provider does not resize them again. Each request stores the pixel sizes of its
images so that decoding is exact. Writes runs/<subset>/requests_<stage>.jsonl.
"""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrgui as G
ROOT = G.ROOT
MODELS = ['gpt61', 'sonnet55', 'gem38']
MODE = {'gpt61': 'grid', 'gem38': 'grid', 'sonnet55': 'pixel'}
os.chdir(ROOT)
_src = {}


def src(path):
    if path not in _src:
        if len(_src) > 8: _src.clear()
        _src[path] = G.load(path)
    return _src[path]


def fitm(im, m): return G.fit_anthropic(im) if MODE[m] == 'pixel' else G.fit(im)


def view(m, path, key, crop=None, draw=None, marker=None):
    """Render (or reuse) the model-facing image; returns (relative path, (W, H))."""
    suf = 'a' if MODE[m] == 'pixel' else 'g'
    out = f'data/views2/{key}_{suf}.png'
    if os.path.exists(out) or os.path.exists(out[:-4] + '.jpg'):
        p = out if os.path.exists(out) else out[:-4] + '.jpg'
        from PIL import Image
        with Image.open(p) as im: return p, im.size
    im = src(path); frame = (0, 0, 1, 1)
    if crop is not None: im = G.crop_native(im, crop); frame = crop
    im = fitm(im, m)
    if draw: im = G.draw_boxes(im, draw, frame=frame)
    if marker: im = G.draw_marker(im, marker[0], marker[1], frame=frame)
    return G.save(im, out), im.size


def subset_items(subset):
    P = json.load(open('data/parse_items.json')); C = json.load(open('data/click_items_prepped.json'))
    if subset == 'pilot':
        Pp = [p for g in sorted({p['group'] for p in P}) for p in [x for x in P if x['group'] == g][:1]]
        Cp = [c for c in C if c['source'] == 'sspro'][::20][:10] + [c for c in C if c['source'] == 'ssv2'][:2]
        return Pp, Cp
    return P, C


def load_resp(subset, stage):
    out = {}
    for m in MODELS:
        f = f'runs/{subset}/responses_{stage}_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': out[r['request_id']] = r
    return out


def load_req(subset, stage):
    f = f'runs/{subset}/requests_{stage}.jsonl'
    return {json.loads(l)['request_id']: json.loads(l) for l in open(f)} if os.path.exists(f) else {}


def req(rid, m, ims, system, user, sizes, **meta):
    return dict(request_id=rid, model_key=m, setting='min', images=ims, system=system, user=user, meta=dict(meta, mode=MODE[m], sizes=sizes))


def scale_of(m, size): return None if MODE[m] == 'grid' else tuple(size)


def full_draft(subset, m, it, A, Areq):
    rid = f"{it['id']}|{m}|D1"; r = A.get(rid)
    if r is None: return None, None
    fs = Areq[rid]['meta']['sizes'][0]
    return G.decode_list(r['raw_text'], scale_of(m, fs))[0], fs


def stage_a(P, C):
    R = []
    for m in MODELS:
        Pm = G.prompts(MODE[m])
        for it in P:
            fv, fs = view(m, it['src_image'], f"{it['id']}/full")
            for k in (1, 2, 3): R.append(req(f"{it['id']}|{m}|D{k}", m, [fv], Pm['draft'], G.user_draft(MODE[m], fs), [fs], item=it['id'], arm='draft', rep=k))
            cv = [view(m, it['src_image'], f"{it['id']}/crop{k + 1}", crop=c) for k, c in enumerate(G.CROPS)]
            for k in (1, 2): R.append(req(f"{it['id']}|{m}|T{k}", m, [v[0] for v in cv], Pm['tiles'], G.user_tiles(MODE[m], [v[1] for v in cv]), [v[1] for v in cv], item=it['id'], arm='tiles', rep=k))
        for c in C:
            fv, fs = view(m, c['image'], f"{c['id']}/full")
            for k in (1, 2): R.append(req(f"{c['id']}|{m}|C{k}", m, [fv], Pm['click'], G.user_click(c['instruction'], c['platform'], MODE[m], fs), [fs], item=c['id'], arm='click', rep=k))
    return R


def stage_b(P, C, subset):
    A = load_resp(subset, 'A'); Areq = load_req(subset, 'A'); R = []; miss = 0
    for m in MODELS:
        Pm = G.prompts(MODE[m])
        for it in P:
            draft, fs = full_draft(subset, m, it, A, Areq)
            if draft is None: miss += 1; continue
            tags = [(b, str(i + 1), (220, 0, 0)) for i, (b, n) in enumerate(draft)]
            fv, _ = view(m, it['src_image'], f"{it['id']}/full")
            ov, _ = view(m, it['src_image'], f"{subset}/{m}/{it['id']}/overlay_full", draw=tags)
            R.append(req(f"{it['id']}|{m}|A3audit", m, [fv, ov], Pm['audit_full'], G.user_audit(draft, MODE[m], fs), [fs, fs], item=it['id'], arm='A3audit', n_draft=len(draft)))
            cv = [view(m, it['src_image'], f"{subset}/{m}/{it['id']}/overlay_crop{k + 1}", crop=c, draw=tags) for k, c in enumerate(G.CROPS)]
            R.append(req(f"{it['id']}|{m}|A4audit", m, [v[0] for v in cv], Pm['audit_crop'], G.user_audit(draft, MODE[m], fs, [v[1] for v in cv]),
                         [fs] + [v[1] for v in cv], item=it['id'], arm='A4audit', n_draft=len(draft)))
        for c in C:
            rid = f"{c['id']}|{m}|C1"; r = A.get(rid)
            if r is None: miss += 1; continue
            p, _ = G.decode_click(r['raw_text'], scale_of(m, Areq[rid]['meta']['sizes'][0]))
            if p is None: continue                                    # no click proposed: nothing to verify or zoom around
            w3 = G.window(p[0], p[1], 0.2)
            f3, s3 = view(m, c['image'], f"{subset}/{m}/{c['id']}/verify", crop=w3, marker=p)
            R.append(req(f"{c['id']}|{m}|B3", m, [f3], Pm['vclick'], G.size_note(MODE[m], [s3], ['crop']) + f'Instruction: "{c["instruction"]}". Is the marked click correct?',
                         [s3], item=c['id'], arm='B3', window=w3, click=p))
            w4 = G.window(p[0], p[1], 1 / 3)
            f4, s4 = view(m, c['image'], f"{subset}/{m}/{c['id']}/zoom", crop=w4)
            R.append(req(f"{c['id']}|{m}|B4", m, [f4], Pm['zoom'], G.size_note(MODE[m], [s4], ['crop']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the click point in this crop.',
                         [s4], item=c['id'], arm='B4', window=w4, click=p))
    print('stage B missing upstream', miss)
    return R


def stage_c(P, subset):
    A = load_resp(subset, 'A'); Areq = load_req(subset, 'A'); B = load_resp(subset, 'B'); Breq = load_req(subset, 'B'); R = []
    for m in MODELS:
        Pm = G.prompts(MODE[m])
        for it in P:
            draft, fs = full_draft(subset, m, it, A, Areq)
            if draft is None: continue
            for arm, crop_mode in (('A3', False), ('A4', True)):
                rid = f"{it['id']}|{m}|{arm}audit"; rb = B.get(rid)
                if rb is None: continue
                sz = Breq[rid]['meta']['sizes']
                edits, _ = G.decode_edits(rb['raw_text'], len(draft), crop_mode, scale_of(m, sz[0]), [scale_of(m, s) for s in sz[1:]] if crop_mode else None)
                if not edits: continue                                 # nothing to verify: stop after the audit
                ch = G.changes_for_verify(edits, draft); tags = []
                for k, x in enumerate(ch, 1):
                    tags += [(b, f'{k}-', (220, 0, 0)) for i, b in x['old']] + [(b, f'{k}+', (0, 160, 0)) for b, n in x['new']]
                if not crop_mode:
                    fv, _ = view(m, it['src_image'], f"{it['id']}/full")
                    ov, _ = view(m, it['src_image'], f"{subset}/{m}/{it['id']}/changes_full", draw=tags)
                    ims, sizes, sysm = [fv, ov], [fs, fs], Pm['verify_full']
                else:
                    cv = [view(m, it['src_image'], f"{subset}/{m}/{it['id']}/changes_crop{k + 1}", crop=c, draw=tags) for k, c in enumerate(G.CROPS)]
                    ims, sizes, sysm = [v[0] for v in cv], [v[1] for v in cv], Pm['verify_crop']
                R.append(req(f"{it['id']}|{m}|{arm}verify", m, ims, sysm, G.user_verify(ch, MODE[m], fs), sizes, item=it['id'], arm=f'{arm}verify',
                             n_changes=len(ch), edits=edits))
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--subset', default='pilot'); a = ap.parse_args()
    P, C = subset_items(a.subset); os.makedirs(f'runs/{a.subset}', exist_ok=True)
    R = stage_a(P, C) if a.stage == 'A' else (stage_b(P, C, a.subset) if a.stage == 'B' else stage_c(P, a.subset))
    with open(f'runs/{a.subset}/requests_{a.stage}.jsonl', 'w') as f:
        for r in R: f.write(json.dumps(r) + '\n')
    print(a.subset, a.stage, len(R), collections.Counter((r['model_key'], r['meta']['arm']) for r in R))

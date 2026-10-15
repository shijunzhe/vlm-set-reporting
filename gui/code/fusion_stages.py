"""Request builders for the fusion confirmation and the two new sensors (no model calls).

Subsets
  main  the 260 click items of the alarm analysis (pilot). New stages: R, RS, V.
  conf  600 new ScreenSpot-Pro items (data/conf_items.json, code/sample_conf.py). Stages: A (first click C1), B (zoom B4 around
        C1), S (every zoomed candidate scored by every judge), R, RS, V.

Zoomed point of model k (as in stageB/analysis/autoselect.py): the B4 point mapped back to the screenshot; the first click when
the zoom returned no point; None when the first click returned no point.

Hypotheses: the zoomed points of the three models clustered greedily (models in the order gpt61, sonnet55, gem38; a point joins
the first cluster with a member within DELTA = 1.5% of the longer side). Centre of a hypothesis: the mean of its points.

Stage R (hypothesis-targeted re-grounding): on items with >= 2 hypotheses, each model zooms around every hypothesis that does not
contain its own zoomed point. Same crop size (1/3), prompt and user text as B4; only the crop centre differs.
Stage RS: every re-grounded point is scored by every judge with the stage-S prompt (same crop 0.2 and marker).
Stage V (same-element verifier, Gemini 3.8): for every pair of hypotheses, three images (downscaled full screenshot with marker A
red and marker B blue; native crop 0.2 around A with A marked; the same around B with B marked), in both orders (A/B swapped).

Usage: python3 -I code/fusion_stages.py --subset main|conf --stage A|B|S|R|RS|V   (writes runs/<subset>/requests_<stage>.jsonl)
"""
import argparse, collections, itertools, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir('/home/claude/gui')
import build_gui as B
import fsrgui as G
from PIL import ImageDraw

M = B.MODELS; DELTA = 0.015; VERIFIER = 'gem38'
SYSTEM_S = json.loads(open('runs/main/requests_S.jsonl').readline())['system']      # identical judge prompt to stage S
SYSTEM_V = ('You are checking two candidate clicks proposed by computer-use agents for the same instruction. Image 1 is the full '
            'screenshot, downscaled, with two labelled markers: A (red) and B (blue). Image 2 is a zoomed crop of the screenshot at native '
            'resolution around marker A (red circle with a cross); image 3 is the same around marker B (blue circle with a cross). Decide '
            'which marker, if clicked at its center, would hit the UI element the instruction refers to. Do not propose another point.\n\n'
            'Output contract: one JSON object {"same_element": true or false, "choice": "A" or "B" or "both" or "neither", "p_A": a number '
            'from 0 to 1, "p_B": a number from 0 to 1}, where same_element says whether the two markers lie on the same UI element, and '
            'p_A and p_B are your probabilities that clicking at the center of marker A, or of marker B, hits the intended element. Give '
            'your best estimate even when unsure. No prose, no code fence.')
RED, BLUE = (255, 0, 0), (0, 90, 255)


def items_of(subset):
    if subset == 'main':
        return {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}
    p = 'data/conf_items_prepped.json'
    if not os.path.exists(p):
        C = json.load(open('data/conf_items.json'))
        for c in C:
            im = G.load(c['image']); W, H = im.size; c['width'], c['height'] = W, H
            b = c['bbox']; c['bbox_norm'] = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
        json.dump(C, open(p, 'w'), indent=1)
    return {c['id']: c for c in json.load(open(p))}


def load(subset, stage):
    return B.load_req(subset, stage), B.load_resp(subset, stage)


def first_click(subset, A, Areq, cid, m):
    k = f'{cid}|{m}|C1'
    if k not in A: return 'missing'
    return G.decode_click(A[k]['raw_text'], B.scale_of(m, Areq[k]['meta']['sizes'][0]))[0]


def zoomed_points(subset, C):
    Areq, A = load(subset, 'A'); Breq, Bz = load(subset, 'B'); Z = {}
    for cid in C:
        for m in M:
            p1 = first_click(subset, A, Areq, cid, m)
            if p1 == 'missing': Z[(cid, m)] = 'missing'; continue
            k = f'{cid}|{m}|B4'; r = Bz.get(k)
            if r is None or p1 is None: Z[(cid, m)] = p1; continue
            q = Breq[k]; p, _ = G.decode_click(r['raw_text'], B.scale_of(m, q['meta']['sizes'][0])); w = q['meta']['window']
            Z[(cid, m)] = [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p1
    return Z


def dist(p, q, c):
    return math.hypot((p[0] - q[0]) * c['width'], (p[1] - q[1]) * c['height']) / max(c['width'], c['height'])


def hypotheses(c, pts):
    """pts: {model: point or None}; returns list of (members, centre)"""
    H = []
    for m in M:
        p = pts.get(m)
        if p is None: continue
        for h in H:
            if any(dist(p, q, c) <= DELTA for _, q in h): h.append((m, p)); break
        else: H.append([(m, p)])
    return [([m for m, _ in h], [sum(q[0] for _, q in h) / len(h), sum(q[1] for _, q in h) / len(h)]) for h in H]


def complete(Z, cid): return all(Z.get((cid, m)) != 'missing' for m in M)


def zoom_user(c, m, s): return G.size_note(B.MODE[m], [s], ['crop']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the click point in this crop.'
def judge_user(c, m, s): return G.size_note(B.MODE[m], [s], ['crop']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Is the marked click correct?'


def marker(im, x, y, frame, col, label=None, div=80):
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size; fx1, fy1, fx2, fy2 = frame
    px = (x - fx1) / (fx2 - fx1) * W; py = (y - fy1) / (fy2 - fy1) * H; r = max(8, round(max(W, H) / div)); lw = max(2, r // 4)
    d.ellipse([px - r, py - r, px + r, py + r], outline=col, width=lw)
    for a, b in (((px - 2 * r, py), (px - r // 2, py)), ((px + r // 2, py), (px + 2 * r, py)), ((px, py - 2 * r), (px, py - r // 2)), ((px, py + r // 2), (px, py + 2 * r))):
        d.line([a, b], fill=col, width=lw)
    if label:
        tw = d.textlength(label, font=G.FONT); tx = min(W - tw - 4, px + r + 2); ty = max(0, py - 2 * r - 16)
        d.rectangle([tx, ty, tx + tw + 4, ty + 16], fill=col); d.text((tx + 2, ty), label, fill=(255, 255, 255), font=G.FONT)
    return im


def render(m, c, key, crop=None, marks=()):
    """model-facing image with coloured markers; marks: [(point, colour, label)]"""
    suf = 'a' if B.MODE[m] == 'pixel' else 'g'; out = f'data/views2/{key}_{suf}.png'
    for p in (out, out[:-4] + '.jpg'):
        if os.path.exists(p):
            from PIL import Image
            with Image.open(p) as im: return p, im.size
    im = B.src(c['image']); frame = (0, 0, 1, 1)
    if crop is not None: im = G.crop_native(im, crop); frame = crop
    im = B.fitm(im, m)
    for p, col, lab in marks: im = marker(im, p[0], p[1], frame, col, lab, div=80 if crop is not None else 200)   # smaller markers on the full view
    return G.save(im, out), im.size


def stage_A(sub, C):
    R = []
    for m in M:
        Pm = G.prompts(B.MODE[m])
        for cid, c in C.items():
            fv, fs = B.view(m, c['image'], f'{cid}/full')
            R.append(B.req(f'{cid}|{m}|C1', m, [fv], Pm['click'], G.user_click(c['instruction'], c['platform'], B.MODE[m], fs), [fs], item=cid, arm='click', rep=1))
    return R


def stage_B(sub, C):
    Areq, A = load(sub, 'A'); R = []
    for m in M:
        Pm = G.prompts(B.MODE[m])
        for cid, c in C.items():
            p = first_click(sub, A, Areq, cid, m)
            if p in (None, 'missing'): continue
            w4 = G.window(p[0], p[1], 1 / 3); f4, s4 = B.view(m, c['image'], f'{sub}/{m}/{cid}/zoom', crop=w4)
            R.append(B.req(f'{cid}|{m}|B4', m, [f4], Pm['zoom'], zoom_user(c, m, s4), [s4], item=cid, arm='B4', window=w4, click=p))
    return R


def stage_S(sub, C):
    Z = zoomed_points(sub, C); R = []
    for cid, c in C.items():
        if not complete(Z, cid): continue
        for k in M:
            p = Z[(cid, k)]
            if p is None: continue
            w = G.window(p[0], p[1], 0.2)
            for j in M:
                f, s = B.view(j, c['image'], f'{sub}/{j}/{cid}/score_{k}', crop=w, marker=p)
                R.append(B.req(f'{cid}|{j}|S|{k}', j, [f], SYSTEM_S, judge_user(c, j, s), [s], item=cid, arm='S', judge=j, candidate_model=k, window=w, click=p))
    return R


def multi(sub, C):
    Z = zoomed_points(sub, C); out = {}
    for cid, c in C.items():
        if not complete(Z, cid): continue
        H = hypotheses(c, {m: Z[(cid, m)] for m in M})
        if len(H) >= 2: out[cid] = H
    return out


def stage_R(sub, C):
    R = []
    for cid, H in multi(sub, C).items():
        c = C[cid]
        for m in M:
            Pm = G.prompts(B.MODE[m])
            for i, (mem, ctr) in enumerate(H):
                if m in mem: continue
                w = G.window(ctr[0], ctr[1], 1 / 3); f, s = B.view(m, c['image'], f'{sub}/{m}/{cid}/reground_h{i}', crop=w)
                R.append(B.req(f'{cid}|{m}|R|h{i}', m, [f], Pm['zoom'], zoom_user(c, m, s), [s], item=cid, arm='R', hyp=i, members=mem, centre=ctr, window=w))
    return R


def reground_points(sub):
    Rq, Rr = load(sub, 'R'); out = {}
    for rid, r in Rr.items():
        q = Rq[rid]; m = q['model_key']; p, _ = G.decode_click(r['raw_text'], B.scale_of(m, q['meta']['sizes'][0])); w = q['meta']['window']
        out[rid] = [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else None   # None: not in crop
    return out


def stage_RS(sub, C):
    R = []
    for rid, p in sorted(reground_points(sub).items()):
        if p is None: continue
        cid, k, _, h = rid.split('|'); c = C[cid]; w = G.window(p[0], p[1], 0.2)
        for j in M:
            f, s = B.view(j, c['image'], f'{sub}/{j}/{cid}/score_R_{k}_{h}', crop=w, marker=p)
            R.append(B.req(f'{cid}|{j}|RS|{k}|{h}', j, [f], SYSTEM_S, judge_user(c, j, s), [s], item=cid, arm='RS', judge=j, candidate_model=k, hyp=h, window=w, click=p))
    return R


def stage_V(sub, C):
    R = []; m = VERIFIER
    for cid, H in multi(sub, C).items():
        c = C[cid]
        for i, j in itertools.combinations(range(len(H)), 2):
            for order, (a, b) in (('ab', (i, j)), ('ba', (j, i))):
                pa, pb = H[a][1], H[b][1]
                fv, fs = render(m, c, f'{sub}/{m}/{cid}/verify_full_h{a}h{b}', marks=((pa, RED, 'A'), (pb, BLUE, 'B')))
                wa, wb = G.window(pa[0], pa[1], 0.2), G.window(pb[0], pb[1], 0.2)
                ca, sa = render(m, c, f'{sub}/{m}/{cid}/verify_crop_h{a}', crop=wa, marks=((pa, RED, None),))
                cb, sb = render(m, c, f'{sub}/{m}/{cid}/verify_cropB_h{b}', crop=wb, marks=((pb, BLUE, None),))
                user = G.size_note(B.MODE[m], [fs, sa, sb], ['image 1', 'image 2', 'image 3']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Which marker is correct?'
                R.append(B.req(f'{cid}|{m}|V|h{i}h{j}|{order}', m, [fv, ca, cb], SYSTEM_V, user, [fs, sa, sb], item=cid, arm='V', order=order,
                               hyp_A=a, hyp_B=b, point_A=pa, point_B=pb, window_A=wa, window_B=wb))
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); ap.add_argument('--stage', required=True); a = ap.parse_args()
    C = items_of(a.subset)
    if a.subset == 'main':                                   # the 260 items of the alarm analysis (those scored in stage S)
        keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
    os.makedirs(f'runs/{a.subset}', exist_ok=True)
    R = dict(A=stage_A, B=stage_B, S=stage_S, R=stage_R, RS=stage_RS, V=stage_V)[a.stage](a.subset, C)
    with open(f'runs/{a.subset}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
    print(a.subset, a.stage, 'requests', len(R), collections.Counter((q['model_key'], q['meta']['arm']) for q in R))

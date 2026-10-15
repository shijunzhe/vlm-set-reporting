"""Strong single-model baseline in the style of GUI-Lens (arXiv 2608.03270), 'Efficient' preset, on GPT-6.1 with the default
setting (subset main; no model calls here).
What follows GUI-Lens: the model itself chooses coarse-to-fine crop boxes; each crop gets 8% padding per side, is clipped to
the screen and shown enlarged (at most 5x for an intermediate view, 8x for the final view, long side <= 2048 px); at most two
crop rounds, then a click; clicks are mapped back through the crop's affine map.
What differs (recorded, not a full reproduction): no OCR / UI-detector coordinate priming, no visual verification step,
prompts are ours (the paper's prompts were not available to us), GPT-6.1 instead of GPT-5.5.
Stages: GL1 on the full view; GL2 on the crop chosen in GL1 (only if GL1 returned a crop); GL3 = final click on the crop chosen
in GL2 (only if GL2 returned a crop). A 'not in crop' or unreadable final reply falls back to the centre of the last region.
Variant GLF ("crop first"): round 1 must return a region (closer to coarse-to-fine as described in the paper); rounds 2 and 3 as GL.
Usage: python3 -I code/gl_stages.py --stage GL1|GL2|GL3|GLF1|GLF2|GLF3
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_stages as F
import aligned_stages as AL
B, G = F.B, F.G
SUB = 'main'; MOD = 'gpt61'; SET = 'default'; PAD = 0.08
SYSTEM_GLF1 = ('You are a computer-use agent locating the UI element that an instruction refers to. The image shows the full screenshot, '
               'downscaled. First narrow the search: choose a rectangular region of THIS image that surely contains the target element '
               'together with enough surrounding context to recognise it (typically between a tenth and a third of the image side). '
               + G.coord_sentence('grid', 'IMAGE') + '\n\nOutput contract: one JSON object {"action": "crop", "box": [x1, y1, x2, y2]}. '
               'No prose, no code fence.')
SYSTEM_GL = ('You are a computer-use agent locating the UI element that an instruction refers to. The image shows {part}. '
             'If the target is too small to point at precisely, choose a tighter rectangular region of THIS image that surely contains '
             'the target element together with enough surrounding context to recognise it. If you can already point at the target '
             'precisely, give the click point instead. ' + G.coord_sentence('grid', 'IMAGE') + '\n\nOutput contract: one JSON object, '
             'either {{"action": "crop", "box": [x1, y1, x2, y2]}} or {{"action": "click", "x": ..., "y": ...}}. No prose, no code fence.')


def pad_clip(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - PAD * w), max(0.0, b[1] - PAD * h), min(1.0, b[2] + PAD * w), min(1.0, b[3] + PAD * h)]


def render_region(c, key, region, maxzoom):
    out = f'data/views2/{key}_g.png'
    if not os.path.exists(out):
        im = B.src(c['image']); cr = G.crop_native(im, region)
        s = min(maxzoom, 2048 / max(cr.size))
        from PIL import Image
        cr = cr.resize((max(1, round(cr.width * s)), max(1, round(cr.height * s))), Image.LANCZOS)
        G.save(cr, out)
    from PIL import Image
    p = out if os.path.exists(out) else out[:-4] + '.jpg'
    with Image.open(p) as im: return p, im.size


def decode(raw, frame):
    """returns ('crop', region in screen frame) or ('click', point in screen frame) or (None, None)"""
    v = G.first_json(raw)
    if not isinstance(v, dict): return None, None
    fx = lambda x, y: [frame[0] + x / 1000 * (frame[2] - frame[0]), frame[1] + y / 1000 * (frame[3] - frame[1])]
    try:
        if v.get('action') == 'crop' and isinstance(v.get('box'), list) and len(v['box']) == 4:
            x1, y1, x2, y2 = [float(t) for t in v['box']]
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000): return None, None
            a, b = fx(x1, y1), fx(x2, y2); return 'crop', [a[0], a[1], b[0], b[1]]
        if 'x' in v and 'y' in v:
            x, y = float(v['x']), float(v['y'])
            if 0 <= x <= 1000 and 0 <= y <= 1000: return 'click', fx(x, y)
    except (TypeError, ValueError):
        return None, None
    return None, None


def req(rid, c, img, size, system, user, frame, rnd):
    q = B.req(rid, MOD, [img], system, user, [size], item=c['id'], arm='GL', round=rnd, frame=frame)
    q['setting'] = SET; return q


def stage(n, C, V='GL'):
    R = []
    prev_q, prev_r = (F.load(SUB, f'{V}{n - 1}') if n > 1 else ({}, {}))
    for cid, c in C.items():
        if n == 1:
            fv, fs = B.view(MOD, c['image'], f'{cid}/full')
            sys1 = SYSTEM_GLF1 if V == 'GLF' else SYSTEM_GL.format(part='the full screenshot, downscaled')
            R.append(req(f'{cid}|{MOD}|{V}1', c, fv, fs, sys1,
                         f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". ' + ('Return the region.' if V == 'GLF' else 'Crop or click?'), [0, 0, 1, 1], 1))
            continue
        k = f'{cid}|{MOD}|{V}{n - 1}'
        if k not in prev_r: continue
        act, val = decode(prev_r[k]['raw_text'], prev_q[k]['meta']['frame'])
        if act != 'crop': continue
        region = pad_clip(val); final = n == 3
        img, size = render_region(c, f'{SUB}/{MOD}/{cid}/{V.lower()}{n}', region, 8 if final else 5)
        if final:
            sysm = G.prompts('grid')['zoom']
            user = f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the click point in this crop.'
        else:
            sysm = SYSTEM_GL.format(part='an enlarged region of the screenshot')
            user = f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Crop or click?'
        R.append(req(f'{cid}|{MOD}|{V}{n}', c, img, size, sysm, user, region, n))
    return R


def final_point(cid, C, L, V='GL'):
    """final GL click for item cid from loaded stages L = {n: (Q, R)}; also returns the number of calls"""
    last_region = [0, 0, 1, 1]; calls = 0
    for n in (1, 2, 3):
        Q, R = L[n]; k = f'{cid}|{MOD}|{V}{n}'
        if k not in Q: break
        if k not in R: return None, calls
        calls += 1
        if n == 3:
            p, _ = G.decode_click(R[k]['raw_text'], None); fr = Q[k]['meta']['frame']
            if p is None: return [(fr[0] + fr[2]) / 2, (fr[1] + fr[3]) / 2], calls
            return [fr[0] + p[0] * (fr[2] - fr[0]), fr[1] + p[1] * (fr[3] - fr[1])], calls
        act, val = decode(R[k]['raw_text'], Q[k]['meta']['frame'])
        if act == 'click': return val, calls
        if act is None: return [(last_region[0] + last_region[2]) / 2, (last_region[1] + last_region[3]) / 2], calls
        last_region = pad_clip(val)
    return [(last_region[0] + last_region[2]) / 2, (last_region[1] + last_region[3]) / 2], calls


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); a = ap.parse_args(); C = AL.items_main()
    R = stage(int(a.stage[-1]), C, a.stage[:-1])
    with open(f'runs/{SUB}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
    print(a.stage, 'requests', len(R), collections.Counter((q['model_key'], q['setting']) for q in R))

"""FSR pilot v2: build manifest and execute (resumable). Keys from config.env, never printed.
Arms per scene: F (free list, 1 call), J (all 16 regions, 1 call), S1 (one named region per call, full image, 16),
C1 (one named region per call, only that region's pixels kept, 16). Requested regions are named explicitly
(v1's bit-mask addressing caused wrong-region answers). Output ceiling 16,384 tokens.
Qwen3-VL uses its native 0-1000 scene coordinates in every arm (secondary analysis only).
"""
import argparse, hashlib, json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).parent
import run_pilot as RP          # loads keys and the paper-2 client
RP.G.MAX_TOKENS = 16384
P = {k: (ROOT / 'prompts' / f'{k}.txt').read_text().strip() for k in ['v2_system_local', 'v2_system_qwen', 'v2_user', 'v2_free_qwen']}
P['free'] = (ROOT / 'prompts' / 'free_list.txt').read_text().strip()
SYS_FREE = RP.SYS['free']
PRE = RP.PRE
SEED = 20261009

def build(models):
    samples = [json.loads(s) for s in (ROOT / 'data_v2/samples.jsonl').read_text().splitlines() if s.strip()]
    rows = []
    for s in samples:
        local = int(hashlib.sha256(f'{SEED}:{s["sample_id"]}'.encode()).hexdigest()[:12], 16)
        order = [r['region_id'] for r in s['regions']]; random.Random(local).shuffle(order)
        roster = {r['region_id']: r for r in s['regions']}; reg = [roster[r] for r in order]
        for m in models:
            base = dict(sample_id=s['sample_id'], dose=s['dose'], model_key=m, reference_path=s['reference_path'])
            pay = lambda req: {'task': 'reference-conditioned point detection', 'image_width': s['width'], 'image_height': s['height'],
                               'coordinate_contract': 'scene_0_1000_xy' if m == 'qwen3vl' else 'region_local_uv_half_open_v1',
                               'ordered_registry': reg, 'requested_regions': req}
            rows.append(dict(base, request_id=f'F:{s["sample_id"]}:{m}', arm='F', requested=order, image_path=s['image_path'],
                             task_payload={'image_width': s['width'], 'image_height': s['height'], 'ordered_registry': reg}))
            rows.append(dict(base, request_id=f'J:{s["sample_id"]}:{m}', arm='J', requested=order, image_path=s['image_path'], task_payload=pay(order)))
            for rid in order:
                rows.append(dict(base, request_id=f'S1:{s["sample_id"]}:{m}:{rid}', arm='S1', requested=[rid], image_path=s['image_path'], task_payload=pay([rid])))
                rows.append(dict(base, request_id=f'C1:{s["sample_id"]}:{m}:{rid}', arm='C1', requested=[rid],
                                 image_path=f'data_v2/masked/{s["sample_id"]}_{rid}.png', task_payload=pay([rid])))
    random.Random(SEED).shuffle(rows)
    (ROOT / 'runs_v2').mkdir(exist_ok=True)
    (ROOT / 'runs_v2/requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows

def prompts(q):
    qw = q['model_key'] == 'qwen3vl'
    if q['arm'] == 'F':
        return SYS_FREE, PRE + (P['v2_free_qwen'] if qw else P['free']).replace('{registry_json}', json.dumps(q['task_payload']['ordered_registry']))
    return P['v2_system_qwen' if qw else 'v2_system_local'], PRE + P['v2_user'].replace('{task_payload_json}', json.dumps(q['task_payload']))

def call_qwen(images, system, user):
    import base64, io, openai, os
    with RP._orl:
        if RP._or is None:
            RP._or = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'], timeout=900)
    def b64(im):
        b = io.BytesIO(); im.save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(im)}} for im in images] + [{'type': 'text', 'text': user}]
    r = RP._or.chat.completions.create(model=RP.QWEN, max_tokens=16384, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                                       extra_body={'provider': {'order': ['Alibaba'], 'allow_fallbacks': False}})
    ch = r.choices[0]
    return (ch.message.content or '').strip(), {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
                                               'actual_model': r.model, 'stop_reason': ch.finish_reason}

def execute(q):
    ims = [Image.open(ROOT / q['reference_path']).convert('RGB'), Image.open(ROOT / q['image_path']).convert('RGB')]
    system, user = prompts(q); err = None
    for attempt in range(1, 5):
        try:
            raw, u = call_qwen(ims, system, user) if q['model_key'] == 'qwen3vl' else RP.G.call(q['model_key'], ims, system, user)
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw, 'provider_model_id': u.get('actual_model'),
                    'finish_reason': u.get('stop_reason'), 'usage': u, 'attempts': attempt, 'time': time.time()}
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--model')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    a = ap.parse_args()
    if a.build:
        rows = build(['gpt56', 'sonnet55', 'opus55', 'gpt61', 'qwen3vl']); print(len(rows)); return
    reqs = [json.loads(s) for s in (ROOT / 'runs_v2/requests.jsonl').read_text().splitlines() if s.strip()]
    reqs = [q for q in reqs if q['model_key'] == a.model and (not a.only or q['request_id'].startswith(a.only))]
    out = ROOT / 'runs_v2' / f'responses_{a.model}.jsonl'; done = set()
    if out.exists():
        done = {json.loads(l)['request_id'] for l in out.read_text().splitlines() if json.loads(l)['status'] == 'completed'}
    todo = [q for q in reqs if q['request_id'] not in done]
    if a.limit: todo = todo[:a.limit]
    print(a.model, 'todo', len(todo), 'done', len(done), flush=True)
    lock = threading.Lock(); n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, 'a') as f:
        for fut in as_completed([ex.submit(execute, q) for q in todo]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + '\n'); f.flush(); n += 1
                if n % 50 == 0: print(a.model, n, flush=True)
    print(a.model, 'finished', n, flush=True)

if __name__ == '__main__':
    main()

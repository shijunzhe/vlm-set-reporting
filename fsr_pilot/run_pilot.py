"""Execute FSR pilot requests. Resumable; keys are read from config.env and never printed.
Images are sent as [reference, scene]; a fixed one-line preamble names them identically in every arm.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).parent
sys.path.insert(0, '/home/claude/stok')

def load_env(p='/home/claude/x2/config.env'):
    for line in open(p):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); k = k.strip(); v = v.strip().strip('"').strip("'")
            if v and not os.environ.get(k): os.environ[k] = v
load_env()
import gen_common as G   # paper-2 client: sonnet55, opus55, gpt56, gpt61 (no reasoning / lowest effort)

PRE = 'The first image is the reference symbol; the second image is the full scene.\n\n'
SYS = {k: (ROOT / 'prompts' / f).read_text().strip() for k, f in
       [('regional_local', 'system_regional.txt'), ('regional_global', 'system_regional_global.txt')]}
SYS['free'] = ('You are performing reference-conditioned visual set reporting. There may be zero, one, or several '
               'matching instances anywhere in the scene. Use any visible context needed to identify the requested target.')
USER_REG = (ROOT / 'prompts' / 'user_regional.txt').read_text().strip()
USER_FREE = (ROOT / 'prompts' / 'free_list.txt').read_text().strip()
QWEN = 'qwen/qwen3-vl-235b-a22b-instruct'

def prompts(q):
    if q['prompt'] == 'free':
        return SYS['free'], PRE + USER_FREE.replace('{registry_json}', json.dumps(q['task_payload']['ordered_registry']))
    return SYS[q['prompt']], PRE + USER_REG.replace('{task_payload_json}', json.dumps(q['task_payload']))

_or = None; _orl = threading.Lock()
def call_qwen(images, system, user):
    global _or
    import base64, io, openai
    with _orl:
        if _or is None:
            _or = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'], timeout=600)
    def b64(im):
        b = io.BytesIO(); im.save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(im)}} for im in images] + [{'type': 'text', 'text': user}]
    r = _or.chat.completions.create(model=QWEN, max_tokens=8192, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                                    extra_body={'provider': {'order': ['Alibaba'], 'allow_fallbacks': False}})
    ch = r.choices[0]
    return (ch.message.content or '').strip(), {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
                                               'actual_model': r.model, 'stop_reason': ch.finish_reason, 'provider': getattr(r, 'provider', None)}

def execute(q):
    ims = [Image.open(ROOT / q['reference_path']).convert('RGB'), Image.open(ROOT / q['image_path']).convert('RGB')]
    system, user = prompts(q)
    err = None
    for attempt in range(1, 5):
        try:
            if q['model_key'] == 'qwen3vl': raw, u = call_qwen(ims, system, user)
            else: raw, u = G.call(q['model_key'], ims, system, user)
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw,
                    'provider_model_id': u.get('actual_model'), 'finish_reason': u.get('stop_reason'),
                    'usage': u, 'attempts': attempt, 'time': time.time()}
        except Exception as e:  # transport/provider errors: bounded retries, then an explicit failed record
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--requests', default='runs/pilot_requests.jsonl')
    ap.add_argument('--model', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    a = ap.parse_args()
    reqs = [json.loads(s) for s in (ROOT / a.requests).read_text().splitlines() if s.strip()]
    reqs = [q for q in reqs if q['model_key'] == a.model and (not a.only or q['request_id'].startswith(a.only))]
    out = ROOT / 'runs' / f'responses_{a.model}.jsonl'
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            x = json.loads(l)
            if x['status'] == 'completed': done.add(x['request_id'])
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

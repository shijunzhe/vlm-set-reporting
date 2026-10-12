"""Runner for FSR stage B. Uses the frozen API calls of run_e12b (gpt61, sonnet55, gem38). Nothing is sent without
--execute. Keys come from config.env via run_pilot.load_env and are never printed. Resumable; technical retries only.
  python code/run_b.py --subset b1 --stage main --model gem38 [--execute]
"""
import argparse, json, os, sys, threading, time, hashlib, platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
sys.path.insert(0, '/home/claude/fsr_pilot')
import run_e12b as RB
ROOT = '/home/claude/stageB'


def call_qwen(ims, system, user, max_tokens=16384):
    """Qwen3-VL-235B-A22B instruct via OpenRouter, provider pinned to Alibaba (as in the ARR runs)."""
    import openai, base64, io
    cl = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'], timeout=600)
    content = []
    for im in ims:
        b = io.BytesIO(); im.save(b, format='PNG'); content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()}})
    content.append({'type': 'text', 'text': user})
    r = cl.chat.completions.create(model='qwen/qwen3-vl-235b-a22b-instruct', max_tokens=max_tokens, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                                   extra_body={'provider': {'sort': 'throughput'}})
    ch = r.choices[0]
    return (ch.message.content or '').strip(), {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens, 'actual_model': r.model, 'stop_reason': ch.finish_reason, 'provider': getattr(r, 'provider', None)}


def execute(q):
    try:
        ims = [Image.open(os.path.join(ROOT, p)).convert('RGB') for p in q['images']]
    except Exception as e:
        return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': f'image: {e}', 'attempts': 0, 'time': time.time(), 'meta': q['meta']}
    err = None
    for attempt in range(1, 5):
        try:
            raw, u = (call_qwen(ims, q['system'], q['user']) if q['model_key'] == 'qwen3vl' else RB.call(q['model_key'], q['setting'], ims, q['system'], q['user']))
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw, 'finish_reason': u.get('stop_reason'),
                    'usage': u, 'attempts': attempt, 'time': time.time(), 'meta': q['meta']}
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time(), 'meta': q['meta']}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--stage', required=True)
    ap.add_argument('--subset', default='pilot'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    man = os.path.join(ROOT, f'runs/{a.subset}/requests_{a.stage}.jsonl')
    reqs = [json.loads(l) for l in open(man) if l.strip()]; reqs = [q for q in reqs if q['model_key'] == a.model]
    out = os.path.join(ROOT, f'runs/{a.subset}/responses_{a.stage}_{a.model}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['request_id'] for l in open(out) if l.strip() and json.loads(l)['status'] == 'completed'}
    todo = [q for q in reqs if q['request_id'] not in done]
    print(a.model, a.stage, 'manifest', len(reqs), 'done', len(done), 'todo', len(todo), flush=True)
    if not a.execute:
        print('dry run: nothing sent'); return
    sys.argv = sys.argv[:1]
    import run_pilot                                   # loads keys (never printed)
    reg = dict(RB.REGISTRY.get(a.model, {'provider': 'openrouter', 'model': 'qwen/qwen3-vl-235b-a22b-instruct', 'pin': 'none (Alibaba endpoint no longer offered; OpenRouter routing by throughput, provider logged per call)'}), max_tokens=RB.MAX_TOKENS, python=platform.python_version(), started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               manifest=os.path.basename(man), manifest_sha256=hashlib.sha256(open(man, 'rb').read()).hexdigest())
    with open(os.path.join(ROOT, f'runs/{a.subset}/registry_{a.model}.jsonl'), 'a') as f: f.write(json.dumps(reg) + '\n')
    lock = threading.Lock(); n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, 'a') as f:
        for fut in as_completed([ex.submit(execute, q) for q in todo]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + '\n'); f.flush(); n += 1
                if n % 50 == 0: print(a.model, a.stage, n, flush=True)
    print(a.model, a.stage, 'finished', n, flush=True)


if __name__ == '__main__':
    main()

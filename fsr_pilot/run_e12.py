"""Runner for v5-E12 (workspace only; the FSR package itself has no provider client).
Reads the frozen manifest built by fsr_v51/e12/build_e12.py and executes it resumably.
Nothing is sent unless --execute is given. Keys are loaded by run_pilot from config.env and never printed.
  python run_e12.py --model gpt56                # dry run: counts only
  python run_e12.py --model gpt56 --execute      # send requests
"""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

E12 = Path('/home/claude/fsr_v51/e12')
MAN = E12 / 'runs_e12/requests.jsonl'


# Frozen per-model configuration (registered in runs_e12/model_registry.json at execution time).
# Principle: the lowest reasoning/thinking setting each API accepts.
REGISTRY = {
    'gpt56': {'provider': 'openai', 'model': 'gpt-5.6-sol', 'image_detail': 'auto (= original for GPT-5.5+)', 'reasoning_effort': 'none'},
    'gpt61': {'provider': 'openai', 'model': 'gpt-6.1-sol', 'image_detail': 'auto', 'reasoning_effort': 'low (lowest accepted)'},
    'opus55': {'provider': 'anthropic', 'model': 'claude-opus-5-5', 'thinking': 'adaptive (always on; cannot be disabled)',
               'output_config.effort': 'low', 'display': 'omitted (default)'},
    'sonnet55': {'provider': 'anthropic', 'model': 'claude-sonnet-5-5', 'thinking': 'between_tools (no up-front thinking)',
                 'output_config.effort': 'default (high)'},
}
MAX_TOKENS = 16384


def call_anthropic(key, ims, system, user, RP):
    import base64, io
    def png(im): b = io.BytesIO(); im.save(b, format='PNG'); return base64.standard_b64encode(b.getvalue()).decode()
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': png(im)}} for im in ims] + [{'type': 'text', 'text': user}]
    kw = {'extra_body': {'output_config': {'effort': 'low'}}} if key == 'opus55' else {'thinking': {'type': 'between_tools'}}
    r = RP.G._client('anthropic').messages.create(model=REGISTRY[key]['model'], max_tokens=MAX_TOKENS, system=system,
                                                  messages=[{'role': 'user', 'content': content}], **kw)
    raw = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text').strip()
    det = getattr(r.usage, 'output_tokens_details', None)
    return raw, {'input_tokens': r.usage.input_tokens, 'output_tokens': r.usage.output_tokens, 'actual_model': r.model,
                 'stop_reason': r.stop_reason, 'thinking_tokens': getattr(det, 'thinking_tokens', None),
                 'block_types': [b.type for b in r.content], 'message_id': r.id}


def execute(q, RP):
    ims = [Image.open(E12 / q['reference_path']).convert('RGB'), Image.open(E12 / q['image_path']).convert('RGB')]
    err = None
    for attempt in range(1, 5):                      # technical retries only; content is never retried
        try:
            if REGISTRY[q['model_key']]['provider'] == 'anthropic':
                raw, u = call_anthropic(q['model_key'], ims, q['system'], q['user'], RP)
            else:
                raw, u = RP.G.call(q['model_key'], ims, q['system'], q['user'])
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw, 'provider_model_id': u.get('actual_model'),
                    'finish_reason': u.get('stop_reason'), 'usage': u, 'attempts': attempt, 'time': time.time()}
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='first scene, repetition 0 only; written to runs_e12_smoke/ and never analysed with the main run')
    a = ap.parse_args()
    reqs = [json.loads(l) for l in MAN.read_text().splitlines() if l.strip()]
    reqs = [q for q in reqs if q['model_key'] == a.model]
    rdir = E12 / 'runs_e12'
    if a.smoke:
        allreq = [json.loads(l) for l in MAN.read_text().splitlines() if l.strip()]
        first = sorted({q['sample_id'] for q in allreq})[0]
        smoke_all = [q for q in allreq if q['sample_id'] == first and q['rep'] == 0]   # identical in every process
        reqs = [q for q in smoke_all if q['model_key'] == a.model]
        rdir = E12 / 'runs_e12_smoke'; rdir.mkdir(exist_ok=True)
        tmp = rdir / f'requests.{a.model}.tmp'
        tmp.write_text(''.join(json.dumps(q) + '\n' for q in smoke_all)); tmp.replace(rdir / 'requests.jsonl')
    out = rdir / f'responses_{a.model}.jsonl'
    done = set()
    if out.exists():
        done = {json.loads(l)['request_id'] for l in out.read_text().splitlines() if json.loads(l)['status'] == 'completed'}
    todo = [q for q in reqs if q['request_id'] not in done]
    if a.limit: todo = todo[:a.limit]
    print(a.model, 'manifest', len(reqs), 'done', len(done), 'todo', len(todo), flush=True)
    if not a.execute:
        print('dry run: nothing sent'); return
    import run_pilot as RP                           # loads keys and the paper-2 client
    RP.G.MAX_TOKENS = MAX_TOKENS
    import anthropic, openai, platform
    reg = dict(REGISTRY[a.model], max_tokens=MAX_TOKENS, sdk={'anthropic': anthropic.__version__, 'openai': openai.__version__},
               python=platform.python_version(), started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               manifest_sha256=__import__('hashlib').sha256(MAN.read_bytes()).hexdigest(),
               retry_rule='technical errors only, up to 4 attempts with exponential backoff; content never retried')
    rf = rdir / f'model_registry_{a.model}.json'                # one file per model: no cross-process race
    allreg = json.loads(rf.read_text()) if rf.exists() else []
    allreg.append(reg); rf.write_text(json.dumps(allreg, indent=1))
    lock = threading.Lock(); n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, 'a') as f:
        for fut in as_completed([ex.submit(execute, q, RP) for q in todo]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + '\n'); f.flush(); n += 1
                if n % 50 == 0: print(a.model, n, flush=True)
    print(a.model, 'finished', n, flush=True)


if __name__ == '__main__':
    main()

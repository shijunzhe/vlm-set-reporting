"""Runner for E3 (workspace only). Uses the per-model settings and API calls of run_e12b (gpt61, sonnet55, gem38; 'min' =
lowest reasoning setting each API accepts, 'default' = no reasoning parameter). Nothing is sent without --execute. Keys come
from config.env via run_pilot.load_env and are never printed. Resumable.
  python run_e3.py --model gem38 --manifest runs/requests_smoke.jsonl --out runs_smoke [--execute]
"""
import argparse, hashlib, json, platform, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/home/claude/fsr_pilot')
import run_e12b as RB

E3 = Path('/home/claude/fsr_v51/e3')


def execute(q):
    ims = [Image.open(E3 / p).convert('RGB') for p in q['images']]
    err = None
    for attempt in range(1, 5):                      # technical retries only; content is never retried
        try:
            raw, u = RB.call(q['model_key'], q['setting'], ims, q['system'], q['user'])
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw, 'provider_model_id': u.get('actual_model'),
                    'finish_reason': u.get('stop_reason'), 'usage': u, 'attempts': attempt, 'time': time.time()}
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    man = E3 / a.manifest; rdir = E3 / a.out; rdir.mkdir(parents=True, exist_ok=True)
    reqs = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    reqs = [q for q in reqs if q['model_key'] == a.model]
    out = rdir / f'responses_{a.model}.jsonl'
    done = {json.loads(l)['request_id'] for l in out.read_text().splitlines() if l.strip() and json.loads(l)['status'] == 'completed'} if out.exists() else set()
    todo = [q for q in reqs if q['request_id'] not in done]
    if a.limit: todo = todo[:a.limit]
    print(a.model, 'manifest', len(reqs), 'done', len(done), 'todo', len(todo), flush=True)
    if not a.execute:
        print('dry run: nothing sent'); return
    sys.argv = sys.argv[:1]
    import run_pilot                                  # loads keys (never printed)
    reg = dict(RB.REGISTRY[a.model], max_tokens=RB.MAX_TOKENS, python=platform.python_version(), started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               manifest=str(man.name), manifest_sha256=hashlib.sha256(man.read_bytes()).hexdigest(),
               retry_rule='technical errors only, up to 4 attempts with exponential backoff; content never retried')
    rf = rdir / f'model_registry_{a.model}.json'
    allreg = json.loads(rf.read_text()) if rf.exists() else []
    allreg.append(reg); rf.write_text(json.dumps(allreg, indent=1))
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

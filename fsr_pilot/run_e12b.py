"""Runner for v5.2-E12b (workspace only). Reads the frozen manifest built by fsr_v51/e12b/build_e12b.py
(or the repair manifest) and executes it resumably. Nothing is sent without --execute. Keys come from config.env
via run_pilot.load_env and are never printed.
  python run_e12b.py --model gem38                     # dry run: counts only
  python run_e12b.py --model gem38 --smoke --execute   # smoke subset -> runs_e12b_smoke/
  python run_e12b.py --model gem38 --execute           # main run
"""
import argparse, base64, hashlib, io, json, os, platform, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

E12B = Path('/home/claude/fsr_v51/e12b')

# Frozen settings. 'min' = lowest reasoning/thinking setting each API accepts (as in E12; Gemini thinking_budget 0
# verified to return no thought tokens). 'default' = no reasoning parameter sent at all (API default).
REGISTRY = {
    'gpt56': {'provider': 'openai', 'model': 'gpt-5.6-sol', 'image_detail': 'auto',
              'min': {'reasoning_effort': 'none'}, 'default': {}},
    'gpt61': {'provider': 'openai', 'model': 'gpt-6.1-sol', 'image_detail': 'auto',
              'min': {'reasoning_effort': 'low'}, 'default': {}},
    'opus55': {'provider': 'anthropic', 'model': 'claude-opus-5-5',
               'min': {'output_config.effort': 'low', 'thinking': 'adaptive (cannot be disabled)'}, 'default': {}},
    'sonnet55': {'provider': 'anthropic', 'model': 'claude-sonnet-5-5',
                 'min': {'thinking': 'between_tools'}, 'default': {}},
    'gem38': {'provider': 'gemini', 'model': 'gemini-3.8-flash', 'media_resolution': 'default',
              'min': {'thinking_budget': 0}, 'default': {}},
}
MAX_TOKENS = {'min': 16384, 'default': 32768, 'default16k': 16384}   # default16k: default reasoning with the 'min' output cap (v5.4 control)
_lock = threading.Lock(); _clients = {}


def client(kind):
    with _lock:
        if kind not in _clients:
            if kind == 'anthropic':
                import anthropic; _clients[kind] = anthropic.Anthropic(timeout=900)
            elif kind == 'openai':
                import openai; _clients[kind] = openai.OpenAI(timeout=900)
            else:
                from google import genai; from google.genai import types
                _clients[kind] = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(timeout=900000))
        return _clients[kind]


def png(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return b.getvalue()


def call(key, setting, ims, system, user):
    R = REGISTRY[key]; mt = MAX_TOKENS[setting]
    if R['provider'] == 'anthropic':
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64.standard_b64encode(png(im)).decode()}} for im in ims]
        content.append({'type': 'text', 'text': user})
        kw = {}
        if setting == 'min':
            kw = {'extra_body': {'output_config': {'effort': 'low'}}} if key == 'opus55' else {'thinking': {'type': 'between_tools'}}
        r = client('anthropic').messages.create(model=R['model'], max_tokens=mt, system=system, messages=[{'role': 'user', 'content': content}], **kw)
        raw = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text').strip()
        det = getattr(r.usage, 'output_tokens_details', None)
        return raw, {'input_tokens': r.usage.input_tokens, 'output_tokens': r.usage.output_tokens, 'actual_model': r.model,
                     'stop_reason': r.stop_reason, 'thinking_tokens': getattr(det, 'thinking_tokens', None), 'message_id': r.id}
    if R['provider'] == 'openai':
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png(im)).decode(), 'detail': 'auto'}} for im in ims]
        content.append({'type': 'text', 'text': user})
        kw = dict(model=R['model'], max_completion_tokens=mt, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])
        kw.update(R['default' if setting.startswith('default') else 'min'])
        r = client('openai').chat.completions.create(**kw); ch = r.choices[0]
        det = getattr(r.usage, 'completion_tokens_details', None)
        return (ch.message.content or '').strip(), {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
                                                     'actual_model': r.model, 'stop_reason': ch.finish_reason,
                                                     'reasoning_tokens': getattr(det, 'reasoning_tokens', None) if det else None}
    from google.genai import types
    parts = [types.Part.from_bytes(data=png(im), mime_type='image/png') for im in ims] + [user]
    cfg = dict(system_instruction=system, max_output_tokens=mt)
    if setting == 'min': cfg['thinking_config'] = types.ThinkingConfig(thinking_budget=0)
    r = client('gemini').models.generate_content(model=R['model'], contents=parts, config=types.GenerateContentConfig(**cfg))
    cand = r.candidates[0] if r.candidates else None
    raw = ''.join(p.text for p in cand.content.parts if getattr(p, 'text', None)) if cand and cand.content and cand.content.parts else ''
    um = r.usage_metadata
    return raw.strip(), {'input_tokens': um.prompt_token_count, 'output_tokens': (um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
                         'visible_output_tokens': um.candidates_token_count, 'thinking_tokens': um.thoughts_token_count,
                         'actual_model': getattr(r, 'model_version', None) or R['model'],
                         'stop_reason': str(cand.finish_reason) if cand else None}


def execute(q):
    ims = [Image.open(E12B / q['reference_path']).convert('RGB'), Image.open(E12B / q['image_path']).convert('RGB')]
    err = None
    for attempt in range(1, 5):                      # technical retries only; content is never retried
        try:
            raw, u = call(q['model_key'], q['setting'], ims, q['system'], q['user'])
            return {'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw, 'provider_model_id': u.get('actual_model'),
                    'finish_reason': u.get('stop_reason'), 'usage': u, 'attempts': attempt, 'time': time.time()}
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:300]}'; time.sleep(min(60, 5 * 2 ** attempt))
    return {'request_id': q['request_id'], 'status': 'failed', 'raw_text': '', 'error': err, 'attempts': 4, 'time': time.time()}


def smoke_subset(allreq):
    """First A scene of each dose, rep 0, every arm and grid; one B1 and one B2 request per kind; both settings."""
    first = {d: sorted(q['sample_id'] for q in allreq if q['block'] == 'A' and q['dose'] == d)[0] for d in (96, 192)}
    keep = [q for q in allreq if q['block'] in ('A', 'Adef') and q['rep'] == 0 and q['sample_id'] in first.values() and q['dose'] == 96]
    keep += [q for q in allreq if q['block'] == 'A' and q['rep'] == 0 and q['sample_id'] == first[192] and q['grid'] == 'on']
    b1 = sorted({q['sample_id'] for q in allreq if q['block'] == 'B1'})[0]
    keep += [q for q in allreq if q['block'] == 'B1' and q['sample_id'] == b1 and q['rep'] == 0]
    for cell in ('c024', 'g32'):
        sid = sorted({q['sample_id'] for q in allreq if q['block'] == 'B2' and q['cell'] == cell})[0]
        keep += [q for q in allreq if q['sample_id'] == sid and q['arm'] == 'A2']
    return keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--manifest', default='runs_e12b/requests.jsonl')
    ap.add_argument('--out', default='runs_e12b')
    a = ap.parse_args()
    man = E12B / a.manifest
    allreq = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    rdir = E12B / a.out
    if a.smoke:
        allreq = smoke_subset(allreq); rdir = E12B / 'runs_e12b_smoke'; rdir.mkdir(exist_ok=True)
        tmp = rdir / f'requests.{a.model}.tmp'; tmp.write_text(''.join(json.dumps(q) + '\n' for q in allreq)); tmp.replace(rdir / 'requests.jsonl')
    reqs = [q for q in allreq if q['model_key'] == a.model]
    rdir.mkdir(exist_ok=True)
    out = rdir / f'responses_{a.model}.jsonl'
    done = {json.loads(l)['request_id'] for l in out.read_text().splitlines() if l.strip() and json.loads(l)['status'] == 'completed'} if out.exists() else set()
    todo = [q for q in reqs if q['request_id'] not in done]
    if a.limit: todo = todo[:a.limit]
    print(a.model, 'manifest', len(reqs), 'done', len(done), 'todo', len(todo), flush=True)
    if not a.execute:
        print('dry run: nothing sent'); return
    import sys; sys.argv = sys.argv[:1]
    import run_pilot                                  # loads keys from config.env (never printed)
    import anthropic, openai
    from google import genai as _g
    reg = dict(REGISTRY[a.model], max_tokens=MAX_TOKENS, python=platform.python_version(),
               sdk={'anthropic': anthropic.__version__, 'openai': openai.__version__, 'google-genai': getattr(_g, '__version__', '?')},
               started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), manifest=str(man.name),
               manifest_sha256=hashlib.sha256(man.read_bytes()).hexdigest(),
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

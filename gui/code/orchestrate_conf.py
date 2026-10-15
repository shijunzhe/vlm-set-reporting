"""Operational driver for the confirmation run (protocol v3, section 7). It only calls the frozen builders
(code/conf_stages.py) and runner (code/run_gui.py); it adds the budget checks and the log. Not part of the method.
Usage: python3 code/orchestrate_conf.py --batch first100|all [--execute]
Budget: before each stage, used + reserve(stage) + expected(later stages) must stay within $60 and 21,000 physical attempts;
otherwise the driver stops and writes PAUSED to the log (no item is dropped)."""
import argparse, glob, json, os, subprocess, sys, time
sys.path.insert(0, '/home/claude/fsr_v51/e3'); from score_e3 import PRICE
ROOT = '/home/claude/gui'; SUB = 'conf_v1'; CAP_USD, CAP_ATT = 60.0, 21000
LOG = f'{ROOT}/runs/{SUB}/orchestrate.log'
STEPS = [['A', 'DA', 'GLF1'], ['B', 'DB', 'GLF2'], ['GLF3', 'S', 'SD'], ['CMPA_M', 'CMPA_D']]
MODELS = {'A': ['gpt61', 'sonnet55', 'gem38'], 'B': ['gpt61', 'sonnet55', 'gem38'], 'DA': ['gpt61', 'sonnet55'], 'DB': ['gpt61', 'sonnet55'],
          'GLF1': ['gpt61'], 'GLF2': ['gpt61'], 'GLF3': ['gpt61'], 'S': ['gem38'], 'SD': ['gem38'], 'CMPA_M': ['gem38'], 'CMPA_D': ['gem38']}
# expected cost per item and stage from pilot unit costs (USD), used only for the reserve
EXP = {'A': 0.01254, 'B': 0.00697, 'DA': 0.0216, 'DB': 0.0083, 'GLF1': 0.0062, 'GLF2': 0.0053, 'GLF3': 0.0005,
       'S': 0.0053, 'SD': 0.0035, 'CMPA_M': 0.0021, 'CMPA_D': 0.0018}


def log(msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%S ') + msg
    print(line, flush=True); open(LOG, 'a').write(line + '\n')


def spent():
    usd = 0.0; att = 0; n = 0
    for f in glob.glob(f'{ROOT}/runs/{SUB}/responses_*_*.jsonl'):
        for l in open(f):
            r = json.loads(l); att += r.get('attempts', 1) or 1
            if r['status'] == 'completed':
                m = r['request_id'].split('|')[1]; u = r['usage']; usd += (u['input_tokens'] * PRICE[m][0] + u['output_tokens'] * PRICE[m][1]) / 1e6; n += 1
    return usd, att, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', required=True); ap.add_argument('--execute', action='store_true'); a = ap.parse_args()
    n_items = 100 if a.batch == 'first100' else 600
    os.makedirs(f'{ROOT}/runs/{SUB}', exist_ok=True)
    log(f'start batch={a.batch} execute={a.execute}')
    for si, step in enumerate(STEPS):
        for st in step:
            r = subprocess.run([sys.executable, '-I', 'code/conf_stages.py', '--stage', st, '--check', '--batch', a.batch], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0: log(f'BUILD FAILED {st}: {r.stderr[-800:]}'); return
            log(f'built {st}: ' + ' | '.join(x for x in r.stdout.strip().splitlines()[-2:]))
        usd, att, n = spent()
        done_items = {st: len({json.loads(l)['request_id'].split('|')[0] for f in glob.glob(f'{ROOT}/runs/{SUB}/responses_{st}_*.jsonl') for l in open(f)}) for s2 in STEPS for st in s2}
        pending = sum(EXP[st] * max(0, n_items - done_items[st]) for s2 in STEPS[si:] for st in s2)
        if a.batch == 'first100': pending += 500 * sum(EXP.values())                    # the rest of the run
        projected = usd + pending
        log(f'budget check before step {si + 1}: spent ${usd:.2f}, attempts {att}, completed {n}; expected remaining ${pending:.2f}; projected ${projected:.2f} (cap ${CAP_USD})')
        if projected > CAP_USD or att > CAP_ATT:
            log('PAUSED: projected spending or attempts exceed the cap; no item is dropped'); return
        if not a.execute: continue
        procs = []
        for st in step:
            for m in MODELS[st]:
                lf = open(f'{ROOT}/runs/{SUB}/log_{st}_{m}.txt', 'a')
                procs.append(subprocess.Popen([sys.executable, 'code/run_gui.py', '--subset', SUB, '--stage', st, '--model', m, '--workers', '8', '--execute'],
                                              cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT))
        for p in procs: p.wait()
        usd, att, n = spent(); log(f'step {si + 1} done: spent ${usd:.2f}, attempts {att}, completed {n}')
        fails = 0
        for f in glob.glob(f'{ROOT}/runs/{SUB}/responses_*_*.jsonl'):
            ok = set(); bad = set()
            for l in open(f):
                r = json.loads(l); (ok if r['status'] == 'completed' else bad).add(r['request_id'])
            fails += len(bad - ok)
        log(f'unresolved failed requests so far: {fails}')
    usd, att, n = spent(); log(f'finished batch={a.batch}: spent ${usd:.2f}, attempts {att}, completed {n}; per item ${usd / n_items:.4f} (cumulative)')


if __name__ == '__main__':
    main()

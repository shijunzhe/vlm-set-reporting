"""E3 repair manifest (2 x 2 design), built from MAIN/AGG responses without reading labels.
Source reports: rep-0 reports of L-min and R-min with any observable flag (empty, truncated, unknown, missing rows, invalid
object, unparseable).
Recurrence (label-free): another call of the same source and candidate (L-min: the AGG repetitions; R-min: the second
screening call) also carries a flag. Reports without a second call are not used.
Action, assigned alternately within each (dataset, model, recurrence) stratum in a seeded order:
  RETRY   identical re-request (fresh call)
  CHANGE  L: follow-up with the previous answer attached; R: re-query only the flagged regions (all regions if the answer was
          empty or unreadable)
Cap: 40 reports per dataset x model (seeded sample, balanced over recurrence strata when possible).
"""
import hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
from score_e3 import load_items

ROOT = Path(__file__).parent
FOLLOW = ("\n\nA previous answer for this image is given below. It may be incomplete or may violate the output contract. Re-inspect the "
          "image and return the complete answer, following the output contract exactly.\n\nPrevious answer:\n{prev}")
CAP = 40


def main():
    items = load_items()
    man = {json.loads(l)['request_id']: json.loads(l) for l in (ROOT / 'runs/requests_main.jsonl').read_text().splitlines() if l.strip()}
    resp = {}
    for f in (ROOT / 'runs_main').glob('responses_*.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in resp: resp[r['request_id']] = r
    by = defaultdict(dict)                      # (item, model, cand) -> rep -> (q, r, flags, au)
    for rid, q in man.items():
        if q['block'] not in ('MAIN', 'AGG') or q['cand'] not in ('L-min', 'R-min'): continue
        r = resp.get(rid)
        if r is None: continue
        it = items[q['item_id']]; ok = r['status'] == 'completed'; raw = r.get('raw_text', '') if ok else ''
        recs, au = L.decode(it, q['arm'], raw, q['requested']); fl = L.flags(it, q['arm'], raw, r.get('finish_reason'), ok, au)
        by[(q['item_id'], q['model_key'], q['cand'])][q['rep']] = (q, r, fl, au, raw)
    cand = defaultdict(list)
    for (iid, m, c), reps in by.items():
        if 0 not in reps or len(reps) < 2: continue
        q, r, fl, au, raw = reps[0]
        if not fl['any']: continue
        recur = any(v[2]['any'] for k, v in reps.items() if k != 0)
        cand[(q['dataset'], m)].append(dict(q=q, raw=raw, fl=fl, au=au, recur=recur))
    out = []
    for (ds, m), lst in sorted(cand.items()):
        rng = random.Random(int(hashlib.sha256(f'e3-repair:{ds}:{m}'.encode()).hexdigest()[:12], 16))
        strata = defaultdict(list)
        for x in lst: strata[x['recur']].append(x)
        for v in strata.values(): rng.shuffle(v)
        take = []; per = CAP // max(1, len(strata))
        for k, v in strata.items(): take += v[:per]
        rest = [x for k, v in strata.items() for x in v[per:]]; take += rest[:max(0, CAP - len(take))]
        cnt = defaultdict(int)
        for x in take:
            q = x['q']; it = items[q['item_id']]; k = cnt[x['recur']]; cnt[x['recur']] += 1
            action = 'RETRY' if k % 2 == 0 else 'CHANGE'
            base = dict(q, block='REPAIR', base_id=q['request_id'], recur=x['recur'], action=action, flags=x['fl'])
            if action == 'RETRY':
                base['request_id'] = 'E3:REPAIR:RETRY:' + q['request_id']
            elif q['arm'] == 'L':
                base.update(request_id='E3:REPAIR:CHANGE:' + q['request_id'], user=q['user'] + FOLLOW.format(prev=x['raw'][:30000]))
            else:
                rows = {}
                recs, au = L.decode(it, 'R', x['raw'], q['requested'])
                # flagged regions: missing rows, unknown rows; whole image if empty / unreadable
                import re
                seen, unk = set(), set()
                for mm in re.finditer(r'"region_id"\s*:\s*"(R\d{3})"[^\n]*?"status"\s*:\s*"(\w+)"', x['raw']):
                    seen.add(mm.group(1))
                    if mm.group(2) == 'unknown': unk.add(mm.group(1))
                flagged = [g for g in q['requested'] if g not in seen or g in unk]
                if not flagged or x['fl']['empty'] or x['fl']['failed']: flagged = q['requested']
                sysm = q['system'].replace('one JSON object per registered region', 'one JSON object per requested region').replace(
                    'in registry order, covering EVERY region in the registry', 'in the order of requested_regions, covering EVERY requested region and no other region')
                payload = json.loads(q['user'][q['user'].index('{'):]); payload['requested_regions'] = flagged
                pre = q['user'][:q['user'].index('{')]
                base.update(request_id='E3:REPAIR:CHANGE:' + q['request_id'], system=sysm, user=pre + json.dumps(payload), requested=flagged)
            base['prompt_sha256'] = hashlib.sha256((base['system'] + '\n---\n' + base['user']).encode()).hexdigest()
            out.append(base)
    p = ROOT / 'runs/requests_repair.jsonl'; p.write_text(''.join(json.dumps(r) + '\n' for r in out))
    from collections import Counter
    print(len(out), 'repair requests', hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    print(Counter((r['dataset'], r['model_key']) for r in out)); print(Counter((r['recur'], r['action']) for r in out))
    print('flagged candidates per cell:', {k: len(v) for k, v in cand.items()})


if __name__ == '__main__':
    main()

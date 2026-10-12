"""E12b-C: capped repair. Builds the repair manifest from completed base responses (no labels are read).

Base reports
  E12b  block A, rep 0, grid on, lowest setting: A2 (region rows) and A3 (single list); 5 models.
  E12   J-global (NDJSON rows), A-global (array rows), F-down (free list); all repetitions; 4 models.
Step 0 (0 calls)  deterministic lenient readout: split the text at each region_id key and read every x/y pair of
                  the segment; lists: every x/y pair. Pairs outside [0,1) are contract violations.
Triggers (label-free, after step 0)
  rows:  flagged regions = regions without a row, rows with status unknown, regions containing a contract violation;
         every region when the answer is failed or empty; plus the last emitted region when the answer was truncated.
  lists: status unknown, truncated, failed or empty answer, or any contract violation.
Action (at most one call per report, same model and setting as the base)
  rows:  one call that requests only the flagged regions (same protocol family); its rows replace the flagged rows.
  lists: one follow-up call with the previous answer attached ("re-inspect and return the complete answer"); its
         list replaces the base list.
  Fallback: if the repair call fails or returns nothing readable, the step-0 report is kept.
Usage: python build_e12b_repair.py   ->  runs_e12b/repair_requests.jsonl (+ repair_bases.jsonl with step-0 reports)
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent; E12 = ROOT.parent / 'e12'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(E12))
import build_e12b as BB
import build_e12 as B12

RID = re.compile(r'"region_id"\s*:\s*"(R0\d\d)"')
STAT = re.compile(r'"status"\s*:\s*"(\w+)"')
PAIR = re.compile(r'"x"\s*:\s*(-?[0-9.eE+]+)\s*,\s*"y"\s*:\s*(-?[0-9.eE+]+)')
TRUNC = {'length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'}
FOLLOW = ("\n\nA previous answer for this scene is given below. It may be incomplete or may violate the output contract. "
          "Re-inspect the scene and return the complete answer, following the output contract exactly.\n\nPrevious answer:\n{prev}")


def pairs(seg):
    good, bad = [], 0
    for a, b in PAIR.findall(seg):
        try:
            x, y = float(a), float(b)
        except ValueError:
            bad += 1; continue
        if 0 <= x < 1 and 0 <= y < 1: good.append((1024 * x, 1024 * y))
        else: bad += 1
    return good, bad


def lenient_rows(raw, requested):
    """{rid: dict(status, pts, bad)} for every requested region that has a row in the text."""
    out = {}; ms = list(RID.finditer(raw))
    for i, m in enumerate(ms):
        rid = m.group(1)
        if rid not in requested: continue
        seg = raw[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(raw)]
        st = STAT.search(seg); p, bad = pairs(seg)
        e = out.setdefault(rid, dict(status=None, pts=[], bad=0, last=False))
        e['status'] = e['status'] or (st.group(1) if st else None); e['pts'] += p; e['bad'] += bad
    if ms and ms[-1].group(1) in out: out[ms[-1].group(1)]['last'] = True
    return out


def lenient_list(raw):
    st = STAT.search(raw); p, bad = pairs(raw)
    return dict(status=st.group(1) if st else None, pts=p, bad=bad)


def triggers(kind, requested, raw, status, finish):
    dead = status != 'completed' or not raw.strip()
    trunc = str(finish) in TRUNC
    if kind == 'rows':
        rows = lenient_rows(raw, requested)
        flagged = list(requested) if dead else [r for r in requested if r not in rows or rows[r]['status'] == 'unknown' or rows[r]['bad'] > 0
                                                or (trunc and rows[r]['last'])]
        return rows, flagged
    lst = lenient_list(raw)
    return lst, bool(dead or trunc or lst['status'] == 'unknown' or lst['bad'] > 0)


def rows_repair_e12b(q, flagged):
    """A2 protocol restricted to the flagged regions (contract text adapted to a subset)."""
    sysm = BB.system('A2').replace('one JSON object per registered region', 'one JSON object per requested region').replace(
        'in registry order, covering EVERY region in the registry', 'in the order of requested_regions, covering EVERY requested region and no other region')
    reg = q['registry']
    payload = {'task': 'reference-conditioned point detection', 'image_width': 1024, 'image_height': 1024,
               'coordinate_contract': 'scene_normalized_xy_half_open_v1', 'ordered_registry': reg, 'requested_regions': flagged}
    return sysm, BB.PRE + BB.USER.format(payload=json.dumps(payload))


def rows_repair_e12(q, flagged, sample):
    payload = {'task': 'reference-conditioned point detection', 'image_width': 1024, 'image_height': 1024,
               'coordinate_contract': 'scene_normalized_xy_half_open_v1', 'ordered_registry': sample['regions'], 'requested_regions': flagged}
    return B12.system_rows(q['format'], 'global'), B12.PRE + B12.USER.format(payload=json.dumps(payload))


def main():
    bases, reps = [], []
    # ---- E12b sources
    man = {json.loads(l)['request_id']: json.loads(l) for l in (ROOT / 'runs_e12b/requests.jsonl').read_text().splitlines() if l.strip()}
    for m in BB.__dict__.get('MODELS', ['gpt56', 'gpt61', 'opus55', 'sonnet55', 'gem38']):
        f = ROOT / f'runs_e12b/responses_{m}.jsonl'
        resp = {}
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r['status'] == 'completed' or r['request_id'] not in resp: resp[r['request_id']] = r
        for rid, q in man.items():
            if not (q['model_key'] == m and q['block'] == 'A' and q['rep'] == 0 and q['grid'] == 'on' and q['arm'] in ('A2', 'A3')): continue
            r = resp[rid]; raw = r.get('raw_text', '') if r['status'] == 'completed' else ''
            kind = 'rows' if q['arm'] == 'A2' else 'list'
            parsed, flag = triggers(kind, q['requested'], raw, r['status'], r.get('finish_reason'))
            b = dict(base_id=rid, source='E12b', src_arm=q['arm'], kind=kind, model_key=m, sample_id=q['sample_id'], data='e12b',
                     requested=q['requested'], step0=parsed, flagged=flag)
            bases.append(b)
            if flag:
                if kind == 'rows': sysm, user = rows_repair_e12b(q, flag)
                else: sysm, user = q['system'], q['user'] + FOLLOW.format(prev=raw[:30000])
                reps.append(dict(q, request_id='E12bC:' + rid, block='C', base_id=rid, kind='rows' if kind == 'rows' else 'free',
                                 requested=flag if kind == 'rows' else q['requested'], system=sysm, user=user,
                                 prompt_sha256=hashlib.sha256((sysm + '\n---\n' + user).encode()).hexdigest()))
    # ---- E12 sources
    man12 = {json.loads(l)['request_id']: json.loads(l) for l in (E12 / 'runs_e12/requests.jsonl').read_text().splitlines() if l.strip()}
    samples = {json.loads(l)['sample_id']: json.loads(l) for l in (E12 / 'data_e12/samples.jsonl').read_text().splitlines() if l.strip()}
    for m in ['gpt56', 'gpt61', 'opus55', 'sonnet55']:
        resp = {json.loads(l)['request_id']: json.loads(l) for l in (E12 / f'runs_e12/responses_{m}.jsonl').read_text().splitlines() if l.strip()}
        for rid, q in man12.items():
            if q['model_key'] != m or q['condition'] not in ('J-global', 'A-global', 'F-down'): continue
            r = resp[rid]; raw = r.get('raw_text', '') if r['status'] == 'completed' else ''
            kind = 'list' if q['kind'] == 'free' else 'rows'
            parsed, flag = triggers(kind, q['requested'], raw, r['status'], r.get('finish_reason'))
            bases.append(dict(base_id=rid, source='E12', src_arm=q['condition'], kind=kind, model_key=m, sample_id=q['sample_id'], data='e12',
                              requested=q['requested'], step0=parsed, flagged=flag))
            if flag:
                if kind == 'rows': sysm, user = rows_repair_e12(q, flag, samples[q['sample_id']])
                else: sysm, user = q['system'], q['user'] + FOLLOW.format(prev=raw[:30000])
                reps.append(dict(q, request_id='E12bC:' + rid, block='C', base_id=rid, setting='min', arm=q['condition'], grid='on', granularity=4,
                                 cell='e12', glyph=22, kind='rows' if kind == 'rows' else 'free',
                                 requested=flag if kind == 'rows' else q['requested'],
                                 image_path=str(E12 / q['image_path']), reference_path=str(E12 / q['reference_path']),
                                 system=sysm, user=user, prompt_sha256=hashlib.sha256((sysm + '\n---\n' + user).encode()).hexdigest()))
    (ROOT / 'runs_e12b/repair_bases.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in bases))
    out = ROOT / 'runs_e12b/repair_requests.jsonl'; out.write_text(''.join(json.dumps(r) + '\n' for r in reps))
    from collections import Counter
    print(len(bases), 'base reports;', len(reps), 'repair calls', hashlib.sha256(out.read_bytes()).hexdigest())
    print(Counter((b['source'], b['src_arm']) for b in bases if b['flagged']))
    print(Counter(r['model_key'] for r in reps))


if __name__ == '__main__':
    main()

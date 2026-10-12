"""Third-party annotations: redaction (maintainers) and byte-exact rebuild from the official downloads (users).

Labels from DOTA v1.0, SKU-110K and FSC-147 are not redistributed in this repository, because their terms do not clearly
allow it. Every file that held them is stored with each such value replaced by a placeholder
    {"__tp__": {"src": ..., ...parameters...}}            (optionally with "__append__": [our own records])
that says where the value comes from in the official release and how it was transformed (window, scale, rounding).
`rebuild` reads the official files, fills every placeholder, and checks the SHA-256 of each rebuilt file against
thirdparty/manifest.json, so the rebuilt files are byte-identical to the ones used for the paper.

    python3 -I thirdparty/tp.py rebuild --sources /path/to/sources      # users: fill placeholders, verify checksums
    python3 -I thirdparty/tp.py status                                    # which files are redacted or rebuilt
    python3 -I thirdparty/tp.py redact --sources /path/to/sources        # maintainers: strip labels, write manifest

Expected layout under --sources (see thirdparty/SOURCES.md for download links):
    dota/DOTAv1/labels/val_original/P0003.txt ...                      DOTA v1.0 val labels, original format
      (or dota/labelTxt/P0003.txt ...)                                  same files from the official DOTA v1.0 release
    sku110k/SKU110K_fixed/annotations/annotations_test.csv              SKU-110K
    fsc147/annotation_FSC147_384.json                                   FSC-147
"""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / 'thirdparty' / 'manifest.json'
DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor',
                'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
DOTA_DIRS = ['dota/DOTAv1/labels/val_original', 'dota/labelTxt']      # Ultralytics DOTAv1.zip layout, or official v1.0 val labelTxt
SKU_CSV = 'sku110k/SKU110K_fixed/annotations/annotations_test.csv'
FSC_JSON = 'fsc147/annotation_FSC147_384.json'
# files that hold third-party labels, relative to the repository root
FILES = [f'fsr_v51/e3/data/{ds}/{n}.jsonl' for ds in ('dota', 'sku110k', 'fsc147') for n in ('items', 'items_agn', 'items_corr')] + \
        ['stageB/runs/nr/items.json', 'stageB/runs/b1p/items.json', 'stageB/runs/b1ppilot/items.json']


# ------------------------------------------------------------------------------------------------ official sources
class Sources:
    def __init__(self, root):
        self.root = Path(root); self._sku = None; self._fsc = None; self._dota = {}

    def dota_objects(self, label_file):
        if label_file not in self._dota:
            p = self.dota_dir() / label_file; objs = []
            for line in open(p):
                t = line.split()
                if len(t) < 10: continue
                xs = [float(v) for v in t[0:8:2]]; ys = [float(v) for v in t[1:8:2]]
                objs.append(([min(xs), min(ys), max(xs), max(ys)], t[8], int(t[9])))
            self._dota[label_file] = objs
        return self._dota[label_file]

    def dota_dir(self):
        for d in DOTA_DIRS:
            if (self.root / d).is_dir(): return self.root / d
        raise SystemExit(f'DOTA v1.0 val labels not found under {self.root}; expected one of {DOTA_DIRS} (see thirdparty/SOURCES.md)')

    def files_used(self):
        """official files read so far, relative to the sources root, with their SHA-256"""
        out = {f'dota/{lf}': sha((self.dota_dir() / lf).read_bytes()) for lf in sorted(self._dota)}
        if self._sku is not None: out[SKU_CSV] = sha((self.root / SKU_CSV).read_bytes())
        if self._fsc is not None: out[FSC_JSON] = sha((self.root / FSC_JSON).read_bytes())
        return out

    def dota_window(self, label_file, window):
        """objects at least half inside the window and not difficult are targets; the others that intersect it are ignore
        regions (as in fsr_v51/e3/prep_e3.py and stageB/code/build_nr.py)"""
        x0, y0, ww, wh = window; tg, ig = [], []
        for b, cl, diff in self.dota_objects(label_file):
            area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1])); c = [max(b[0], x0), max(b[1], y0), min(b[2], x0 + ww), min(b[3], y0 + wh)]
            if c[2] <= c[0] or c[3] <= c[1]: continue
            frac = (c[2] - c[0]) * (c[3] - c[1]) / area; cb = [round(c[0] - x0, 1), round(c[1] - y0, 1), round(c[2] - x0, 1), round(c[3] - y0, 1)]
            if diff or frac < 0.5 or cl not in DOTA_CLASSES: ig.append(cb)
            else: tg.append(dict(label=cl, box=cb))
        return tg, ig

    def sku_boxes(self, image_file):
        if self._sku is None:
            import csv
            self._sku = {}
            for r in csv.reader(open(self.root / SKU_CSV)):
                if len(r) < 8: continue
                self._sku.setdefault(r[0], []).append([float(r[1]), float(r[2]), float(r[3]), float(r[4])])
        return self._sku[image_file]

    def fsc(self, image_file):
        if self._fsc is None: self._fsc = json.load(open(self.root / FSC_JSON))
        return self._fsc[image_file]

    def resolve(self, spec):
        src, field = spec['src'], spec['field']
        if src == 'dota':
            tg, ig = self.dota_window(spec['label_file'], spec['window'])
            if field == 'targets': return [dict(label='obj', box=t['box']) for t in tg] if spec.get('agn') else tg
            return ig
        if src == 'sku':
            s = spec['scale']
            bx = [[round(b[0] * s, 1), round(b[1] * s, 1), round(b[2] * s, 1), round(b[3] * s, 1)] for b in self.sku_boxes(spec['image_file'])]
            return [dict(label='obj' if spec.get('agn') else 'product', box=b) for b in bx]
        if src == 'fsc':
            a = self.fsc(spec['image_file'])
            if field == 'targets': return [dict(label=spec['label'], point=[float(p[0]), float(p[1])]) for p in a['points']]
            if field == 'points': return [[float(p[0]), float(p[1])] for p in a['points']]
            if field == 'exemplars':
                return [[min(p[0] for p in b), min(p[1] for p in b), max(p[0] for p in b), max(p[1] for p in b)] for b in a['box_examples_coordinates']]
        raise ValueError(f'unknown placeholder {spec}')


# ------------------------------------------------------------------------------------------------ serialization
def sha(b): return hashlib.sha256(b).hexdigest()


SERIALIZERS = {
    'jsonl': lambda objs: ''.join(json.dumps(o) + '\n' for o in objs),
    'json': lambda o: json.dumps(o),
    'json_indent1': lambda o: json.dumps(o, indent=1),
    'json_indent2': lambda o: json.dumps(o, indent=2),
}


def load(path):
    t = Path(path).read_text()
    return [json.loads(l) for l in t.splitlines() if l.strip()] if str(path).endswith('.jsonl') else json.loads(t)


def detect_serializer(path):
    raw = Path(path).read_bytes(); obj = load(path)
    names = ['jsonl'] if str(path).endswith('.jsonl') else ['json', 'json_indent1', 'json_indent2']
    for n in names:
        if SERIALIZERS[n](obj).encode() == raw: return n
    raise RuntimeError(f'no serializer reproduces {path} byte for byte')


def fill(o, S):
    """replace every placeholder in o by its value from the official sources"""
    if isinstance(o, dict):
        if '__tp__' in o:
            v = S.resolve(o['__tp__'])
            return v + copy.deepcopy(o['__append__']) if '__append__' in o else v
        return {k: fill(v, S) for k, v in o.items()}
    if isinstance(o, list): return [fill(v, S) for v in o]
    return o


def has_placeholder(o):
    if isinstance(o, dict): return '__tp__' in o or any(has_placeholder(v) for v in o.values())
    if isinstance(o, list): return any(has_placeholder(v) for v in o)
    return False


# ------------------------------------------------------------------------------------------------ redaction rules
def ph(spec, append=None):
    d = {'__tp__': spec}
    if append: d['__append__'] = append
    return d


def redact_e3(rec, kind, SRC):
    ds = rec['dataset']; r = copy.deepcopy(rec); agn = kind == 'items_agn'
    if ds == 'dota':
        win = rec['meta']['window']; lf = rec['meta']['source_image'].rsplit('.', 1)[0] + '.txt'
        base = dict(src='dota', label_file=lf, window=win)
        r['targets'] = ph(dict(base, field='targets', **({'agn': True} if agn else {})))
        # ignore regions that come from DOTA lead the list; regions appended after them (items_corr) are our own audit regions
        n_src = len(SRC.dota_window(lf, win)[1])
        if n_src:
            r['ignore'] = ph(dict(base, field='ignore'), rec['ignore'][n_src:])
    elif ds == 'sku110k':
        r['targets'] = ph(dict(src='sku', image_file=rec['id'][4:] + '.jpg', scale=rec['meta']['scale'], field='targets'))  # items_agn keeps 'product'
    elif ds == 'fsc147':
        f = rec['id'][4:] + '.jpg'
        r['targets'] = ph(dict(src='fsc', image_file=f, label=rec['targets'][0]['label'] if rec['targets'] else rec['labels'][0], field='targets'))
        r['meta'] = dict(r['meta']); r['meta']['exemplars'] = ph(dict(src='fsc', image_file=f, field='exemplars'))
    return r


def redact_file(rel, S):
    path = REPO / rel; obj = load(path); name = Path(rel).stem
    if rel.startswith('fsr_v51/e3/data/'):
        out = [redact_e3(x, name, S) for x in obj]
    elif rel == 'stageB/runs/nr/items.json':
        out = []
        for x in obj:
            r = copy.deepcopy(x)
            if x['dataset'] == 'dota':
                base = dict(src='dota', label_file=x['meta']['source_image'].rsplit('.', 1)[0] + '.txt', window=x['meta']['window'])
                r['targets'] = ph(dict(base, field='targets')); r['ignore'] = ph(dict(base, field='ignore'))
            out.append(r)
    elif rel.startswith('stageB/runs/b1p'):
        out = {k: dict(v, points=ph(dict(src='fsc', image_file=v['item'], field='points'))) for k, v in obj.items()}
    else:
        raise ValueError(rel)
    return out


# ------------------------------------------------------------------------------------------------ commands
def cmd_redact(a):
    """verify every round trip first; write nothing unless all files pass"""
    S = Sources(a.sources); man = []; todo = []
    for rel in FILES:
        path = REPO / rel; raw = path.read_bytes()
        if has_placeholder(load(path)): raise SystemExit(f'{rel} is already redacted; rebuild first')
        ser = detect_serializer(path); red = redact_file(rel, S)
        if SERIALIZERS[ser](fill(red, S)).encode() != raw: raise SystemExit(f'round trip failed for {rel}; nothing written')
        red_bytes = SERIALIZERS[ser](red).encode(); todo.append((path, red_bytes))
        man.append(dict(path=rel, serializer=ser, sha256_full=sha(raw), sha256_redacted=sha(red_bytes)))
        print('round trip ok', rel)
    if a.dry_run: return
    for path, b in todo: path.write_bytes(b)
    MANIFEST.write_text(json.dumps(dict(files=man, sources=S.files_used()), indent=1) + '\n'); print(f'redacted {len(todo)} files; wrote {MANIFEST.relative_to(REPO)}')


def cmd_rebuild(a):
    S = Sources(a.sources); M = json.load(open(MANIFEST)); bad = 0
    for m in M['files']:
        path = REPO / m['path']; raw = path.read_bytes()
        if sha(raw) == m['sha256_full']: print('already rebuilt', m['path']); continue
        if sha(raw) != m['sha256_redacted']: print('unexpected content, skipped', m['path']); bad += 1; continue
        try: full = SERIALIZERS[m['serializer']](fill(load(path), S)).encode()
        except (OSError, KeyError) as e: print('source missing, not written', m['path'], '-', e); bad += 1; continue
        if sha(full) != m['sha256_full']: print('CHECKSUM MISMATCH, not written', m['path']); bad += 1; continue
        path.write_bytes(full); print('rebuilt', m['path'])
    if bad:
        # name the official files that differ from the ones we used, to help find a wrong download
        for rel, h in M['sources'].items():
            p = S.dota_dir() / rel[5:] if rel.startswith('dota/') else S.root / rel
            got = sha(p.read_bytes()) if p.exists() else None
            if got != h: print('source differs from the release we used:', rel, '(missing)' if got is None else '')
    sys.exit(1 if bad else 0)


def cmd_status(a):
    for m in json.load(open(MANIFEST))['files']:
        h = sha((REPO / m['path']).read_bytes())
        print({m['sha256_full']: 'rebuilt ', m['sha256_redacted']: 'redacted'}.get(h, 'modified'), m['path'])


def rebuilt():
    """True when every third-party file has been rebuilt (used by reproduce.sh)"""
    return all(sha((REPO / m['path']).read_bytes()) == m['sha256_full'] for m in json.load(open(MANIFEST))['files'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('rebuild'); r.add_argument('--sources', required=True)
    d = sp.add_parser('redact'); d.add_argument('--sources', required=True); d.add_argument('--dry-run', action='store_true')
    sp.add_parser('status'); sp.add_parser('check')
    a = ap.parse_args()
    if a.cmd == 'check': sys.exit(0 if rebuilt() else 1)
    {'rebuild': cmd_rebuild, 'redact': cmd_redact, 'status': cmd_status}[a.cmd](a)

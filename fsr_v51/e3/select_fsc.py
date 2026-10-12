"""Select 60 FSC-147 test images not used in paper 2 or earlier experiments, n <= 300, 20 per count tertile (seeded),
download them from the Hugging Face mirror (isentropic/FSC147). Output: e3data/fsc/{images/, selection.json}."""
import json, os, random, subprocess, sys
D = '/home/claude/e3data/fsc'; os.makedirs(f'{D}/images', exist_ok=True)
a = json.load(open('/home/claude/fsc/annotation_FSC147_384.json')); s = json.load(open('/home/claude/fsc/Train_Test_Val_FSC_147.json'))
cls = dict(l.strip().split('\t') for l in open('/home/claude/fsc/ImageClasses_FSC147.txt') if '\t' in l)
used = set(os.listdir('/home/claude/fsc/images')) | set(os.listdir('/home/claude/x2/arr/data_e4/fsc147/images'))
pool = sorted([x for x in s['test'] if x not in used and len(a[x]['points']) <= 300], key=lambda x: len(a[x]['points']))
t1, t2 = len(pool) // 3, 2 * len(pool) // 3
strata = [pool[:t1], pool[t1:t2], pool[t2:]]
rng = random.Random(20261010); sel = []
for k, st in enumerate(strata):
    cand = st[:]; rng.shuffle(cand); got = []
    for x in cand:
        if len(got) == 20: break
        url = f'https://huggingface.co/datasets/isentropic/FSC147/resolve/main/images_384_VarV2/{x}'
        r = subprocess.run(['curl', '-sS', '-L', '-f', '-o', f'{D}/images/{x}', url], capture_output=True)
        if r.returncode == 0 and os.path.getsize(f'{D}/images/{x}') > 1000: got.append(x)
    sel += [dict(file=x, stratum=k, n=len(a[x]['points']), cls=cls.get(x)) for x in got]
json.dump(dict(source='huggingface.co/datasets/isentropic/FSC147 (images_384_VarV2), annotations annotation_FSC147_384.json',
               tertile_bounds=[len(a[pool[t1]]['points']), len(a[pool[t2]]['points'])], items=sel), open(f'{D}/selection.json', 'w'), indent=1)
print(len(sel), 'images', [len([x for x in sel if x['stratum'] == k]) for k in range(3)])

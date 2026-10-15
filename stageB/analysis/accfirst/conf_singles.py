"""Single-model flows on the 600 confirmation items, including Gemini 3.8 at its lowest setting, and the number of items on
which each flow returned no point (counted as misses). No model calls. Output: accfirst/conf_singles_conf_v1.json"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; GUI = '/home/claude/gui'; sys.path.insert(0, GUI + '/code')
cwd = os.getcwd(); import fusion_stages as F, gl_stages as GLS, decision as DC; os.chdir(GUI)
G, B = F.G, F.B; C = F.items_of('conf'); run = DC.Run(GUI, 'conf_v1', G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
FL = {'GPT-min-zoom': lambda c: DC.single_zoom(run, 'M', 'gpt61', c), 'Sonnet-min-zoom': lambda c: DC.single_zoom(run, 'M', 'sonnet55', c),
      'Gemini-min-zoom': lambda c: DC.single_zoom(run, 'M', 'gem38', c), 'GPT-default-zoom': lambda c: DC.single_zoom(run, 'D', 'gpt61', c),
      'Sonnet-default-zoom': lambda c: DC.single_zoom(run, 'D', 'sonnet55', c), 'GPTx3': lambda c: DC.gpt_x3(run, c, C[c]),
      'GLF': lambda c: DC.glf(run, c, GLS.final_point, C), 'M-S': lambda c: DC.s_rule(run, 'M', c, C[c]), 'M-C': lambda c: DC.c_rule(run, 'M', c, C[c]),
      'D-S': lambda c: DC.s_rule(run, 'D', c, C[c]), 'D-C': lambda c: DC.c_rule(run, 'D', c, C[c])}
res = {}
for k, fn in FL.items():
    P = {c: fn(c)[0] for c in C}
    res[k] = dict(errors=sum(1 - hit(p, c) for c, p in P.items()), no_point=sum(p is None for p in P.values()))
os.chdir(cwd); json.dump(res, open(HERE / 'conf_singles_conf_v1.json', 'w'), indent=1); print(res)

"""EXP-RC16: readout combination of method v5.16 step 1 (1a + 1b + 1c) run as one arm. No model calls.
Arm v516 (SCP-dense, 50 images, the B1 images; setting 'min', as every B1 arm):
  1a list first, counts_by_type after (B1 arm c format and closing sentence)
  1b output contract names the five allowed type names (G ENUM text); code maps synonyms (diamond / rhombus -> square)
  1c row-by-row scan order (B1 SCAN text)
  base: same system template and two-sided sentence as B1 arms b-l and G.
Readouts (analysis side): code count (synonym-mapped), degeneracy-check readout (code count unless the list is degenerate,
then the written count), written count.
Usage: python -I build_rc16.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
R = C.ROOT
TYPES = 'circle, triangle, square, star, pentagon'
TWO = 'Locate every shape carefully. Count systematically; do not miss any and do not invent shapes that are not there.'
SCAN = 'List the shapes in reading order: scan the image row by row from top to bottom, and within a row from left to right.'
ENUM = ('The "type" value must be exactly one of: circle, triangle, square, star, pentagon. A square rotated by any angle '
        'is still a "square"; do not use any other type name.')
EX_SHAPE = {'grid': '{"type": "circle", "x": 402, "y": 311}', 'pixel': '{"type": "circle", "x": 412, "y": 318}'}
COUNTS = '"counts_by_type": {"circle": 5, "triangle": 3, "square": 0, "star": 2, "pentagon": 4}'


def system(mode):
    fmt = '{"shapes": [' + EX_SHAPE[mode] + ', ...], ' + COUNTS + '}'
    return ('You are a precise visual counting and localization assistant.\n\nGiven an image containing geometric shapes, identify '
            'every shape.\n' + C.coord_sentence(mode, 1024, 1024) + '\n' + SCAN + '\n' + ENUM + '\n'
            '\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + fmt + '\n\n'
            f'Shape types to recognize: {TYPES}.\n' + TWO +
            '\nWrite the full shapes list first; after it, give counts_by_type for the shapes you listed.')


USER = 'List every geometric shape in this image with its center, then counts by type. Respond with JSON only.'
reqs = []
for i in range(50):
    n = json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['total_shapes']
    for m in C.MODELS:
        mode = C.MODE[m]
        reqs.append(dict(request_id=f'scpd|{i:03d}|{m}|v516|r0', model_key=m, setting='min', images=[f'data/scpd/{i:03d}.png'],
                         system=system(mode), user=USER,
                         meta=dict(task='scpd', item=i, arm='v516', rep=0, setting='min', mode=mode, w=1024, h=1024, sent=[1024, 1024], n=n)))
C.write_manifest(f'{R}/runs/rc16/requests_main.jsonl', reqs)
print(len(reqs))

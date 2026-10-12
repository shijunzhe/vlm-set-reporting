"""Experiment G: does a grid (or a similar reference) help, and why. No model calls.
All arms are list-only (no count field) and use two-sided wording, as B1 arm d. 'enum' = the output contract names the
five allowed type names and says a rotated square is still a square (fixes the label vocabulary, not the visual input).
Arms:
  d_enum      plain image, enum
  g_enum      + grid '+' markers (B1 layout: 100 px lattice), marker sentence, enum
  ruler_enum  + coordinate ruler: ticks with numeric labels along the top and left borders, in the model's own
              coordinate units (0-1000 grid for GPT / Gemini, pixels for Sonnet), ruler sentence, enum
  cell_enum   grid image + instruction to list shapes cell by cell (rows of cells top to bottom, left to right)
  scan_enum   plain image + row-by-row scan instruction (B1 arm i), enum
  gscan_enum  grid image + scan instruction, enum
  req_enum    plain image, enum + 'every record must contain type, x and y'
Design (setting 'min' unless noted):
  SCP 100        Gemini: all 7 arms; Sonnet: d_enum, g_enum, ruler_enum
  SCP-dense 40   GPT, Sonnet, Gemini: d_enum, g_enum
  SCP 50         Gemini at 'default' (API default thinking): d_enum, g_enum
Usage: python -I build_g.py"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
# constants copied verbatim from build_b1.py (importing it would rebuild the B1 manifest)
class B1:
    TYPES = 'circle, triangle, square, star, pentagon'
    TWO = 'Locate every shape carefully. Count systematically; do not miss any and do not invent shapes that are not there.'
    MARK = ("Faint semi-transparent magenta '+' markers are drawn in the image as visual reference points. They are reference markers "
            "only, NOT shapes to count. Use them to estimate each shape's coordinate as accurately as possible.")
    SCAN = 'List the shapes in reading order: scan the image row by row from top to bottom, and within a row from left to right.'
    GRID = [(100 * i, 100 * j) for i in range(1, 11) for j in range(1, 11)]
R = C.ROOT

ENUM = ('The "type" value must be exactly one of: circle, triangle, square, star, pentagon. A square rotated by any angle '
        'is still a "square"; do not use any other type name.')
REQ = 'Every record must contain all three fields "type", "x" and "y".'
RULER = ('Coordinate tick marks with numeric labels are drawn along the top and left borders of the image, in the same '
         'coordinate units you must report. They are reference marks only, NOT shapes to count. Use them to estimate each '
         "shape's coordinate as accurately as possible.")
CELL = ("The '+' markers lie on a regular lattice and divide the image into cells. Go through the cells row by row, from the "
        'top row of cells to the bottom, and from left to right within a row; list all shapes in a cell before moving to the next cell.')
FMT = '{"shapes": [' + '{"type": "<circle|triangle|square|star|pentagon>", "x": <int>, "y": <int>}' + ', ...]}'


def system(mode, extras):
    s = ('You are a precise visual counting and localization assistant.\n\nGiven an image containing geometric shapes, identify '
         'every shape.\n' + C.coord_sentence(mode, 1024, 1024) + '\n' + ''.join(e + '\n' for e in extras) +
         '\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + FMT + '\n\n'
         f'Shape types to recognize: {B1.TYPES}.\n' + B1.TWO)
    return s


USER = 'List every geometric shape in this image with its center. Respond with JSON only.'
ARM_EXTRAS = {'d_enum': [ENUM], 'g_enum': [B1.MARK, ENUM], 'ruler_enum': [RULER, ENUM], 'cell_enum': [B1.MARK, CELL, ENUM],
              'scan_enum': [B1.SCAN, ENUM], 'gscan_enum': [B1.MARK, B1.SCAN, ENUM], 'req_enum': [ENUM, REQ]}
ARM_IMAGE = {'d_enum': 'none', 'g_enum': 'g', 'ruler_enum': 'ruler', 'cell_enum': 'g', 'scan_enum': 'none', 'gscan_enum': 'g', 'req_enum': 'none'}


def draw_ruler(im, mode):
    """ticks every 100 units of the model's own coordinate system, labels at each tick, along top and left borders."""
    im = im.convert('RGBA'); ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception: font = ImageFont.load_default()
    col = (255, 0, 255, 170); W, H = im.size
    for v in range(100, 1001 if mode == 'grid' else 1001, 100):
        p = v * W / 1000 if mode == 'grid' else v
        if p >= W - 5: continue
        d.line([(p, 0), (p, 12)], fill=col, width=2); d.text((p + 2, 12), str(v), fill=col, font=font)
        q = v * H / 1000 if mode == 'grid' else v
        d.line([(0, q), (12, q)], fill=col, width=2); d.text((14, q - 7), str(v), fill=col, font=font)
    return Image.alpha_composite(im, ov).convert('RGB')


def image_path(task, i, kind, mode):
    if kind == 'none': return f'data/{task}/{i:03d}.png'
    if kind == 'g' and task == 'scp': return f'data/scp/{i:03d}_g.png'
    p = f'data/{task}/{i:03d}_{kind}{"_" + mode if kind == "ruler" else ""}.png'
    if not os.path.exists(f'{R}/{p}'):
        base = Image.open(f'{R}/data/{task}/{i:03d}.png').convert('RGB')
        (draw_ruler(base, mode) if kind == 'ruler' else C.draw_markers(base, B1.GRID)).save(f'{R}/{p}')
    return p


def main():
    reqs = []
    def add(task, i, m, arm, setting='min'):
        mode = C.MODE[m]; n = json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json' if task == 'scp' else f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['total_shapes']
        reqs.append(dict(request_id=f'{task}|{i:03d}|{m}|{arm}|{setting}', model_key=m, setting=setting, images=[image_path(task, i, ARM_IMAGE[arm], mode)],
                         system=system(mode, ARM_EXTRAS[arm]), user=USER,
                         meta=dict(task=task, item=i, arm=arm, rep=0, setting=setting, mode=mode, w=1024, h=1024, sent=[1024, 1024], n=n)))
    for i in range(100):
        for arm in ARM_EXTRAS: add('scp', i, 'gem38', arm)
        for arm in ('d_enum', 'g_enum', 'ruler_enum'): add('scp', i, 'sonnet55', arm)
    for i in range(40):
        for m in C.MODELS:
            for arm in ('d_enum', 'g_enum'): add('scpd', i, m, arm)
    for i in range(50):
        for arm in ('d_enum', 'g_enum'): add('scp', i, 'gem38', arm, 'default')
    C.write_manifest(f'{R}/runs/g1/requests_main.jsonl', reqs)
    import collections
    print(len(reqs), collections.Counter((q['model_key'], q['meta']['task'], q['setting']) for q in reqs))


if __name__ == '__main__':
    main()

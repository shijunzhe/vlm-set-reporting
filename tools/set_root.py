"""Point every script in this repository at the repository itself.

The experiment and analysis code was written with absolute paths rooted at one working directory.
This script rewrites that root in every .py and .sh file to the absolute path of this checkout.
It is idempotent: the current root is stored in .fsr_root and replaced on every run.
Usage: python3 tools/set_root.py"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / '.fsr_root'
old = STATE.read_text().strip() if STATE.exists() else '/home/claude'
new = str(REPO)
n = 0
olds = [o for o in {old, '/home/claude'} if o != new]
for p in REPO.rglob('*'):
    if p.suffix in ('.py', '.sh') and p.is_file() and p.name != 'set_root.py':
        s = p.read_text(encoding='utf-8', errors='surrogateescape'); t = s
        for o in olds: t = t.replace(o, new)
        if t != s:
            p.write_text(t, encoding='utf-8', errors='surrogateescape'); n += 1
STATE.write_text(new + '\n')
print(f'root set to {new} ({n} files rewritten)')

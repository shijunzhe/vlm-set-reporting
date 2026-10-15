"""Check the SHA-256 digests recorded before the click runs (protocol v3 for the 600-item evaluation, and the four-source
rules). tools/set_root.py rewrites the original root (ORIG below) in .py and .sh files to this checkout's path; the check
hashes each such file with that rewrite undone, so it verifies the files as they were run.
Usage: python3 tools/check_protocol_hashes.py"""
import hashlib
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ORIG = '/home/' + 'claude'          # written in two parts so that tools/set_root.py does not rewrite it
STATE = REPO / '.fsr_root'; root = STATE.read_text().strip() if STATE.exists() else ORIG
LISTS = [(REPO / 'stageB/analysis/accfirst/FSR_600_protocol_v3.sha256', REPO), (REPO / 'gui/runs/conf_v1/d4_rules.sha256', REPO / 'gui')]
bad = 0; n = 0
for lst, base in LISTS:
    for line in lst.read_text().splitlines():
        if not line.strip(): continue
        digest, rel = line.split(None, 1); p = base / rel.strip()
        data = p.read_bytes()
        if p.suffix in ('.py', '.sh') and root != ORIG:
            data = data.decode('utf-8', 'surrogateescape').replace(root, ORIG).encode('utf-8', 'surrogateescape')
        ok = hashlib.sha256(data).hexdigest() == digest; n += 1; bad += not ok
        if not ok: print('MISMATCH', p.relative_to(REPO))
print(f'protocol hashes: {n - bad} of {n} match')

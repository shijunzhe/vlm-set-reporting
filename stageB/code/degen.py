"""Degeneracy flags of draft 3.3 (list-level L3 checks), side-effect free. Same function as analyze_diag.degenerate."""
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']


def degenerate(recs, written_total, r, typed):
    n = len(recs); flags = []
    if typed and n and sum(1 for x in recs if x['type'] not in T5) / n > 0.1: flags.append('schema')
    if written_total and (n > 1.5 * written_total or n < 0.5 * written_total): flags.append('length')
    seen = set(); rep = 0
    for x in recs:
        k = (round(x['cx']), round(x['cy']))
        rep += k in seen; seen.add(k)
    if n and rep / n > 0.1: flags.append('repeat')
    if str(r.get('finish_reason')) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'): flags.append('trunc')
    return flags

"""Provider-independent FSR scoring. No inference, label-dependent repair, or NMS.
Records are distinct even when their coordinates coincide. Uses maximum-cardinality
matching on a declared correctness relation, not distance assignment then thresholding.
Python 3.10+; standard library only.
"""
from __future__ import annotations
from collections import Counter
from math import hypot, isfinite
from typing import Callable, Any

Record = dict[str, Any]

def maximum_matching(adj: list[list[bool]]) -> tuple[int, list[tuple[int, int]]]:
    """Return cardinality and (target index, prediction index) pairs."""
    n = len(adj)
    r = len(adj[0]) if n else 0
    if any(len(row) != r for row in adj):
        raise ValueError('Adjacency must be rectangular')
    # Iterative augmenting paths avoid recursion limits for dense reports.
    match_r = [-1] * r
    for start in range(n):
        queue = [start]
        seen_l = {start}
        seen_r: set[int] = set()
        parent_r: dict[int, int] = {}
        tail = None
        for u in queue:
            for v, edge in enumerate(adj[u]):
                if not edge or v in seen_r:
                    continue
                seen_r.add(v)
                parent_r[v] = u
                if match_r[v] == -1:
                    tail = v
                    break
                old_u = match_r[v]
                if old_u not in seen_l:
                    seen_l.add(old_u)
                    queue.append(old_u)
            if tail is not None:
                break
        if tail is not None:
            while tail is not None:
                u = parent_r[tail]
                previous = next((v for v, a in enumerate(match_r) if a == u), None)
                match_r[tail] = u
                tail = previous
    pairs = [(u, v) for v, u in enumerate(match_r) if u >= 0]
    return len(pairs), pairs

def point_relation(tau: float, require_label: bool = True) -> Callable[[Record, Record], bool]:
    if not isfinite(tau) or tau <= 0:
        raise ValueError('tau must be positive and finite')
    def valid(a: Record, b: Record) -> bool:
        if not b.get('valid', True):
            return False
        if require_label and a.get('label') != b.get('label'):
            return False
        try:
            ax, ay = map(float, a['point'])
            bx, by = map(float, b['point'])
        except (KeyError, ValueError, TypeError):
            return False
        return all(map(isfinite, (ax, ay, bx, by))) and hypot(ax-bx, ay-by) <= tau
    return valid

def box_relation(iou_threshold: float = 0.5, require_label: bool = True,
                 require_text: bool = False) -> Callable[[Record, Record], bool]:
    if not 0 < iou_threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    def valid(a: Record, b: Record) -> bool:
        if not b.get('valid', True):
            return False
        if require_label and a.get('label') != b.get('label'):
            return False
        if require_text and a.get('text') != b.get('text'):
            return False  # Any text normalization must happen in a frozen adapter.
        try:
            x1,y1,x2,y2 = map(float,a['box'])
            u1,v1,u2,v2 = map(float,b['box'])
        except (KeyError, ValueError, TypeError):
            return False
        if not all(map(isfinite,(x1,y1,x2,y2,u1,v1,u2,v2))):
            return False
        aa=max(0.,x2-x1)*max(0.,y2-y1)
        bb=max(0.,u2-u1)*max(0.,v2-v1)
        inter=max(0.,min(x2,u2)-max(x1,u1))*max(0.,min(y2,v2)-max(y1,v1))
        union=aa+bb-inter
        return aa>0 and bb>0 and union>0 and inter/union >= iou_threshold
    return valid

def score_adjacency(adj: list[list[bool]], target_owners: list[str],
                    prediction_owners: list[str]) -> dict[str, Any]:
    n,r = len(target_owners),len(prediction_owners)
    if len(adj)!=n or any(len(row)!=r for row in adj):
        raise ValueError('Adjacency dimensions do not agree with owners')
    tp,pairs=maximum_matching(adj)
    # maximum_matching cannot infer r for n=0; tp=0 remains correct.
    owners=sorted(set(target_owners)|set(prediction_owners))
    local={}
    sum_t=0
    for owner in owners:
        ii=[i for i,x in enumerate(target_owners) if x==owner]
        jj=[j for j,x in enumerate(prediction_owners) if x==owner]
        tj,_=maximum_matching([[adj[i][j] for j in jj] for i in ii])
        nj,rj=len(ii),len(jj)
        local[owner]={'n':nj,'r':rj,'tp':tj,'fp':rj-tj,'fn':nj-tj,
                      'd':nj+rj-2*tj,'ell':min(nj,rj)-tj}
        sum_t+=tj
    cross=[[edge and target_owners[i]!=prediction_owners[j]
            for j,edge in enumerate(row)] for i,row in enumerate(adj)]
    cross_max,_=maximum_matching(cross)
    eset=n+r-2*tp
    B=sum(v['d'] for v in local.values())
    chi=tp-sum_t
    assert B==eset+2*chi
    f1=2*tp/(n+r) if n+r else 1.
    return {'n':n,'r':r,'tp':tp,'fp':r-tp,'fn':n-tp,'set_error':eset,
            'f1':f1,'recall':tp/n if n else None,
            'exact_set':eset==0,'empty_false_alarm':bool(n==0 and r>0),
            'B':B,'chi':chi,'cross_matching_cap':cross_max,
            'normalized_set_error':eset/max(n,1),
            'f1_upper_surrogate':min(1.,B/(n+r)) if n+r else 0.,
            'local':local,'matched_pairs':pairs}

def score(targets: list[Record], predictions: list[Record],
          relation: Callable[[Record, Record], bool]) -> dict[str, Any]:
    if any('owner' not in x for x in targets+predictions):
        raise ValueError('Every record needs a frozen ownership assignment')
    result=score_adjacency([[relation(t,p) for p in predictions] for t in targets],
                           [t['owner'] for t in targets],[p['owner'] for p in predictions])
    keys=[p.get('record_id') for p in predictions if p.get('record_id') is not None]
    result['duplicate_record_ids']=sum(v-1 for v in Counter(keys).values())
    # This does NOT label all repeated physical objects: unmatched proximity is ambiguous.
    return result

def compare_components(joint: dict, split: dict, final: dict | None = None,
                       free: dict | None = None) -> dict[str, float]:
    """Exact per-image accounting; not an empirical proof of assumption Q."""
    if final is None: final=split
    if joint['n']!=split['n'] or joint['n']!=final['n']:
        raise ValueError('Reports must refer to the same target set')
    local_delta=split['B']-joint['B']
    cross_delta=2*(joint['chi']-split['chi'])
    assembly_delta=final['set_error']-split['set_error']
    format_delta=0 if free is None else joint['set_error']-free['set_error']
    baseline=joint if free is None else free
    actual=final['set_error']-baseline['set_error']
    rhs=local_delta+cross_delta+assembly_delta+format_delta
    assert actual==rhs
    return {'actual_delta':actual,'local_delta':local_delta,'cross_delta':cross_delta,
            'assembly_delta':assembly_delta,'format_delta':format_delta,
            'identity_residual':actual-rhs,
            'conservative_upper':format_delta+local_delta+2*joint['chi']+max(0,assembly_delta)}

"""Paired source-level inference for completed canonical runs.
Bootstrap intervals are approximate, NOT finite-sample certifications of Q.
Averages repeats per image and images per source (equal-source estimand).
No missing-pair deletion; specify one model/domain per invocation.
"""
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean,stdev,NormalDist
import argparse,json,math,random

def quantile(x,p):
    xs=sorted(x); k=(len(xs)-1)*p; lo=int(k); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(k-lo)

def source_values(rows,condition,metric):
    by_image=defaultdict(list); owner={}
    for r in rows:
        if r['condition']!=condition: continue
        if r.get(metric) is None: raise ValueError(f'Undefined {metric}; declare stratum before analysis')
        image=r['sample_id']; source=r['source_id']
        if image in owner and owner[image]!=source: raise ValueError('Image/source mapping changed')
        owner[image]=source; by_image[image].append(float(r[metric]))
    sources=defaultdict(list)
    for image,values in by_image.items(): sources[owner[image]].append(mean(values))
    return {source:mean(values) for source,values in sources.items()}

def paired(rows,a,b,metric,seed=1,draws=10000,alpha=.05,family=1,margin=0.):
    ka=Counter((r['sample_id'],r['source_id'],r.get('repeat')) for r in rows if r['condition']==a)
    kb=Counter((r['sample_id'],r['source_id'],r.get('repeat')) for r in rows if r['condition']==b)
    if ka!=kb: raise ValueError('Missing paired image/repeat runs; add explicit failed outcomes rather than drop pairs')
    va,vb=source_values(rows,a,metric),source_values(rows,b,metric)
    if set(va)!=set(vb) or not va: raise ValueError('Missing paired sources; add failed runs, do not drop pairs')
    ids=sorted(va); d=[va[i]-vb[i] for i in ids]; n=len(d)
    if n<2: raise ValueError('At least two independent sources required')
    rng=random.Random(seed)
    boot=[mean(d[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    lo,hi=quantile(boot,alpha/(2*family)),quantile(boot,1-alpha/(2*family))
    # Metric must be lower-is-better for this label. F1 deltas should be interpreted separately.
    decision='supported_noninferiority' if hi<=margin else ('contradicted' if lo>margin else 'inconclusive')
    return {'comparison':f'{a} - {b}','metric':metric,'independent_sources':n,'delta':mean(d),
            'paired_sd':stdev(d),'interval':[lo,hi],'margin':margin,'decision_lower_is_better':decision,
            'alpha':alpha,'family_size':family,'method':'paired source bootstrap, Bonferroni quantiles; approximate',
            'replicate_draws':draws,'source_ids':ids}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--a',required=True); p.add_argument('--b',required=True)
    p.add_argument('--metric',default='set_error'); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True); p.add_argument('--family',type=int,default=1)
    p.add_argument('--margin',type=float,default=0); p.add_argument('--seed',type=int,default=20261008)
    a=p.parse_args()
    rows=[r for r in (json.loads(s) for s in a.scores.read_text().splitlines() if s.strip()) if r['model_key']==a.model]
    ans=paired(rows,a.a,a.b,a.metric,a.seed,family=a.family,margin=a.margin)
    a.out.write_text(json.dumps(ans,indent=2)); print(json.dumps(ans,indent=2))
if __name__=='__main__': main()

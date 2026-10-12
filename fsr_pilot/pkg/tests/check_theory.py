"""Finite checks for FSR v3 only. Not model experiments or formal proof certification."""
import sys,json,random,math,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fsr_metrics import score_adjacency,maximum_matching,compare_components,score,point_relation
from fsr_ledger import Region,Ledger,decode_ndjson
from freeze_plan import missing
rng=random.Random(20261008)
counts={}

def brute(adj):
    n=len(adj); r=len(adj[0]) if n else 0
    def f(i,used):
        if i==n: return 0
        best=f(i+1,used)
        for j in range(r):
            if adj[i][j] and j not in used: best=max(best,1+f(i+1,used|{j}))
        return best
    return f(0,set())

for _ in range(1000):
    n,r=rng.randrange(6),rng.randrange(6)
    adj=[[rng.random()<.4 for j in range(r)] for i in range(n)]
    assert maximum_matching(adj)[0]==brute(adj)
counts['matching_vs_bruteforce']=1000

q_cases=0
for _ in range(6000):
    n=rng.randrange(10); owners=[str(rng.randrange(4)) for i in range(n)]
    def trial():
        r=rng.randrange(13); po=[str(rng.randrange(4)) for i in range(r)]
        adj=[[rng.random()<.25 for j in range(r)] for i in range(n)]
        return score_adjacency(adj,owners,po)
    F,G,L,V=trial(),trial(),trial(),trial()
    comp=compare_components(G,L,V,F)
    assert comp['actual_delta']<=comp['conservative_upper']
    for x in (F,G,L,V):
        assert 0<=x['chi']<=x['cross_matching_cap']
        assert 1-x['f1']<=x['f1_upper_surrogate']+1e-12
    if L['B']<=G['B']:
        q_cases+=1
        assert V['set_error']<=G['set_error']+2*G['chi']+max(0,V['set_error']-L['set_error'])
counts['central_identity_boundary_and_contract_checks']=6000
counts['subset_with_pathwise_Q']=q_cases

# Cross-region rescue counterexample: local Q without a boundary term is insufficient.
G=score_adjacency([[False,True],[True,False]],['a','b'],['a','b'])
L=score_adjacency([[False,False],[False,False]],['a','b'],['a','b'])
assert G['B']==L['B']==4 and G['set_error']==0 and L['set_error']==4 and G['chi']==2
counts['boundary_counterexample']=1
# Set-error improvement is not F1 improvement.
x=score_adjacency([[j==i for j in range(18)] for i in range(10)],['a']*10,['a']*18)
y=score_adjacency([[j==i for j in range(4)] for i in range(10)],['a']*10,['a']*4)
assert y['set_error']<x['set_error'] and y['f1']<x['f1']
counts['F1_counterexample']=1

for _ in range(3000):
    n,r=rng.randrange(10),rng.randrange(1,12)
    adj=[[rng.random()<.4 for _ in range(r)] for _ in range(n)]
    e=n+r-2*maximum_matching(adj)[0]
    col=rng.randrange(r)
    removed=[[x for j,x in enumerate(row) if j!=col] for row in adj]
    er=n+r-1-2*maximum_matching(removed)[0]
    assert abs(e-er)<=1
    changed=[row[:] for row in adj]
    for row in changed: row[col]=rng.random()<.4
    ec=n+r-2*maximum_matching(changed)[0]
    assert abs(ec-e)<=2
counts['edit_sensitivity_checks']=3000

for _ in range(10000):
    y=rng.randrange(2); p0,p1=rng.random(),rng.random(); gamma=rng.random()*.5+.00001
    m0,m1=(2*y-1)*(p0-.5),(2*y-1)*(p1-.5)
    d=max(0,m0-m1)
    witness=min(1.,int(0<=m0<gamma)+d/gamma)
    regress=int(int(p0>=.5)==y and int(p1>=.5)!=y)
    assert regress<=witness+1e-12
counts['margin_witness_checks']=10000

for _ in range(1000):
    T=rng.randrange(2,200); delta=rng.random()*12
    differences=[rng.uniform(-delta,delta) for _ in range(T-1)]
    a=1/(1+sum(math.exp(-d) for d in differences))
    upper=1/(1+(T-1)*math.exp(-delta))
    assert a<=upper+1e-12
counts['softmax_mass_bound_checks']=1000

reg={'R001':Region('R001',0,0,100,100),'R002':Region('R002',100,0,100,100)}
ledger=Ledger(reg,'h')
row='{"region_id":"R001","status":"found","objects":[{"u":0.5,"v":0.5}]}'
ledger.apply('a',['R001','R002'],row,'h')
assert ledger.report()['count']==1 and ledger.unresolved()==['R002']
assert ledger.apply('a',['R001','R002'],row,'h')['idempotent_replay']
ledger.apply('b',['R002'],'{"region_id":"R002","status":"unknown","objects":[{"u":0.4,"v":0.4}]}','h')
assert ledger.report()['count']==2 and ledger.unresolved()==['R002']
ledger.apply('c',['R002'],'{"region_id":"R002","status":"empty","objects":[]}','h')
assert ledger.report()['count']==1 and not ledger.unresolved()
try: ledger.apply('d',['R002'],row,'wrong'); raise AssertionError('hash check absent')
except ValueError: pass
bad=decode_ndjson('{broken\n'+row+'\n'+row,['R001'],reg,'bad')
assert bad.audit['malformed_lines']==1 and bad.audit['duplicate_region_rows']==1
assert bad.rows['R001']['status']=='unknown' and len(bad.rows['R001']['records'])==2
invalid=decode_ndjson('{"region_id":"R002","status":"found","objects":[{"u":2,"v":0.4}]}',['R002'],reg,'invalid')
assert invalid.audit['invalid_objects']==1 and invalid.rows['R002']['status']=='unknown'
assert missing({}) and not missing({k:0 for k in __import__('freeze_plan').REQUIRED})
counts['controller_scenarios']=8

out={'version':'FSR-v3','seed':20261008,'checks':counts,'all_assertions_passed':True,
     'status':'finite algebra and implementation tests only; no model inference, no training, no data replication',
     'counterexamples':{'cross_region':{'joint':G,'split':L},
                       'set_error_not_F1':{'joint_F1':x['f1'],'alternative_F1':y if isinstance(y,dict) else 8/14}}}
# Keep artifact summary compact; y was reused in margin checks.
out['counterexamples']['set_error_not_F1']={'joint_F1':20/28,'alternative_F1':8/14,'joint_set_error':8,'alternative_set_error':6}
path=Path(__file__).resolve().parents[1]/'checks'/'v3_check_results.json'
path.write_text(json.dumps(out,indent=2))
print(json.dumps({'checks':counts,'all_assertions_passed':True,'result':str(path)},indent=2))

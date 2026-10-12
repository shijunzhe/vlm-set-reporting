"""Minimal FSR responsibility ledger and frozen, label-free point decoder.
No API client, no visual inference, no confidence-based semantic filtering.
The decoder recovers complete NDJSON rows, never guesses scales/coordinates.
Malformed rows remain in the audit and leave their obligations unresolved.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json, math

@dataclass(frozen=True)
class Region:
    region_id: str
    x0: float
    y0: float
    width: float
    height: float
    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x0,self.y0,self.width,self.height)):
            raise ValueError('Nonfinite region')
        if self.width<=0 or self.height<=0: raise ValueError('Nonpositive region')

@dataclass
class Decoded:
    rows: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def decode_ndjson(raw: str, requested: list[str], registry: dict[str,Region],
                  request_id: str, label: str='target') -> Decoded:
    if len(set(requested))!=len(requested) or not set(requested)<=registry.keys():
        raise ValueError('Requested IDs must be unique and registered')
    rows:dict[str,dict[str,Any]]={}
    audit={'malformed_lines':0,'unrequested_rows':0,'invalid_objects':0,
           'duplicate_region_rows':0,'contradictory_rows':0,'missing_ids':[],
           'raw_nonempty_lines':0}
    # A code fence is permitted only as a standalone boundary, never by content guessing.
    lines=raw.strip().splitlines()
    if lines and lines[0].strip() in ('```','```json','```jsonl'):
        lines=lines[1:]
    if lines and lines[-1].strip()=='```': lines=lines[:-1]
    for row_index,line in enumerate(lines):
        if not line.strip(): continue
        audit['raw_nonempty_lines']+=1
        try: obj=json.loads(line)
        except (ValueError,TypeError):
            audit['malformed_lines']+=1; continue
        if not isinstance(obj,dict) or not isinstance(obj.get('region_id'),str):
            audit['malformed_lines']+=1; continue
        rid=obj['region_id']
        if rid not in requested:
            audit['unrequested_rows']+=1; continue
        reg=registry[rid]
        status=obj.get('status')
        items=obj.get('objects')
        if not isinstance(items,list):
            audit['malformed_lines']+=1; continue
        records=[]
        bad=False
        for k,item in enumerate(items):
            if not isinstance(item,dict):
                audit['invalid_objects']+=1; bad=True; continue
            try:
                u,v=float(item['u']),float(item['v'])
                ok=math.isfinite(u) and math.isfinite(v) and 0<=u<1 and 0<=v<1
            except (KeyError,ValueError,TypeError): ok=False
            if not ok:
                audit['invalid_objects']+=1; bad=True; continue
            records.append({'record_id':f'{request_id}:{row_index}:{k}',
                            'owner':rid,'label':item.get('label',label),
                            'point':[reg.x0+reg.width*u,reg.y0+reg.height*v],
                            'valid':True})
        compatible=(status=='empty' and len(items)==0) or (status=='found' and len(items)>0) or status=='unknown'
        if not compatible or bad:
            audit['contradictory_rows']+=1; status='unknown'
        if rid in rows:
            audit['duplicate_region_rows']+=1
            # Preserve recoverable records for raw scoring; mark unresolved.
            rows[rid]['records'].extend(records)
            rows[rid]['status']='unknown'
        else:
            rows[rid]={'status':status,'records':records}
    audit['missing_ids']=[rid for rid in requested if rid not in rows]
    return Decoded(rows,audit)

@dataclass
class Ledger:
    registry: dict[str,Region]
    evidence_hash: str
    states: dict[str,str] = field(init=False)
    records: dict[str,list[dict[str,Any]]] = field(init=False)
    versions: dict[str,int] = field(init=False)
    processed_requests: set[str] = field(default_factory=set)
    events: list[dict[str,Any]] = field(default_factory=list)
    def __post_init__(self):
        if not self.registry: raise ValueError('Need at least one responsibility')
        self.states={k:'pending' for k in self.registry}
        self.records={k:[] for k in self.registry}
        self.versions={k:0 for k in self.registry}
    def apply(self, request_id:str, requested:list[str], raw:str,
              evidence_hash:str, allow_review:bool=False) -> dict[str,Any]:
        if evidence_hash!=self.evidence_hash:
            raise ValueError('Evidence changed: create a new treatment/run')
        if request_id in self.processed_requests:
            return {'idempotent_replay':True}
        if not allow_review and any(self.states[x]=='resolved' for x in requested):
            raise ValueError('Resolved regions require an explicit review condition')
        decoded=decode_ndjson(raw,requested,self.registry,request_id)
        self.processed_requests.add(request_id)
        for rid in requested:
            row=decoded.rows.get(rid)
            if row is None:
                if self.states[rid]!='resolved': self.states[rid]='unresolved'
                continue  # Invalid attempts do not silently erase prior valid partial output.
            self.versions[rid]+=1
            # Atomic replacement, not append. A retry can lose true records; audit it.
            self.records[rid]=row['records']
            self.states[rid]='resolved' if row['status'] in ('empty','found') else 'unresolved'
        self.events.append({'request_id':request_id,'requested':requested,
                            'audit':decoded.audit,'versions':dict(self.versions),
                            'states':dict(self.states)})
        return decoded.audit
    def unresolved(self)->list[str]:
        return [r for r,s in self.states.items() if s!='resolved']
    def report(self)->dict[str,Any]:
        records=[p for rid in self.registry for p in self.records[rid]]
        return {'predictions':records,'count':len(records),
                'coverage_process_complete':not self.unresolved(),
                'unresolved_regions':self.unresolved(),'states':dict(self.states)}

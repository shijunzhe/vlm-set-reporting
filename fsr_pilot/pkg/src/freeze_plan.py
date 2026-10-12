"""Refuse confirmatory freeze until actual models, splits, budgets, estimands are set.
This creates a LOCAL dated/hash manifest, not a public preregistration.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,re
REQUIRED=['model_snapshots','provider_parameters','source_split_manifest_sha256',
          'confirmatory_sample_sizes','fixed_batch_size','decode_version',
          'primary_contrasts','primary_metric','q_noninferiority_margin',
          'family_error_control','total_budget_rule','truncation_rule','failure_rule',
          'replicates','frozen_prompts_sha256','annotation_audit_rule']
def missing(config):
    return [k for k in REQUIRED if k not in config or config[k] is None or config[k]=='' or config[k]==[] or config[k]=={}]
def validate(config):
    errors=missing(config)
    if errors: return errors
    models=config['model_snapshots']
    if not isinstance(models,dict) or any(not isinstance(v,str) or not v.strip() or v.startswith('TO_') for v in models.values()):
        errors.append('model_snapshots must map model keys to actual nonblank snapshot IDs')
    for field in ('source_split_manifest_sha256','frozen_prompts_sha256'):
        if not isinstance(config[field],str) or not re.fullmatch(r'[0-9a-fA-F]{64}',config[field]):
            errors.append(field+' must be an actual SHA-256 hex digest')
    sizes=config['confirmatory_sample_sizes']
    if not isinstance(sizes,dict) or any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in sizes.values()):
        errors.append('confirmatory_sample_sizes must map named cohorts to positive integers')
    if config['fixed_batch_size'] not in (1,4,16): errors.append('fixed_batch_size must be 1, 4, or 16 for this frozen core')
    margin=config['q_noninferiority_margin']
    if not isinstance(margin,(int,float)) or margin<0: errors.append('q_noninferiority_margin must be a nonnegative number')
    if not isinstance(config['replicates'],int) or config['replicates']<=0: errors.append('replicates must be a positive integer')
    if not isinstance(config['provider_parameters'],dict): errors.append('provider_parameters must be a mapping')
    if not isinstance(config['primary_contrasts'],list): errors.append('primary_contrasts must be a list')
    return errors

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('freeze_manifest.json')); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); bad=validate(cfg)
    if bad:
        print('Not ready for confirmatory freeze. Fill: '+', '.join(bad)); sys.exit(2)
    ans={'created_utc':datetime.now(timezone.utc).isoformat(),
         'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
         'status':'locally_frozen_not_publicly_registered','configuration':cfg}
    a.out.write_text(json.dumps(ans,indent=2,ensure_ascii=False)); print(str(a.out))
if __name__=='__main__': main()

"""Probe Claude 5.5 thinking behaviour under the configurations used in paper 2 / FSR and the candidates for E12.
Keys are loaded from config.env and never printed. 10 calls."""
import base64, io, json, sys, time
sys.path.insert(0, '/home/claude/fsr_pilot')
import run_pilot as RP
from PIL import Image
G = RP.G
E12 = '/home/claude/fsr_v51/e12/'
q = [json.loads(l) for l in open(E12 + 'runs_e12/requests.jsonl') if '"J-local"' in l][0]
ims = [Image.open(E12 + q['reference_path']).convert('RGB'), Image.open(E12 + q['image_path']).convert('RGB')]
def png(im): b = io.BytesIO(); im.save(b, format='PNG'); return base64.standard_b64encode(b.getvalue()).decode()
content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': png(im)}} for im in ims] + [{'type': 'text', 'text': q['user']}]
CFG = {
 'sonnet_omitted (paper-2/FSR baseline)': ('claude-sonnet-5-5', {}),
 'sonnet_adaptive_high (paper-2 sonnet55t)': ('claude-sonnet-5-5', {'thinking': {'type': 'adaptive'}, 'extra_body': {'output_config': {'effort': 'high'}}}),
 'sonnet_between_tools': ('claude-sonnet-5-5', {'thinking': {'type': 'between_tools'}}),
 'opus_omitted (paper-2/FSR baseline)': ('claude-opus-5-5', {}),
 'opus_effort_low': ('claude-opus-5-5', {'extra_body': {'output_config': {'effort': 'low'}}}),
}
cl = G._client('anthropic'); out = []
for name, (model, kw) in CFG.items():
    for rep in range(2):
        t = time.time()
        try:
            r = cl.messages.create(model=model, max_tokens=16384, system=q['system'], messages=[{'role': 'user', 'content': content}], **kw)
            types = [b.type for b in r.content]
            text = ''.join(b.text for b in r.content if b.type == 'text')
            rec = dict(config=name, rep=rep, model=r.model, block_types={t_: types.count(t_) for t_ in set(types)},
                       output_tokens=r.usage.output_tokens, text_chars=len(text), stop=r.stop_reason, secs=round(time.time() - t, 1),
                       n_rows=sum(1 for l in text.splitlines() if l.strip().startswith('{')))
        except Exception as e:
            rec = dict(config=name, rep=rep, error=f'{type(e).__name__}: {str(e)[:300]}')
        print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open('/home/claude/fsr_pilot/probe/thinking_probe_results.json', 'w'), indent=1)

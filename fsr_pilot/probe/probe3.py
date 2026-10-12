import base64, io, json, sys
sys.path.insert(0, '/home/claude/fsr_pilot')
import run_v2 as RV
from PIL import Image
RP = RV.RP; G = RP.G
reqs = {json.loads(l)['request_id']: json.loads(l) for l in open('runs_v2/requests.jsonl')}
out = []
for rid in ['S1:v2_n064_0:opus55:R014', 'C1:v2_n192_4:opus55:R004']:
    q = reqs[rid]; system, user = RV.prompts(q)
    ims = [Image.open(RV.ROOT / q['reference_path']).convert('RGB'), Image.open(RV.ROOT / q['image_path']).convert('RGB')]
    def png(im): b = io.BytesIO(); im.save(b, format='PNG'); return base64.standard_b64encode(b.getvalue()).decode()
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': png(im)}} for im in ims] + [{'type': 'text', 'text': user}]
    for name, model, kw in [('opus_omitted', 'claude-opus-5-5', {}), ('opus_effort_low', 'claude-opus-5-5', {'extra_body': {'output_config': {'effort': 'low'}}}),
                            ('sonnet_omitted', 'claude-sonnet-5-5', {}), ('sonnet_between_tools', 'claude-sonnet-5-5', {'thinking': {'type': 'between_tools'}})]:
        r = G._client('anthropic').messages.create(model=model, max_tokens=16384, system=system, messages=[{'role': 'user', 'content': content}], **kw)
        text = ''.join(b.text for b in r.content if b.type == 'text')
        rec = dict(request=rid, config=name, blocks=[b.type for b in r.content], out=r.usage.output_tokens,
                   thinking=getattr(r.usage.output_tokens_details, 'thinking_tokens', None), chars=len(text))
        print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open('probe/thinking_probe3.json', 'w'), indent=1)

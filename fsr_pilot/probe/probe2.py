import base64, io, json, sys, hashlib
sys.path.insert(0, '/home/claude/fsr_pilot')
import run_pilot as RP
from PIL import Image
G = RP.G; E12 = '/home/claude/fsr_v51/e12/'
q = [json.loads(l) for l in open(E12 + 'runs_e12/requests.jsonl') if '"J-local"' in l][0]
ims = [Image.open(E12 + q['reference_path']).convert('RGB'), Image.open(E12 + q['image_path']).convert('RGB')]
def png(im): b = io.BytesIO(); im.save(b, format='PNG'); return base64.standard_b64encode(b.getvalue()).decode()
content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': png(im)}} for im in ims] + [{'type': 'text', 'text': q['user']}]
cl = G._client('anthropic')
for model, kw in [('claude-sonnet-5-5', {}), ('claude-opus-5-5', {'extra_body': {'output_config': {'effort': 'low'}}})]:
    r = cl.messages.create(model=model, max_tokens=16384, system=q['system'], messages=[{'role': 'user', 'content': content}], **kw)
    text = ''.join(b.text for b in r.content if b.type == 'text')
    print(model, r.id, r.usage, hashlib.sha256(text.encode()).hexdigest()[:12]); print(text[:300])
# a trivially different prompt to test caching/mocking
r = cl.messages.create(model='claude-sonnet-5-5', max_tokens=200, messages=[{'role': 'user', 'content': 'Reply with the single word: banana'}])
print('control', r.id, r.usage, [b.type for b in r.content], ''.join(getattr(b,'text','') for b in r.content))
print(q['sample_id'], q['request_id'])

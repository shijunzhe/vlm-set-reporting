#!/usr/bin/env python3
"""
Experiment G: new-generation models. Direct API calls (no release wrappers), no reasoning where the API
allows it ("Setup A: pure vision", as for the first generation).

Vendors (key -> model id, options):
  sonnet55    claude-sonnet-5-5                       Anthropic, no extended thinking
  opus55      claude-opus-5-5                         Anthropic, no extended thinking (A1 only)
  gpt55       gpt-5.5, detail=auto                    OpenAI, lowest reasoning effort accepted
  gpt56       gpt-5.6-sol, detail=auto
  gpt56orig   gpt-5.6-sol, detail=original            (within-model budget manipulation)
  gem38       gemini-3.8-flash, media_resolution default
  gem38uh     gemini-3.8-flash, media_resolution ultra_high (within-model budget manipulation)
call(vendor, images, system, user) -> (raw_text, usage), same contract as arr_common.call.
"""
import base64, io, os, threading
from PIL import Image

VENDORS = {
    'sonnet55': ('anthropic', 'claude-sonnet-5-5', {}),
    'opus55': ('anthropic', 'claude-opus-5-5', {}),
    'gpt55': ('openai', 'gpt-5.5', {'detail': 'auto'}),
    'gpt56': ('openai', 'gpt-5.6-sol', {'detail': 'auto'}),
    'gpt56orig': ('openai', 'gpt-5.6-sol', {'detail': 'original'}),
    'gem38': ('gemini', 'gemini-3.8-flash', {}),
    'gpt61': ('openai', 'gpt-6.1-sol', {'detail': 'auto'}),
    'gem38uh': ('gemini', 'gemini-3.8-flash', {'part_res': 'MEDIA_RESOLUTION_ULTRA_HIGH'}),
    # reasoning-enabled variants (experiment X3)
    'sonnet55t': ('anthropic', 'claude-sonnet-5-5', {'thinking': 16000}),
    'gpt56r': ('openai', 'gpt-5.6-sol', {'detail': 'auto', 'effort': 'high'}),
    'gem38t': ('gemini', 'gemini-3.8-flash', {'think': {'thinking_budget': -1}}),
    # X9 confound controls: reasoning effort matched to the zoom agent (GPT medium, Claude adaptive/high, Gemini dynamic)
    'gpt56m': ('openai', 'gpt-5.6-sol', {'detail': 'auto', 'effort': 'medium'}),
    'gpt56origm': ('openai', 'gpt-5.6-sol', {'detail': 'original', 'effort': 'medium'}),
    # T1 within-model token budget at fixed pixels (Gemini 3.8 media resolution)
    'gem38lo': ('gemini', 'gemini-3.8-flash', {'part_res': 'MEDIA_RESOLUTION_LOW'}),
}
MAX_TOKENS = 8192
_lock = threading.Lock(); _clients = {}
REASONING = {'gpt-6.1-sol': 'low'}   # filled by probe(): lowest accepted reasoning setting per provider/model (GPT-6.1 Sol accepts low at minimum)


def _png(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return b.getvalue()


def _client(kind):
    with _lock:
        if kind not in _clients:
            if kind == 'anthropic':
                import anthropic; _clients[kind] = anthropic.Anthropic(timeout=600)
            elif kind == 'openai':
                import openai; _clients[kind] = openai.OpenAI(timeout=600)
            else:
                from google import genai; from google.genai import types as _t; _clients[kind] = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=_t.HttpOptions(timeout=180000))
        return _clients[kind]


def call(vendor, images, system, user):
    kind, model, opt = VENDORS[vendor]
    if kind == 'anthropic':
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                 'data': base64.standard_b64encode(_png(im)).decode()}} for im in images]
        content.append({'type': 'text', 'text': user})
        kw = dict(model=model, max_tokens=MAX_TOKENS, system=system, messages=[{'role': 'user', 'content': content}])
        if 'thinking' in opt:
            kw['thinking'] = {'type': 'adaptive'}; kw['extra_body'] = {'output_config': {'effort': 'high'}}; kw['max_tokens'] = MAX_TOKENS + opt['thinking']
        r = _client(kind).messages.create(**kw)
        raw = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text').strip()
        return raw, {'input_tokens': r.usage.input_tokens, 'output_tokens': r.usage.output_tokens,
                     'actual_model': r.model, 'stop_reason': r.stop_reason}
    if kind == 'openai':
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(_png(im)).decode(),
                                                       'detail': opt['detail']}} for im in images]
        content.append({'type': 'text', 'text': user})
        kw = dict(model=model, max_completion_tokens=MAX_TOKENS,
                  messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])
        eff = opt.get('effort', REASONING.get(model, 'none'))
        if 'effort' in opt:
            kw['max_completion_tokens'] = 64000
        if eff:
            kw['reasoning_effort'] = eff
        r = _client(kind).chat.completions.create(**kw)
        ch = r.choices[0]
        u = {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
             'actual_model': r.model, 'stop_reason': ch.finish_reason, 'reasoning_effort': eff}
        det = getattr(r.usage, 'completion_tokens_details', None)
        if det is not None and getattr(det, 'reasoning_tokens', None):
            u['reasoning_tokens'] = det.reasoning_tokens
        return (ch.message.content or '').strip(), u
    from google.genai import types
    if 'part_res' in opt:   # Gemini 3: per-part resolution (the request-level enum has no ULTRA_HIGH)
        parts = [types.Part(inline_data=types.Blob(data=_png(im), mime_type='image/png'),
                            media_resolution=types.PartMediaResolution(level=opt['part_res'])) for im in images] + [user]
    else:
        parts = [types.Part.from_bytes(data=_png(im), mime_type='image/png') for im in images] + [user]
    th = opt.get('think', REASONING.get(model, {'thinking_budget': 0}))
    cfg = dict(system_instruction=system, max_output_tokens=MAX_TOKENS if 'think' not in opt else 65536, thinking_config=types.ThinkingConfig(**th))
    r = _client(kind).models.generate_content(model=model, contents=parts, config=types.GenerateContentConfig(**cfg))
    raw = ''.join(p.text for p in r.candidates[0].content.parts if getattr(p, 'text', None)) if r.candidates and r.candidates[0].content else ''
    um = r.usage_metadata
    return raw.strip(), {'input_tokens': um.prompt_token_count, 'output_tokens': um.candidates_token_count,
                         'thinking_tokens': getattr(um, 'thoughts_token_count', None), 'actual_model': model,
                         'stop_reason': str(r.candidates[0].finish_reason) if r.candidates else None, 'thinking': th}

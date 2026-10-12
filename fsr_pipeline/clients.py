"""Clients for the pipeline. ReplayClient serves stored responses and never calls a model."""
from pipeline import Call, call_cost


class ReplayClient:
    """Serves the stored response that the adapter maps a Request to (adapter.lookup).
    overrides: {(task, item, model, kind, rep, view): response dict or 'transport_failed'} for boundary cases;
    a stored dict needs keys raw_text and optionally finish_reason and usage.
    A request with no stored response returns status 'unavailable' (for example a follow-up that was never sent)."""

    def __init__(self, adapter, overrides=None):
        self.adapter, self.overrides, self.log = adapter, overrides or {}, []

    def key(self, req):
        return (req.task, req.item, req.model, req.kind, req.rep, req.view)

    def call(self, req):
        k = self.key(req)
        if k in self.overrides:
            r = self.overrides[k]
            if r == 'transport_failed': c = Call(req, 'transport_failed')
            else: c = Call(req, 'completed', r.get('raw_text', ''), r.get('finish_reason'), call_cost(req.model, r.get('usage')))
        else:
            r = self.adapter.lookup(req)
            if r is None: c = Call(req, 'unavailable')
            elif r.get('status') != 'completed': c = Call(req, 'transport_failed')
            else: c = Call(req, 'completed', r.get('raw_text') or '', r.get('finish_reason'), call_cost(req.model, r.get('usage')))
        self.log.append(c); return c


class LiveClient:
    """Hook for live use: send Request objects to a model API and return Call objects. Not implemented here, because
    this release makes no model calls; the request builders used in the experiments are in stageB/code and gui/code."""

    def call(self, req):
        raise NotImplementedError('LiveClient is a hook; this package replays stored responses only')

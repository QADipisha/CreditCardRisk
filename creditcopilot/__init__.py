"""CreditCopilot package."""

import json, numpy as np
_orig_default = json.JSONEncoder.default
def _np_default(self, o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return _orig_default(self, o)
json.JSONEncoder.default = _np_default

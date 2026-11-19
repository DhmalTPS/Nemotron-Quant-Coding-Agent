"""Minimal OpenAI-compatible mock of the House route for plumbing tests (no real model).

Scenario for t1-EXAMPLE-bs-greeks-pde: the first program writes the WRONG filename with an index,
the repair returns a SEARCH/REPLACE patch, the review passes. Usage:
    python dev/mock_house.py 8765
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BUGGY = '''```python
import numpy as np, pandas as pd
from scipy.stats import norm

df = pd.read_parquet("/app/data/options.parquet")
S, K, T, r, s = (df[c].to_numpy(float) for c in ["S", "K", "T", "r", "sigma"])
call = df["option_type"].to_numpy() == "call"
d1 = (np.log(S / K) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
price = np.where(call, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1)
gamma = norm.pdf(d1) / (S * s * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
th = -S * norm.pdf(d1) * s / (2 * np.sqrt(T)) + np.where(call, -r * K * np.exp(-r * T) * norm.cdf(d2), r * K * np.exp(-r * T) * norm.cdf(-d2))
out = pd.DataFrame({"option_id": df["option_id"], "price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": th / 365})
out.to_parquet("/app/output/result.parquet", index=True)
print("rows", len(out))
```'''

PATCH = '''The file name is wrong and the index is written.
<<<<<<< SEARCH
out.to_parquet("/app/output/result.parquet", index=True)
=======
out.to_parquet("/app/output/results.parquet", index=False)
>>>>>>> REPLACE'''


class H(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(403)
            self.end_headers()
            return
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            self.send_response(401)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if REJECT_SEED and "seed" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"unknown field seed"}')
            return
        last = body["messages"][-1]["content"]
        if "Write an implementation plan" in last:
            content, reasoning = "## 1. Deliverables\n- results.parquet ...\n" * 5, "thinking about BS"
        elif "Now write the COMPLETE program" in last:
            content, reasoning = BUGGY, ""
        elif "SEARCH/REPLACE edits against the CURRENT PROGRAM" in last:
            content, reasoning = PATCH, "diagnosing"
        elif "strict senior quant reviewer" in last:
            content, reasoning = "VERDICT: PASS", "review"
        else:
            content, reasoning = "?", ""
        resp = {
            "id": "x", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}],
            "usage": {"prompt_tokens": len(last) // 4, "completion_tokens": len(content) // 4},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # noqa: ANN002
        pass


REJECT_SEED = len(sys.argv) > 2 and sys.argv[2] == "reject-seed"

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()

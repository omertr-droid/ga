"""Sam3Client — thin HTTP client to the MedSAM3 service running on Google Colab (Phase 2).

The reader stays torch-free: the model lives in Colab (GPU + gated SAM3 weights) behind a cloudflared
tunnel; this client just POSTs a B-scan PNG + a concept phrase and gets a mask PNG back. Stdlib only
(urllib) so the reader gains no dependency. The service contract (see segmenter_service/serve.py):

    GET  /health                       -> {"ok": true, "device": "...", "model": "..."}
    POST /segment?concept=..&threshold=..   body = B-scan PNG bytes
                                       -> mask PNG bytes (grayscale 0/255, or RGBA w/ alpha)
"""
import json
import urllib.error
import urllib.parse
import urllib.request


class SegmenterError(RuntimeError):
    pass


class Sam3Client:
    def __init__(self, base_url: str):
        self.base = (base_url or "").rstrip("/")

    def health(self, timeout=8) -> dict:
        if not self.base:
            raise SegmenterError("no endpoint set")
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SegmenterError(f"health check failed: {e}")

    def segment(self, png_bytes: bytes, concept: str, threshold: float = 0.5, timeout=180) -> bytes:
        """POST a B-scan PNG + concept -> mask PNG bytes."""
        if not self.base:
            raise SegmenterError("no endpoint set")
        q = urllib.parse.urlencode({"concept": concept or "", "threshold": float(threshold)})
        req = urllib.request.Request(self.base + "/segment?" + q, data=bytes(png_bytes),
                                     method="POST", headers={"Content-Type": "image/png"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            raise SegmenterError(f"segment failed (HTTP {e.code}) {detail}")
        except (urllib.error.URLError, OSError, ValueError) as e:   # ValueError = malformed URL (no scheme)
            raise SegmenterError(f"segment failed: {e}")


class Sam2Client:
    """HTTP client to the SAM2 box/point service (segmenter_service/sam2_serve.py) — the annotation
    accelerator. POSTs a B-scan PNG + a box (or points) and gets a mask PNG back. Stdlib only."""
    def __init__(self, base_url: str):
        self.base = (base_url or "").rstrip("/")

    def health(self, timeout=8) -> dict:
        if not self.base:
            raise SegmenterError("no endpoint set")
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SegmenterError(f"health check failed: {e}")

    def segment(self, png_bytes: bytes, box=None, points=None, timeout=120) -> bytes:
        if not self.base:
            raise SegmenterError("no endpoint set")
        q = {}
        if box:
            q["box"] = ",".join(str(float(v)) for v in box)
        if points:
            q["points"] = ";".join(",".join(str(float(v)) for v in p) for p in points)
        url = self.base + "/segment" + ("?" + urllib.parse.urlencode(q) if q else "")
        req = urllib.request.Request(url, data=bytes(png_bytes), method="POST",
                                     headers={"Content-Type": "image/png"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            raise SegmenterError(f"sam2 failed (HTTP {e.code}) {detail}")
        except (urllib.error.URLError, OSError) as e:
            raise SegmenterError(f"sam2 failed: {e}")

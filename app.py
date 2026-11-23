"""Basic web page: python app.py, then open http://127.0.0.1:5000 in your browser."""
 
import base64
import io
 
from flask import Flask, render_template_string, request
from PIL import Image
 
from checker import HouseDoorChecker
 
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads
checker = HouseDoorChecker()
 
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Door Checker</title>
<style>
  body{font-family:Segoe UI,system-ui,sans-serif;background:#f4f5f7;color:#1d1f23;margin:0}
  main{max-width:640px;margin:40px auto;padding:0 16px}
  h1{font-size:1.6rem;margin:0 0 4px} p.sub{color:#5b606a;margin:0 0 24px}
  form{background:#fff;border:1px solid #dde0e5;border-radius:10px;padding:20px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  button{background:#2457d6;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:1rem;cursor:pointer}
  .card{margin-top:24px;background:#fff;border:1px solid #dde0e5;border-radius:10px;overflow:hidden}
  .card img{display:block;width:100%;max-height:420px;object-fit:contain;background:#000}
  .verdict{padding:18px 20px;font-size:2rem;font-weight:700;letter-spacing:.05em}
  .ACCEPT{color:#127a3a} .REJECT{color:#c0262d}
  .meta{padding:0 20px 18px;color:#5b606a} .err{color:#c0262d;margin-top:16px}
</style></head><body><main>
<h1>House Door Checker</h1>
<p class="sub">Upload a photo. It is <b>accepted</b> only if it shows a house with its door or gate visible.</p>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Check image</button>
</form>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
{% if result %}
<div class="card">
  <img src="data:image/jpeg;base64,{{ preview }}" alt="uploaded image">
  <div class="verdict {{ result.label }}">{{ result.label }}</div>
  <div class="meta">{{ filename }} &middot; confidence score {{ '%.2f' % result.probability }} (accept at {{ '%.2f' % threshold }} or above) &middot; model v{{ version }}</div>
</div>
{% endif %}
</main></body></html>"""
 
 
@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"result": None, "error": None, "threshold": checker.threshold, "version": checker.version}
    if request.method == "POST":
        f = request.files.get("image")
        try:
            img = Image.open(f.stream)
            img.load()
        except Exception:
            ctx["error"] = "That file could not be read as an image."
            return render_template_string(PAGE, **ctx)
        result = checker.check(img)
        print(f"{f.filename}: {result.label} (score {result.probability:.2f})", flush=True)
        thumb = img.convert("RGB")
        thumb.thumbnail((900, 900))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=85)
        ctx.update(result=result, filename=f.filename, preview=base64.b64encode(buf.getvalue()).decode())
    return render_template_string(PAGE, **ctx)
 
 
if __name__ == "__main__":
    print(f"House Door Checker, model v{checker.version}")
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000)
 
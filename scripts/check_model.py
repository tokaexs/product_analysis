import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import settings
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(timeout=30000))
models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
for m in models:
    t = time.time()
    try:
        r = client.models.generate_content(
            model=m, contents="Return JSON with ok true",
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        print(m, "OK", round(time.time() - t, 1), "s", r.text)
    except Exception as e:
        print(m, "FAIL", round(time.time() - t, 1), "s", str(e)[:90])

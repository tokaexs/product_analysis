import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import settings
from google import genai
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for i in range(8):
    try:
        c.models.generate_content(model=settings.MODEL, contents="Say hi")
        print("call", i + 1, "ok")
    except Exception as e:
        print("call", i + 1, "FAILED:")
        print(str(e)[:1500])
        break

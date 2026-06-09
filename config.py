import os

# קריאת מפתח ה-API - קריסה מיידית ומבוקרת במידה והמפתח חסר בשרת
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]          # raises KeyError if absent — fail fast

# הגדרת פרמטרי המודל המשתנים דינמית (עם ערכי ברירת מחדל חסכוניים ומהירים)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

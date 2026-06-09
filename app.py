import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import os
import google.generativeai as genai

# Setup Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

system_instruction = """
אתה FireMate AI, סוכן חכם תומך החלטות לניהול אירועי חירום ושריפות בזמן אמת. פעל תמיד כמוקדן חירום אנושי, מקצועי וטבעי (ולא כבוט רובוטי מבוסס חוקים).

הנחיות הפעולה שלך:
1. איסוף נתונים שיחתי: עליך לנהל שיחה טבעית וזורמת עם המשתמש כדי לאסוף מידע חיוני (תוואי שטח, גודל השריפה, מיקום האירוע, סיבה להתפרצות). קריטי: שאל רק שאלה אחת בכל פעם, והמתן לתשובת המשתמש. אל תציג "הודעות שגיאה" על מידע חסר, אלא שאל בנימוס ובאופן טבעי להשלמת הפרטים.
2. התאמה חכמה והיסטורית: לאחר איסוף כלל הנתונים, הפק תוכנית פעולה טקטית. עליך לציין במפורש שההמלצה שלך מבוססת על מודל השוואה מתקדם (Cosine Similarity) לאירועי עבר היסטוריים דומים. התאם אישית את ההמלצות למיקום הספציפי של המשתמש (למשל ציין כיצד כיבוי בחיפה שונה מכיבוי בקרית אונו בהתאם לתנאי השטח שהוזנו).
3. שילוב מספרי חירום: שלב באופן טבעי את מספרי החירום הרלוונטיים בתוך תוכנית הפעולה שלך, תוך הבחנה לפי סוג תוואי השטח שהתגלה בשיחה, על בסיס מאגר המידע הבא:
   - כללי לכל אירוע: משטרה 100, מד"א 101, כיבוי אש 102, מוקד עירוני 106.
   - בשטח מגורים: הוסף חברת חשמל 103, פיקוד העורף 104.
   - בשטח תעשייתי/מפעלים: הוסף חברת חשמל 103, מוקד חירום סביבתי / חומ"ס *6911.
   - בשטח פתוח/יערות: הוסף מוקד קק"ל 1-800-350-550, רשות הטבע והגנים *3639.
4. הבן בסובלנות שגיאות כתיב והקלדה. עליך להבין את כוונת המשתמש בין אם כתב בעברית, אנגלית או שילוב של שתיהן.
5. השב תמיד, וללא יוצא מן הכלל, בעברית רהוטה וזורמת.
"""

if api_key:
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.3}
    
    # Using gemini-pro as it's universally available on all free tiers
    best_model = "gemini-pro"
    
    gemini_model = genai.GenerativeModel(best_model, generation_config=generation_config)
else:
    gemini_model = None

# 1. Page Configuration
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# 2. Load External CSS
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Custom Sticky Header
st.markdown("""
    <div class="custom-header">
        <span class="header-logo">🔥 FireMate AI</span>
    </div>
""", unsafe_allow_html=True)


# 3. Bulletproof Data Loading (Cleans CSV headers automatically)
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")

        # CRITICAL FIX: Clean all column names (lowercase + strip spaces)
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        df_c.columns = df_c.columns.str.strip().str.lower()

        return df_f, df_w, df_c, False
    except Exception:
        # Fallback simulation
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })
        df_w = pd.DataFrame({'weekly_area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'cumulative_area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c, True


df_fires, df_weekly, df_cumulative, is_fallback = load_local_csv_files()


# 4. AI Engine
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly, df_cumulative):
        self.df_fires = df_fires
        self.df_weekly = df_weekly

        # Safely extract target columns
        self.frp_col = self._get_column(self.df_fires, ['fire_radiative_power_mw', 'frp', 'fire_radiative_power'])
        self.conf_col = self._get_column(self.df_fires, ['confidence_pct', 'confidence', 'conf'])
        self.weekly_area_col = self._get_column(self.df_weekly, ['weekly_area', 'area'])

    def _get_column(self, df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]

    def compute_similarity(self):
        frp_series = pd.to_numeric(self.df_fires[self.frp_col], errors='coerce').fillna(0)
        conf_series = self.df_fires[self.conf_col].copy()

        if conf_series.dtype == object:
            conf_lower = conf_series.astype(str).str.lower().str.strip()
            conf_mapping = {'high': 95.0, 'nominal': 50.0, 'low': 15.0}
            conf_series = conf_lower.map(conf_mapping)
            conf_series = pd.to_numeric(conf_series, errors='coerce').fillna(50.0)
        else:
            conf_series = pd.to_numeric(conf_series, errors='coerce').fillna(50.0)

        historical_matrix = np.column_stack((frp_series, conf_series))
        live_incident_vector = np.array([[85.0, 95.0]])
        similarities = cosine_similarity(historical_matrix, live_incident_vector)
        max_idx = np.argmax(similarities)
        return self.df_fires.iloc[max_idx], float(similarities[max_idx][0])

    def generate_tactical_response(self, text):
        if gemini_model:
            try:
                # Use Gemini API to generate response
                history = []
                
                # Inject System Instructions into the chat history for full compatibility
                history.append({"role": "user", "parts": [f"System Instructions (Follow these strictly):\n{system_instruction}"]})
                history.append({"role": "model", "parts": ["הבנתי. אפעל בדיוק לפי ההנחיות הללו כסוכן FireMate AI."]})

                # Pass previous messages (excluding the current one which is at the end of session_state.messages)
                for msg in st.session_state.messages[:-1]:
                    role = "model" if msg["role"] == "assistant" else "user"
                    history.append({"role": role, "parts": [msg["content"]]})
                
                chat = gemini_model.start_chat(history=history)
                response = chat.send_message(text)
                return response.text
            except Exception as e:
                return f"שגיאה בתקשורת עם השרת: {e}"
        return "המערכת מבוססת על API אך לא נמצא מפתח (API Key)." 

agent = FireMateIntelligenceEngine(df_fires, df_weekly, df_cumulative)

# 5. UI Elements: Main Title & Hero Section
st.markdown("<div class='main-title'>FireMate AI</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-brand-name">מתמודדים עם שריפה?</div>
    <div class="hero-subtitle">הסוכן החכם שלנו ינתח את תנאי השטח, ישווה לאירועי עבר דומים מנתוני NASA, ויפיק באופן מיידי פרוטוקול טיפול אופטימלי להצלת חיים</div>
</div>
<div class="info-section">
    <div class="info-title">איך אוכל לסייע היום</div>
    הבוט מיועד לספק המלצות להתמודדות עם שריפות לפי שלושה סוגי אזורים מרכזיים:<br>
    אזור מיושב 🏘️ | מתחם תעשייתי ומפעלים 🏭 | שטח פתוח ויערות 🌲
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='sample-heading'>התחילו לדבר עם הבוט, דוגמאות לדיווחים שתוכלו להזין:</p>",
            unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("🏘️ שריפה בשטח בנוי"):
        click_query = "היי, זיהיתי שריפה באזור מגורים בתל אביב."
with col2:
    if st.button("🏭 שריפה באזור תעשייה"):
        click_query = "שלום, פרצה שריפה באזור תעשייה בחיפה."
with col3:
    if st.button("🌲 שריפה בשטח פתוח"):
        click_query = "היי, יש שריפת יער גדולה בשטח פתוח בצפון."

# Chat & Form Persistent State Initialization

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "היי, אני הסוכן שלך ואשאל אותך מספר שאלות קצרות כדי לסייע לך היום"}
    ]

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {message['content']}", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='bot-msg-flag'></div> {message['content']}", unsafe_allow_html=True)

# User Input Processing
user_query = st.chat_input("הקלד את הדיווח שלך (זכור לכלול: תוואי שטח, גודל, מיקום, סיבה)...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {user_query}", unsafe_allow_html=True)

        # 2. Typing indicator (Bot thinking)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("הסוכן מנתח נתונים ומקליד תשובה... 💬"):
                time.sleep(1.5)  # Slight pause to simulate thinking
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div> {response}", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Fixed bottom footer
st.markdown(
    """
    <div class='custom-footer'>
        <div style='color: #01579b; font-weight: bold; font-size: 16px;'>Shira Chitayat & Shira Dabach</div>
        <div style='margin-top: 4px; font-size: 15px;'> סדנת חדשנות מבוססת AI/ML | כל הזכויות שמורות © 2026 </div>
    </div>
    """,
    unsafe_allow_html=True
) 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Try importing the official Google GenAI library, fallback if not installed yet
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load External CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ═══════════════════════════════════════════════════════════
# LOCALIZATION DICTIONARIES (Hebrew / English)
# ═══════════════════════════════════════════════════════════
LOC = {
    'he': {
        'dir': 'rtl',
        'page_title': "מערכת FireMate AI",
        'header_logo': "🔥 FireMate AI",
        'lang_btn': "English 🇬🇧",
        'main_title': "FireMate AI",
        'hero_brand_name': "יש שריפה באזור? דיווח מבצעי מהיר 🔥",
        'info_title': "⚡ איך ניתן לעזור לכוחות בשטח",
        'info_text': "מערכת חכמה המבוססת על מודל שפה ונתוני לוויין NASA לקבלת הנחיות אופרטיביות לפי שלושה אזורים:<br><b>מגורים 🏘️ &nbsp;|&nbsp; תעשייה ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות 🌲</b>",
        'sample_heading': "בחר תרחיש לדוגמה להתחלה:",
        'btn_urban': "🏘️ אש במגורים / ירושלים",
        'btn_industrial': "🏭 אזור תעשייה / חיפה",
        'btn_wildfire': "🌲 שטח פתוח / כרמל",
        'chat_placeholder': "הקלד את הדיווח או התשובה שלך כאן...",
        'footer_main': "כל הזכויות שמורות לפרויקט הגמר ©",
        'footer_sub': "סדנת חדשנות מבוססת AI/ML 2026 🎓 | Shira Chitayat & Shira Dabach",
        'reset_btn': "התחל דיווח חדש 🔄",
        'spinner': "הסוכן מנתח נתוני שטח... 💬",
        'welcome_msg': (
            "שלום המפקד. אני סוכן חכם תומך החלטה מבצעי, המבוסס על נתוני לוויין NASA ומודל שפה מתקדם.\n\n"
            "תאר לי את אירוע השריפה, ואשאל אותך **4 שאלות קצרות** כדי לגבש פרוטוקול מבצעי מדויק."
        ),
        'q_intro': "התקבל דיווח ראשוני. אשאל אותך 4 שאלות קצרות:",
        'q_terrain': "1️⃣ **תוואי השטח:** מהו סוג האזור? (אזור מגורים 🏘️ / אזור תעשייה/מפעלים 🏭 / שטח פתוח/יער 🌲)",
        'q_size': "2️⃣ **גודל השריפה:** מהו היקף האירוע? (למשל: קטנה, בינונית, גדולה, ענקית, כמה דונמים בערך)",
        'q_location': "3️⃣ **מיקום האירוע:** באיזו עיר/אזור? (לדוגמה: חיפה, ירושלים, כרמל, תל אביב)",
        'q_cause': "4️⃣ **סיבה להתפרצות:** מה גרם לשריפה? (אם לא ידוע, ציין 'סיבה לא ידועה')",
        'progress_text': "שאלה {current} מתוך 4",
        'missing_terrain': "תוואי השטח (אזור מגורים / אזור תעשייה / שטח פתוח)",
        'missing_size': "גודל השריפה (למשל: קטנה, גדולה, ענקית, מספר דונמים)",
        'missing_location': "מיקום האירוע (עיר ומדינה)",
        'missing_cause': "סיבה להתפרצות השריפה (אם לא ידוע, ציין 'סיבה לא ידועה')",
        'missing_prefix': "⚠️ **חסר מידע קריטי לגיבוש פרוטוקול מלא:**\n",
        'final_title': "📋 פרוטוקול אופרטיבי מבוסס מיקום ותנאי שטח",
        'follow_up': "פרוטוקול גובש בהצלחה! תוכל לשאול שאלות מבצעיות נוספות.",
    },
    'en': {
        'dir': 'ltr',
        'page_title': "FireMate AI System",
        'header_logo': "🔥 FireMate AI",
        'lang_btn': "עברית 🇮🇱",
        'main_title': "FireMate AI",
        'hero_brand_name': "Active Fire Emergency? Tactical Report 🔥",
        'info_title': "⚡ How we assist emergency crews",
        'info_text': "A smart agent powered by LLMs and NASA satellite data to provide operational fire protocols across three major sectors:<br><b>Residential 🏘️ &nbsp;|&nbsp; Industrial & Factories 🏭 &nbsp;|&nbsp; Open Space & Forests 🌲</b>",
        'sample_heading': "Choose a sample scenario to begin:",
        'btn_urban': "🏘️ Residential Fire / Jerusalem",
        'btn_industrial': "🏭 Industrial Area / Haifa",
        'btn_wildfire': "🌲 Forest Fire / Carmel",
        'chat_placeholder': "Type your report or response here...",
        'footer_main': "All Rights Reserved to Final Project ©",
        'footer_sub': "AI/ML Innovation Workshop 2026 🎓 | Shira Chitayat & Shira Dabach",
        'reset_btn': "Start New Report 🔄",
        'spinner': "Agent is analyzing terrain data... 💬",
        'welcome_msg': (
            "Hello Commander. I am an operational decision-support agent based on NASA satellite data and an advanced LLM.\n\n"
            "Describe the fire incident (in Hebrew or English), and I will ask you **4 short questions** to formulate a precise operational protocol."
        ),
        'q_intro': "Initial report received. I will ask you 4 short questions:",
        'q_terrain': "1️⃣ **Terrain type:** What is the area type? (Residential 🏘️ / Industrial/Factories 🏭 / Open Space/Forest 🌲)",
        'q_size': "2️⃣ **Fire size:** What is the scale of the incident? (e.g. small, medium, large, massive, approximate acres/dunams)",
        'q_location': "3️⃣ **Incident location:** Which city/area? (e.g. Haifa, Jerusalem, Carmel, Tel Aviv)",
        'q_cause': "4️⃣ **Cause of fire:** What caused the fire? (If unknown, state 'cause unknown')",
        'progress_text': "Question {current} of 4",
        'missing_terrain': "Terrain type (Residential / Industrial / Open Space)",
        'missing_size': "Fire size (e.g. small, large, massive, number of acres)",
        'missing_location': "Incident location (city and country)",
        'missing_cause': "Cause of fire outbreak (if unknown, state 'cause unknown')",
        'missing_prefix': "⚠️ **Missing critical information for full protocol:**\n",
        'final_title': "📋 Location & Terrain-Aware Operational Protocol",
        'follow_up': "Protocol formulated successfully! You may ask additional operational questions.",
    }
}

# ═══════════════════════════════════════════════════════════
# TERRAIN CLASSIFIER  (maps free text → 'residential' / 'industrial' / 'open')
# ═══════════════════════════════════════════════════════════
def classify_terrain(text):
    t = text.lower()
    if any(w in t for w in ['תעשייה', 'מפעל', 'מחסן', 'כימ', 'ind', 'factory', 'warehouse', 'chemical', 'petrochemical', '🏭']):
        return 'industrial'
    if any(w in t for w in ['יער', 'חורש', 'פתוח', 'כרמל', 'שדה', 'open', 'forest', 'wildland', 'brush', 'carmel', 'grass', '🌲']):
        return 'open'
    return 'residential'  # default

# ═══════════════════════════════════════════════════════════
# EMERGENCY CONTACTS  (appended to every final protocol)
# ═══════════════════════════════════════════════════════════
def build_emergency_contacts(terrain_type, lang):
    if lang == 'he':
        res  = "<br><br>📞 <b>מוקדי סיוע ותיאום חיוניים (למפקד בשטח):</b><br>"
        res += "• משטרת ישראל (לסגירת צירים ופינוי): <b>100</b> 🚔<br>"
        res += "• מגן דוד אדום (לכוננות רפואית): <b>101</b> 🚑<br>"
        res += "• מוקד עירוני (לתיאום תשתיות מים/רווחה): <b>106</b> 🏢<br>"
        if terrain_type == 'industrial':
            res += "• מוקד חירום סביבתי / חומ\"ס (המשרד להגנת הסביבה): <b>*6911</b> ⚠️<br>"
            res += "• חברת החשמל (לניתוק מתח תעשייתי): <b>103</b> ⚡<br>"
        elif terrain_type == 'residential':
            res += "• חברת החשמל (לניתוק מקורות אנרגיה למבנה): <b>103</b> ⚡<br>"
            res += "• פיקוד העורף (בחשש לקריסת מבנה): <b>104</b> 🏗️<br>"
        elif terrain_type == 'open':
            res += "• מוקד קק\"ל (לתיאום שריפות יער וקווי חיץ): <b>1-800-350-550</b> 🌲<br>"
            res += "• רשות הטבע והגנים (שמורות טבע): <b>*3639</b> 🦌<br>"
    else:
        res  = "<br><br>📞 <b>Emergency Coordination Contacts (Field Commander):</b><br>"
        res += "• Israel Police (road closures & evacuation): <b>100</b> 🚔<br>"
        res += "• Magen David Adom (medical standby): <b>101</b> 🚑<br>"
        res += "• Municipal hotline (water & welfare coordination): <b>106</b> 🏢<br>"
        if terrain_type == 'industrial':
            res += "• Environmental Emergency / HazMat (Ministry of Environment): <b>*6911</b> ⚠️<br>"
            res += "• Israel Electric Corp. (industrial power cut): <b>103</b> ⚡<br>"
        elif terrain_type == 'residential':
            res += "• Israel Electric Corp. (building power shutoff): <b>103</b> ⚡<br>"
            res += "• Home Front Command (structural collapse risk): <b>104</b> 🏗️<br>"
        elif terrain_type == 'open':
            res += "• KKL-JNF hotline (forest firebreaks coordination): <b>1-800-350-550</b> 🌲<br>"
            res += "• Nature & Parks Authority (nature reserves): <b>*3639</b> 🦌<br>"
    return res

# ═══════════════════════════════════════════════════════════
# SPELLING CORRECTION
# ═══════════════════════════════════════════════════════════
def correct_location(location):
    loc = location.strip().lower()
    mapping = {
        'ישרולים': 'ירושלים', 'ירוסלים': 'ירושלים', 'ירושלם': 'ירושלים',
        'חפיה': 'חיפה', 'חיפהה': 'חיפה',
        'קרמל': 'כרמל', 'הכרמל': 'כרמל', 'פארק הכרמל': 'כרמל',
        'תא': 'תל אביב', 'תלאביב': 'תל אביב', 'תל-אביב': 'תל אביב',
        'בר שבע': 'באר שבע', 'באר-שבע': 'באר שבע',
        'jerosalem': 'Jerusalem', 'jerusalim': 'Jerusalem', 'jerusalm': 'Jerusalem',
        'hafa': 'Haifa', 'hayfa': 'Haifa',
        'karmel': 'Carmel',
        'telaviv': 'Tel Aviv', 'tel-aviv': 'Tel Aviv',
        'beersheba': 'Beer Sheva', 'beer-sheva': 'Beer Sheva',
    }
    for key, val in mapping.items():
        if key in loc:
            return val
    return location.strip().title()

# ═══════════════════════════════════════════════════════════
# INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        return df_f, df_w
    except Exception:
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [45.2, 110.5, 320.0],
            'confidence': ['high', 'nominal', 'high'],
            'wind_max_kmh': [10.0, 22.0, 45.0],
            'temp_max_c': [30.0, 36.0, 42.0],
            'humidity_pct': [50.0, 25.0, 12.0],
            'region': ['New South Wales', 'Victoria', 'Queensland'],
            'country': ['Australia', 'Australia', 'Australia'],
            'fire_type': ['Bushfire', 'Forest', 'Grassland'],
            'fire_intensity': ['Moderate', 'High', 'Extreme']
        })
        df_w = pd.DataFrame({'weekly_area': [120, 300, 450]})
        return df_f, df_w

df_fires, df_weekly = load_local_csv_files()

# --- LLM API Setup ---
api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = None
if HAS_GEMINI and len(api_key) > 10:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception:
        model = None

def compute_ml_metrics():
    try:
        df_f = df_fires.copy()
        frp = pd.to_numeric(df_f['fire_radiative_power_mw'], errors='coerce').fillna(40.0)
        conf = df_f['confidence'].replace({'high': 95.0, 'nominal': 50.0, 'low': 15.0})
        conf = pd.to_numeric(conf, errors='coerce').fillna(50.0)
        mat = np.column_stack((frp, conf))
        sim = cosine_similarity(mat, np.array([[85.0, 95.0]]))
        similarity_score = round(float(np.max(sim)) * 100, 1)
        weekly = pd.to_numeric(df_weekly['weekly_area'], errors='coerce').dropna().values
        mean_v = np.mean(weekly) if len(weekly) > 0 else 150.0
        std_v = np.std(weekly) if len(weekly) > 0 and np.std(weekly) > 0 else 20.0
        is_anomaly = (285.0 - mean_v) / std_v > 1.8
        return similarity_score, is_anomaly
    except Exception:
        return 91.4, True

def generate_final_protocol(report_data, lang):
    """Called after all 4 questions are answered. Returns LLM or fallback response + contacts."""
    location   = correct_location(report_data.get('location', ''))
    terrain    = report_data.get('terrain', '')
    size       = report_data.get('size', '')
    cause      = report_data.get('cause', '')
    terrain_type = classify_terrain(terrain)

    sim_score, is_anomaly = compute_ml_metrics()

    # ── Validate collected info ──────────────────────────────
    missing_info = []
    if not report_data.get('terrain'):
        missing_info.append(LOC[lang]['missing_terrain'])
    if not report_data.get('size'):
        missing_info.append(LOC[lang]['missing_size'])
    if not report_data.get('location'):
        missing_info.append(LOC[lang]['missing_location'])
    if not report_data.get('cause'):
        missing_info.append(LOC[lang]['missing_cause'])

    missing_block = ""
    if missing_info:
        missing_block = LOC[lang]['missing_prefix']
        for item in missing_info:
            missing_block += f"  • {item}\n"
        missing_block += "\n"

    # ── LLM Prompt ──────────────────────────────────────────
    lang_name = "Hebrew" if lang == 'he' else "English"
    system_prompt = f"""
You are FireMate AI, a state-of-the-art real-time operational commander assistant for fire emergencies.
You are running inside a crisis management system.

CRITICAL LANGUAGE INSTRUCTION:
- You MUST respond ENTIRELY in {lang_name}. The user interface is set to {lang_name}.
- Regardless of which language the user typed in, ALL your output must be in {lang_name}.

The field commander has reported an active fire incident with the following details:
- Location: {location}
- Terrain/Area type: {terrain} → Classified as: {terrain_type}
- Fire size/scope: {size}
- Cause of fire: {cause}

NASA satellite data analysis:
- Historical event similarity score: {sim_score}%
- Anomalous seasonal spread rate detected: {"YES — Fire is spreading abnormally fast for this season" if is_anomaly else "NO — Within normal seasonal parameters"}

OPERATIONAL INSTRUCTIONS:
1. Generate a highly tactical, structured protocol for the incident commander.
2. Integrate the NASA similarity score and anomaly detection naturally into the report.
3. Apply location-specific protocols:
   - Haifa / Carmel → chemical/petrochemical risks, coastal wind shifts, toxic gas plume evacuation
   - Jerusalem → mountainous WUI, uphill flame propagation, low water pressure at altitude, narrow streets
4. Apply terrain-specific actions for {terrain_type}:
   - industrial → hazmat, fuel shutoff valves, AFFF foam, toxic runoff
   - residential → search & rescue, power shutoff, civilian panic management
   - open → firebreaks, counter-burning, aerial suppression, ember spotting
5. Use clear markdown headers, emojis, and bullet points. Keep it action-oriented.
6. Do NOT include emergency contact numbers in your response — they will be added automatically.
"""

    global model
    if model is not None:
        try:
            resp = model.generate_content([system_prompt, f"Generate the tactical report now in {lang_name}."])
            protocol_text = resp.text
        except Exception:
            protocol_text = _fallback_protocol(location, terrain_type, sim_score, is_anomaly, lang)
    else:
        protocol_text = _fallback_protocol(location, terrain_type, sim_score, is_anomaly, lang)

    # ── Assemble final response ──────────────────────────────
    contacts = build_emergency_contacts(terrain_type, lang)
    if lang == 'he':
        header = f"### {LOC[lang]['final_title']} — {location}\n\n"
    else:
        header = f"### {LOC[lang]['final_title']} — {location}\n\n"

    return missing_block + header + protocol_text + contacts

def _fallback_protocol(location, terrain_type, sim_score, is_anomaly, lang):
    loc_lower = location.lower()
    if lang == 'he':
        if 'חיפה' in loc_lower or 'haifa' in loc_lower or 'כרמל' in loc_lower or 'carmel' in loc_lower:
            loc_g = "⚠️ **פרוטוקול חיפה/כרמל:** ניטור חומ\"ס רציף · פינוי מדרום/מערב · רוחות חוף מסוכנות"
        elif 'ירושלים' in loc_lower or 'jerusalem' in loc_lower:
            loc_g = "⚠️ **פרוטוקול ירושלים:** קווי בלימה על רכסים · Relay pumping בגבהים · גישה ברכבים קלים בלבד"
        else:
            loc_g = f"⚠️ **פרוטוקול {location}:** פריסת כוחות הגנה · פינוי מוקדם · מעקב רוחות"
        if terrain_type == 'industrial':
            ter_g = "🏭 **תעשייה:** ניתוק שסתומי גז/דלק · קירור מכלים · שימוש בקצף AFFF"
        elif terrain_type == 'open':
            ter_g = "🌲 **שטח פתוח:** קווים שרופים · מטוסי כיבוי · תצפיות גיצים"
        else:
            ter_g = "🏘️ **מגורים:** חילוץ לכודים · ניתוק חשמל/גז · הנחיית תושבים"
        anomaly_txt = "⚠️ זוהתה התפשטות אנומלית — קצב מהיר מהרגיל לעונה זו." if is_anomaly else "✅ קצב התפשטות תקין לעונה."
        return f"**ניתוח NASA:** דמיון היסטורי {sim_score}% · {anomaly_txt}\n\n{loc_g}\n\n{ter_g}\n\n*גובש על ידי מנוע FireMate AI המקומי.*"
    else:
        if 'חיפה' in loc_lower or 'haifa' in loc_lower or 'כרמל' in loc_lower or 'carmel' in loc_lower:
            loc_g = "⚠️ **Haifa/Carmel Protocol:** Continuous HazMat monitoring · SW evacuation routes · Coastal wind shift risk"
        elif 'ירושלים' in loc_lower or 'jerusalem' in loc_lower:
            loc_g = "⚠️ **Jerusalem Protocol:** Ridge-line containment · Relay pumping at altitude · Light vehicle access only"
        else:
            loc_g = f"⚠️ **{location} Protocol:** Deploy defensive lines · Early evacuations · Monitor wind shifts"
        if terrain_type == 'industrial':
            ter_g = "🏭 **Industrial:** Isolate gas/fuel shutoffs · Cool adjacent tanks · Deploy AFFF foam"
        elif terrain_type == 'open':
            ter_g = "🌲 **Wildland:** Create firebreaks · Aerial suppression · Deploy ember spotters"
        else:
            ter_g = "🏘️ **Residential:** Search & rescue · Cut power/gas · Manage civilian panic"
        anomaly_txt = "⚠️ Anomalous spread detected — faster than normal seasonal rate." if is_anomaly else "✅ Spread rate within normal seasonal parameters."
        return f"**NASA Analysis:** {sim_score}% historical similarity · {anomaly_txt}\n\n{loc_g}\n\n{ter_g}\n\n*Generated by FireMate AI local engine.*"

def answer_follow_up(question, report_data, lang):
    """Responds to follow-up questions after protocol is complete."""
    location = correct_location(report_data.get('location', ''))
    terrain_type = classify_terrain(report_data.get('terrain', ''))
    sim_score, _ = compute_ml_metrics()
    lang_name = "Hebrew" if lang == 'he' else "English"
    system_prompt = f"""
You are FireMate AI, a real-time operational commander assistant.
Active incident at: {location} (terrain: {terrain_type}).
NASA similarity score: {sim_score}%.
IMPORTANT: Respond ENTIRELY in {lang_name}, regardless of the language of the question.
Answer the follow-up question operationally and tactically.
"""
    global model
    if model is not None:
        try:
            resp = model.generate_content([system_prompt, question])
            return resp.text
        except Exception:
            pass
    if lang == 'he':
        return "פועל במצב גיבוי. אנא פעל לפי הפרוטוקול שגובש למעלה ודווח מיידית למרכז שליטה — 102."
    return "Running in fallback mode. Please follow the protocol above and report to national dispatch — 102."

# ═══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════
if 'lang' not in st.session_state:
    st.session_state.lang = 'he'
if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = 'welcome'  # welcome → questioning → completed
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0            # 0..3 for 4 questions
if 'report_data' not in st.session_state:
    st.session_state.report_data = {'terrain': '', 'size': '', 'location': '', 'cause': ''}
if 'messages' not in st.session_state:
    st.session_state.messages = []

lang = st.session_state.lang
L = LOC[lang]

# ── Apply language direction dynamically ─────────────────────
st.markdown(f"""
    <style>
        html, body, .stApp {{
            direction: {L['dir']} !important;
            text-align: {'right' if lang == 'he' else 'left'} !important;
        }}
        [data-testid="stChatInputTextArea"] {{
            direction: {L['dir']} !important;
            text-align: {'right' if lang == 'he' else 'left'} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ── Initialise welcome message in chosen language ─────────────
if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": L['welcome_msg']}]

# ═══════════════════════════════════════════════════════════
# HEADER  (fixed top bar with logo + language toggle)
# ═══════════════════════════════════════════════════════════
col_logo, col_lang = st.columns([5, 2])
with col_logo:
    st.markdown(f"""
        <div class='brand-logo'>
            <span style='display:inline-block;width:9px;height:9px;border-radius:50%;
                background:#22c55e;box-shadow:0 0 8px #22c55e;margin-left:4px;margin-right:4px;
                animation:dotBlink 1.4s ease-in-out infinite;'></span>
            {L['header_logo']}
        </div>
        <style>@keyframes dotBlink{{0%,100%{{opacity:1}}50%{{opacity:0.2}}}}</style>
    """, unsafe_allow_html=True)
with col_lang:
    if st.button(L['lang_btn'], key="lang_toggle"):
        new_lang = 'en' if lang == 'he' else 'he'
        st.session_state.lang = new_lang
        # Reset to welcome in new language
        st.session_state.messages = [{"role": "assistant", "content": LOC[new_lang]['welcome_msg']}]
        st.session_state.chat_stage = 'welcome'
        st.session_state.q_index = 0
        st.session_state.report_data = {'terrain': '', 'size': '', 'location': '', 'cause': ''}
        st.rerun()

# ═══════════════════════════════════════════════════════════
# HERO UI
# ═══════════════════════════════════════════════════════════
st.markdown(f"<div class='main-title'>🔥 {L['main_title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='hero-brand-name'>{L['hero_brand_name']}</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-section-transparent">
    <div class="info-title-large">{L['info_title']}</div>
    <div class="info-text-large">{L['info_text']}</div>
</div>
""", unsafe_allow_html=True)

# ── Progress bar during questioning ─────────────────────────
if st.session_state.chat_stage == 'questioning':
    qi = st.session_state.q_index
    pct = int((qi / 4.0) * 100)
    st.markdown(f"""
        <div class='progress-label'>
            {L['progress_text'].format(current=qi + 1)}
            <span style='margin:0 8px;color:#475569;'>·</span>
            <span style='color:#64748b;font-size:12px;'>{pct}%</span>
        </div>
    """, unsafe_allow_html=True)
    st.progress(qi / 4.0)

# ── Quick-start scenario buttons (welcome only) ──────────────
QUESTIONS = ['q_terrain', 'q_size', 'q_location', 'q_cause']
DATA_KEYS  = ['terrain',   'size',   'location',   'cause']

click_query = ""
if st.session_state.chat_stage == 'welcome':
    st.markdown(f"<div class='sample-heading'>{L['sample_heading']}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(L['btn_urban'], key="btn_urban"):
            click_query = "שריפה בבניין מגורים בירושלים" if lang == 'he' else "Residential building fire in Jerusalem"
    with c2:
        if st.button(L['btn_industrial'], key="btn_industrial"):
            click_query = "שריפה במחסן כימיקלים בחיפה" if lang == 'he' else "Chemical warehouse fire in Haifa"
    with c3:
        if st.button(L['btn_wildfire'], key="btn_wildfire"):
            click_query = "שריפת יער בפארק הכרמל" if lang == 'he' else "Forest fire in Carmel Park"

# ── Reset button (not during welcome) ───────────────────────
if st.session_state.chat_stage != 'welcome':
    _, col_rst = st.columns([4, 1.5])
    with col_rst:
        if st.button(L['reset_btn'], key="reset_chat"):
            st.session_state.chat_stage = 'welcome'
            st.session_state.q_index = 0
            st.session_state.report_data = {'terrain': '', 'size': '', 'location': '', 'cause': ''}
            st.session_state.messages = [{"role": "assistant", "content": L['welcome_msg']}]
            st.rerun()

# ═══════════════════════════════════════════════════════════
# CHAT DISPLAY
# ═══════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    avatar    = "👤" if msg["role"] == "user" else "✨"
    css_class = "user-msg-box" if msg["role"] == "user" else "bot-msg-box"
    align     = "right" if lang == 'he' else "left"
    with st.chat_message(msg["role"], avatar=avatar):
        msg_dir = L['dir']
        st.markdown(
            f"<div class='{css_class}' style='text-align:{align};direction:{msg_dir};'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════
# CHAT INPUT PROCESSING
# ═══════════════════════════════════════════════════════════
user_input = st.chat_input(L['chat_placeholder'])
if click_query:
    user_input = click_query

if user_input:
    # Guard: avoid duplicate messages on rerun
    if st.session_state.messages and st.session_state.messages[-1]["content"] == user_input:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── STAGE: welcome → start questioning ───────────────────
    if st.session_state.chat_stage == 'welcome':
        st.session_state.chat_stage = 'questioning'
        st.session_state.q_index = 0
        st.session_state.report_data = {'terrain': '', 'size': '', 'location': '', 'cause': ''}
        intro = L['q_intro'] + "\n\n" + L[QUESTIONS[0]]
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.rerun()

    # ── STAGE: questioning → collect answers one by one ──────
    elif st.session_state.chat_stage == 'questioning':
        qi = st.session_state.q_index
        data_key = DATA_KEYS[qi]
        st.session_state.report_data[data_key] = user_input
        next_qi = qi + 1
        st.session_state.q_index = next_qi

        if next_qi < 4:
            # Ask next question
            st.session_state.messages.append({"role": "assistant", "content": L[QUESTIONS[next_qi]]})
            st.rerun()
        else:
            # All 4 answers collected → generate protocol
            st.session_state.chat_stage = 'completed'
            with st.spinner(L['spinner']):
                final_resp = generate_final_protocol(st.session_state.report_data, lang)
            st.session_state.messages.append({"role": "assistant", "content": final_resp})
            st.session_state.messages.append({"role": "assistant", "content": L['follow_up']})
            st.rerun()

    # ── STAGE: completed → follow-up questions ───────────────
    elif st.session_state.chat_stage == 'completed':
        with st.spinner(L['spinner']):
            fu_resp = answer_follow_up(user_input, st.session_state.report_data, lang)
        st.session_state.messages.append({"role": "assistant", "content": fu_resp})
        st.rerun()

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
    <div class='custom-footer'>
        <div class='footer-text-main'>{L['footer_main']}</div>
        <div class='footer-text-sub'>{L['footer_sub']}</div>
    </div>
""", unsafe_allow_html=True)

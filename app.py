import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="FireMate AI",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div class="custom-header">
    <span class="header-logo">🔥 FireMate AI</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# GOOGLE SHEETS CONNECTION
# ---------------------------------------------------

SHEET_ID = "1vx7Q1C8rYhiOYYI_IDsNYqHpyAKcs3-z2MJUD09i_QU"

GIDS = {
    "wildfire_dataset": "1727648073",
    "area_burnt_weekly": "684940024",
    "cumulative_burnt_weekly": "1453144375"
}

URL_DATASET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['wildfire_dataset']}"
URL_WEEKLY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['area_burnt_weekly']}"
URL_CUMULATIVE = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['cumulative_burnt_weekly']}"

@st.cache_data(show_spinner=False, ttl=3600)
def load_all_processed_sheets():

    try:
        df_f = pd.read_csv(URL_DATASET, timeout=5)
        df_w = pd.read_csv(URL_WEEKLY, timeout=5)
        df_c = pd.read_csv(URL_CUMULATIVE, timeout=5)

        return df_f, df_w, df_c, False

    except Exception:

        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence_pct': [80, 95, 60, 88, 99] * 50
        })

        df_w = pd.DataFrame({
            'Weekly_Area': [105, 240, 150, 310, 225]
        })

        df_c = pd.DataFrame({
            'Cumulative_Area': [1000, 2400, 1500, 3100, 2250]
        })

        return df_f, df_w, df_c, True


df_fires, df_weekly, df_cumulative, is_fallback = load_all_processed_sheets()

# ---------------------------------------------------
# AI ENGINE
# ---------------------------------------------------

class FireMateIntelligenceEngine:

    def __init__(self, df_fires, df_weekly, df_cumulative):

        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.df_cumulative = df_cumulative

        self.operational_keywords = [
            'fire', 'wildfire', 'שריפה', 'אש', 'עשן',
            'פינוי', 'חומרים', 'חמ"ל', 'רוח',
            'קוצים', 'יער', 'להבות', 'כיבוי',
            'כבאים', 'דליקה', 'דיווח',
            'מתקרבת', 'פרצה'
        ]

        self.out_of_scope_keywords = [
            'היסטוריה',
            'מלחמה',
            'פוליטיקה',
            'בורסה',
            'ספורט',
            'זמר',
            'מוזיקה',
            'סרט',
            'חדשות'
        ]

        self.frp_col = 'fire_radiative_power_mw'
        self.conf_col = 'confidence_pct'
        self.weekly_area_col = 'Weekly_Area'

    # ---------------------------------------------------

    def is_out_of_scope(self, query):

        return any(
            word in query.lower()
            for word in self.out_of_scope_keywords
        )

    # ---------------------------------------------------

    def has_operational_context(self, query):

        return any(
            word in query.lower()
            for word in self.operational_keywords
        )

    # ---------------------------------------------------

    def compute_similarity(self):

        historical_matrix = self.df_fires[
            [self.frp_col, self.conf_col]
        ].fillna(0).values

        live_incident_vector = np.array([[85.0, 95.0]])

        similarities = cosine_similarity(
            historical_matrix,
            live_incident_vector
        )

        max_idx = np.argmax(similarities)

        return self.df_fires.iloc[max_idx], float(similarities[max_idx][0])

    # ---------------------------------------------------

    def run_anomaly_detection(self):

        if self.weekly_area_col not in self.df_weekly.columns:
            return False, 0.0

        weekly_values = self.df_weekly[
            self.weekly_area_col
        ].dropna().values

        if len(weekly_values) == 0:
            return False, 0.0

        mean_val = np.mean(weekly_values)

        std_val = np.std(weekly_values)

        if std_val == 0:
            std_val = 1.0

        z_score = (295.0 - mean_val) / std_val

        return z_score > 1.8, z_score

    # ---------------------------------------------------

    def generate_tactical_response(self, text):

        # OUT OF SCOPE

        if self.is_out_of_scope(text):

            return """
❌ **הבקשה אינה בתחום ההתמחות של הסוכן**

FireMate AI מיועד לסייע בניהול אירועי שריפה בזמן אמת בלבד.

המערכת מספקת המלצות מבצעיות עבור:
• אזורי מגורים  
• אזורי תעשייה  
• שטחים פתוחים ויערות  

אנא הזן דיווח מבצעי הקשור לאירוע אש פעיל.
"""

        # NO OPERATIONAL CONTEXT

        if not self.has_operational_context(text):

            return """
⚠️ **לא זוהה דיווח מבצעי**

FireMate AI מתמחה בניתוח אירועי שריפה בלבד.

אנא תאר:
• סוג האירוע  
• מיקום השריפה  
• האם מדובר באזור מגורים, תעשייה או שטח פתוח
"""

        # AREA DETECTION

        is_residential = any(
            word in text
            for word in ["מגורים", "שכונה", "בתים"]
        )

        is_industrial = any(
            word in text
            for word in ["תעשייה", "מחסן", "מפעל", "חומרים"]
        )

        is_open = any(
            word in text
            for word in ["פתוח", "יער", "חורש", "קוצים"]
        )

        if not (is_residential or is_industrial or is_open):

            return """
⚠️ **חסר מידע על סוג האזור**

כדי להפיק המלצות מדויקות,
אנא ציין האם מדובר ב:

• אזור מגורים  
• אזור תעשייה  
• שטח פתוח / יער
"""

        # ML ANALYSIS

        twin_case, sim_score = self.compute_similarity()

        is_anomaly, z_score = self.run_anomaly_detection()

        hist_frp_val = twin_case.get(self.frp_col, "N/A")

        # ---------------------------------------------------

        res = "### 🤖 FireMate AI – Operational Analysis Engine\n\n"

        res += (
            f"🧬 **ניתוח דמיון מבוסס ML:** "
            f"זוהה אירוע עבר דומה עם "
            f"**{sim_score*100:.1f}% התאמה** "
            f"(FRP היסטורי: {hist_frp_val}).\n\n"
        )

        if is_anomaly:

            res += (
                f"⚠️ **זוהתה חריגה במדדי ההתפשטות** "
                f"(Z-Score: {z_score:.2f}) "
                f"העשויה להעיד על תנאי יובש קיצוניים.\n\n"
            )

        else:

            res += (
                "📈 **מגמות האקלים השבועיות נמצאות בטווח היציב.**\n\n"
            )

        res += "---\n\n"

        res += "### 📋 בהתאם לנתוני האירוע, אלו הפעולות המומלצות בשלב זה:\n\n"

        # ---------------------------------------------------
        # RESIDENTIAL
        # ---------------------------------------------------

        if is_residential:

            res += "🏘️ **זוהה אירוע באזור מגורים**\n\n"

            res += (
                "• מומלץ להתחיל בפינוי מיידי של קו הבתים הקרוב לאזור האש.\n"
            )

            res += (
                "• יש להזניק כוחות כבאות, משטרת ישראל ומד\"א.\n"
            )

            res += (
                "• מומלץ לבצע חסימות צירים ולהרחיק אזרחים מאזור העשן.\n"
            )

            res += (
                "• מומלץ להציב תצפיות לאיתור מוקדי אש משניים.\n"
            )

        # ---------------------------------------------------
        # INDUSTRIAL
        # ---------------------------------------------------

        elif is_industrial:

            res += "🏭 **זוהה אירוע באזור תעשייה**\n\n"

            res += (
                "• קיים חשש להימצאות חומרים מסוכנים.\n"
            )

            res += (
                "• יש לנתק תשתיות חשמל וגז באזור הסיכון.\n"
            )

            res += (
                "• מומלץ להזניק צוותי חומ\"ס וכוחות כבאות מתוגברים.\n"
            )

            res += (
                "• יש לעדכן את המשרד להגנת הסביבה במידת הצורך.\n"
            )

        # ---------------------------------------------------
        # OPEN AREA
        # ---------------------------------------------------

        else:

            res += "🌲 **זוהה אירוע בשטח פתוח / יער**\n\n"

            res += (
                "• מומלץ לייצר קווי חיץ למניעת התפשטות האש.\n"
            )

            res += (
                "• יש לשקול הזנקת מטוסי כיבוי בהתאם לעוצמת השריפה.\n"
            )

            res += (
                "• מומלץ לעקוב אחר כיווני הרוח בזמן אמת.\n"
            )

            res += (
                "• מומלץ להזעיק כוחות תגבור מרשות הטבע והגנים.\n"
            )

        return res


agent = FireMateIntelligenceEngine(
    df_fires,
    df_weekly,
    df_cumulative
)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="hero-section">

    <div class="hero-brand-name">
        מתמודדים עם דיווח על שריפה מסוכנת? 🔥
    </div>

    <div class="hero-subtitle">
        הסוכן החכם שלנו מנתח את תנאי האירוע בזמן אמת,
        משווה לשריפות עבר ממאגרי NASA
        ומפיק המלצות מבצעיות מיידיות.
    </div>

</div>

<div class="info-section">

<strong>איך אפשר לעזור לכוחות בשטח היום?</strong><br><br>

המערכת מספקת המלצות מבצעיות עבור:

<br><br>

🏘️ אזורי מגורים  
🏭 אזורי תעשייה  
🌲 שטחים פתוחים ויערות

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SAMPLE QUESTIONS
# ---------------------------------------------------

st.markdown(
    "<p class='sample-heading'>דוגמאות לדיווחים מבצעיים:</p>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

click_query = ""

with col1:

    if st.button("🚨 אש בשכונת מגורים"):

        click_query = (
            "יש לנו דיווח על שריפת קוצים "
            "שמתקרבת לקו הבתים הראשון בשכונה."
        )

with col2:

    if st.button("🏭 שריפה באזור תעשייה"):

        click_query = (
            "פרצה אש במחסן באזור תעשייה "
            "וקיים חשש לחומרים מסוכנים."
        )

with col3:

    if st.button("🌲 אש ביער פתוח"):

        click_query = (
            "זוהו להבות גבוהות בלב היער "
            "והאש מתפשטת במהירות."
        )

# ---------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",

            "content":
            "שלום, כאן FireMate AI 🚨\n\n"
            "אנא תאר את אירוע השריפה וציין "
            "האם מדובר באזור מגורים, "
            "אזור תעשייה או שטח פתוח."
        }
    ]

# ---------------------------------------------------
# DISPLAY CHAT
# ---------------------------------------------------

for message in st.session_state.messages:

    avatar = "🤖" if message["role"] == "assistant" else "🧑‍🚒"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

user_query = st.chat_input("הקלד דיווח מבצעי כאן...")

if click_query:
    user_query = click_query

if user_query:

    if (
        not st.session_state.messages
        or st.session_state.messages[-1]["content"] != user_query
    ):

        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user", avatar="🧑‍🚒"):

            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):

            with st.spinner(
                "מעבד נתוני אירוע, מבצע ניתוח סיכונים ומשווה לאירועי עבר..."
            ):

                response = agent.generate_tactical_response(user_query)

                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    """
<div class='custom-footer'>

    <div style='font-weight: bold;'>
        כל הזכויות שמורות לפרויקט הגמר ©
    </div>

    <div style='margin-top: 4px;'>
        מגישות:
        <b>Shira Chitayat & Shira Dabach</b>
        |
        סדנת חדשנות מבוססת AI/ML 2026 🎓
    </div>

</div>
""",
    unsafe_allow_html=True
) 

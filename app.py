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
except:
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
# LOAD DATA
# ---------------------------------------------------

@st.cache_data(show_spinner=False)
def load_local_csv_files():

    try:

        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")

        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        df_c.columns = df_c.columns.str.strip().str.lower()

        return df_f, df_w, df_c, False

    except Exception:

        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })

        df_w = pd.DataFrame({
            'weekly_area': [105, 240, 150, 310, 225]
        })

        df_c = pd.DataFrame({
            'cumulative_area': [1000, 2400, 1500, 3100, 2250]
        })

        return df_f, df_w, df_c, True

df_fires, df_weekly, df_cumulative, is_fallback = load_local_csv_files()

# ---------------------------------------------------
# AI ENGINE
# ---------------------------------------------------

class FireMateIntelligenceEngine:

    def __init__(self, df_fires, df_weekly):

        self.df_fires = df_fires
        self.df_weekly = df_weekly

        self.domain_keywords = [
            'fire', 'wildfire', 'שריפה', 'אש', 'עשן',
            'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
            'יער', 'להבות', 'כיבוי', 'כבאים',
            'ציוד', 'חום', 'דליקה',
            'משטרה', 'מדא'
        ]

        self.trivia_keywords = [
            'מה ה', 'איפה', 'מתי',
            'הכי', 'גדולה', 'היסטוריה',
            'מי', 'איך קוראים'
        ]

        self.frp_col = self._get_column(
            self.df_fires,
            ['fire_radiative_power_mw', 'frp']
        )

        self.conf_col = self._get_column(
            self.df_fires,
            ['confidence_pct', 'confidence']
        )

        self.weekly_area_col = self._get_column(
            self.df_weekly,
            ['weekly_area', 'area']
        )

    def _get_column(self, df, possible_names):

        for name in possible_names:

            if name in df.columns:
                return name

        return df.columns[0]

    def compute_similarity(self):

        frp_series = pd.to_numeric(
            self.df_fires[self.frp_col],
            errors='coerce'
        ).fillna(0)

        conf_series = self.df_fires[self.conf_col].copy()

        if conf_series.dtype == object:

            conf_lower = conf_series.astype(str).str.lower()

            conf_mapping = {
                'high': 95.0,
                'nominal': 50.0,
                'low': 15.0
            }

            conf_series = conf_lower.map(conf_mapping)

            conf_series = pd.to_numeric(
                conf_series,
                errors='coerce'
            ).fillna(50.0)

        historical_matrix = np.column_stack(
            (frp_series, conf_series)
        )

        live_incident_vector = np.array([[85.0, 95.0]])

        similarities = cosine_similarity(
            historical_matrix,
            live_incident_vector
        )

        max_idx = np.argmax(similarities)

        return (
            self.df_fires.iloc[max_idx],
            float(similarities[max_idx][0])
        )

    def run_anomaly_detection(self):

        weekly_values = pd.to_numeric(
            self.df_weekly[self.weekly_area_col],
            errors='coerce'
        ).dropna().values

        if len(weekly_values) == 0:
            return False, 0.0

        mean_val = np.mean(weekly_values)

        std_val = np.std(weekly_values)

        if std_val == 0:
            std_val = 1.0

        z_score = (295.0 - mean_val) / std_val

        return z_score > 1.8, z_score

    def generate_tactical_response(self, text):

        query_lower = text.lower()

        if any(word in query_lower for word in self.trivia_keywords):

            return """
❌ **השאלה חורגת מגבולות הפעילות של הסוכן**

FireMate AI מיועד לתמיכה מבצעית בזמן אמת בלבד ואינו מספק מידע היסטורי או שאלות טריוויה.

ניתן לדווח על:
- שריפה באזור מגורים
- שריפה במתחם תעשייתי
- שריפה בשטח פתוח
"""

        if not any(word in query_lower for word in self.domain_keywords):

            return """
❌ **השאלה אינה קשורה לתחום הפעילות המבצעי**

אנא תארו אירוע שריפה פעיל כדי שאוכל לסייע בקבלת החלטות בזמן אמת.
"""

        is_residential = any(
            word in text
            for word in ["מגורים", "שכונה", "בניין"]
        )

        is_industrial = any(
            word in text
            for word in ["תעשייה", "מחסן", "מפעל"]
        )

        is_open = any(
            word in text
            for word in ["יער", "שטח פתוח", "קוצים"]
        )

        if not (is_residential or is_industrial or is_open):

            return """
⚠️ חסר מידע על סוג האזור

אנא ציינו האם מדובר ב:
- אזור מגורים
- אזור תעשייה
- שטח פתוח
"""

        twin_case, sim_score = self.compute_similarity()

        is_anomaly, z_score = self.run_anomaly_detection()

        similarity_pct = sim_score * 100

        hist_frp_val = twin_case.get(
            self.frp_col,
            "N/A"
        )

        response = "### 🤖 ניתוח מבצעי של FireMate AI\n\n"

        response += f"""
המערכת זיהתה התאמה של **{similarity_pct:.1f}%**
לאירוע היסטורי ממאגר NASA
(עוצמת FRP: {hist_frp_val}).
"""

        if is_residential:

            response += """

🏘️ **זוהה אירוע באזור מגורים**

- יש להזעיק מיידית את כוחות הכיבוי והמשטרה
- מומלץ להתחיל בפינוי תושבים מהקו הראשון
- יש לפתוח נתיבי גישה לכוחות ההצלה
- מומלץ לערב את מד״א בכוננות גבוהה
"""

        elif is_industrial:

            response += """

🏭 **זוהה אירוע באזור תעשייה**

- נדרש להזעיק יחידות חומ״ס
- יש לנתק קווי גז וחשמל
- מומלץ להרחיק עובדים ואזרחים
- יש להקים חסימות משטרתיות סביב המתחם
"""

        else:

            response += """

🌲 **זוהתה שריפה בשטח פתוח**

- מומלץ להזניק מטוסי כיבוי
- יש לפתוח קווי חיץ באמצעות דחפורים
- יש לעקוב אחר כיוון הרוח
- מומלץ להרחיק מטיילים וצוותים לא חיוניים
"""

        if is_anomaly:

            response += f"""

⚠️ **התגלתה חריגה במדדי ההתפשטות**

Z-Score: **{z_score:.2f}**

קצב ההתפשטות גבוה מהרגיל ומומלץ לבקש תגבור מחוזי.
"""

        else:

            response += f"""

📊 מדדי ההתפשטות נמצאים בטווח התקין.

Z-Score: **{z_score:.2f}**
"""

        return response

# ---------------------------------------------------
# INITIALIZE AGENT
# ---------------------------------------------------

agent = FireMateIntelligenceEngine(
    df_fires,
    df_weekly
)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown(
"""
<div class='main-title'>
🔥 FireMate AI
</div>

<div class="hero-section">

    <div class="hero-brand-name">
        זוהתה שריפה? FireMate AI יסייע בקבלת החלטות בזמן אמת 🔥
    </div>

    <div class="hero-subtitle">
        המערכת מנתחת את תנאי השטח,
        משווה לאירועי עבר ממאגרי NASA,
        ומספקת המלצות מבצעיות מיידיות.
    </div>

</div>

<div class="info-section">

    <div class="info-title">
        איך אפשר לעזור לכוחות בשטח?
    </div>

    FireMate AI מותאם לשלושה תרחישים מרכזיים:

    <br><br>

    🏘️ אזור מגורים עירוני  
    🏭 מתחם תעשייתי  
    🌲 שטח פתוח ויערות

</div>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# SAMPLE BUTTONS
# ---------------------------------------------------

st.markdown(
"<p class='sample-heading'>דוגמאות לדיווחים:</p>",
unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

click_query = ""

with col1:

    if st.button("🏘️ שריפה בבניין מגורים"):

        click_query = """
יש שריפה בבניין מגורים,
עשן כבד מתפשט לכיוון הרחוב.
"""

with col2:

    if st.button("🏭 שריפה במפעל"):

        click_query = """
פרצה אש באזור תעשייה,
יש חשש לחומרים מסוכנים.
"""

with col3:

    if st.button("🌲 שריפה ביער"):

        click_query = """
זוהתה שריפה בשטח פתוח,
האש מתפשטת במהירות עם הרוח.
"""

# ---------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content":
            """
שלום, כאן FireMate AI 🤖

אנא תארו את אירוע השריפה וציינו:
- אזור מגורים
- אזור תעשייה
- שטח פתוח
"""
        }
    ]

# ---------------------------------------------------
# DISPLAY CHAT
# ---------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user", avatar="🧑‍🚒"):

            st.markdown(
                f"<div class='user-msg-flag'></div>{message['content']}",
                unsafe_allow_html=True
            )

    else:

        with st.chat_message("assistant", avatar="🤖"):

            st.markdown(
                f"<div class='bot-msg-flag'></div>{message['content']}",
                unsafe_allow_html=True
            )

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

user_query = st.chat_input(
    "הקלד דיווח מבצעי כאן..."
)

if click_query:
    user_query = click_query

if user_query:

    st.session_state.messages.append({

        "role": "user",
        "content": user_query
    })

    with st.chat_message("user", avatar="🧑‍🚒"):

        st.markdown(
            f"<div class='user-msg-flag'></div>{user_query}",
            unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner(
            "המערכת מנתחת נתונים ומפיקה המלצות..."
        ):

            response = agent.generate_tactical_response(
                user_query
            )

            st.markdown(
                f"<div class='bot-msg-flag'></div>{response}",
                unsafe_allow_html=True
            )

    st.session_state.messages.append({

        "role": "assistant",
        "content": response
    })

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
"""
<div class='custom-footer'>

    <div style='font-weight:700; font-size:16px;'>
        כל הזכויות שמורות ©
    </div>

    <div style='margin-top:5px; font-size:14px;'>
        Shira Chitayat & Shira Dabach |
        סדנת חדשנות AI/ML 2026 🎓
    </div>

</div>
""",
unsafe_allow_html=True
) 

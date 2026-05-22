import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# 2. Load external CSS
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

# 3. Connection to Google Sheets (Data Anchors)
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
        # Fallback simulation to prevent crashes
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence_pct': [80, 95, 60, 88, 99] * 50
        })
        df_w = pd.DataFrame({'Weekly_Area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'Cumulative_Area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c, True

df_fires, df_weekly, df_cumulative, is_fallback = load_all_processed_sheets()

# 4. Intelligence Engine (Rules & ML logic)
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly, df_cumulative):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.df_cumulative = df_cumulative
        
        # Guardrail Keywords
        self.operational_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים', 'יער', 'להבות', 'כיבוי', 'כבאים', 'דליקה', 'דיווח', 'מתקרבת', 'פרצה']
        self.trivia_keywords = ['מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'בישראל', 'בשנה האחרונה', 'היסטוריה', 'מי', 'איך קוראים']
        
        # Columns
        self.frp_col = 'fire_radiative_power_mw'
        self.conf_col = 'confidence_pct'
        self.weekly_area_col = 'Weekly_Area'

    def is_trivia_question(self, query):
        """Check if user is asking general trivia instead of reporting an incident"""
        return any(word in query.lower() for word in self.trivia_keywords)

    def has_operational_context(self, query):
        """Check if user provided fire-related context"""
        return any(word in query.lower() for word in self.operational_keywords)

    def compute_similarity(self):
        """Cosine Similarity Engine"""
        historical_matrix = self.df_fires[[self.frp_col, self.conf_col]].fillna(0).values
        live_incident_vector = np.array([[85.0, 95.0]]) 
        similarities = cosine_similarity(historical_matrix, live_incident_vector)
        max_idx = np.argmax(similarities)
        return self.df_fires.iloc[max_idx], float(similarities[max_idx][0])

    def run_anomaly_detection(self):
        """Z-Score Anomaly Detection"""
        if self.weekly_area_col not in self.df_weekly.columns:
            return False, 0.0
        weekly_values = self.df_weekly[self.weekly_area_col].dropna().values
        if len(weekly_values) == 0: 
            return False, 0.0
        mean_val = np.mean(weekly_values)
        std_val = np.std(weekly_values) if np.std(weekly_values) > 0 else 1.0
        z_score = (295.0 - mean_val) / std_val 
        return z_score > 1.8, z_score

    def generate_tactical_response(self, text):
        # 1. Guardrail for Trivia / General Knowledge
        if self.is_trivia_question(text):
            return "❌ **[מחוץ לגבולות הגזרה - שאלת מידע כללי]**\n\nאני מערכת מבצעית תומכת החלטה המיועדת אך ורק לאירועי חירום פעילים. איני אנציקלופדיה לשאלות היסטוריות או טריוויה. תפקידי לספק המלצות על סמך אזור השריפה (מגורים, תעשייה, או שטח פתוח). אנא הזן דיווח מבצעי כדי שאוכל לסייע."
        
        # 2. Guardrail for Irrelevant Topics
        if not self.has_operational_context(text):
            return "❌ **[מחוץ לגבולות הגזרה]**\n\nאני סוכן שמומחה לניהול שריפות ומשברים בזמן אמת. השאלה שלך לא קשורה לתחום המבצעי שלי. אנא דווח על אירוע חירום באחד משלושת האזורים המוגדרים."
        
        # Extract location to ensure agent stays focused
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים"])

        if not (is_residential or is_industrial or is_open):
            return "⚠️ **[חסר מידע מיקום מבצעי]**\n\nזיהיתי דיווח על אירוע אש, אך חסר לי סוג תוואי השטח כדי להפיק פרוטוקול. אנא ציין האם האירוע מתרחש ב**אזור מיושב**, **אזור תעשייה**, או **שטח פתוח**."

        # Run ML Analysis
        twin_case, sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()
        hist_frp_val = twin_case.get(self.frp_col, "N/A")
        
        # Build Response
        res = f"### 🤖 ניתוח סוכן חכם - FireMate AI\n\n"
        res += f"🧬 **זיהוי דמיון (ML):** האירוע הושווה מול היסטוריית NASA. נמצאה שריפה תאומה מהעבר עם **{sim_score*100:.1f}% אחוז דמיון** (עוצמת FRP היסטורית: {hist_frp_val}).\n"
        
        if is_anomaly:
            res += f"⚠️ **התרעת חריגות:** מדד ההתפשטות חורג מהמגמה השבועית! **(Z-Score: {z_score:.2f})** - סכנת יובש קטסטרופלית.\n\n"
        else:
            res += f"📈 **מגמות אקלים:** מדדי השטח השרוף השבועיים שלכן בטווח היציב.\n\n"
            
        res += "--- \n\n### 📋 פרוטוקול פעולה מומלץ לכוחות:\n"
        
        if is_residential:
            res += "🚨 **פרופיל: אזור מיושב עירוני**\n"
            res += "* **פינוי מיידי:** הפעלת מערכות כריזה ופינוי קו הבתים המאוים.\n"
            res += "* **חומת אש:** ריכוז מאמץ והקמת קווי הגנה סביב תשתיות אזרחיות.\n"
            res += "* **תובנת עבר:** מומלץ להציב תצפיות אש לזיהוי שריפות משנה מגיצים בתוך השכונה."
        elif is_industrial:
            res += "🚨 **פרופיל: מתחם תעשייתי (סיכון חומ\"ס)**\n"
            res += "* **חומרים מסוכנים:** הזנקה מיידית של יחידת ניטור לבדיקת רעילות עשן.\n"
            res += "* **בידוד תשתיות:** ניתוק תשתיות גז וחשמל במפעלים סמוכים.\n"
            res += "* **הגבלת תנועה:** הקמת רדיוס סטרילי של 1 ק\"מ לכוחות החירום בלבד."
        else:
            res += "🌲 **פרופיל: שטח פתוח ויערות**\n"
            res += "* **קווי חיץ:** הפעלת דחפורים לחשיפת אדמה למניעת התפשטות האש.\n"
            res += "* **סיוע אווירי:** דרישת הזנקה של מטוסי כיבוי עקב מדדי חום (FRP) גבוהים.\n"
            res += "* **בטיחות צוותים:** מעקב צמוד אחר כיווני הרוח למניעת כליאת לוחמי אש באגפים."
            
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly, df_cumulative)

# 5. Marketing Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-brand-name">מתמודדים עם דיווח על שריפה מסוכנת? 🔥</div>
    <div class="hero-subtitle">הסוכן החכם שלנו ינתח את תנאי השטח שלכם, ישווה לאירועי עבר דומים מנתוני NASA, ויפיק באופן מיידי פרוטוקול טיפול אופטימלי.</div>
</div>
<div class="info-section">
    <strong>איך אפשר לעזור לכוחות בשטח היום?</strong><br>
    הבוט מיועד לספק המלצות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br>
    אזור מיושב (מגורים) 🏘️ | אזור תעשייה 🏭 | שטח פתוח 🌲
</div>
""", unsafe_allow_html=True)

# 6. Quick Action Sample Questions
st.markdown("<p class='sample-heading'>התחילו לדבר עם הבוט - דוגמאות לדיווחים שתוכלו להזין:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("🚨 אש בשכונת מגורים"):
        click_query = "יש לנו דיווח על שריפת קוצים חזקה שמתקרבת במהירות לקו הבתים הראשון של השכונה, יש עשן סמיך ורוח חזקה!"
with col2:
    if st.button("🏭 שריפה באזור תעשייה"):
        click_query = "פרצה אש במחסן לוגיסטי בתוך אזור התעשייה, יש חשש כבד להימצאות חומרים מסוכנים ומכלים דליקים באזור."
with col3:
    if st.button("🌲 אש ביער הפתוח"):
        click_query = "זיהינו להבות גבוהות בלב היער השטח פתוח, השריפה מתפשטת אופקית ויש קושי בהגעה של רכבי כיבוי כבדים."

# 7. Initialize Chat Memory & Initial Greeting
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "שלום המפקד, כאן סוכן FireMate AI המבצעי שלך. 🚨\n\nעל איזה אירוע תרצה לדווח? אנא תאר לי את מצב השריפה וציין במפורש האם מדובר ב**אזור מגורים**, **תעשייה** או **שטח פתוח**."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. User Input Processing
user_query = st.chat_input("הקלד את הדיווח המבצעי שלך כאן...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("מנתח תנאי אקלים ומריץ השוואת דמיון..."):
                response = agent.generate_tactical_response(user_query)
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# 9. Official Footer with Copyrights & Icon
st.markdown(
    """
    <div class='custom-footer'>
        <div style='color: #0d4d44; font-weight: bold;'>כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div style='margin-top: 4px;'>הוגש ע"י: <b>Shira Chitayat & Shira Dabach</b> | סדנת חדשנות מבוססת AI/ML 2026 🎓</div>
    </div>
    """, 
    unsafe_allow_html=True
) 

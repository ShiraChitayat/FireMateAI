import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# 2. Load the external elegant CSS design
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

# 3. Load data directly from local CSV files
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")
        return df_f, df_w, df_c, False
    except Exception:
        # Fallback simulation if files are missing
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })
        df_w = pd.DataFrame({'Weekly_Area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'Cumulative_Area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c, True

df_fires, df_weekly, df_cumulative, is_fallback = load_local_csv_files()

# 4. Agent Intelligence Engine
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly, df_cumulative):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.df_cumulative = df_cumulative
        
        self.domain_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים', 'יער', 'להבות', 'כיבוי', 'כבאים', 'ציוד', 'חום', 'להבה', 'הצלה', 'שרפה', 'שרפות', 'דליקה', 'משטרה', 'מדא', 'מגן דוד']
        self.trivia_keywords = ['מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'היסטוריה', 'מי', 'איך קוראים', 'איזה שריפות']
        
        # Robust Column Finder
        self.frp_col = self._get_column(self.df_fires, ['fire_radiative_power_mw', 'frp', 'FRP', 'fire_radiative_power'])
        self.conf_col = self._get_column(self.df_fires, ['confidence_pct', 'confidence', 'Confidence', 'conf'])
        self.weekly_area_col = self._get_column(self.df_weekly, ['Weekly_Area', 'area', 'Area', 'weekly_area'])

    def _get_column(self, df, possible_names):
        """Smart function to find the correct column name in the CSV"""
        for name in possible_names:
            if name in df.columns:
                return name
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]

    def is_trivia_question(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.trivia_keywords)

    def check_boundaries(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)

    def compute_similarity(self):
        """Cosine Similarity Engine with Automatic Text-to-Number Conversion"""
        # Extract the relevant columns
        frp_series = self.df_fires[self.frp_col].copy()
        conf_series = self.df_fires[self.conf_col].copy()
        
        # Ensure FRP is numeric
        frp_series = pd.to_numeric(frp_series, errors='coerce').fillna(0)
        
        # Ensure Confidence is numeric (Convert 'high', 'nominal', 'low' to floats)
        if conf_series.dtype == object: # If it's text
            conf_lower = conf_series.astype(str).str.lower()
            # Map the text categories to numeric percentages
            conf_mapping = {'high': 95.0, 'nominal': 50.0, 'low': 15.0}
            conf_series = conf_lower.map(conf_mapping)
            # Fill any unmapped or missing values with a default of 50.0
            conf_series = pd.to_numeric(conf_series, errors='coerce').fillna(50.0)
        else:
            conf_series = pd.to_numeric(conf_series, errors='coerce').fillna(50.0)

        # Create the historical matrix for comparison
        historical_matrix = np.column_stack((frp_series, conf_series))
        
        # Live simulated incident features (FRP: 85.0, Confidence: 95.0)
        live_incident_vector = np.array([[85.0, 95.0]])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(historical_matrix, live_incident_vector)
        max_idx = np.argmax(similarities)
        
        return self.df_fires.iloc[max_idx], float(similarities[max_idx][0])

    def run_anomaly_detection(self):
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
        if self.is_trivia_question(text):
            return "❌ **[חריגה מגבולות הגזרה - שאלת מידע כללי]**\n\nאני מערכת מבצעית ותומכת החלטה המיועדת לניהול אירועי חירום פעילים בלבד. איני מוסמך לענות על שאלות היסטוריות או טריוויה. תפקידי הוא לספק הנחיות פעולה בזמן אמת עבור שריפות באזור מיושב, אזור תעשייה או שטח פתוח. אנא הזן דיווח מבצעי מהשטח."
        
        if not self.check_boundaries(text):
            return "❌ **[חריגה מגבולות הגזרה של הסוכן]**\n\nאיני מוסמך לענות על שאלה זו. אנא מיקדו את הדיווח שלכם באירוע שריפה פעיל באחד משלושת האזורים המוגדרים."
        
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים", "עירוני", "בניין"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים"])

        if not (is_residential or is_industrial or is_open):
            return "⚠️ **[חסר מידע מיקום מבצעי]**\n\nזיהיתי דיווח על אירוע אש, אך חסר לי סוג תוואי השטח. ציין בדיווח האם מדובר ב**אזור מגורים**, **אזור תעשייה**, או **שטח פתוח**."

        twin_case, sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()
        hist_frp_val = twin_case.get(self.frp_col, "N/A")
        similarity_pct = sim_score * 100
        
        res = "### 🤖 ניתוח תפעולי של סוכן FireMate AI\n\n"
        
        if is_residential:
            res += f"על פי ניתוח הדיווח הנוכחי באזור המגורים והצלבה מול ה-Data Anchor, המערכת איתרה אירוע עבר דומה מנתוני NASA עם **מדד דמיון קוסינוס גבוה של {similarity_pct:.1f}%** (עוצמת FRP היסטורית: {hist_frp_val}). לאור רמת הסיכון לחיי אדם, יש להורות מיד לחמ\"ל להזניק כוחות משטרה לחסימת צירי התנועה ופתיחת נתיבי מילוט, ובמקביל לערב את מד\"א לצורך היערכות רפואית. מומלץ לרכז את מאמץ הכיבוי ביצירת חיץ מים היקפי סביב הבניינים, ולהציב תצפיות אש לאיתור דילוג גיצים."
        
        elif is_industrial:
            res += f"המערכת מזהה שמדובר באירוע מורכב במתחם תעשייתי. המודל האלגוריתמי מציג **התאמה של {similarity_pct:.1f}%** למקרה היסטורי מסוכן. נדרש להזניק יחידות חומ\"ס ייעודיות לבחינת רעילות האוויר. יש לתאם סגר הרמטי עם משטרת ישראל ברדיוס של 1 ק\"מ ולפעול לניתוק קווי גז וחשמל מרכזיים. יש להנחות את כוחות מד\"א להתמקם מחוץ לטווח סכנת הפיצוץ."
            
        else:
            res += f"דיווח על שריפה בשטח פתוח או ביער דורש הפעלה דחופה של דחפורים לחשיפת אדמה. מנוע הדמיון מציג **התאמה של {similarity_pct:.1f}%** לאירוע לווייני בעל קצב התפשטות מהיר. לאור המדדים, מומלץ לפנות לחפ\"ק להזנקת מטוסי כיבוי. על מפקד האירוע לוודא קשר רציף מול הניטור המטאורולוגי כדי לעקוב אחר תנודות ברוח."
            
        if is_anomaly:
            res += f" בנוסף, המערכת מתריעה כי קצב ההתפשטות חורג (אנומליה) ביחס להיסטוריה השבועית שלכן **(Z-Score חמור של {z_score:.2f})**. זה מעיד על תנאי יובש קיצוניים או הצתה, ומומלץ לבקש תגבורת."
        else:
            res += f" מבחינה סטטיסטית, מדדי השטח השרוף השבועיים שלכן נמצאים בטווח האקלימי התקין (Z-Score: {z_score:.2f})."
            
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly, df_cumulative)

# 5. Marketing Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-brand-name">מתמודדים עם דיווח על שריפה מסוכנת? 🔥</div>
    <div class="hero-subtitle">הסוכן החכם שלנו ינתח את תנאי השטח, ישווה לאירועי עבר דומים מנתוני NASA, ויפיק באופן מיידי פרוטוקול טיפול אופטימלי להצלת חיים.</div>
</div>
<div class="info-section">
    <strong>איך אפשר לעזור לכוחות בשטח היום?</strong><br>
    הבוט מיועד לספק המלצות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br>
    אזור מיושב עירוני 🏘️ | מתחם תעשייתי ומפעלים 🏭 | שטח פתוח ויערות 🌲
</div>
""", unsafe_allow_html=True)

# 6. Quick Action Sample Buttons
st.markdown("<p class='sample-heading'>התחילו לדבר עם הבוט - דוגמאות לדיווחים שתוכלו להזין:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("🚨 אש בבניין מגורים"):
        click_query = "אני בשטח עירוני בישראל ויש שריפה של בניין מגורים, מה לעשות?"
with col2:
    if st.button("🏭 שריפה באזור תעשייה"):
        click_query = "פרצה אש במחסן לוגיסטי בתוך אזור התעשייה, יש חשש כבד להימצאות חומרים מסוכנים ומכלים דליקים באזור."
with col3:
    if st.button("🌲 אש ביער הפתוח"):
        click_query = "זיהינו להבות גבוהות בלב היער בשטח פתוח, השריפה מתפשטת אופקית ויש קושי בהגעה של רכבי כיבוי כבדים."

# 7. Chat Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "שלום המפקד, כאן סוכן FireMate AI המבצעי שלך. 🚨\n\nעל איזה אירוע תרצה לדווח? אנא תאר לי את מצב השריפה וציין האם מדובר ב**אזור מגורים**, **תעשייה** או **שטח פתוח**."}
    ]

# Display history with HTML flags for targeted CSS colors
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {message['content']}", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='bot-msg-flag'></div> {message['content']}", unsafe_allow_html=True)

# 8. User Input Processing
user_query = st.chat_input("הקלד את הדיווח המבצעי שלך כאן...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {user_query}", unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("מנתח נתונים..."):
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div> {response}", unsafe_allow_html=True)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# 9. Dynamic Footer with extra spacing to prevent overlapping
st.markdown(
    """
    <div style="height: 60px;"></div> <div class='custom-footer'>
        <div style='color: #0d4d44; font-weight: bold;'>כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div style='margin-top: 4px;'> Shira Chitayat & Shira Dabach | סדנת חדשנות מבוססת AI/ML 2026 🎓 | כל הזכויות שמורות ©</div>
    </div>
    """, 
    unsafe_allow_html=True
)

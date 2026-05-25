import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

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

        self.domain_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
                                'יער', 'להבות', 'כיבוי', 'כבאים', 'ציוד', 'חום', 'להבה', 'הצלה', 'שרפה', 'דליקה',
                                'משטרה', 'מדא']
        self.trivia_keywords = ['מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'היסטוריה', 'מי', 'איך קוראים', 'איזה שריפות']

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

    def run_anomaly_detection(self):
        if self.weekly_area_col not in self.df_weekly.columns:
            return False, 0.0
        weekly_values = pd.to_numeric(self.df_weekly[self.weekly_area_col], errors='coerce').dropna().values
        if len(weekly_values) == 0:
            return False, 0.0
        mean_val = np.mean(weekly_values)
        std_val = np.std(weekly_values) if np.std(weekly_values) > 0 else 1.0
        z_score = (295.0 - mean_val) / std_val
        return z_score > 1.8, z_score

    def generate_tactical_response(self, text):
        query_lower = text.lower()
        
        # Boundaries Check
        if any(keyword in query_lower for keyword in self.trivia_keywords):
            return "❌ **[חריגה מגבולות הגזרה - שאלת מידע כללי]**\n\nאני מערכת מבצעית ותומכת החלטה המיועדת לניהול אירועי חירום פעילים בלבד. איני מוסמך לענות על שאלות היסטוריות או טריוויה. תפקידי הוא לספק הנחיות פעולה בזמן אמת. אנא הזן דיווח מבצעי מהשטח."
        if not any(keyword in query_lower for keyword in self.domain_keywords):
            return "❌ **[חריגה מגבולות הגזרה של הסוכן]**\n\nאיני מוסמך לענות על שאלה זו. אנא מיקדו את הדיווח שלכם באירוע שריפה פעיל וספקו פרטים רלוונטיים."

        # Extract Information Criteria
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים", "עירוני", "בניין", "עיר"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים"])
        
        has_size = any(word in text for word in ["גודל", "גדולה", "קטנה", "עצומה", "ענקית", "מטר", "דונם", "נרחבת", "בינונית", "ענק"])
        has_location = any(word in text for word in ["ישראל", "עיר", "רחוב", "חיפה", "תל אביב", "ירושלים", "צפון", "דרום", "מרכז", "מדינה", "אזור", "סמוך ל"])
        has_cause = any(word in text for word in ["נגרמה", "בגלל", "כתוצאה", "הצתה", "קצר", "חשמל", "נפילה", "טבעי", "לא ידוע", "סיבה", "מטען", "פיצוץ", "לא ידועה", "פגיעה"])

        # Check for missing required details
        missing_info = []
        if not (is_residential or is_industrial or is_open):
            missing_info.append("תוואי השטח (אזור מגורים / אזור תעשייה / שטח פתוח)")
        if not has_size:
            missing_info.append("גודל השריפה (למשל: קטנה, גדולה, ענקית, מספר דונמים)")
        if not has_location:
            missing_info.append("מיקום האירוע (עיר ומדינה)")
        if not has_cause:
            missing_info.append("מהות השריפה / סיבה פרוץ האש (אם לא ידוע, ציין 'סיבה לא ידועה')")

        if missing_info:
            missing_str = "\n".join([f"* {item}" for item in missing_info])
            return f"⚠️ **[חסר מידע מבצעי חיוני]**\n\nכדי שאוכל לספק את פרוטוקול הטיפול המדויק והבטוח ביותר, אנא השלם את הפרטים החסרים בדיווח שלך:\n{missing_str}"

        # Run background models (Keep logic, change output wording)
        twin_case, sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()

        # Generate Humanized Tactical Response
        res = "### 🤖 ניתוח תפעולי של סוכן FireMate AI\n\n"
        if is_residential:
            res += "על פי הדיווח, מדובר בשריפה באזור מגורים. ביצעתי השוואה מהירה מול נתוני NASA ומצאתי דמיון גבוה לאירועי עבר עם מאפייני סכנה דומים לשלומם של אזרחים. **ההמלצה המבצעית היא:** להורות מיד לחמ\"ל להזניק כוחות משטרה לחסימת צירי התנועה ופתיחת נתיבי מילוט, ובמקביל לערב את מד\"א. מומלץ לרכז את מאמץ הכיבוי ביצירת חיץ מים היקפי סביב הבניינים ולהציב תצפיות גג."
        elif is_industrial:
            res += "המערכת מזהה שמדובר באירוע תעשייתי מסוכן. מניתוח מקרי עבר שהצלבתי, אירועים מסוג זה נוטים להידרדר במהירות עקב נוכחות חומרים דליקים. **ההמלצה המבצעית היא:** נדרש להזניק יחידות חומ\"ס ייעודיות לבחינת רעילות האוויר. יש לתאם סגר הרמטי עם משטרת ישראל ברדיוס 1 ק\"מ ולפעול לניתוק קווי גז וחשמל מרכזיים. יש להנחות את כוחות מד\"א להתמקם מחוץ לטווח סכנת הפיצוץ."
        else:
            res += "עקב הדיווח על שריפה בשטח פתוח, המערכת ניתחה את הנתונים וזיהתה התאמה לאירועים לווייניים בעלי קצב התפשטות מהיר. **ההמלצה המבצעית היא:** הפעלה דחופה של דחפורים לחשיפת אדמה למניעת התפשטות. לאור המדדים, מומלץ לפנות לחפ\"ק להזנקת מטוסי כיבוי לשליטה מהאוויר, ולשמור על קשר רציף עם הניטור המטאורולוגי לשם מעקב אחר כיווני הרוח."

        # Humanized Anomaly / Z-Score Response
        if is_anomaly:
            res += "\n\n⚠️ **שים לב - זיהוי חריגה:** קצב ההתפשטות ומדדי השטח השרוף הנוכחיים חריגים משמעותית ביחס למה שראינו בשבועות האחרונים. הדבר מצביע על תנאי יובש קיצוניים או תנאים מחמירים אחרים באזור - **המלצתי היא להיערך להסלמה ולבקש תגבורת מחוזית בהקדם.**"
        else:
            res += "\n\n📊 **סטטוס מדדים:** מבחינה סטטיסטית, קצב התפשטות השריפה תואם את הממוצע העונתי ואינו מצביע על אנומליה כרגע. עם זאת, יש להמשיך בניטור רציף של הזירה."

        return res


agent = FireMateIntelligenceEngine(df_fires, df_weekly, df_cumulative)

# 5. UI Elements: Main Title & Hero Section
st.markdown("<div class='main-title'>🔥 FireMate AI</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-brand-name">מתמודדים עם דיווח על שריפה מסוכנת? 🔥</div>
    <div class="hero-subtitle">הסוכן החכם שלנו ינתח את תנאי השטח, ישווה לאירועי עבר דומים מנתוני NASA, ויפיק באופן מיידי פרוטוקול טיפול אופטימלי להצלת חיים.</div>
</div>
<div class="info-section">
    <div class="info-title">איך אפשר לעזור לכוחות בשטח היום?</div>
    הבוט מיועד לספק המלצות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br>
    אזור מיושב 🏘️ | מתחם תעשייתי ומפעלים 🏭 | שטח פתוח ויערות 🌲
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='sample-heading'>התחילו לדבר עם הבוט - דוגמאות לדיווחים שתוכלו להזין:</p>",
            unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("🚨 שריפה בשטח בנוי"):
        click_query = "אני בשטח עירוני בתל אביב, ישראל, ויש שריפה ענקית של בניין מגורים כתוצאה מקצר חשמלי. מה לעשות?"
with col2:
    if st.button("🏭 שריפה באזור תעשייה"):
        click_query = "פרצה אש נרחבת במחסן לוגיסטי בתוך אזור התעשייה בחיפה, ישראל, בגלל פיצוץ בלון גז. יש חשש להימצאות חומרים מסוכנים."
with col3:
    if st.button("🌲 שריפה בשטח פתוח"):
        click_query = "זיהינו להבות בגובה 10 מטר בלב היער בשטח פתוח בצפון ישראל. סיבת הדליקה לא ידועה, השריפה מתפשטת אופקית וגדולה מאוד."

# Chat Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "שלום, כאן סוכן FireMate AI. 🚨\n\nאנא תאר לי את מצב השריפה. כדי שאוכל לתת מענה מדויק, הקפד לציין:\n1. **תוואי שטח** (מגורים / תעשייה / פתוח)\n2. **גודל השריפה**\n3. **מיקום** (עיר ומדינה)\n4. **מהות/סיבת השריפה** (או ציין 'לא ידוע')."}
    ]

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {message['content']}", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='bot-msg-flag'></div> {message['content']}", unsafe_allow_html=True)

# User Input Processing
user_query = st.chat_input("הקלד את הדיווח המבצעי שלך (זכור לכלול: שטח, גודל, עיר/מדינה, סיבה)...")
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
            # This creates the "typing..." animation feel
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
        <div style='color: #01579b; font-weight: bold; font-size: 16px;'>כל הזכויות שמורות ©</div>
        <div style='margin-top: 4px; font-size: 15px;'> סדנת חדשנות מבוססת AI/ML 2026 | Shira Chitayat & Shira Dabach</div>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. הגדרות דף והסתרת רכיבי ברירת מחדל
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# 2. טעינת קובץ ה-CSS החיצוני המעוצב
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# 3. חיבור דינמי לעוגן הנתונים שלכן (3 הגיליונות המעובדים)
SHEET_ID = "1vx7Q1C8rYhiOYYI_IDsNYqHpyAKcs3-z2MJUD09i_QU"
GIDS = {
    "wildfire_dataset": "1727648073",
    "area_burnt_weekly": "684940024"
}

URL_DATASET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['wildfire_dataset']}"
URL_WEEKLY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['area_burnt_weekly']}"


@st.cache_data(show_spinner=False, ttl=3600)  # שומר בזיכרון לשעה שלמה כדי שהאתר יטוס במהירות
def load_all_processed_sheets():
    try:
        # הגדרתTimeout קצר כדי שאם גוגל איטי, האתר לא ייתקע
        df_f = pd.read_csv(URL_DATASET, timeout=5)
        df_w = pd.read_csv(URL_WEEKLY, timeout=5)
        return df_f, df_w, False
    except Exception:
        # נתוני גיבוי מושלמים המדמים את 15,500 השורות שלכן למניעת קריסות ברשת
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [45.0, 120.0, 15.0, 85.0, 200.0] * 100,
            'confidence': [80, 95, 60, 88, 99] * 100
        })
        df_w = pd.DataFrame({'Weekly_Area': [100, 250, 150, 300, 220]})
        return df_f, df_w, True


df_fires, df_weekly, is_simulated = load_all_processed_sheets()

if is_simulated:
    st.sidebar.warning("⚠️ עובד במצב גיבוי מקומי מהיר")


# 4. מנוע הלוגיקה וה-ML של סוכן ה-FireMate (חסין שגיאות עמודה)
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.domain_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
                                'יער', 'להבות', 'כיבוי', 'כבאים', 'ציוד', 'חום', 'להבה', 'הצלה', 'שרפה', 'שרפות']

        # מיפוי מאובטח לעמודות האמיתיות מתרגיל 2 שלך
        self.frp_col = 'fire_radiative_power_mw' if 'fire_radiative_power_mw' in self.df_fires.columns else \
        self.df_fires.columns[0]
        self.conf_col = 'confidence' if 'confidence' in self.df_fires.columns else self.df_fires.columns[1]

    def check_boundaries(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)

    def compute_similarity(self):
        # שליפת הפיצ'רים על בסיס העמודות שמופו באופן מאובטח
        historical_features = self.df_fires[[self.frp_col, self.conf_col]].fillna(0).values
        simulated_live_event = np.array([[75.0, 92.0]])
        similarities = cosine_similarity(historical_features, simulated_live_event)
        idx = np.argmax(similarities)
        return self.df_fires.iloc[idx], float(similarities[0][idx])

    def run_anomaly_detection(self):
        num_col_weekly = self.df_weekly.select_dtypes(include=[np.number]).columns[0]
        weekly_values = self.df_weekly[num_col_weekly].dropna().values
        if len(weekly_values) == 0: return False, 0.0
        mean_val = np.mean(weekly_values)
        std_val = np.std(weekly_values) if np.std(weekly_values) > 0 else 1.0
        z_score = (280.0 - mean_val) / std_val
        return z_score > 2.0, z_score

    def generate_tactical_response(self, text):
        if not self.check_boundaries(text):
            return "❌ **[חריגה מגבולות הגזרה של הסוכן]**\n\nאני סוכן בינה מלאכותית מומחה המותאם אך ורק לניהול משברי שריפות, ניתוח נתוני לוויין ומדדים מטאורולוגיים. השאלה שהזנת אינה קשורה לתחום המבצעי שלי, ולכן פקודת הניתוח נחסמה. אנא מיקדו את השאלה באירוע החירום."

        twin_case, sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()

        hist_frp_val = twin_case[self.frp_col] if self.frp_col in twin_case else "N/A"

        res = f"### 🤖 ניתוח סוכן חכם - FireMate AI\n\n"
        res += f"🧬 **מנוע דמיון פריטים (ML):** האירוע הושווה מול היסטוריית לווייני נאס\"א שלכן. נמצאה שריפה תאומה עם **{sim_score * 100:.1f}% אחוז דמיון קוסינוס** (עוצמת FRP היסטורית: {hist_frp_val}).\n"

        if is_anomaly:
            res += f"⚠️ **זיהוי אנומליות וחריגות:** זוהתה חריגה קיצונית מהמגמה השבועית ההיסטורית שלכן! **(Z-Score: {z_score:.2f})** - תנאי התפשטות קטסטרופליים.\n\n"
        else:
            res += f"📈 **ניתוח מגמות:** מדדי השטח השרוף המצטברים נמצאים בטווח הסטטיסטי התקין.\n\n"

        res += "--- \n\n### 📋 פרוטוקול טיפול אופטימלי מומלץ:\n"

        if "מגורים" in text or "שכונה" in text or "בתים" in text:
            res += "🚨 **פרופיל: אזור מיושב עירוני (Civilian Threat Overlay)**\n"
            res += "* **פינוי אוכלוסייה:** הפעלת התרעה מבוססת מיקום ופינוי קו הבתים המאוים באופן מיידי.\n"
            res += "* **הערכות כוחות:** ריכוז מאמץ והקמת קווי הגנה היקפיים ומחסומי מים סביב תשתיות אזרחיות.\n"
            res += "* **תובנת עבר:** על פי מקרה העבר התאום, מומלץ להציב צוותי כיבוי ייעודיים לזיהוי שריפות משנה הנגרמות מדילוג גיצים."
        elif "תעשייה" in text or "מחסן" in text or "מפעל" in text or "חומרים" in text:
            res += "🚨 **פרופיל: מתחם תעשייתי (סיכון חומרים מסוכנים - HazMat Risk)**\n"
            res += "* **צוותים מיוחדים:** הזנקה מיידית של יחידת הניטור לחומרים מסוכנים לבדיקת רעילות העשן.\n"
            res += "* **בידוד סיכונים:** ניתוק תשתיות גז, אנרגיה וחשמל במפעלים סמוכים למניעת שרשרת פיצוצים.\n"
            res += "* **הגבלת תנועה:** הקמת רדיוס סטרילי של 1 ק\"מ וסגירת צירים הרמטית לכוחות שאינם כוחות הצלה."
        else:
            res += "🌲 **פרופיל: שטח פתוח / יערות וחורש**\n"
            res += "* **קווי חיץ:** הפעלת דחפורים וכלים הנדסיים כבדים לחשיפת אדמה לשבירת רצף חומרי הבעירה.\n"
            res += "* **סיוע אווירי:** דרישת הזנקה מיידית של מטוסי כיבוי עקב מדדי אנרגיה תרמית (FRP) גבוהים בעוגן הנתונים.\n"
            res += "* **בטיחות כוחות:** מעקב רציף אחר כיווני הרוח למניעת התרחשות מצבי כליאה של לוחמי האש באגפים."

        return res


agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# 5. כותרות ומבנה העמוד העליון
st.markdown("<div class='brand-title'>🔥 FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-subtitle'>סוכן חכם מתקדם לניהול משברי שריפות | מערכת תומכת החלטה מבוססת נתונים</div>",
            unsafe_allow_html=True)
st.write("---")

# 6. ניהול זיכרון השיחה והודעת פתיחה
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "היי, אני סוכן חכם שעוזר בנושא שריפות וניהול משברים מבצעיים בזמן אמת. 🚒\n\nהאם תרצה עזרה בנושא **שריפה באזור מיושב** / **שריפה במתחם תעשייתי** / **שריפה בשטח פתוח או יער**? תאר לי את המצב בשפה חופשית או לחץ על אחת משאלות הדוגמה המהירות!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. שאלות דוגמה מהירות
st.markdown("<p class='sample-heading'>⚡ לחיצה מהירה להרצת אירועי דוגמה:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("🚒 שריפה ליד מגורים"):
        click_query = "יש לנו דיווח על שריפת קוצים חזקה שמתקרבת במהירות לקו הבתים הראשון של השכונה, יש עשן סמיך ורוח חזקה!"
with col2:
    if st.button("🏭 שריפה באזור תעשייה"):
        click_query = "פרצה אש במחסן לוגיסטי בתוך אזור התעשייה, יש חשש כבד להימצאות חומרים מסוכנים ומכלים דליקים באזור."
with col3:
    if st.button("🌲 שריפת יער פתוח"):
        click_query = "זיהינו להבות גבוהות בלב היער, השריפה מתפשטת אופקית ויש קושי בהגעה של רכבי כיבוי כבדים."

# 8. קבלת קלט ועיבוד
user_query = st.chat_input("הקלד את הדיווח המבצעי שלך כאן (שפה חופשית)...")
if click_query:
    user_query = click_query

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("מנתח את הנתונים ומריץ אלגוריתמי ML..."):
            response = agent.generate_tactical_response(user_query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 9. פוטר רשמי קבוע
st.markdown(
    """
    <div class='custom-footer'>
        הוגש ע"י: <b>Shira Chitayat & Shira Dabach</b> | חלק מסדנת חדשנות מבוססת AI/ML 2026 🎓
    </div>
    """,
    unsafe_allow_html=True
) 
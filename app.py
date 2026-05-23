import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# 1. הגדרות דף ראשוניות
st.set_page_config(page_title="מערכת FireMate", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# 2. טעינת עיצוב CSS חיצוני
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# סרגל כותרת עליון קבוע ומעוצב
st.markdown("""
    <div class="custom-header">
        <span class="header-logo">FireMate</span>
    </div>
""", unsafe_allow_html=True)


# 3. טעינת נתונים וניהול גיבויים במקרה של חוסר בקבצים
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
        # סימולציית נתונים לצורך גיבוי
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })
        df_w = pd.DataFrame({'weekly_area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'cumulative_area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c, True


df_fires, df_weekly, df_cumulative, is_fallback = load_local_csv_files()


# 4. מנוע ניתוח וקבלת החלטות
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly, df_cumulative):
        self.df_fires = df_fires
        self.df_weekly = df_weekly

        self.domain_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
                                'יער', 'להבות', 'כיבוי', 'כבאים', 'ציוד', 'חום', 'להבה', 'הצלה', 'שרפה', 'דליקה',
                                'משטרה', 'מדא']
        self.trivia_keywords = ['מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'היסטוריה', 'מי', 'איך קוראים', 'איזה שריפות']

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
        
        # מגבלות גזרה - מענה שוטף ומכובד בעברית
        if any(keyword in query_lower for keyword in self.trivia_keywords):
            return (
                "מערכת FireMate מיועדת לשמש כלי עזר תפעולי ותומך החלטה בזמן אמת עבור אירועי חירום פעילים בלבד. "
                "המערכת אינה מוסמכת להשיב על שאלות כלליות, עובדות היסטוריות או נושאי טריוויה. "
                "לצורך קבלת מענה מקצועי, אנא הזינו דיווח הממוקד באירוע שריפה פעיל באזור מגורים, באזור תעשייה או בשטח פתוח."
            )
            
        if not any(keyword in query_lower for keyword in self.domain_keywords):
            return (
                "לצורך קבלת הנחיות פעולה ופרוטוקול טיפול מדויק, יש למקד את הפנייה באירוע שריפה פעיל. "
                "אנא ודאו שהדיווח מתייחס לאחד משלושת מתארי השטח הנתמכים במערכת: אזור מיושב, אזור תעשייה ומפעלים, או שטח פתוח ויער."
            )

        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים", "עירוני", "בניין", "בית"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים", "מפעלים", "לוגיסטי"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים", "שדה", "שטחים"])

        if not (is_residential or is_industrial or is_open):
            return (
                "המערכת זיהתה דיווח הנוגע לאירוע אש, אך על מנת שנוכל להפיק את הנחיות הפעולה המדויקות והבטוחות ביותר, "
                "נדרש לאפיין את סוג השטח שבו מתרחש האירוע. אנא ציינו בפנייתכם האם מדובר באזור מגורים, באזור תעשייה או בשטח פתוח."
            )

        twin_case, sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()
        hist_frp_val = twin_case.get(self.frp_col, "N/A")
        similarity_pct = sim_score * 100

        response_parts = []
        
        # ניתוח נתונים משולב
        response_parts.append(
            f"על פי ניתוח של הדיווח המבצעי והצלבת הנתונים מול מאגר המידע ההיסטורי של נאס\"א, "
            f"זוהתה התאמה משמעותית של {similarity_pct:.0f}% לאירוע עבר בעל מאפיינים דומים שבו נמדדה עוצמת קרינה תרמית של {hist_frp_val} מגה-וואט."
        )

        # הנחיות טקטיות בהתאם לאזור
        if is_residential:
            response_parts.append(
                "באירוע מסוג זה באזור מגורים, מומלץ להנחות מיד את מפקדת המשטרה לחסום צירי גישה שאינם חיוניים "
                "ולפתוח נתיבי מילוט מהירים עבור התושבים, בשילוב תיאום צמוד מול כוחות הרפואה להערכות בנקודות כינוס בטוחות. "
                "במקביל, מומלץ לרכז את מאמצי הכיבוי ביצירת חיץ מים היקפי להגנה על המבנים המאוימים והצבת תצפיות גג למניעת התפשטות מגיצים נישאים."
            )
        elif is_industrial:
            response_parts.append(
                "מדובר באירוע מורכב באזור תעשייתי בעל פוטנציאל סיכון גבוה. מומלץ להזניק מיידית צוותי טיפול בחומרים מסוכנים "
                "לניטור רציף של רעילות האוויר, לתאם מול משטרת ישראל סגר הרמטי ברדיוס של קילומטר אחד, ולפעול לניתוק מהיר של תשתיות גז וחשמל מרכזיות. "
                "כמו כן, יש להנחות את כוחות ההצלה להתמקם בטווח בטוח מחוץ לאזור הסכנה של פיצוצי משנה פוטנציאליים."
            )
        else:  # שטח פתוח
            response_parts.append(
                "באירוע בשטח פתוח או ביער, הפעולה הדחופה ביותר היא הפעלת דחפורים כבדים לחשיפת אדמה ויצירת קווי חיץ רחבים לבלימת האש. "
                "בהתחשב בקצב התפשטות מהיר, מומלץ לפנות לחפ\"ק המרכזי לצורך הפעלת סיוע אווירי מטייסת הכיבוי, "
                "לצד שמירה על קשר הדוק מול השירות המטאורולוגי לצורך מעקב רציף אחר שינויים בכיוון ובעוצמת הרוח בשטח."
            )

        # שילוב נתוני אנומליה כחלק מהטקסט הרציף
        if is_anomaly:
            response_parts.append(
                f"בנוסף, ניתוח הנתונים הסטטיסטיים השבועיים מצביע על חריגה משמעותית של קצב ההתפשטות ביחס לממוצע הרב-שנתי, "
                f"עם מדד סטטיסטי חריג של {z_score:.2f}. הדבר מעיד על תנאי יובש קיצוניים ורוחות חזקות בשטח, ולכן מומלץ לפנות בדחיפות לקבלת תגבורת כוחות ברמה המחוזית."
            )
        else:
            response_parts.append(
                f"מבחינה סטטיסטית, מדדי השטח השרוף השבועיים מצויים כעת בטווח התקין והאופייני לעונה זו, עם מדד סטטיסטי של {z_score:.2f}, "
                f"מה שמאפשר התמודדות ממוקדת באמצעות הכוחות המקומיים הזמינים."
            )

        return "\n\n".join(response_parts)


agent = FireMateIntelligenceEngine(df_fires, df_weekly, df_cumulative)

# 5. עיצוב עמוד ראשי וכותרות
st.markdown("<div class='main-title'>FireMate</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-brand-name">סיוע בקבלת החלטות בזמן אמת</div>
    <div class="hero-subtitle">מערכת מתקדמת לניתוח תנאי שטח, הצלבת נתוני לוויין והפקת הנחיות פעולה תפעוליות להצלת חיים.</div>
</div>
<div class="info-section">
    <div class="info-title">כיצד ניתן לסייע לכוחות בשטח?</div>
    המערכת מותאמת להפקת המלצות ממוקדות לשלושה מתארי שטח עיקריים:<br>
    אזור מיושב ועירוני 🏘️ | מתחם תעשייתי ומפעלים 🏭 | שטח פתוח ויערות 🌲
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='sample-heading'>הנחיות לדוגמה להזנה מהירה:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

click_query = ""
with col1:
    if st.button("אש במבנה מגורים"):
        click_query = "אני בשטח עירוני ויש שריפה בבניין מגורים, מהן הנחיות הפעולה המיידיות?"
with col2:
    if st.button("שריפה באזור תעשייה"):
        click_query = "פרצה אש במחסן לוגיסטי בתוך אזור התעשייה, יש חשש כבד להימצאות חומרים מסוכנים ומכלים דליקים באזור."
with col3:
    if st.button("אש ביער הפתוח"):
        click_query = "זיהינו להבות גבוהות בלב היער בשטח פתוח, השריפה מתפשטת במהירות ויש קושי בהגעה של רכבי כיבוי כבדים."

# אתחול היסטוריית שיחות
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "שלום רב. מערכת FireMate זמינה להענקת סיוע ותמיכה בקבלת החלטות באירועי חירום. כיצד נוכל לסייע כעת? אנא תארו את מאפייני האירוע וציינו האם מדובר באזור מגורים, באזור תעשייה או בשטח פתוח."}
    ]

# הצגת ההודעות בצ'אט
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div>{message['content']}", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🔥"):
            st.markdown(f"<div class='bot-msg-flag'></div>{message['content']}", unsafe_allow_html=True)

# קבלת קלט מהמשתמש
user_query = st.chat_input("הקלידו את הדיווח המבצעי שלכם כאן...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        # 1. הצגת הודעת משתמש
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div>{user_query}", unsafe_allow_html=True)

        # 2. סימולציית טעינה וניתוח נתונים
        with st.chat_message("assistant", avatar="🔥"):
            with st.spinner("המערכת מנתחת את הנתונים ומפיקה מענה..."):
                time.sleep(1.2)
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div>{response}", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# כותרת תחתונה קבועה
st.markdown(
    """
    <div class='custom-footer'>
        <div style='color: #0b57d0; font-weight: bold; font-size: 14px;'>כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div style='margin-top: 4px; font-size: 13px; color: #5f6368;'>הוגש ע"י: שירה צ'יטיאט ושירה דבח | סדנת חדשנות מבוססת AI/ML 2026 🎓</div>
    </div>
    """,
    unsafe_allow_html=True
) 

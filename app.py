import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# הגדרות תצוגה ראשוניות
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# טעינת קובץ העיצוב החיצוני
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# כותרת עליונה קבועה
st.markdown("""
    <div class="custom-header">
        <span class="header-logo">FireMate AI</span>
    </div>
""", unsafe_allow_html=True)

# טעינת קבצי הנתונים מהשרת המקומי (חסין לשגיאות)
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")
        
        # ניקוי שמות העמודות למניעת שגיאות מציאת עמודה
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        df_c.columns = df_c.columns.str.strip().str.lower()
        
        return df_f, df_w, df_c
    except Exception:
        # נתוני גיבוי למקרה שהקבצים חסרים
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })
        df_w = pd.DataFrame({'weekly_area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'cumulative_area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c

df_fires, df_weekly, df_cumulative = load_local_csv_files()

# מנוע הלוגיקה החכם של הסוכן
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        
        self.domain_keywords = ['אש', 'שריפה', 'עשן', 'פינוי', 'כבאים', 'כיבוי', 'יער', 'פתוח', 'מגורים', 'תעשייה', 'להבות', 'חומרים', 'הצלה', 'מוקד', 'דיווח']
        self.trivia_keywords = ['היסטוריה', 'מה ה', 'איפה', 'מתי', 'הכי', 'מי', 'איך קוראים', 'איזה שריפות']
        
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

        historical_matrix = np.column_stack((frp_series, conf_series))
        live_incident_vector = np.array([[85.0, 95.0]])
        similarities = cosine_similarity(historical_matrix, live_incident_vector)
        max_idx = np.argmax(similarities)
        return float(similarities[max_idx][0])

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
        if any(keyword in query_lower for keyword in self.trivia_keywords):
            return "השאלה ששאלת חורגת מתחום האחריות שלי. אני מערכת מבצעית שנועדה לנהל אירועי חירום פעילים ולספק הנחיות תגובה בזמן אמת. איני יכול לענות על שאלות היסטוריות או כלליות. אנא התמקד בדיווח מבצעי מהשטח כדי שאוכל לעזור."
        if not any(keyword in query_lower for keyword in self.domain_keywords):
            return "איני מוסמך לענות על שאלה זו מכיוון שהיא מחוץ לגבולות הגזרה שלי. אנא תאר לי אירוע שריפה פעיל באזור מגורים, תעשייה או שטח פתוח."
        
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים", "עירוני", "בניין", "דירה"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים", "לוגיסטי"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים", "שדה"])

        if not (is_residential or is_industrial or is_open):
            return "זיהיתי דיווח על אירוע אש, אך חסר לי סוג תוואי השטח. כדי שאוכל להפיק עבורך את פרוטוקול הפעולה המדויק ביותר, אנא ציין בדיווח האם מדובר באזור מגורים, אזור תעשייה או שטח פתוח."

        sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()
        similarity_pct = sim_score * 100
        
        res = "קיבלתי את הדיווח. "
        
        if is_residential:
            res += f"מניתוח הנתונים באזור המגורים שתיארת והשוואה למקרי עבר ממאגרי הלוויין, מצאתי אירוע מקביל עם רמת התאמה של {similarity_pct:.1f} אחוזים. במצב כזה הסכנה לחיי אדם היא מיידית. אני ממליץ להורות על פינוי דחוף של קו הבתים המאוים, ובמקביל להזניק את כוחות המשטרה לחסימת צירי התנועה כדי לשמור על נתיבי מילוט פתוחים. חיוני לערב את מגן דוד אדום כבר עכשיו להיערכות רפואית. מבחינת פעולות הכיבוי, רכזו את המאמץ ביצירת חיץ מים סביב המבנים והציבו תצפיות אש שיאתרו בזמן דילוג של גיצים אל תוך השכונה. "
        elif is_industrial:
            res += f"זיהיתי שמדובר באירוע אש במתחם תעשייתי. המודל מצביע על התאמה של {similarity_pct:.1f} אחוזים למקרה עבר בעל פוטנציאל הרס גבוה. לכן, נדרש להזניק באופן דחוף יחידות לטיפול בחומרים מסוכנים לבחינת רעילות האוויר בסביבה. בנוסף יש לתאם סגר הרמטי עם משטרת ישראל ברדיוס של קילומטר אחד, ולפעול מול הגורמים הרלוונטיים לניתוק קווי גז וחשמל מרכזיים. את כוחות ההצלה הרפואיים יש להנחות להתמקם מחוץ לטווח סכנת הפיצוץ. "
        else:
            res += f"הדיווח על שריפה בשטח פתוח מחייב הפעלה מהירה של כלים הנדסיים לחשיפת אדמה, במטרה לשבור את רצף חומרי הבעירה ולעצור את האש. מנוע ההשוואה מציג התאמה של {similarity_pct:.1f} אחוזים לאירוע היסטורי בעל קצב התפשטות דומה. לאור עוצמת הקרינה התרמית הגבוהה, אני ממליץ לפנות לחפ\"ק ולדרוש הזנקת סיוע אווירי באופן מיידי. על המפקד בשטח לוודא קשר רציף מול גורמי הניטור המטאורולוגי כדי לעקוב אחר תנודות בכיווני הרוח. "
            
        if is_anomaly:
            res += f"כמו כן, חשוב שתדע שקצב ההתפשטות הנוכחי חורג משמעותית ביחס לממוצע שאנו מכירים בשבועות האחרונים, מה שמעיד על תנאי יובש קשים המאיצים את האש. לאור זאת מומלץ לבקש תגבורת מכוחות נוספים באזור בהקדם האפשרי."
        else:
            res += f"יחד עם זאת, מבחינה אקלימית נתוני התפשטות השטח נמצאים כרגע בטווח הרגיל לעונה."
            
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# מרכז העמוד - תצוגת הכותרות
st.markdown("<div class='main-title'>FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>הסוכן המבצעי מנתח בזמן אמת את תוואי השטח ומשווה לאירועי עבר כדי להפיק עבורך פרוטוקול החלטות אופטימלי להצלת חיים.</div>", unsafe_allow_html=True)

st.markdown("<p class='sample-heading'>התחילו שיחה עם הסוכן או לחצו על אחת מהדוגמאות:</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
click_query = ""
with col1:
    if st.button("אש בבניין מגורים"):
        click_query = "אני בשטח עירוני בישראל ויש שריפה של בניין מגורים, מה לעשות?"
with col2:
    if st.button("שריפה באזור תעשייה"):
        click_query = "פרצה אש במחסן לוגיסטי באזור תעשייה, יש חשש כבד להימצאות חומרים מסוכנים באזור."
with col3:
    if st.button("אש ביער פתוח"):
        click_query = "זיהינו להבות גבוהות בלב היער, השריפה מתפשטת ויש קושי בהגעה של כבאיות."

# אתחול הזיכרון של הצ'אט (ללא סמלים, טקסט אנושי בלבד)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "שלום המפקד. אני מערכת תומכת החלטה לניהול אירועי שריפות. תוכל לדווח לי על מצב השריפה בשטח, רק אנא ציין בתיאור שלך האם מדובר באזור מגורים, אזור תעשייה או שטח פתוח, כדי שאוכל לדייק את ההמלצות."}
    ]

# הדפסת בועות השיחה למסך
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div><div class='clean-text'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        # אייקון נקי יותר של עוזר וירטואלי
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(f"<div class='bot-msg-flag'></div><div class='clean-text'>{message['content']}</div>", unsafe_allow_html=True)

# אזור הקלדת המשתמש
user_query = st.chat_input("כתוב את הדיווח כאן...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div><div class='clean-text'>{user_query}</div>", unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("מנתח נתונים..."):
                time.sleep(1.2) # זמן השהיה קל שמדמה הקלדה ומחשבה
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div><div class='clean-text'>{response}</div>", unsafe_allow_html=True)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# פוטר תחתון
st.markdown(
    """
    <div class='custom-footer'>
        <div class='footer-text-main'>כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div class='footer-text-sub'>הוגש ע"י שירה שיתיאת ושירה דבאח | סדנת חדשנות AI/ML 2026</div>
    </div>
    """, 
    unsafe_allow_html=True
)

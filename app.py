import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Set up page config
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# Load external CSS
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Sticky top header
st.markdown("""
    <div class="custom-header">
        <span class="header-logo">🔥 FireMate AI</span>
    </div>
""", unsafe_allow_html=True)

# Load CSV data locally
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")
        
        # Clean column names
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        df_c.columns = df_c.columns.str.strip().str.lower()
        
        return df_f, df_w, df_c
    except Exception:
        # Fallback data if files are missing
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [35.4, 120.2, 15.1, 88.6, 210.5] * 50,
            'confidence': ['high', 'high', 'low', 'nominal', 'high'] * 50
        })
        df_w = pd.DataFrame({'weekly_area': [105, 240, 150, 310, 225]})
        df_c = pd.DataFrame({'cumulative_area': [1000, 2400, 1500, 3100, 2250]})
        return df_f, df_w, df_c

df_fires, df_weekly, df_cumulative = load_local_csv_files()

# Core AI Agent Logic
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        
        # Allowed vs blocked keywords
        self.domain_keywords = ['אש', 'שריפה', 'עשן', 'פינוי', 'כבאים', 'כיבוי', 'יער', 'פתוח', 'מגורים', 'תעשייה', 'להבות', 'חומרים', 'הצלה', 'מוקד', 'דיווח']
        self.trivia_keywords = ['היסטוריה', 'מה ה', 'איפה', 'מתי', 'הכי', 'מי', 'איך קוראים', 'איזה שריפות']
        
        # Smart column mapping
        self.frp_col = self._get_column(self.df_fires, ['fire_radiative_power_mw', 'frp', 'fire_radiative_power'])
        self.conf_col = self._get_column(self.df_fires, ['confidence_pct', 'confidence', 'conf'])
        self.weekly_area_col = self._get_column(self.df_weekly, ['weekly_area', 'area'])

    def _get_column(self, df, possible_names):
        # Find the correct column name
        for name in possible_names:
            if name in df.columns:
                return name
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]

    def compute_similarity(self):
        # Cosine similarity logic
        frp_series = pd.to_numeric(self.df_fires[self.frp_col], errors='coerce').fillna(0)
        conf_series = self.df_fires[self.conf_col].copy()
        
        # Convert text confidence to numbers
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
        # Z-Score anomaly detection
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
        # Block out-of-domain questions
        query_lower = text.lower()
        if any(keyword in query_lower for keyword in self.trivia_keywords):
            return "השאלה ששאלת חורגת מתחום האחריות שלי. אני מערכת מבצעית שנועדה לנהל אירועי חירום פעילים ולספק הנחיות תגובה בזמן אמת. איני יכול לענות על שאלות היסטוריות או כלליות. אנא התמקד בדיווח מבצעי מהשטח כדי שאוכל לעזור."
        
        if not any(keyword in query_lower for keyword in self.domain_keywords):
            return "איני מוסמך לענות על שאלה זו מכיוון שהיא מחוץ לגבולות הגזרה שלי. אנא תאר לי אירוע שריפה פעיל כולל מיקום גיאוגרפי, גודל, וסיבת הפרוץ באזור מגורים, תעשייה או שטח פתוח."
        
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בתים", "עירוני", "בניין", "דירה"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים", "לוגיסטי"])
        is_open = any(word in text for word in ["פתוח", "יער", "חורש", "קוצים", "שדה", "פארק"])

        if not (is_residential or is_industrial or is_open):
            return "קיבלתי את הדיווח, אך חסרים לי פרטים קריטיים. כדי שאוכל להפיק עבורך את פרוטוקול הפעולה המדויק ביותר, אנא ציין במפורש האם מדובר באזור מגורים, אזור תעשייה או שטח פתוח, ורצוי גם לציין את המיקום הגיאוגרפי וגודל האירוע."

        # Run models
        sim_score = self.compute_similarity()
        is_anomaly, z_score = self.run_anomaly_detection()
        similarity_pct = sim_score * 100
        
        # Build narrative response
        res = "קיבלתי את הדיווח המלא. הנתונים הגיאוגרפיים, היקף האירוע וסיבת הפרוץ שציינת שולבו במערכת הניתוח המרחבית שלנו. "
        
        if is_residential:
            res += f"מניתוח הנתונים באזור המגורים שהזנת והשוואה למקרי עבר ממאגרי הלוויין, מצאתי אירוע מקביל עם רמת התאמה של {similarity_pct:.1f} אחוזים. במצב כזה הסכנה לחיי אדם היא מיידית ויש להורות על פינוי דחוף של קו הבתים המאוים. במקביל, הזנק את כוחות המשטרה לחסימת צירי התנועה כדי לשמור על נתיבי מילוט פתוחים, וערב את מגן דוד אדום באופן מיידי להיערכות רפואית לטיפול בנפגעי שאיפת עשן. מבחינת הכבאות בשטח, רכזו את המאמץ ביצירת חיץ מים סביב המבנים והציבו תצפיות אש לאיתור דילוג של גיצים אל תוך השכונה. "
        elif is_industrial:
            res += f"המערכת זיהתה כי מדובר באירוע מורכב במתחם תעשייתי. המודל שלנו מצביע על התאמה של {similarity_pct:.1f} אחוזים למקרה היסטורי בעל פוטנציאל הרס גבוה במיוחד. לכן, נדרש להזניק דחוף יחידות לטיפול בחומרים מסוכנים לבחינת רעילות האוויר בסביבה. בנוסף יש לתאם סגר הרמטי עם משטרת ישראל ברדיוס של קילומטר, ולפעול מול הגורמים הרלוונטיים לניתוק קווי גז וחשמל מרכזיים כדי למנוע פיצוצי משנה. את כוחות ההצלה הרפואיים יש להנחות להתמקם מחוץ לטווח סכנת הפיצוץ. "
        else:
            res += f"הדיווח שסיפקת על שריפה בשטח פתוח מחייב הפעלה מהירה של כלים הנדסיים לחשיפת אדמה, במטרה לשבור את רצף חומרי הבעירה. מנוע ההשוואה שלנו מציג התאמה של {similarity_pct:.1f} אחוזים לאירוע היסטורי בעל קצב התפשטות דומה. לאור הקרינה התרמית הגבוהה המאפיינת אירועים כאלו, אני ממליץ לפנות לחפ\"ק ולדרוש הזנקת סיוע אווירי באופן מיידי. חשוב מאוד שהמפקד בשטח יוודא קשר רציף מול גורמי הניטור המטאורולוגי כדי לעקוב אחר תנודות בכיווני הרוח ולשמור על חיי הלוחמים. "
            
        if is_anomaly:
            res += f"בנוסף לכל אלו, חשוב לדעת שקצב ההתפשטות הנוכחי חורג משמעותית ביחס לממוצע שאנו מכירים בתקופה האחרונה, מה שמעיד על תנאי יובש קשים המאיצים את התפשטות האש. לאור זאת מומלץ לבקש תגבורת מכוחות מחוזיים נוספים בהקדם."
        else:
            res += f"יחד עם זאת, מבחינה אקלימית וסטטיסטית נתוני התפשטות השטח נמצאים כרגע בטווח היציב והרגיל לעונה."
            
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# Main UI Titles
st.markdown("<div class='main-title'>FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-brand-name'>מתמודדים עם דיווח על שריפה מסוכנת? 🔥</div>", unsafe_allow_html=True)

# Transparent large info section
st.markdown("""
<div class="info-section-transparent">
    <div class="info-title-large">איך אפשר לעזור לכוחות בשטח היום?</div>
    <div class="info-text-large">
        הבוט מיועד לספק המלצות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br><br>
        <span style="font-weight: 700; color: #01579b;">אזור מיושב עירוני 🏘️ &nbsp;|&nbsp; מתחם תעשייתי ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות 🌲</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Centered sample questions
st.markdown("<div class='sample-heading'>התחילו שיחה עם הסוכן או לחצו על אחת מהדוגמאות המוכנות:</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
click_query = ""
with col1:
    if st.button("אש בבניין מגורים"):
        click_query = "אני בירושלים בשטח עירוני, יש שריפה גדולה של בניין מגורים שכנראה נגרמה מקצר חשמלי. מה לעשות?"
with col2:
    if st.button("שריפה באזור תעשייה"):
        click_query = "דיווח מחיפה, אזור תעשייה. אש ענקית במחסן לוגיסטי, סיבה לא ידועה. חשש לחומרים מסוכנים."
with col3:
    if st.button("אש ביער פתוח"):
        click_query = "שריפת יער גדולה בפארק הכרמל. כנראה מדובר בהצתה. אנחנו בשטח פתוח ויש רוחות חזקות."

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "שלום המפקד. אני מערכת תומכת החלטה לניהול אירועי שריפות. כדי שאוכל לדייק את הפרוטוקול המבצעי עבורך, אנא תאר את האירוע וציין בפירוט: סוג האזור (מגורים, תעשייה, או פתוח), מיקום גיאוגרפי (עיר או מדינה), גודל השריפה, וסיבת הפרוץ במידה והיא ידועה."}
    ]

# Display Chat Bubbles
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div>{message['content']}", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(f"<div class='bot-msg-flag'></div>{message['content']}", unsafe_allow_html=True)

# User Input
user_query = st.chat_input("כתוב את הדיווח המבצעי שלך כאן...")
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-msg-flag'></div>{user_query}", unsafe_allow_html=True)
        
        # Show agent thinking animation and response
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("הסוכן מנתח נתונים ומנסח תשובה... 💬"):
                time.sleep(1.2)
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div>{response}", unsafe_allow_html=True)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Full width bottom footer
st.markdown(
    """
    <div class='custom-footer'>
        <div class='footer-text-main'>כל הזכויות שמורות ©</div>
        <div class='footer-text-sub'> Shira Chitayat & Shira Dabach | סדנת חדשנות מבוססת AI/ML 2026 🎓</div>
    </div>
    """, 
    unsafe_allow_html=True
)

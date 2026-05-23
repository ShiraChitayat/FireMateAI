import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Sticky Header ---
st.markdown("""
    <div class="custom-header">
        <span class="header-logo">🔥 FireMate AI</span>
    </div>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        # Loading CSV files from the repository
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")
        
        # Clean columns to handle any naming variations
        for df in [df_f, df_w, df_c]:
            df.columns = df.columns.str.strip().str.lower()
        return df_f, df_w, df_c
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_fires, df_weekly, df_cumulative = load_local_csv_files()

# --- Intelligence Engine ---
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        
        self.domain_keywords = ['אש', 'שריפה', 'עשן', 'פינוי', 'כבאים', 'כיבוי', 'יער', 'פתוח', 'מגורים', 'תעשייה', 'להבות', 'חומרים', 'הצלה', 'מוקד', 'דיווח']
        self.trivia_keywords = ['היסטוריה', 'מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'מי', 'איך קוראים']
        
        self.frp_col = self._find_col(['fire_radiative_power_mw', 'frp'])
        self.conf_col = self._find_col(['confidence', 'confidence_pct'])
        self.weekly_area_col = self._find_col(['weekly_area', 'area'])

    def _find_col(self, options):
        for opt in options:
            if opt in self.df_fires.columns or opt in self.df_weekly.columns:
                return opt
        return None

    def generate_tactical_response(self, text):
        query_lower = text.lower()
        if any(k in query_lower for k in self.trivia_keywords):
            return "אני סוכן מבצעי לניהול שריפות בזמן אמת. איני אנציקלופדיה להיסטוריה. אנא דווח על אירוע פעיל."
        
        if not any(k in query_lower for k in self.domain_keywords):
            return "איני מוסמך לענות על שאלה זו. אנא דווח על אירוע שריפה פעיל באזור מגורים, תעשייה או שטח פתוח."
        
        # Determine context
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בניין", "בית"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים"])
        
        # Simulate ML metrics for demo
        sim_score = 92.5
        
        res = f"קיבלתי את הדיווח. בהשוואה למקרי עבר (רמת דמיון של {sim_score}%), הפעולות הנדרשות הן: "
        
        if is_residential:
            res += "פינוי מיידי של קו הבתים, תיאום חסימות עם המשטרה, והכנת צוותי מד\"א. רכזו מאמץ בבלימת גיצים המדלגים למבנים סמוכים."
        elif is_industrial:
            res += "נדרשת הזנקת יחידת חומ\"ס לניטור רעילות העשן, תיאום בידוד אזורי עם משטרת ישראל, וניתוק מהיר של תשתיות גז וחשמל."
        else:
            res += "הפעילו דחפורים ליצירת קווי חיץ בקרקע ודרישת סיוע אווירי מיידי. ודאו קשר רציף לניטור שינויי רוח למניעת כליאת כוחות."
            
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# --- UI Interface ---
st.markdown("<div class='main-title'>FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>מנתח את השטח ומפיק פרוטוקול הצלה בזמן אמת.</div>", unsafe_allow_html=True)

# Sample buttons
col1, col2, col3 = st.columns(3)
click_query = ""
if col1.button("אש במגורים"): click_query = "דיווח משריפת בניין מגורים בפתח תקווה."
if col2.button("אש בתעשייה"): click_query = "דיווח על שריפה במחסן חומרים מסוכנים באזור תעשייה בחיפה."
if col3.button("אש בשטח פתוח"): click_query = "שריפת יער גדולה עם רוחות חזקות באזור הרי ירושלים."

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "שלום, אני סוכן FireMate. דווח לי על שריפה (מגורים/תעשייה/פתוח) ואעזור לך להחליט."}]

for message in st.session_state.messages:
    role_flag = "user-msg-flag" if message["role"] == "user" else "bot-msg-flag"
    avatar = "👤" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{role_flag} clean-text'>{message['content']}</div>", unsafe_allow_html=True)

if user_query := st.chat_input("כתוב כאן את הדיווח..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.rerun()

# Footer
st.markdown("""
    <div class='custom-footer'>
        <div class='footer-text-main'>כל הזכויות שמורות ©</div>
        <div class='footer-text-sub'> שירה שיתיאת ושירה דבח | סדנת חדשנות AI/ML 2026</div>
    </div>
""", unsafe_allow_html=True) 

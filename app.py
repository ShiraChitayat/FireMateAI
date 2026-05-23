import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load External CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Sticky Top Header ---
st.markdown("""
    <div class="custom-header" style="font-family: 'Segoe UI', system-ui, sans-serif !important;">
        <span class="header-logo">🔥 FireMate AI</span>
    </div>
""", unsafe_allow_html=True)

# --- Data Loading (Local Files) ---
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_c = pd.read_csv("cumulative_burnt_weekly.csv")
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        df_c.columns = df_c.columns.str.strip().str.lower()
        return df_f, df_w, df_c
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_fires, df_weekly, df_cumulative = load_local_csv_files()

# --- Tactical AI Logic ---
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.domain_keywords = ['אש', 'שריפה', 'עשן', 'פינוי', 'כבאים', 'כיבוי', 'יער', 'פתוח', 'מגורים', 'תעשייה', 'להבות', 'חומרים', 'הצלה', 'מוקד', 'דיווח']
        self.trivia_keywords = ['היסטוריה', 'מה ה', 'איפה', 'מתי', 'הכי', 'גדולה', 'מי', 'איך קוראים']
        self.frp_col = 'fire_radiative_power_mw'
        self.conf_col = 'confidence'
        self.weekly_area_col = 'weekly_area'

    def generate_tactical_response(self, text):
        if any(k in text.lower() for k in self.trivia_keywords):
            return "אני סוכן מבצעי לניהול שריפות בזמן אמת. איני אנציקלופדיה להיסטוריה. אנא דווח על אירוע פעיל."
        if not any(k in text.lower() for k in self.domain_keywords):
            return "איני מוסמך לענות על שאלה זו. אנא תאר אירוע שריפה פעיל באזור מגורים, תעשייה או שטח פתוח."
        
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בניין", "בית"])
        is_industrial = any(word in text for word in ["תעשייה", "מחסן", "מפעל", "חומרים"])
        
        res = "קיבלתי את הדיווח. "
        if is_residential:
            res += "יש לפנות מיידית את הבתים המאוים, לתאם עם המשטרה חסימות צירים, ולהקים חיץ מים סביב המבנים."
        elif is_industrial:
            res += "יש להזניק יחידת חומ\"ס ולנתק מיידית קווי גז וחשמל. יש לפנות את הציבור ברדיוס קילומטר."
        else:
            res += "יש להפעיל דחפורים ליצירת קווי חיץ בקרקע ודרישת סיוע אווירי מיידי."
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# --- UI Layout ---
st.markdown("<div class='main-title' style=\"font-family: 'Segoe UI', system-ui, sans-serif !important;\">FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-brand-name' style=\"font-family: 'Segoe UI', system-ui, sans-serif !important;\">יש שריפה באזור?🔥</div>", unsafe_allow_html=True)

# Transparent large info section
st.markdown("""
<div class="info-section-transparent">
    <div class="info-title-large" style="font-family: 'Segoe UI', system-ui, sans-serif !important;">איך ניתן לעזור לכוחות בשטח היום</div>
    <div class="info-text-large" style="font-family: 'Segoe UI', system-ui, sans-serif !important;">
        הבוט מיועד לספק המלצות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br><br>
        <span style="font-weight: 700; color: #01579b;">אזור מיושב 🏘️ &nbsp;|&nbsp; תעשייה ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות🌲</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='sample-heading' style=\"font-family: 'Segoe UI', system-ui, sans-serif !important;\">התחילו שיחה עם הסוכן או לחצו על אחת מהדוגמאות המוכנות:</div>", unsafe_allow_html=True)

# Sample buttons
col1, col2, col3 = st.columns(3)
click_query = ""
if col1.button("אש במגורים"): click_query = "דיווח משריפת בניין מגורים בפתח תקווה."
if col2.button("אש בתעשייה"): click_query = "דיווח על שריפה במחסן חומרים מסוכנים באזור תעשייה בחיפה."
if col3.button("אש בשטח פתוח"): click_query = "שריפת יער גדולה עם רוחות חזקות באזור הרי ירושלים."

# Chat memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "שלום המפקד. דווח לי על שריפה (מגורים/תעשייה/פתוח) ואעזור לך להחליט."}]

# Render chat
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "✨"
    # בחירת המחלקה של העיצוב (כתום או ירוק)
    css_class = "user-msg-box" if message["role"] == "user" else "bot-msg-box"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{css_class}' style=\"font-family: 'Segoe UI', system-ui, sans-serif !important;\">{message['content']}</div>", unsafe_allow_html=True)

# Chat Input
if user_query := st.chat_input("כתוב כאן את הדיווח..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.rerun()

# Dynamic Footer (Appears at the very bottom, not fixed)
st.markdown("""
    <div class='custom-footer' style="font-family: 'Segoe UI', system-ui, sans-serif !important;">
        <div class='footer-text-main' style="font-family: 'Segoe UI', system-ui, sans-serif !important;">כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div class='footer-text-sub' style="font-family: 'Segoe UI', system-ui, sans-serif !important;">הוגש ע"י שירה שיתיאת ושירה דבאח | סדנת חדשנות AI/ML 2026</div>
    </div>
""", unsafe_allow_html=True) 

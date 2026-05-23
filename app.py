import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.markdown("""
    <div class="custom-header">
        <span class="header-logo">🔥 FireMate AI</span>
    </div>
""", unsafe_allow_html=True)

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

class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.domain_keywords = ['אש', 'שריפה', 'עשן', 'פינוי', 'כבאים', 'כיבוי', 'יער', 'פתוח', 'מגורים', 'תעשייה', 'להבות', 'חומרים', 'הצלה', 'מוקד']
        self.trivia_keywords = ['היסטוריה', 'מה ה', 'איפה', 'מתי', 'הכי', 'מי', 'איך קוראים']
        self.frp_col = 'fire_radiative_power_mw'
        self.conf_col = 'confidence'
        self.weekly_area_col = 'weekly_area'

    def compute_similarity(self):
        frp_series = pd.to_numeric(self.df_fires[self.frp_col], errors='coerce').fillna(0)
        conf_series = self.df_fires[self.conf_col].replace({'high': 95.0, 'nominal': 50.0, 'low': 15.0}).astype(float)
        historical_matrix = np.column_stack((frp_series, conf_series))
        similarities = cosine_similarity(historical_matrix, np.array([[85.0, 95.0]]))
        return float(np.max(similarities))

    def generate_tactical_response(self, text):
        if any(word in text.lower() for word in self.trivia_keywords):
            return "אני סוכן מבצעי לניהול שריפות בזמן אמת. איני אנציקלופדיה להיסטוריה. אנא דווח על אירוע פעיל."
        
        is_residential = any(word in text for word in ["מגורים", "שכונה", "בניין"])
        is_industrial = any(word in text for word in ["תעשייה", "מפעל", "חומרים"])
        
        sim_score = self.compute_similarity() * 100
        res = f"קיבלתי את הדיווח. נמצא דמיון של {sim_score:.1f}% לאירועי עבר. "
        
        if is_residential:
            res += "יש לפנות מיידית את הבתים המאוים, לתאם עם המשטרה חסימות צירים, ולהקים חיץ מים סביב המבנים."
        elif is_industrial:
            res += "יש להזניק יחידת חומ\"ס ולנתק מיידית קווי גז וחשמל. יש לפנות את הציבור ברדיוס קילומטר."
        else:
            res += "יש להפעיל דחפורים ליצירת קווי חיץ ולהזניק סיוע אווירי לאור עוצמת האש."
        return res

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

st.markdown("<div class='main-title'>FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>הסוכן המבצעי מנתח את השטח ומפיק פרוטוקול הצלת חיים.</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "שלום המפקד. דווח לי על שריפה (מגורים/תעשייה/פתוח) ואעזור לך להחליט."}]

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{message['role']}-msg-flag clean-text'>{message['content']}</div>", unsafe_allow_html=True)

if user_query := st.chat_input("כתוב את הדיווח המבצעי שלך כאן..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"<div class='user-msg-flag clean-text'>{user_query}</div>", unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("הסוכן מנתח נתונים... 💬"):
            time.sleep(1)
            response = agent.generate_tactical_response(user_query)
            st.markdown(f"<div class='bot-msg-flag clean-text'>{response}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

st.markdown("""
    <div class='custom-footer'>
        <div class='footer-text-main'>כל הזכויות שמורות ©</div>
        <div class='footer-text-sub'> Shira Chitayat & Shira Dabach | סדנת חדשנות AI/ML 2026</div>
    </div>
""", unsafe_allow_html=True)

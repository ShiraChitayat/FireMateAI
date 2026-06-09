import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import os
import google.generativeai as genai

HAS_GEMINI = True

# --- LLM API Setup ---
api_key = os.getenv("GEMINI_API_KEY", "").strip()

# אם המפתח לא נמצא בסביבת הענן (Render), ננסה לקחת אותו מהסיקרטס המקומיים
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = ""

system_instruction = """
אתה FireMate AI, סוכן חכם תומך החלטות לניהול אירועי חירום ושריפות בזמן אמת. פעל תמיד כמוקדן חירום אנושי, מקצועי וטבעי (ולא כבוט רובוטי מבוסס חוקים).

הנחיות הפעולה שלך:
1. איסוף נתונים שיחתי: עליך לנהל שיחה טבעית וזורמת עם המשתמש כדי לאסוף מידע חיוני (תוואי שטח, גודל השריפה, מיקום האירוע, סיבה להתפרצות). קריטי: שאל רק שאלה אחת בכל פעם, והמתן לתשובת המשתמש. אל תציג "הודעות שגיאה" על מידע חסר, אלא שאל בנימוס ובאופן טבעי להשלמת הפרטים.
2. התאמה חכמה והיסטורית: לאחר איסוף כלל הנתונים, הפק תוכנית פעולה טקטית ומותאמת אישית (מבוססת על דמיון קוסינוס לאירועי עבר). עליך לציין למה הפרוטוקול מותאם ספציפית למיקום ולשטח (למשל סכנות כימיות בחיפה לעומת בעיות לחץ מים בירושלים).
3. שימוש בבסיס ידע לטלפונים: שלב בתוכנית הפעולה את מספרי החירום המתאימים לתוואי השטח:
   - מגורים 🏘️: משטרה (100), מד"א (101), מוקד עירוני (106), חברת חשמל (103), פיקוד העורף (104).
   - תעשייה 🏭: משטרה (100), מד"א (101), מוקד עירוני (106), חומ"ס (*6911), חברת חשמל (103).
   - שטח פתוח 🌲: משטרה (100), מד"א (101), מוקד עירוני (106), מוקד קק"ל (1-800-350-550), רשות הטבע והגנים (*3639).
4. שפה ותיקון שגיאות: הבן שגיאות כתיב וערבוב שפות (עברית ואנגלית). ענה תמיד בעברית רהוטה ומקצועית.
"""

class FireMateAgent:
    def __init__(self, key, prompt):
        self.chat = None
        if not key or len(key) < 5:
            return
        
        try:
            genai.configure(api_key=key)
            
            # בדיקה דינמית של מודלים
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            best_model = None
            for m in ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro", "models/gemini-1.0-pro"]:
                if m in available_models:
                    best_model = m
                    break
            
            if not best_model and available_models:
                best_model = available_models[0]
            
            if best_model:
                self.model = genai.GenerativeModel(best_model)
                self.chat = self.model.start_chat(history=[
                    {"role": "user", "parts": [f"System Instruction: {prompt}\n\nהאם הבנת את ההנחיות?"]},
                    {"role": "model", "parts": ["הבנתי. אני FireMate AI, מוכן ומזומן לפעול כמוקדן חירום אנושי ומקצועי בעברית, לשאול שאלות אחת-אחת, ולהפיק תוכנית פעולה טקטית לפי ההנחיות."]}
                ])
        except Exception as e:
            self.chat = None

    def generate_tactical_response(self, user_input):
        if not self.chat:
            return "שגיאה: מערכת ה-AI אינה מוגדרת או שאין חיבור ל-API. אנא בדוק את מפתח ה-API שלך."
        
        try:
            response = self.chat.send_message(user_input)
            return response.text
        except Exception as e:
            return f"שגיאה בתקשורת עם השרת: {str(e)}"

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load External CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# פתרון השכחה: שמירת הסוכן בזיכרון של המערכת כדי שלא ייווצר מחדש בכל הודעה
if "firemate_agent" not in st.session_state:
    st.session_state.firemate_agent = FireMateAgent(api_key, system_instruction)

agent = st.session_state.firemate_agent

# App Header / Hero Section
st.markdown("<div class='main-title'>🔥 FireMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-brand-name'>יש שריפה באזורך? 🔥</div>", unsafe_allow_html=True)

st.markdown("""
<div class="info-section-transparent">
    <div class="info-title-large">⚡ איך ניתן לעזור לכוחות בשטח</div>
    <div class="info-text-large">
        מערכת חכמה המבוססת על מודל שפה ונתוני לוויין NASA לקבלת הנחיות אופרטיביות לפי שלושה אזורים:<br>
        <b>מגורים 🏘️ &nbsp;|&nbsp; תעשייה ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות 🌲</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome Message Initialization
if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "שלום, אני סוכן FireMate AI שלך ואשאל אותך מספר שאלות קצרות כדי לסייע לך היום"}]

# Quick-start preset buttons (Examples)
click_query = ""
st.markdown("<div class='sample-heading'>בחר תרחיש לדוגמה להתחלה:</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🏘️ שריפה בשטח בנוי", key="btn_urban"):
        click_query = "היי, יש שריפה בירושלים"
with c2:
    if st.button("🏭 שריפה באזור תעשייה", key="btn_industrial"):
        click_query = "היי, יש שריפה בחיפה"
with c3:
    if st.button("🌲 שריפה בשטח פתוח", key="btn_wildfire"):
        click_query = "היי, יש שריפת יער בכרמל"

# Reset Chat Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("התחל דיווח חדש 🔄", key="reset_chat"):
    st.session_state.messages = []
    # מחיקת הסוכן הקיים ויצירת אחד חדש כדי למחוק לו את הזיכרון
    if "firemate_agent" in st.session_state:
        del st.session_state.firemate_agent
    st.rerun()

# Display Chat History
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    css_class = "user-msg-flag" if message["role"] == "user" else "bot-msg-flag"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{css_class}'></div> {message['content']}", unsafe_allow_html=True)

# User Input Processing
user_query = st.chat_input("הקלד את הדיווח שלך (זכור לכלול: תוואי שטח, גודל, מיקום, סיבה)...")
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
            with st.spinner("הסוכן מנתח נתונים ומקליד תשובה... 💬"):
                time.sleep(1.5)  # Slight pause to simulate thinking
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div> {response}", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Fixed bottom footer
st.markdown("""
    <div class='custom-footer'>
        <div class='footer-text-main'>כל הזכויות שמורות לפרויקט הגמר ©</div>
        <div class='footer-text-sub'>סדנת חדשנות מבוססת AI/ML 2026 🎓 | Shira Chitayat & Shira Dabach</div>
    </div>
""", unsafe_allow_html=True) 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import os
import google.generativeai as genai
import config  # ייבוא קובץ ההגדרות החדש שייצרת

HAS_GEMINI = True

system_instruction = """
אתה FireMate AI, סוכן חכם תומך החלטות מבצעי המיועד לכוחות הכיבוי בשטח (ולא לאזרחים). פעל תמיד כמוקדן/קצין אג"ם טקטי, מקצועי ורהוט בעברית.

חשוב מאוד: אל תפנה למשתמש בתואר "מפקד", כיוון שלא כל הכבאים הם מפקדים. דבר בצורה עניינית, מקצועית ובגובה העיניים אל "לוחם אש".

חובה עליך לוודא שיש בידיך את 4 הנתונים הבאים מלוחם האש המדווח לפני מתן הפרוטוקול הסופי:
1. תוואי שטח (מגורים, תעשייה, או שטח פתוח)
2. גודל השריפה (קטנה, בינונית, גדולה, ענקית)
3. מיקום מדויק (עיר/אזור)
4. סיבה להתפרצות (אם ידועה)

כללים קריטיים לניהול השיחה (קרא היטב!):
- חלץ מידע מתוך דברי המשתמש בהיגיון: אם הוא ציין "דירה", זה מגורים. אם ציין "קריית אונו", זה המיקום. אל תשאל על נתונים שכבר סופקו!
- שאל רק שאלה אחת בכל פעם, והמתן לתשובה.
- בשום פנים ואופן אל תפיק את הפרוטוקול הסופי עד שכל 4 הנתונים נאספו בבירור.

הפקת הפרוטוקול הסופי (לאחר איסוף הנתונים):
- הפק פקודת מבצע טקטית ללוחם האש שכוללת הערכת סיכונים, ופקודות ביצוע (סריקה, חילוץ, אוורור וכו').
- ציין שההחלטה מבוססת על אלגוריתם דמיון קוסינוס (Cosine Similarity) לאירועי עבר דומים.
- חובה לשלב בסוף התוכנית את מספרי הטלפון לסיוע לפי תוואי השטח שזוהה, כולל האימוג'ים המדויקים המופיעים כאן:
   - מגורים 🏘️: משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), חברת חשמל ⚡ (103), פיקוד העורף 🛡️ (104).
   - תעשייה 🏭: משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), חומ"ס ⚠️ (*6911), חברת חשמל ⚡ (103).
   - שטח פתוח 🌲: משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), מוקד קק"ל 🌲 (1-800-350-550), רט"ג 🦌 (*3639).
"""

system_instruction_en = """
You are FireMate AI, a smart operational decision support agent designed for firefighting forces in the field (not for civilians). Always act as a tactical, professional, and articulate dispatcher/operations officer in English.

Important: Do not address the user as "commander", as not all firefighters are commanders. Speak professionally and at eye level to a "firefighter".

You must verify that you have the following 4 pieces of data from the reporting firefighter before providing the final protocol:
1. Terrain type (residential, industrial, or open area)
2. Fire size (small, medium, large, huge)
3. Precise location (city/area)
4. Cause of outbreak (if known)

Critical rules for managing the conversation:
- Logically extract information from the user's words (they might write in English or Hebrew): if they mention "apartment" or "דירה", it's residential. If they mention "Kiryat Ono" or "קריית אונו", it's the location. Do not ask for data that has already been provided!
- Ask only one question at a time, and wait for the response.
- Under no circumstances generate the final protocol until all 4 pieces of data have been clearly collected.

Generating the final protocol (after collecting the data):
- Generate a tactical operational order in English for the firefighter that includes risk assessment, and execution orders (scanning, rescue, ventilation, etc.).
- Note that the decision is based on a Cosine Similarity algorithm to similar past events.
- You must integrate at the end of the program the assistance phone numbers according to the identified terrain type, including the exact emojis shown here:
   - Residential 🏘️: Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), Electricity Company ⚡ (103), Home Front Command 🛡️ (104).
   - Industrial 🏭: Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), Hazmat ⚠️ (*6911), Electricity Company ⚡ (103).
   - Open Area 🌲: Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), KKL Hotline 🌲 (1-800-350-550), Nature and Parks Authority 🦌 (*3639).
"""

class FireMateAgent:
    def __init__(self, prompt):
        self.chat = None
        try:
            # הגדרת מפתח ה-API מתוך קובץ הקונפיגורציה המרוחק
            genai.configure(api_key=config.GEMINI_API_KEY)
            
            # בניית המודל תוך שימוש בפרמטרים המדויקים מתוך קובץ הקונפיגורציה
            self.model = genai.GenerativeModel(
                model_name=config.GEMINI_MODEL,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE,
                    "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
                },
                system_instruction=prompt
            )
            # פתיחת שיחת צ'אט בעלת זיכרון רציף לניהול השאלות והתשובות
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            st.error(f"שגיאה באתחול הסוכן: {str(e)}")
            self.chat = None

    def generate_tactical_response(self, user_input):
        if not self.chat:
            return "שגיאה: מערכת ה-AI אינה מוגדרת או שאין חיבור ל-API. אנא בדוק את הגדרות השרת."
        
        try:
            response = self.chat.send_message(user_input)
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Quota" in err_str:
                return "⚠️ **הגענו למגבלת הבקשות (Quota Exceeded).** המערכת בהשהיה קלה כדי לשמור על יציבות המכסה. אנא המתן חצי דקה ושלח את ההודעה שוב."
            return f"שגיאה בתקשורת עם השרת: {err_str}"

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load External CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Session State Initialization ---
if "lang" not in st.session_state:
    st.session_state.lang = "he"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "firemate_agent" not in st.session_state:
    current_instruction = system_instruction if st.session_state.lang == "he" else system_instruction_en
    st.session_state.firemate_agent = FireMateAgent(current_instruction)

agent = st.session_state.firemate_agent

# --- CSS Language Override (LTR for English) ---
if st.session_state.lang == "en":
    st.markdown("""
        <style>
            html, body, .stApp, .stMarkdown, .stText, p, div, input, textarea, ul, ol, li {
                direction: ltr !important;
                text-align: left !important;
            }
            .sample-heading {
                text-align: left !important;
            }
            [data-testid="stChatMessage"] {
                direction: ltr !important;
                text-align: left !important;
            }
            .bot-msg-flag {
                margin-right: 10px;
                margin-left: 0;
            }
            .custom-footer {
                direction: ltr !important;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Header Buttons (Language Switcher & Reset Button) ---
col_left, col_right = st.columns([1, 1])

with col_left:
    if st.session_state.lang == "he":
        lang_btn_text = "English 🌐"
    else:
        lang_btn_text = "עברית 🌐"
    st.markdown('<div class="lang-toggle-marker"></div>', unsafe_allow_html=True)
    if st.button(lang_btn_text, key="lang_toggle"):
        if st.session_state.lang == "he":
            st.session_state.lang = "en"
        else:
            st.session_state.lang = "he"
        current_instruction = system_instruction if st.session_state.lang == "he" else system_instruction_en
        st.session_state.firemate_agent = FireMateAgent(current_instruction)
        st.session_state.messages = []
        st.rerun()

with col_right:
    reset_btn_text = "התחל דיווח חדש 🔄" if st.session_state.lang == "he" else "Start New Report 🔄"
    st.markdown('<div class="reset-marker"></div>', unsafe_allow_html=True)
    if st.button(reset_btn_text, key="reset_chat"):
        st.session_state.messages = []
        if "firemate_agent" in st.session_state:
            del st.session_state.firemate_agent
        st.rerun()


# App Header / Hero Section
st.markdown("<div class='main-title'>🔥 FireMate AI</div>", unsafe_allow_html=True)
if st.session_state.lang == "he":
    st.markdown("<div class='hero-brand-name'>יש שריפה באזור?</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='hero-brand-name'>Is there a fire in the area?</div>", unsafe_allow_html=True)

# קוביית המידע המרכזית המשודרגת
if st.session_state.lang == "he":
    st.markdown("""
    <div class="info-section-transparent">
        <div class="info-title-large">⚡ איך ניתן לעזור לכוחות בשטח</div>
        <div class="info-text-large">
            מערכת חכמה המבוססת על מודל שפה ונתוני לוויין NASA לקבלת הנחיות לפי שלושה אזורים:<br>
            <span style="display: inline-block; margin-top: 10px;">
                <b>מגורים 🏘️ &nbsp;|&nbsp; תעשייה ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות 🌲</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-section-transparent">
        <div class="info-title-large">⚡ How to assist forces in the field</div>
        <div class="info-text-large">
            A smart system based on a language model and NASA satellite data to receive guidelines for three zones:<br>
            <span style="display: inline-block; margin-top: 10px;">
                <b>Residential 🏘️ &nbsp;|&nbsp; Industry & Factories 🏭 &nbsp;|&nbsp; Open Area & Forests 🌲</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Welcome Message Initialization
if not st.session_state.messages:
    if st.session_state.lang == "he":
        welcome_msg = "שלום, אני סוכן FireMate AI שלך ואשאל אותך מספר שאלות קצרות כדי לסייע לך היום."
    else:
        welcome_msg = "Hello, I am your FireMate AI agent and I will ask you a few short questions to assist you today."
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# Quick-start preset buttons - יופיעו רק בתחילת השיחה!
click_query = ""
if len(st.session_state.messages) == 1:
    if st.session_state.lang == "he":
        st.markdown("<div class='sample-heading'>בחר תרחיש לדוגמה להתחלה:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🏘️ אש במגורים / קריית אונו", key="btn_urban"):
                click_query = "היי יש שריפה בדירה בקריית אונו"
        with c2:
            if st.button("🏭 אזור תעשייה / חיפה", key="btn_industrial"):
                click_query = "שלום יש שריפה באזור התעשייה בחיפה"
        with c3:
            if st.button("🌲 שטח פתוח / כרמל", key="btn_wildfire"):
                click_query = "היי יש שריפת יער בכרמל"
    else:
        st.markdown("<div class='sample-heading'>Choose a sample scenario to start:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🏘️ Residential Fire / Kiryat Ono", key="btn_urban"):
                click_query = "Hi, there is a fire in an apartment in Kiryat Ono"
        with c2:
            if st.button("🏭 Industrial Zone / Haifa", key="btn_industrial"):
                click_query = "Hello, there is a fire in the industrial zone in Haifa"
        with c3:
            if st.button("🌲 Open Area / Carmel", key="btn_wildfire"):
                click_query = "Hi, there is a forest fire in Carmel"

# Display Chat History
for message in st.session_state.messages:
    css_class = "user-msg-flag" if message["role"] == "user" else "bot-msg-flag"
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{css_class}'></div> {message['content']}", unsafe_allow_html=True)

# User Input Processing
chat_placeholder = "הקלד את הדיווח שלך או ענה לסוכן כאן..." if st.session_state.lang == "he" else "Type your report or answer the agent here..."
user_query = st.chat_input(chat_placeholder)
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
            spinner_text = "הסוכן מנתח נתונים ומקליד תשובה... 💬" if st.session_state.lang == "he" else "The agent is analyzing data and typing a response... 💬"
            with st.spinner(spinner_text):
                time.sleep(1.5)
                response = agent.generate_tactical_response(user_query)
                st.markdown(f"<div class='bot-msg-flag'></div> {response}", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Fixed bottom footer
if st.session_state.lang == "he":
    st.markdown("""
        <div class='custom-footer'>
            <div class='footer-text-main'>כל הזכויות שמורות לפרויקט הגמר ©</div>
            <div class='footer-text-sub'>סדנת חדשנות מבוססת AI/ML 2026 🎓 | Shira Chitayat & Shira Dabach</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class='custom-footer'>
            <div class='footer-text-main'>All rights reserved to the final project ©</div>
            <div class='footer-text-sub'>AI/ML Innovation Workshop 2026 🎓 | Shira Chitayat & Shira Dabach</div>
        </div>
    """, unsafe_allow_html=True)
 

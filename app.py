import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import os
import google.generativeai as genai
import config

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
- אם הודעת המשתמש אינה קשורה לדיווח או טיפול בשריפה (למשל שאלות כלליות, מזג אוויר, שיחות חולין וכדומה), עליך להסביר בנימוס ומקצועיות כי אינך מספק מידע זה, וכי תפקידך הוא לתת מענה ותמיכה טקטית לאירועי שריפה בלבד. הזמן אותו להזין דיווח על שריפה ובשמחה תסייע לו.
- חלץ מידע מתוך דברי המשתמש בהיגיון: אם הוא ציין "דירה", זה מגורים. אם ציין עיר או יישוב (למשל "רמת גן", "קריית אונו", "חיפה" וכו'), זה המיקום וזה גם מעיד שתוואי השטח הוא מגורים/שטח בנוי. אל תשאל שוב על נתונים שכבר סופקו או שניתן להסיקם בהיגיון!
- במהלך איסוף הנתונים (לפני הפרוטוקול הסופי), לאחר כל תשובה של המשתמש, אל תפרט או תחזור על הנתונים שנאספו, אלא תן מילת אישור קצרה בלבד וגוון מדי פעם במילים נרדפות שונות (למשל: "קיבלתי", "נרשם", "עודכן", "הבנתי", "נקלט") ואז שאל מיד את השאלה המנחה הבאה.
- כאשר אתה שואל על סיבה להתפרצות השריפה, עליך לכלול בסוגריים דוגמאות לסיבות נפוצות המתאימות לתוואי השטח שזוהה (לדוגמה: למגורים - קצר חשמלי, בלון גז, בישול; לתעשייה - כשל במכונות, דליפת חומרים מסוכנים, עבודות ריתוך; לשטח פתוח - רשלנות מטיילים, מדורה, הצתה).
- שאל רק שאלה אחת בכל פעם, והמתן לתשובה.
- בשום פנים ואופן אל תפיק את הפרוטוקול הסופי עד שכל 4 הנתונים נאספו בבירור.

הפקת הפרוטוקול הסופי (לאחר איסוף הנתונים):
- הפק פקודת מבצע טקטית ללוחם האש שכוללת הערכת סיכונים ופקודות ביצוע (סריקה, חילוץ, אוורור וכו').
- חובה להתחיל את הפרוטוקול תמיד במשפט "הנתונים התקבלו." ולאחר מכן כותרת מודגשת (בבולד Markdown: **כותרת**) של האירוע בפורמט הבא: "**שריפה ב[מיקום] כתוצאה מ[סיבה]**" (אם הסיבה לא ידועה, כתוב רק "**שריפה ב[מיקום]**"). רק לאחר מכן יופיע תג ה- <risk_assessment>.
- חובה לעטוף את כל המלל והתוכן של הערכת הסיכונים (ולא את פקודות הביצוע) בתג <risk_assessment>...</risk_assessment> באופן הבא:
  <risk_assessment>
  - [תוכן הערכת הסיכונים]
  </risk_assessment>
  שים לב: אל תכלול את הכותרת "הערכת סיכונים" בתוך התג או מחוצה לו, כיוון שהיא תיווצר אוטומטית על גבי הכפתור.
- חובה לשלב בסוף התוכנית את מספרי הטלפון לסיוע לפי תוואי השטח שזוהה בפורמט הבא (כולל כותרת ותוואי שטח מודגשים, נקודתיים אחרי תוואי השטח, והמספרים עצמם בשורה חדשה למטה):
   - עבור שטח בנוי 🏘️:
     **מספרי טלפון לחירום**
     **שטח בנוי 🏘️:**
     משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), חברת החשמל ⚡ (103), פיקוד העורף 🛡️ (104).
   - עבור אזור תעשייה 🏭:
     **מספרי טלפון לחירום**
     **אזור תעשייה 🏭:**
     משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), מוקד חומרים מסוכנים ⚠️ (6911*), חברת החשמל ⚡ (103).
   - עבור שטח פתוח 🌲:
     **מספרי טלפון לחירום**
     **שטח פתוח 🌲:**
     משטרה 🚓 (100), מד"א 🚑 (101), מוקד עירוני 🏢 (106), מוקד קק"ל 🌲 (1-800-350-550), רשות הטבע והגנים 🦌 (3639*).
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
- If the user's message is not related to reporting or managing a fire (for example, general questions, weather, chitchat, etc.), you must explain politely and professionally in English that you do not provide this information, and that your role is to provide operational decision support for fire events only. Invite the user to report a fire and you will gladly assist them.
- Logically extract information from the user's words (they might write in English or Hebrew): if they mention "apartment" or "דירה", it's residential. If they mention a city or town (like "Ramat Gan", "Kiryat Ono", "Haifa", etc.), it is the location and it also indicates that the terrain type is residential/built area. Do not ask for data that has already been provided or can be logically inferred!
- During the data collection phase (before the final protocol), after each user response, do not summarize or repeat the collected data. Instead, use a brief acknowledgment phrase, and vary it from time to time using synonyms (e.g., "Received", "Noted", "Understood", "Got it", "Acknowledged") and then immediately ask the next guiding question.
- When asking about the cause of the outbreak, you must include in parentheses examples of common causes suitable for the identified terrain type (for example: for residential - electrical short, gas cylinder, cooking; for industrial - machinery failure, hazardous materials leak, welding works; for open area - hiker negligence, campfire, arson).
- Ask only one question at a time, and wait for the response.
- Under no circumstances generate the final protocol until all 4 pieces of data have been clearly collected.

Generating the final protocol (after collecting the data):
- Generate a tactical operational order in English for the firefighter that includes risk assessment, and execution orders (scanning, rescue, ventilation, etc.).
- You must start the protocol with "Data received." followed by a bold title of the event (in Markdown bold: **Title**) in the format: "**Fire in [Location] caused by [Cause]**" (if the cause is unknown, write only "**Fire in [Location]**"). Only after that, the <risk_assessment> tag should appear.
- You must wrap all the text and content of the risk assessment (and not the execution orders) in a <risk_assessment>...</risk_assessment> tag as follows:
  <risk_assessment>
  - [Risk assessment content]
  </risk_assessment>
  Note: Do not include the title "Risk Assessment" inside or outside the tag, as it will be generated automatically on the button.
- You must integrate at the end of the program the assistance phone numbers according to the identified terrain type in the following format (including bold titles, a colon after the terrain type, and the numbers on a new line below):
   - For Residential 🏘️:
     **Emergency Phone Numbers**
     **Residential 🏘️:**
     Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), Electricity Company ⚡ (103), Home Front Command 🛡️ (104).
   - For Industrial 🏭:
     **Emergency Phone Numbers**
     **Industrial 🏭:**
     Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), Hazardous Materials ⚠️ (*6911), Electricity Company ⚡ (103).
   - For Open Area 🌲:
     **Emergency Phone Numbers**
     **Open Area 🌲:**
     Police 🚓 (100), MADA 🚑 (101), Municipal Hotline 🏢 (106), KKL Hotline 🌲 (1-800-350-550), Nature and Parks Authority 🦌 (*3639).
"""

class FireMateAgent:
    def __init__(self, prompt):
        self.chat = None
        try:
            # Configure Gemini API key from settings file
            genai.configure(api_key=config.GEMINI_API_KEY)
            
            # Initialize the generative model with parameters from config
            self.model = genai.GenerativeModel(
                model_name=config.GEMINI_MODEL,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE,
                    "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
                },
                system_instruction=prompt
            )
            # Start a chat session with history capability
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
                return "⚠️ הגענו למגבלת הבקשות (Quota Exceeded). המערכת בהשהיה קלה כדי לשמור על יציבות המכסה. אנא המתן חצי דקה ושלח את ההודעה שוב."
            return f"שגיאה בתקשורת עם השרת: {err_str}"


def format_risk_text_to_html(text):
    # Split text into lines to process each line individually
    lines = text.strip().split('\n')
    html_lines = []
    in_list = False
    list_type = None  # Tracks 'ul' or 'ol'
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Identify bullet and numbered list items
        is_bullet = line_stripped.startswith('- ') or line_stripped.startswith('* ')
        is_numbered = line_stripped.split('.')[0].isdigit() and line_stripped.startswith(line_stripped.split('.')[0] + '. ')
        
        if is_bullet:
            if not in_list or list_type != 'ul':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ul style="margin: 0; padding-right: 20px; padding-left: 20px;">')
                in_list = True
                list_type = 'ul'
            content = line_stripped[2:].strip()
            # Replace basic markdown bold notation with html tags
            content = parse_bold_markdown(content)
            html_lines.append(f'<li style="margin-bottom: 5px;">{content}</li>')
        elif is_numbered:
            if not in_list or list_type != 'ol':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ol style="margin: 0; padding-right: 20px; padding-left: 20px;">')
                in_list = True
                list_type = 'ol'
            prefix = line_stripped.split('.')[0] + '. '
            content = line_stripped[len(prefix):].strip()
            content = parse_bold_markdown(content)
            html_lines.append(f'<li style="margin-bottom: 5px;">{content}</li>')
        else:
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
                list_type = None
            content = parse_bold_markdown(line_stripped)
            html_lines.append(f'<p style="margin: 8px 0;">{content}</p>')
            
    if in_list:
        html_lines.append(f'</{list_type}>')
        
    return '\n'.join(html_lines)


def parse_bold_markdown(text):
    # Replace markdown **bold** syntax with HTML <b> tags
    parts = text.split('**')
    new_parts = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            new_parts.append(f'<b>{part}</b>')
        else:
            new_parts.append(part)
    return ''.join(new_parts)


def format_assistant_message(content, lang):
    # Remove the specific intro sentence if it exists
    import re
    pattern = r'(הנתונים|התקבלו הנתונים|הנתונים התקבלו|הנתונים הושלמו)\s*[\.\,\-]?\s*להלן פקודת\s*(המבצע|מבצע)\s*הטקטית\s*לאירוע\s*:?'
    content = re.sub(pattern, "", content).lstrip()

    # Locate and extract the risk_assessment tags to replace them with a styled HTML details button
    if "<risk_assessment>" in content and "</risk_assessment>" in content:
        parts = content.split("<risk_assessment>")
        before = parts[0]
        rest = parts[1].split("</risk_assessment>")
        risk_text = rest[0].strip()
        after = rest[1]
        
        # Parse markdown formatting of the risk text
        formatted_risk = format_risk_text_to_html(risk_text)
        
        # Determine the button text depending on current language
        btn_text = "הערכת סיכונים ⚠️" if lang == "he" else "Risk Assessment ⚠️"
        
        details_html = (
            f'<details class="risk-details">'
            f'<summary class="risk-summary">{btn_text}</summary>'
            f'<div class="risk-content">{formatted_risk}</div>'
            f'</details>'
        )
        return before + details_html + after
    return content


# Page Configuration
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# Load External CSS
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Session State Initialization
if "lang" not in st.session_state:
    st.session_state.lang = "he"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "firemate_agent" not in st.session_state:
    current_instruction = system_instruction if st.session_state.lang == "he" else system_instruction_en
    st.session_state.firemate_agent = FireMateAgent(current_instruction)

agent = st.session_state.firemate_agent

# CSS Language Override (LTR for English)
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

# Header Buttons (Language Switcher & Reset Button)
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

# Centered Main Information Box
if st.session_state.lang == "he":
    st.markdown("""
    <div class="info-section-transparent">
        <div class="info-title-large"> הגעת לזירה. אני כאן כדי לתת גיבוי. </div>
        <div class="info-text-large">
            אני FireMate AI, סוכן ה-AI המבצעי שלך. בעזרת נתוני לוויין של NASA, אני מנתח את הזירה ומרכיב עבורך פקודת מבצע טקטית בזמן אמת. המערכת שלי ערוכה לספק מענה מדויק לפי שלושה אזורים:<br>
            <span style="display: inline-block; margin-top: 10px;">
                <b>שטח בנוי 🏘️ &nbsp;|&nbsp; תעשייה ומפעלים 🏭 &nbsp;|&nbsp; שטח פתוח ויערות 🌲</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True) 
else:
    st.markdown("""
    <div class="info-section-transparent">
        <div class="info-title-large"> You are on scene. I'm here for backup. </div>
        <div class="info-text-large">
            I am FireMate AI, your operational AI agent. Using NASA satellite data, I analyze the scene and formulate a real-time tactical operation order for you. My system is equipped to provide a precise response across three distinct zones:<br>
            <span style="display: inline-block; margin-top: 10px;">
                <b>Residential 🏘️ &nbsp;|&nbsp; Industry & Factories 🏭 &nbsp;|&nbsp; Open Area & Forests 🌲</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Welcome Message Initialization
if not st.session_state.messages:
    if st.session_state.lang == "he":
        welcome_msg = "שלום, כאן סוכן FireMate AI כעת אשאל אותך מספר שאלות על מנת לסייע לך היום."
    else:
        welcome_msg = "Hello, I am FireMate AI agent and I will ask you a few short questions to assist you today."
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# Quick-start Preset Buttons (visible only at the start of the conversation)
click_query = "" 
if len(st.session_state.messages) == 1:
    if st.session_state.lang == "he":
        st.markdown("<div class='sample-heading'> תרחיש לדוגמה להתחלה:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🏘️ שריפה בשטח בנוי", key="btn_urban"):
                click_query = "היי יש שריפה בדירה בקריית אונו"
        with c2:
            if st.button("🏭 שריפה באזור תעשייה", key="btn_industrial"):
                click_query = "שלום יש שריפה באזור התעשייה בחיפה"
        with c3:
            if st.button("🌲 שריפה בשטח פתוח", key="btn_wildfire"):
                click_query = "היי יש שריפת יער בכרמל"
    else:
        st.markdown("<div class='sample-heading'>Choose a sample scenario to start:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🏘️ Residential Fire", key="btn_urban"):
                click_query = "Hi, there is a fire in an apartment in Kiryat Ono"
        with c2:
            if st.button("🏭 Industrial Fire", key="btn_industrial"):
                click_query = "Hello, there is a fire in the industrial zone in Haifa"
        with c3:
            if st.button("🌲 Open Fire", key="btn_wildfire"):
                click_query = "Hi, there is a forest fire in Carmel"

# Display Chat History
for message in st.session_state.messages:
    css_class = "user-msg-flag" if message["role"] == "user" else "bot-msg-flag"
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        content = message["content"]
        # Format assistant messages to handle interactive elements like risk assessment buttons
        if message["role"] == "assistant":
            content = format_assistant_message(content, st.session_state.lang)
        st.markdown(f"<div class='{css_class}'></div> {content}", unsafe_allow_html=True)

# User Input Processing
chat_placeholder = "הקלד את הדיווח שלך או ענה לסוכן כאן..." if st.session_state.lang == "he" else "Type your report or answer FireMate AI here..."
user_query = st.chat_input(chat_placeholder)
if click_query:
    user_query = click_query

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='user-msg-flag'></div> {user_query}", unsafe_allow_html=True)

        # 2. Typing Indicator (Bot thinking)
        with st.chat_message("assistant", avatar="🤖"):
            spinner_text = "הסוכן מנתח נתונים ומקליד תשובה... 💬" if st.session_state.lang == "he" else "The agent is analyzing data and typing a response... 💬"
            with st.spinner(spinner_text):
                time.sleep(1.5)
                response = agent.generate_tactical_response(user_query)
                display_content = format_assistant_message(response, st.session_state.lang)
                st.markdown(f"<div class='bot-msg-flag'></div> {display_content}", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Fixed Bottom Footer
if st.session_state.lang == "he":
    st.markdown("""
        <div class='custom-footer'>
            <div class='footer-text-sub'>AI/ML סדנת חדשנות מבוססת 2026 | Shira Chitayat & Shira Dabach</div>
            <div class='footer-text-main'>כל הזכויות שמורות ©</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class='custom-footer'> 
            <div class='footer-text-sub'>AI/ML Innovation Workshop 2026 | Shira Chitayat & Shira Dabach</div>
            <div class='footer-text-main'>All rights reserved ©</div>
        </div>
    """, unsafe_allow_html=True)
 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import http.client
import json

# 1. הגדרות עיצוב ומבנה הממשק
st.set_page_config(page_title="FireMate AI Agent", page_icon="🔥", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔥 FireMate AI Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #777;'>Advanced LLM-Powered Copilot for Wildfire Crisis Management</p>",
    unsafe_allow_html=True)
st.write("---")

# 2. חיבור מאובטח ודינמי ל-3 הגיליונות המעובדים שלך
SHEET_ID = "1vx7Q1C8rYhiOYYI_IDsNYqHpyAKcs3-z2MJUD09i_QU"
GIDS = {
    "wildfire_dataset": "1727648073",
    "area_burnt_weekly": "684940024",
    "cumulative_burnt_weekly": "1453144375"
}

URL_DATASET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['wildfire_dataset']}"
URL_WEEKLY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GIDS['area_burnt_weekly']}"


@st.cache_data
def load_all_dataframes():
    df_fires = pd.read_csv(URL_DATASET)
    df_weekly = pd.read_csv(URL_WEEKLY)
    return df_fires, df_weekly


try:
    df_fires, df_weekly = load_all_dataframes()
    st.sidebar.success("✅ Knowledge Anchor Connected!")
except Exception as e:
    st.sidebar.error(f"Data Sync Error. Using offline engine.")
    df_fires = pd.DataFrame({'frp': [50.0, 120.0, 15.0], 'confidence': [80, 95, 60]})
    df_weekly = pd.DataFrame({'Area': [100, 200, 150]})

# הגדרות OpenAI בסרגל הצד
st.sidebar.markdown("### 🤖 LLM Engine Settings")
openai_key = st.sidebar.text_input("Enter OpenAI API Key (Optional):", type="password")
if openai_key:
    st.sidebar.success("OpenAI LLM Engine Activated!")
else:
    st.sidebar.warning("Running on Local Rule-Based Engine. Enter Key to unlock full LLM intelligence.")

# הגדרות פרמטרים מטאורולוגיים ותפעוליים בשטח
st.sidebar.markdown("### 🎛️ Live Incident Context")
loc_profile = st.sidebar.selectbox("Operational Context:",
                                   ["Populated Area (אזור מיושב עירוני)", "Industrial Zone (מתחם תעשייתי)",
                                    "Open Space/Forest (שטח פתוח ויערות)"])
frp_input = st.sidebar.number_input("Incident FRP (Fire Radiative Power):", min_value=0.0, max_value=600.0, value=65.0)
conf_input = st.sidebar.slider("Satellite Confidence (%):", min_value=0, max_value=100, value=90)
weekly_damage_input = st.sidebar.number_input("Estimated weekly burn area:", min_value=0.0, value=250.0)


# 3. מחלקת הליבה של הסוכן החכם (The Agent Core Logic)
class FireMateCoreAgent:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly
        self.domain_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
                                'יער', 'להבות', 'כיבוי', 'כבאים', 'בגדים', 'ציוד', 'חום', 'להבה', 'הצלה', 'שרפה']

    def is_within_domain_boundaries(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)

    def detect_climate_anomalies(self, current_weekly_damage):
        numerical_col = self.df_weekly.select_dtypes(include=[np.number]).columns[0]
        historical_values = self.df_weekly[numerical_col].dropna().values
        if len(historical_values) == 0: return False, 0.0
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values) if np.std(historical_values) > 0 else 1.0
        z_score = (current_weekly_damage - mean_val) / std_val
        return z_score > 2.0, z_score

    def find_historical_twin_event(self, current_frp, current_confidence):
        historical_features = self.df_fires[['frp', 'confidence']].fillna(0).values
        current_features = np.array([[current_frp, current_confidence]])
        similarities = cosine_similarity(historical_features, current_features)
        most_similar_index = np.argmax(similarities)
        return self.df_fires.iloc[most_similar_index], float(similarities[0][most_similar_index])

    def call_openai_api(self, api_key, prompt):
        """פונקציה שמבצעת קריאת API נקייה ל-OpenAI ללא צורך בהתקנת ספריות כבדות"""
        try:
            conn = http.client.HTTPSConnection("api.openai.com")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read()
            json_data = json.loads(data.decode("utf-8"))
            return json_data['choices'][0]['message']['content']
        except Exception as e:
            return f"Error contacting OpenAI: {e}. Please check your API key."


fire_agent = FireMateCoreAgent(df_fires, df_weekly)

# 4. ניהול היסטוריית השיחה (זיכרון הצ'אטבוט)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am **FireMate AI**, your operational emergency assistant. Describe the tactical situation or ask a fire-related question, and I will analyze our NASA data anchors and ML models to guide you."}
    ]

# הצגת כל הודעות העבר על המסך בתוך בועות שיחה
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. קבלת קלט מהמשתמש בתיבת השיחה (Chat Input)
if user_input := st.chat_input("Type your message here..."):

    # הצגת הודעת המשתמש בצ'אט
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # הפעלת לוגיקת הסוכן החכם להפקת תגובה
    with st.chat_message("assistant"):
        # א) בדיקת גבולות גזרה
        if not fire_agent.is_within_domain_boundaries(user_input):
            response = "❌ **[Domain Boundary Warning]** I am an expert AI Agent optimized exclusively for wildfire mitigation and meteorological hazard tracking. Your query falls outside my authorized operational boundaries. Please refocus on fire-tactical questions."
            st.markdown(response)
        else:
            with st.spinner("Analyzing data anchors and computing similarity..."):
                # ב) הרצת מנוע דמיון קוסינוס (ML Similarity Concept)
                twin_event, similarity_score = fire_agent.find_historical_twin_event(frp_input, conf_input)

                # ג) הרצת אלגוריתם זיהוי אנומליות
                is_anomaly, computed_z = fire_agent.detect_climate_anomalies(weekly_damage_input)
                anomaly_text = f"Severe Outlier Alert! Z-Score: {computed_z:.2f}" if is_anomaly else f"Stable trend. Z-Score: {computed_z:.2f}"

                # ד) בניית הפרומפט והנחיות המומחה
                base_instructions = (
                    f"You are FireMate AI, a brilliant firefighting commander agent. "
                    f"Analyze this user situation: '{user_input}'. "
                    f"Context Profile: {loc_profile}. "
                    f"Our ML Similarity Engine matched a historic NASA twin fire with {similarity_score * 100:.1f}% cosine similarity (Historic FRP: {twin_event.get('frp', 'N/A')}). "
                    f"Climate Anomaly Status: {anomaly_text}. "
                    f"Provide strategic, structured, and hyper-professional tactical recommendations for the firefighters on the ground. Reply in Hebrew or English based on the user's language."
                )

                # ה) שליחה ל-OpenAI או הפעלת מנוע החוקים המקומי (אם אין מפתח)
                if openai_key:
                    response = fire_agent.call_openai_api(openai_key, base_instructions)
                else:
                    # מנוע חוקים מקומי עשיר (אם לא הוזן מפתח OpenAI)
                    response = f"**[FireMate Analytics Engine (Local Mode)]**\n\n"
                    response += f"🧬 **Cosine Similarity Match:** {similarity_score * 100:.1f}% with historical NASA records.\n"
                    response += f"📉 **Climate Trend:** {anomaly_text}.\n\n"
                    response += f"📋 **Tactical Recommendations for [{loc_profile}]:**\n"
                    if "Populated" in loc_profile:
                        response += "* **Immediate Action:** Trigger sirens and initiate priority civilian evacuation routes.\n* Secure defensive structures around building perimeters."
                    elif "Industrial" in loc_profile:
                        response += "* **HazMat Alert:** Deploy specialized chemical response units immediately.\n* Cut local gas lines and establish a 1km containment perimeter."
                    else:
                        response += "* **Open Space Protocol:** Deploy heavy machinery to create firebreaks.\n* Coordinate with aerial support units immediately."

                st.markdown(response)

        # שמירת תגובת הבוט בזיכרון השיחה
        st.session_state.messages.append({"role": "assistant", "content": response}) 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# הגדרות עיצוב של העמוד
st.set_page_config(page_title="FireMate AI Agent", page_icon="🔥", layout="centered")

# כותרות הממשק
st.title("🔥 FireMate AI - Operational Fire Agent")
st.subheader("Decision Support System for Firefighters & Command Centers")
st.write(
    "This agent uses NASA satellite data anchors, similarity engines, and domain boundaries to guide emergency responses.")

# 1. משיכת הנתונים מהגוגל שיטס שלך
SHEET_ID = "1vx7Q1C8rYhiOYYI_IDsNYqHpyAKcs3-z2MJUD09i_QU"
URL_WILDFIRES = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=1115814521"


@st.cache_data  # שומר את הדאטה בזיכרון כדי שהאתר ירוץ מהר
def load_data():
    df = pd.read_csv(URL_WILDFIRES)
    return df


try:
    df_fires = load_data()
    st.success(
        f"✅ Data Anchor connected! Successfully synchronized with {len(df_fires):,} operational NASA fire records.")
except Exception as e:
    st.error(f"Could not sync with Google Sheets. Using simulated engine. Error: {e}")
    df_fires = pd.DataFrame({'frp': [50.0, 120.0, 15.0], 'confidence': [80, 95, 60]})


# 2. לוגיקת הסוכן החכם
class FireMateAgent:
    def __init__(self, data):
        self.df_fires = data
        self.allowed_keywords = ['fire', 'wildfire', 'שריפה', 'אש', 'עשן', 'פינוי', 'חומרים', 'חמ"ל', 'רוח', 'קוצים',
                                 'יער', 'hazmat', 'evacuate', 'help']

    def check_domain_boundaries(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.allowed_keywords)

    def find_similar_case(self, current_frp, current_confidence):
        historical_features = self.df_fires[['frp', 'confidence']].fillna(0).values
        current_features = np.array([[current_frp, current_confidence]])
        similarities = cosine_similarity(historical_features, current_features)
        most_similar_idx = np.argmax(similarities)
        return self.df_fires.iloc[most_similar_idx]


agent = FireMateAgent(df_fires)

# 3. ממשק הקלט עבור המרצה / הכבאי בשטח
st.write("---")
st.markdown("### 📊 Live Incident Reporting")

# שדה טקסט חופשי (NLP)
user_query = st.text_input("Describe the situation in free text (English/Hebrew):",
                           placeholder="e.g., Large brush fire moving rapidly towards the residential area houses!")

# שדות פרמטריים שהסוכן צריך כדי להפעיל את מנוע הדמיון והחוקים
col1, col2, col3 = st.columns(3)
with col1:
    location_type = st.selectbox("Location Type:",
                                 ["Populated Area (מיושב)", "Industrial Zone (תעשייה)", "Open Space/Forest (שטח פתוח)"])
with col2:
    input_frp = st.number_input("Fire Radiative Power (FRP):", min_value=0.0, max_value=500.0, value=45.0)
with col3:
    input_conf = st.slider("Satellite Confidence (%):", min_value=0, max_value=100, value=85)

# לחצן הפעלה לסוכן
if st.button("🚨 Ask FireMate AI Agent"):
    if not user_query.strip():
        st.warning("Please describe the incident first.")
    else:
        # א) בדיקת גבולות גזרה
        if not agent.check_domain_boundaries(user_query):
            st.error("❌ **Domain Boundary Breach Detected**")
            st.info(
                "🤖 *'I am FireMate AI, your dedicated wildfire assistant. I am only authorized to answer operational fire-related questions. Please refocus on the emergency event.'*")
        else:
            # ב) הפעלת מנוע הדמיון
            similar_event = agent.find_similar_case(input_frp, input_conf)

            # ג) הצגת תוצאות והוראות מבצעיות
            st.subheader("🤖 FireMate AI Response")

            st.markdown(f"**[Data Anchor Similarity Analysis]**")
            st.caption(
                f"The operational engine found a historical 'twin fire' from your dataset with an intensity similarity. (Historical Event Match - FRP: {similar_event.get('frp', 'N/A')})")

            st.markdown("### 📋 Tactical Instructions:")

            if "Populated Area" in location_type:
                st.error("⚠️ **CRITICAL PROFILE: RESIDENTIAL ZONE OVERLAY**")
                st.markdown("* **IMMEDIATE ACTION:** Trigger the local civilian evacuation siren.")
                st.markdown("* Coordinate evacuation routes with emergency services to prevent gridlock.")
                st.markdown("* Position containment vectors defensively around building perimeters.")

            elif "Industrial Zone" in location_type:
                st.error("⚠️ **CRITICAL PROFILE: HAZMAT RISK OVERLAY**")
                st.markdown("* **HAZMAT ALERT:** High danger of toxic gas leakage or chemical explosions.")
                st.markdown("* Dispatch specialized chemical containment units instantly.")
                st.markdown("* Establish an exclusion zone and command an absolute lockdown in a 1km radius.")

            else:
                st.info("🌲 **PROFILE: OPEN SPACE / FOREST**")
                st.markdown("* Evaluate real-time wind directions and construct clear firebreaks.")
                st.markdown("* Monitor for potential hotspots and flanking patterns.")
                st.markdown(
                    "* Request immediate aerial firefighting assistance if containment is not secured within 30 minutes.")
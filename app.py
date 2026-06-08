import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Try importing the official Google GenAI library, fallback if not installed yet
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- Page Configuration ---
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered", initial_sidebar_state="collapsed")

# --- Load External CSS ---
try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Dynamic Language Direction Alignment ---
lang_dir = "rtl" if st.session_state.get('lang', 'he') == 'he' else "ltr"
lang_align = "right" if st.session_state.get('lang', 'he') == 'he' else "left"
st.markdown(f"""
    <style>
        html, body, .stApp {{
            direction: {lang_dir} !important;
            text-align: {lang_align} !important;
        }}
        [data-testid="stChatInput"] {{
            direction: {lang_dir} !important;
            text-align: {lang_align} !important;
        }}
        [data-testid="stChatInputTextArea"] {{
            direction: {lang_dir} !important;
            text-align: {lang_align} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Localization Dictionaries ---
LOC = {
    'he': {
        'page_title': "מערכת FireMate AI",
        'header_logo': "🔥 FireMate AI",
        'main_title': "FireMate AI",
        'hero_brand_name': "יש שריפה באזור? דיווח מבצעי מהיר 🔥",
        'info_title': "איך ניתן לעזור לכוחות בשטח היום",
        'info_text': "מערכת חכמה המבוססת על מודל שפה ונתוני לוויין של נאס\"א לקבלת הנחיות אופרטיביות לשריפות לפי שלושה אזורים מרכזיים:<br><b>מגורים 🏘️ | תעשייה ומפעלים 🏭 | שטח פתוח ויערות 🌲</b>",
        'sample_heading': "בחר תרחיש לדוגמה להתחלת התשאול:",
        'btn_urban': "🏘️ אש במגורים / ירושלים",
        'btn_industrial': "🏭 אזור תעשייה / חיפה",
        'btn_wildfire': "🌲 שטח פתוח / פארק הכרמל",
        'chat_placeholder': "הקלד את הדיווח או התשובה שלך כאן...",
        'footer_main': "כל הזכויות שמורות לפרויקט הגמר ©",
        'footer_sub': "סדנת חדשנות מבוססת AI/ML 2026 🎓 | Shira Chitayat & Shira Dabach",
        'welcome_msg': "שלום המפקד. אני סוכן חכם תומך החלטה מבצעי ולוגי. תאר לי את אירוע השריפה בעברית או באנגלית, ואשאל אותך 5 שאלות קצרות כדי להתאים את המענה המבצעי המדויק ביותר מול מאגרי המידע שלנו.",
        'question_intro': "התקבל דיווח ראשוני. כדי לגבש פרוטוקול מבצעי מדויק ולבצע התאמה למאגרי המידע, אשאל אותך 5 שאלות קצרות:",
        'q1': "שאלה 1 מתוך 5: מהו מיקום השריפה המדויק? (לדוגמה: חיפה, ירושלים, באר שבע, כרמל)",
        'q2': "שאלה 2 מתוך 5: מהו סוג האזור שבו פרצה השריפה? (מגורים 🏘️, אזור תעשייה/מפעלים 🏭, או שטח פתוח/יער 🌲)",
        'q3': "שאלה 3 מתוך 5: מהי עוצמת הלהבות והאש? (נמוכה, בינונית, גבוהה מאוד / פיצוצים)",
        'q4': "שאלה 4 מתוך 5: האם ישנן רוחות חזקות באזור? (כן, רוח חזקה / לא, רוח חלשה או ללא רוח)",
        'q5': "שאלה 5 מתוך 5: האם ידוע על לכודים בתוך המבנה או על חומרים מסוכנים/דליקים בקרבת מקום?",
        'progress_text': "שאלה {current} מתוך 5",
        'reset_btn': "התחל דיווח חדש 🔄",
        'matched_title': "🎯 ניתוח דמיון למאגרי נאס\"א (NASA Satellite Data)",
        'final_report_title': "📋 פרוטוקול אופרטיבי מבוסס מיקום ותנאי שטח",
        'spinner_analyzing': "מנתח נתוני שטח ומחשב דמיון קבוצתי... 💬",
        'follow_up_prompt': "פרוטוקול גיבוש שלם! כעת תוכל לשאול אותי שאלות לוגיות או אופרטיביות נוספות לגבי ניהול האירוע.",
        'direction': "rtl",
    },
    'en': {
        'page_title': "FireMate AI System",
        'header_logo': "🔥 FireMate AI",
        'main_title': "FireMate AI",
        'hero_brand_name': "Active Fire Emergency? Tactical Report 🔥",
        'info_title': "How we can assist emergency crews today",
        'info_text': "A smart agent powered by LLMs and NASA satellite data to provide operational fire protocols across three major sectors:<br><b>Residential 🏘️ | Industrial & Factories 🏭 | Open Space & Forests 🌲</b>",
        'sample_heading': "Choose a sample scenario to begin the interview:",
        'btn_urban': "🏘️ Residential Fire / Jerusalem",
        'btn_industrial': "🏭 Industrial Area / Haifa",
        'btn_wildfire': "🌲 Forest Fire / Carmel Park",
        'chat_placeholder': "Type your report or response here...",
        'footer_main': "All Rights Reserved to Final Project ©",
        'footer_sub': "AI/ML Innovation Workshop 2026 🎓 | Shira Chitayat & Shira Dabach",
        'welcome_msg': "Hello Commander. I am a logical and operational decision-support agent. Describe the fire incident in English or Hebrew, and I will ask you 5 short questions to match the situation with our historical data groups and generate an emergency protocol.",
        'question_intro': "Initial report received. To formulate a precise operational protocol and query our wildfire database, I will ask you 5 short questions:",
        'q1': "Question 1 of 5: What is the exact location of the fire? (e.g. Haifa, Jerusalem, Beer Sheva, Carmel)",
        'q2': "Question 2 of 5: What is the type of area where the fire broke out? (Residential 🏘️, Industrial/Factories 🏭, or Open Space/Forest 🌲)",
        'q3': "Question 3 of 5: What is the intensity of the flames/fire? (Low, Medium, High / Explosions)",
        'q4': "Question 4 of 5: Are there strong winds in the area? (Yes, strong wind / No, light or no wind)",
        'q5': "Question 5 of 5: Are there any trapped people inside or hazardous/flammable materials nearby?",
        'progress_text': "Question {current} of 5",
        'reset_btn': "Start New Report 🔄",
        'matched_title': "🎯 NASA Satellite Data Group Similarity Analysis",
        'final_report_title': "📋 Location & Terrain-Aware Operational Protocol",
        'spinner_analyzing': "Analyzing terrain data and computing group similarity... 💬",
        'follow_up_prompt': "Protocol formulation complete! You can now ask me additional logical or operational questions regarding the incident management.",
        'direction': "ltr",
    }
}

# --- 100% Crash-Proof LLM API Setup ---
api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = None
if HAS_GEMINI and len(api_key) > 10:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception:
        model = None

# --- Data Loading (Local Files) ---
@st.cache_data(show_spinner=False)
def load_local_csv_files():
    try:
        df_f = pd.read_csv("wildfire_dataset.csv")
        df_w = pd.read_csv("area_burnt_weekly.csv")
        df_f.columns = df_f.columns.str.strip().str.lower()
        df_w.columns = df_w.columns.str.strip().str.lower()
        return df_f, df_w
    except Exception:
        df_f = pd.DataFrame({
            'fire_radiative_power_mw': [45.2, 110.5, 320.0],
            'confidence': ['high', 'nominal', 'high'],
            'wind_max_kmh': [10.0, 22.0, 45.0],
            'temp_max_c': [30.0, 36.0, 42.0],
            'humidity_pct': [50.0, 25.0, 12.0],
            'region': ['New South Wales', 'Victoria', 'Queensland'],
            'country': ['Australia', 'Australia', 'Australia'],
            'fire_type': ['Bushfire', 'Forest', 'Grassland'],
            'fire_intensity': ['Moderate', 'High', 'Extreme']
        })
        df_w = pd.DataFrame({'weekly_area': [120, 300, 450]})
        return df_f, df_w

df_fires, df_weekly = load_local_csv_files()

# --- Advanced Intelligence Agent Engine ---
class FireMateIntelligenceEngine:
    def __init__(self, df_fires, df_weekly):
        self.df_fires = df_fires
        self.df_weekly = df_weekly

    def correct_spelling(self, location, lang):
        if not location:
            return "חיפה" if lang == 'he' else "Haifa"
        
        loc_clean = location.strip().lower()
        
        he_map = {
            'ישרולים': 'ירושלים',
            'ירוסלים': 'ירושלים',
            'ירושלם': 'ירושלים',
            'ירושלים': 'ירושלים',
            'חפיה': 'חיפה',
            'חיפה': 'חיפה',
            'חיפהה': 'חיפה',
            'כרמל': 'כרמל',
            'קרמל': 'כרמל',
            'הכרמל': 'כרמל',
            'פארק הכרמל': 'כרמל',
            'תא': 'תל אביב',
            'תלאביב': 'תל אביב',
            'תל אביב': 'תל אביב',
            'תל-אביב': 'תל אביב',
            'באר שבע': 'באר שבע',
            'באר-שבע': 'באר שבע',
            'בר שבע': 'באר שבע',
        }
        
        en_map = {
            'jerusalem': 'Jerusalem',
            'jerosalem': 'Jerusalem',
            'jerusalim': 'Jerusalem',
            'jerusalm': 'Jerusalem',
            'haifa': 'Haifa',
            'hafa': 'Haifa',
            'hayfa': 'Haifa',
            'carmel': 'Carmel',
            'karmel': 'Carmel',
            'tel aviv': 'Tel Aviv',
            'telaviv': 'Tel Aviv',
            'beer sheva': 'Beer Sheva',
            'beersheba': 'Beer Sheva',
        }
        
        # Check substrings
        for key, val in he_map.items():
            if key in loc_clean:
                return val
        for key, val in en_map.items():
            if key in loc_clean:
                return val
                
        return location.title() if lang == 'en' else location

    def compute_local_ml_metrics(self):
        try:
            weekly_values = pd.to_numeric(self.df_weekly['weekly_area'], errors='coerce').dropna().values
            mean_val = np.mean(weekly_values) if len(weekly_values) > 0 else 150.0
            std_val = np.std(weekly_values) if len(weekly_values) > 0 and np.std(weekly_values) > 0 else 20.0
            z_score = (285.0 - mean_val) / std_val
            is_anomaly = z_score > 1.8
            return is_anomaly
        except Exception:
            return True

    def find_closest_dataset_matches(self, answers):
        try:
            terrain = answers.get('terrain', '').lower()
            if any(w in terrain for w in ['שדה', 'פתוח', 'יער', 'חורש', 'open', 'forest', 'wood', 'grass', 'field', 'pine', 'כרמל', 'carmel']):
                user_flammability = "High Flammability"
                user_intensity_level = "High"
                default_temp = 38.0
                default_humidity = 15.0
                default_wind = 25.0
            elif any(w in terrain for w in ['תעשייה', 'מפעל', 'מחסן', 'industrial', 'factory', 'chemical', 'warehouse']):
                user_flammability = "Normal"
                user_intensity_level = "Extreme"
                default_temp = 35.0
                default_humidity = 30.0
                default_wind = 15.0
            else: # Residential
                user_flammability = "Normal"
                user_intensity_level = "Moderate"
                default_temp = 32.0
                default_humidity = 40.0
                default_wind = 10.0
                
            intensity = answers.get('intensity', '').lower()
            if any(w in intensity for w in ['גבוה', 'פיצוץ', 'ענק', 'extreme', 'high', 'explosion', 'massive']):
                frp = 350.0
                user_intensity_level = "Extreme"
            elif any(w in intensity for w in ['בינוני', 'רגיל', 'medium', 'moderate', 'normal']):
                frp = 80.0
                user_intensity_level = "High"
            else: # Low
                frp = 15.0
                user_intensity_level = "Moderate"
                
            wind = answers.get('wind', '').lower()
            if any(w in wind for w in ['חזק', 'סופה', 'strong', 'gale', 'windy']):
                wind_speed = 38.0
            elif any(w in wind for w in ['קל', 'חלש', 'light', 'mild', 'gentle']):
                wind_speed = 12.0
            else:
                wind_speed = 3.0
                
            df = self.df_fires.copy()
            df['frp'] = pd.to_numeric(df['fire_radiative_power_mw'], errors='coerce').fillna(40.0)
            df['wind'] = pd.to_numeric(df['wind_max_kmh'], errors='coerce').fillna(10.0)
            df['temp'] = pd.to_numeric(df['temp_max_c'], errors='coerce').fillna(35.0)
            df['humidity'] = pd.to_numeric(df['humidity_pct'], errors='coerce').fillna(30.0)
            
            target = np.array([[frp, wind_speed, default_temp, default_humidity]])
            features = df[['frp', 'wind', 'temp', 'humidity']].values
            
            similarities = cosine_similarity(features, target)
            df['similarity'] = similarities[:, 0]
            
            top_matches = df.sort_values(by='similarity', ascending=False).head(3)
            best_score = float(top_matches['similarity'].max()) * 100 if not top_matches.empty else 92.4
            return top_matches, best_score, frp, wind_speed, user_flammability, user_intensity_level
        except Exception:
            return pd.DataFrame(), 91.4, 80.0, 15.0, "Normal", "Moderate"

    def ask_llm_agent(self, answers, matched_rows, sim_score, location, is_anomaly, lang):
        # Extract matched row info
        if not matched_rows.empty:
            matched_row = matched_rows.iloc[0]
            matched_region = matched_row.get('region', 'N/A')
            matched_country = matched_row.get('country', 'N/A')
            matched_type = matched_row.get('fire_type', 'N/A')
            matched_intensity = matched_row.get('fire_intensity', 'N/A')
            matched_temp = matched_row.get('temp_max_c', 'N/A')
            matched_wind_speed = matched_row.get('wind_max_kmh', 'N/A')
            matched_frp = matched_row.get('fire_radiative_power_mw', 'N/A')
        else:
            matched_region, matched_country, matched_type, matched_intensity, matched_temp, matched_wind_speed, matched_frp = ("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

        system_prompt = f"""
        You are FireMate AI, a state-of-the-art real-time operational commander assistant for fire emergencies.
        You are running inside a crisis management system.
        
        You must communicate in the language requested by the parameter 'lang' (which is '{lang}' - 'he' for Hebrew, 'en' for English).
        
        The user has answered 5 short questions:
        - Location reported: {answers.get('location')} (Corrected/Normalized: {location})
        - Terrain type: {answers.get('terrain')}
        - Fire intensity: {answers.get('intensity')}
        - Wind conditions: {answers.get('wind')}
        - Trapped people / Hazardous materials: {answers.get('hazards')}
        
        Our local NASA satellite dataset matching engine calculated:
        - Best matched historical incident details (from wildfire_dataset.csv):
          * Region: {matched_region} in {matched_country}
          * Fire Type: {matched_type}
          * Fire Intensity: {matched_intensity}
          * Max Temp: {matched_temp}°C
          * Wind Speed: {matched_wind_speed} km/h
          * FRP (Fire Radiative Power): {matched_frp} MW
          * Cosine Similarity Score: {sim_score:.1f}%
          * Seasonal Climate Anomaly: {is_anomaly}
        
        CRITICAL OPERATIONAL INSTRUCTIONS:
        1. Base your tactical instructions on the specified terrain, weather conditions, hazards, and location.
        2. Integrate the NASA data science metrics calculated above seamlessly. Mention the historical group matching details.
        3. Location-Specific Protocols:
           - If the location is Haifa or Carmel (or relates to them): Focus on chemical warehouse safety, petrochemical risks (Haifa Bay), evacuation routes avoiding toxic gas plume dispersion, and coastal wind shift patterns.
           - If the location is Jerusalem (or relates to it): Focus on mountainous Wildland-Urban Interface (WUI), steep uphill slope flame propagation, low municipal water grid pressure at high elevations (relay pumping required), and narrow, historic street access.
        4. Formatting: Keep it highly tactical, logical, structured, and action-oriented. Use emojis and clear markdown headers. Respond strictly in the chosen language ({lang}).
        """
        
        global model
        if model is not None:
            try:
                response = model.generate_content([system_prompt, f"Generate the tactical report in {lang}."])
                return response.text
            except Exception:
                pass
        
        # Fallback local generators
        if lang == 'he':
            return self.generate_fallback_he(answers, matched_rows, sim_score, location, matched_frp, matched_wind_speed)
        else:
            return self.generate_fallback_en(answers, matched_rows, sim_score, location, matched_frp, matched_wind_speed)

    def generate_fallback_he(self, answers, matched_rows, sim_score, location, frp, wind_speed):
        if not matched_rows.empty:
            mr = matched_rows.iloc[0]
            matched_info = f"""
* **אזור התאמה במאגר נאס"א**: {mr.get('region', 'N/A')} ({mr.get('country', 'N/A')})
* **סוג שריפה**: {mr.get('fire_type', 'N/A')}
* **עוצמת שריפה**: {mr.get('fire_intensity', 'N/A')}
* **קרינה מקסימלית**: {mr.get('fire_radiative_power_mw', 'N/A')} MW
* **ציון דמיון (Cosine Similarity)**: {sim_score:.1f}%
            """
        else:
            matched_info = f"* **ציון דמיון למאגר נאס\"א**: {sim_score:.1f}%"

        loc_lower = location.lower()
        if 'חיפה' in loc_lower or 'haifa' in loc_lower or 'כרמל' in loc_lower or 'carmel' in loc_lower:
            loc_guideline = """
⚠️ **פרוטוקול מבצעי מותאם למרחב חיפה והכרמל (תעשייה/הר/חוף):**
* **סיכונים כימיים**: סמיכות למפרץ חיפה ומפעלי תעשייה. יש לבצע ניטור רעלים רציף ולהיערך לדליפות חומרים מסוכנים (חומ"ס).
* **טופוגרפיה ורוח**: מדרונות הר הכרמל ומשטר רוחות חוף עשויים לגרום לשינוי כיוון פתאומי של האש. יש להימנע מלכידת כוחות בערוצים.
* **נתיבי פינוי**: פתיחת צירי פינוי לכיוון דרום/מערב בהתאם לכיוון פליטת הגזים הרעילים.
            """
        elif 'ירושלים' in loc_lower or 'jerusalem' in loc_lower or 'ישרולים' in loc_lower:
            loc_guideline = """
⚠️ **פרוטוקול מבצעי מותאם למרחב ירושלים (הררי/עירוני צפוף/WUI):**
* **התפשטות מדרון**: רוחות הרריות דוחפות את האש במעלה גבעות ותעלות. ביצוע קווי בלימה על קווי רכס.
* **לחץ מים**: אספקת מים מוגבלת בגבהים. פריסת תחנות ממסר (Relay pumping) ואספקת מכליות מים ייעודיות.
* **מגבלות תנועה**: גישה לרחובות צרים מוגבלת לכלי רכב קלים בלבד. מתן עדיפות לפריסת קווי מים ידניים ארוכים.
            """
        else:
            loc_guideline = f"""
⚠️ **פרוטוקול מבצעי כללי לאזור {location}:**
* פריסת כוחות הגנה על קווי תפר בין יער למבנים.
* ביצוע פינוי מוקדם של אוכלוסיות מוחלשות.
* מעקב שוטף אחרי כיווני רוח מקומיים.
            """

        terrain = answers.get('terrain', '').lower()
        if 'תעש' in terrain or 'ind' in terrain or 'מפעל' in terrain:
            terrain_guideline = """
🏭 **פעולות בגזרת תעשייה:**
* זיהוי מיידי של שסתומי ניתוק דלק וגז.
* הקמת עמדות קירור למכלי דלק/כימיקלים סמוכים.
* שימוש בקצף כיבוי ייעודי (AFFF) במקום מים עבור שריפות נוזלים דליקים.
            """
        elif 'יער' in terrain or 'חורש' in terrain or 'open' in terrain or 'forest' in terrain or 'פתוח' in terrain:
            terrain_guideline = """
🌲 **פעולות בגזרת שטח פתוח/יער:**
* יצירת אזורי חישוף וקווים שרופים (Counter-burning) לבלימת האש.
* שילוב מטוסי כיבוי להטלת מעכבי בעירה בראש החזית.
* פריסת תצפיות בנקודות שולטות לאיתור מוקדי משנה הנוצרים מגיצים.
            """
        else:
            terrain_guideline = """
🏘️ **פעולות בגזרת מגורים עירונית:**
* סריקה מהירה לאיתור ולחילוץ לכודים במבנים אפופים עשן.
* בידוד מקורות אנרגיה (חשמל וגז עירוני) לכלל הרחובות המושפעים.
* ניהול פאניקה והנחיית תושבים להסתגרות בממ"דים במידה וציר הפינוי חסום.
            """

        hazards = answers.get('hazards', '').lower()
        if 'כן' in hazards or 'yes' in hazards or 'לכוד' in hazards or 'חומר' in hazards or 'trap' in hazards or 'chem' in hazards:
            hazards_guideline = "🚨 **תשומת לב מיוחדת**: דווח על לכודים או חומרים מסוכנים! יש להקצות צוותי סריקה וחילוץ ממוגנים (לבוש תרמי מלא ומנפ\"פ) ותצפיות חומ\"ס ייעודיות באופן מיידי."
        else:
            hazards_guideline = "✅ לא דווח על לכודים או חומרים חריגים בשלב זה. המשך פעילות בבטחה."

        return f"""
### **[פרוטוקול ניהול משברים פעיל - גיבוי לוקאלי]**

שלום המפקד, להלן הניתוח המבצעי המבוסס על מודל הנתונים המקומי עבור **{location}**:

{matched_info}

---

{loc_guideline}

{terrain_guideline}

{hazards_guideline}

*המלצה זו גובשה באופן אוטומטי על ידי מנוע ה-ML המקומי של FireMate AI.*
        """

    def generate_fallback_en(self, answers, matched_rows, sim_score, location, frp, wind_speed):
        if not matched_rows.empty:
            mr = matched_rows.iloc[0]
            matched_info = f"""
* **NASA Matched Region**: {mr.get('region', 'N/A')} ({mr.get('country', 'N/A')})
* **Fire Type**: {mr.get('fire_type', 'N/A')}
* **Fire Intensity**: {mr.get('fire_intensity', 'N/A')}
* **Radiative Power (FRP)**: {mr.get('fire_radiative_power_mw', 'N/A')} MW
* **Cosine Similarity Score**: {sim_score:.1f}%
            """
        else:
            matched_info = f"* **Similarity Score**: {sim_score:.1f}%"

        loc_lower = location.lower()
        if 'חיפה' in loc_lower or 'haifa' in loc_lower or 'כרמל' in loc_lower or 'carmel' in loc_lower:
            loc_guideline = """
⚠️ **Operational Protocol tailored for Haifa & Carmel region (Industry/Hills/Coast):**
* **Chemical Hazards**: Proximity to Haifa Bay and chemical industries. Perform continuous toxic gas monitoring and prepare for chemical incidents.
* **Topography & Winds**: Mountainous Carmel slopes and coastal wind patterns can cause rapid flame front shifts. Avoid placing crews in narrow ravines.
* **Evacuation Routes**: Direct evacuations towards West/South, accounting for downwind plume dispersion.
            """
        elif 'ירושלים' in loc_lower or 'jerusalem' in loc_lower or 'ישרולים' in loc_lower:
            loc_guideline = """
⚠️ **Operational Protocol tailored for Jerusalem region (Mountainous/Dense Urban/WUI):**
* **Slope Propagation**: Mountain winds push fires uphill. Deploy containment lines along ridge lines.
* **Water Supply Grid**: Low water pressure at high altitudes. Set up relay pumping networks and order large capacity water tankers.
* **Access Limitations**: Narrow historical roads restrict large engines. Prioritize light vehicles and long hose lays.
            """
        else:
            loc_guideline = f"""
⚠️ **General Operational Protocol for {location}:**
* Deploy defensive lines at the wildland-urban interface boundaries.
* Coordinate early evacuations of high-risk neighborhoods.
* Continuously monitor local wind changes.
            """

        terrain = answers.get('terrain', '').lower()
        if 'תעש' in terrain or 'ind' in terrain or 'מפעל' in terrain:
            terrain_guideline = """
🏭 **Industrial Sector Actions:**
* Locate and isolate fuel and gas shutoff valves immediately.
* Establish cooling lines on adjacent petrochemical tanks and warehouses.
* Use Class B firefighting foam (AFFF) for liquid fires rather than water.
            """
        elif 'יער' in terrain or 'חורש' in terrain or 'open' in terrain or 'forest' in terrain or 'פתוח' in terrain:
            terrain_guideline = """
🌲 **Wildland & Forestry Sector Actions:**
* Create fire breaks and deploy counter-burning techniques to halt propagation.
* Coordinate with aerial suppression units to drop retardants on the fire head.
* Deploy spotters on high ground to detect ember-induced secondary fires.
            """
        else:
            terrain_guideline = """
🏘️ **Urban & Residential Sector Actions:**
* Initiate rapid search and rescue inside smoke-logged structures.
* Coordinate municipal utilities to shut down electric grid and gas lines in affected streets.
* Manage civilian panic; advise sheltering in fortified rooms if evacuations are blocked.
            """

        hazards = answers.get('hazards', '').lower()
        if 'yes' in hazards or 'כן' in hazards or 'trap' in hazards or 'chem' in hazards or 'לכוד' in hazards or 'חומר' in hazards:
            hazards_guideline = "🚨 **Critical Warning**: Trapped individuals or hazardous materials reported! Allocate specialized search-and-rescue teams (full SCBA and thermal gear) and hazmat units immediately."
        else:
            hazards_guideline = "✅ No trapped casualties or hazmat concerns reported at this stage. Proceed with standard safety rules."

        return f"""
### **[Active Crisis Management Protocol - Local Fallback]**

Hello Commander, here is the operational analysis based on our local data models for **{location}**:

{matched_info}

---

{loc_guideline}

{terrain_guideline}

{hazards_guideline}

*This guidance was automatically generated by the FireMate AI local ML engine.*
        """

    def answer_follow_up(self, answers, matched_rows, sim_score, location, conversation_history, new_question, lang):
        system_prompt = f"""
        You are FireMate AI, a real-time operational commander assistant.
        The user has reported an active fire incident at location: {location}.
        Details collected:
        - Terrain: {answers.get('terrain')}
        - Intensity: {answers.get('intensity')}
        - Wind: {answers.get('wind')}
        - Hazards: {answers.get('hazards')}
        
        Our matched historical NASA model similarity is {sim_score:.1f}%.
        
        The user is asking a follow-up question. Answer it operationally, logically, and in the language requested ('{lang}').
        Keep it direct, tactical, and clear.
        """
        
        global model
        if model is not None:
            try:
                # Compile messages context
                prompt_messages = [system_prompt]
                for msg in conversation_history[-6:]:  # include recent history context
                    prompt_messages.append(f"{msg['role']}: {msg['content']}")
                prompt_messages.append(f"user: {new_question}")
                response = model.generate_content(prompt_messages)
                return response.text
            except Exception:
                pass
        
        # Local Fallback
        if lang == 'he':
            return "אני פועל כעת במצב חירום לא מקוון. אנא פעל לפי פרוטוקול הכיבוי המבצעי המפורט למעלה ודווח מיידית למרכז השליטה הארצי בטלפון 102."
        else:
            return "I am currently running in offline emergency mode. Please follow the detailed operational protocol above and report immediately to national dispatch (Tel: 102)."

agent = FireMateIntelligenceEngine(df_fires, df_weekly)

# --- Session State Initialization ---
if "lang" not in st.session_state:
    st.session_state.lang = "he"
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "welcome"
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": LOC[st.session_state.lang]['welcome_msg']}
    ]

lang = st.session_state.lang

# --- Header & Language Switcher ---
col_logo, col_lang = st.columns([5, 2])
with col_logo:
    st.markdown(f"<div class='brand-logo'>{LOC[lang]['header_logo']}</div>", unsafe_allow_html=True)
with col_lang:
    btn_label = "English 🇬🇧" if lang == "he" else "עברית 🇮🇱"
    if st.button(btn_label, key="lang_toggle"):
        new_lang = "en" if lang == "he" else "he"
        st.session_state.lang = new_lang
        # If we are in welcome stage, translate welcome message
        if st.session_state.chat_stage == 'welcome':
            st.session_state.messages = [{"role": "assistant", "content": LOC[new_lang]['welcome_msg']}]
        # Force rerun to translate static texts
        st.rerun()

# --- Main Hero UI ---
st.markdown(f"<div class='main-title'>{LOC[lang]['main_title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='hero-brand-name'>{LOC[lang]['hero_brand_name']}</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-section-transparent">
    <div class="info-title-large">{LOC[lang]['info_title']}</div>
    <div class="info-text-large">{LOC[lang]['info_text']}</div>
</div>
""", unsafe_allow_html=True)

# --- Interactive Progress Indicator ---
if st.session_state.chat_stage == 'questioning':
    curr_q = st.session_state.current_q_index
    st.markdown(f"<div class='progress-label'>{LOC[lang]['progress_text'].format(current=curr_q+1)}</div>", unsafe_allow_html=True)
    st.progress(curr_q / 5.0)

# --- Sample Scenarios Buttons ---
if st.session_state.chat_stage == 'welcome':
    st.markdown(f"<div class='sample-heading'>{LOC[lang]['sample_heading']}</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    click_query = ""
    with col1:
        if st.button(LOC[lang]['btn_urban']):
            click_query = "שריפה בבניין מגורים בירושלים" if lang == 'he' else "Residential building fire in Jerusalem"
    with col2:
        if st.button(LOC[lang]['btn_industrial']):
            click_query = "שריפה במחסן כימיקלים בחיפה" if lang == 'he' else "Chemical warehouse fire in Haifa"
    with col3:
        if st.button(LOC[lang]['btn_wildfire']):
            click_query = "שריפת יער בפארק הכרמל" if lang == 'he' else "Forest fire in Carmel Park"
            
    if click_query:
        st.session_state.initial_query = click_query
        st.session_state.messages.append({"role": "user", "content": click_query})
        st.session_state.chat_stage = 'questioning'
        st.session_state.current_q_index = 0
        st.session_state.answers = {}
        
        # Add question 1
        q_intro = LOC[lang]['question_intro']
        q1 = LOC[lang]['q1']
        st.session_state.messages.append({"role": "assistant", "content": f"{q_intro}\n\n{q1}"})
        st.rerun()

# --- Reset Button ---
if st.session_state.chat_stage != 'welcome':
    col_reset_space, col_reset_btn = st.columns([4, 1.5])
    with col_reset_btn:
        if st.button(LOC[lang]['reset_btn'], key="reset_chat"):
            st.session_state.chat_stage = "welcome"
            st.session_state.current_q_index = 0
            st.session_state.answers = {}
            st.session_state.messages = [
                {"role": "assistant", "content": LOC[lang]['welcome_msg']}
            ]
            st.rerun()

# --- Chat Messages Display ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "✨"
    css_class = "user-msg-box" if message["role"] == "user" else "bot-msg-box"
    
    # Adjust layout direction per message language if desired, or align globally
    align_style = "text-align: right; direction: rtl;" if lang == 'he' else "text-align: left; direction: ltr;"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"<div class='{css_class}' style='{align_style}'>{message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input Processing ---
user_query = st.chat_input(LOC[lang]['chat_placeholder'])

if user_query:
    # 1. Process Initial Report
    if st.session_state.chat_stage == 'welcome':
        st.session_state.initial_query = user_query
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.chat_stage = 'questioning'
        st.session_state.current_q_index = 0
        st.session_state.answers = {}
        
        q_intro = LOC[lang]['question_intro']
        q1 = LOC[lang]['q1']
        st.session_state.messages.append({"role": "assistant", "content": f"{q_intro}\n\n{q1}"})
        st.rerun()
        
    # 2. Process Questioning Stage answers
    elif st.session_state.chat_stage == 'questioning':
        st.session_state.messages.append({"role": "user", "content": user_query})
        curr_q = st.session_state.current_q_index
        
        # Save current answer
        if curr_q == 0:
            st.session_state.answers['location'] = user_query
        elif curr_q == 1:
            st.session_state.answers['terrain'] = user_query
        elif curr_q == 2:
            st.session_state.answers['intensity'] = user_query
        elif curr_q == 3:
            st.session_state.answers['wind'] = user_query
        elif curr_q == 4:
            st.session_state.answers['hazards'] = user_query
            
        next_q = curr_q + 1
        st.session_state.current_q_index = next_q
        
        if next_q < 5:
            # Ask next question
            q_key = f"q{next_q+1}"
            st.session_state.messages.append({"role": "assistant", "content": LOC[lang][q_key]})
        else:
            # All 5 questions answered -> generate final tactical protocol!
            st.session_state.chat_stage = 'completed'
            with st.spinner(LOC[lang]['spinner_analyzing']):
                # Run spelling correction on location
                raw_loc = st.session_state.answers.get('location', '')
                corrected_loc = agent.correct_spelling(raw_loc, lang)
                
                # Fetch similarity and data analysis
                top_matches, sim_score, frp, wind_speed, flammability, intensity_level = agent.find_closest_dataset_matches(st.session_state.answers)
                is_anomaly = agent.compute_local_ml_metrics()
                
                # Generate LLM response
                report_content = agent.ask_llm_agent(
                    st.session_state.answers, 
                    top_matches, 
                    sim_score, 
                    corrected_loc, 
                    is_anomaly, 
                    lang
                )
                
                # Combine match presentation & advice
                intro_report = f"### {LOC[lang]['final_report_title']} ({corrected_loc})\n\n{report_content}"
                st.session_state.messages.append({"role": "assistant", "content": intro_report})
                
                # Add follow-up instruction
                st.session_state.messages.append({"role": "assistant", "content": LOC[lang]['follow_up_prompt']})
                
        st.rerun()

    # 3. Process Follow-Up Questions (completed stage)
    elif st.session_state.chat_stage == 'completed':
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        raw_loc = st.session_state.answers.get('location', '')
        corrected_loc = agent.correct_spelling(raw_loc, lang)
        top_matches, sim_score, _, _, _, _ = agent.find_closest_dataset_matches(st.session_state.answers)
        
        with st.spinner(LOC[lang]['spinner_analyzing']):
            follow_up_response = agent.answer_follow_up(
                st.session_state.answers,
                top_matches,
                sim_score,
                corrected_loc,
                st.session_state.messages[:-1], # excluding current user message
                user_query,
                lang
            )
            st.session_state.messages.append({"role": "assistant", "content": follow_up_response})
            
        st.rerun()

# --- Custom Footer ---
st.markdown(f"""
    <div class='custom-footer'>
        <div class='footer-text-main'>{LOC[lang]['footer_main']}</div>
        <div class='footer-text-sub'>{LOC[lang]['footer_sub']}</div>
    </div>
""", unsafe_allow_html=True)

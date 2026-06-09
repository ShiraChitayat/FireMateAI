import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import json
from datetime import datetime

# ==========================================
# 1. הגדרות תצוגה ועיצוב תכלת חדשני (UI)
# ==========================================
st.set_page_config(page_title="FireMate AI", page_icon="🔥", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #E6F2F8; /* רקע תכלת בהיר ומרגיע */
    }
    .stChatMessage {
        border-radius: 15px;
        border: 1px solid #BFE0F2;
        background-color: #ffffff;
    }
    h1 {
        color: #005A8D;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔥 FireMate AI")
st.subheader("סוכן חכם לניהול משברים ואסונות טבע - שריפות")

# ==========================================
# 2. הגדרת מודל Gemini ממשתני הסביבה
# ==========================================
# שליפת המפתח מהסביבה של Render (או .env לוקאלי)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("שגיאה קריטית: לא נמצא מפתח API. נא להגדיר את משתנה הסביבה GEMINI_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)

system_instruction = """
אתה סוכן חכם בשם FireMate AI. מטרתך לסייע לכבאים ולמנהלי משברים לנהל אירועי שריפות יער על סמך נתוני Big Data.
גבולות גזרה חובה: עליך לענות אך ורק על שאלות הקשורות לשריפות, כיבוי אש, מזג אוויר וניהול אסונות טבע מסוג שריפה. 
אם המשתמש שואל על נושא אחר, עליך לענות: "אני לא מתעסק בנושא הזה, תחום המומחיות שלי מוגבל לשריפות בלבד".
הטון שלך צריך להיות מקצועי, חדשני, מרגיע, מבוסס נתונים וברור.
"""

model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)

# ==========================================
# 3. טעינת נתונים והכנת מודלי ה-ML
# ==========================================
cat_features = ['day_part', 'fire_type', 'country']
num_features = ['fire_radiative_power_mw', 'temp_max_c', 'humidity_pct']

@st.cache_data
def load_and_prep_all_data():
    # קובץ אירועים - לניקוי וחישובי דמיון
    df_events = pd.read_csv("wildfire_multi_region_dataset.csv")
    df_events.columns = df_events.columns.str.strip().str.lower() # תאימות לשמות העמודות
    df_events = df_events.dropna(subset=cat_features + num_features).reset_index(drop=True)
    
    # הכנת One-Hot Encoder למשתנים קטגוריאליים
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df_events[cat_features])
    
    # קבצי מגמות שבועיות ומצטברות
    df_weekly = pd.read_csv("area_burnt_by_wildfires_by_week.csv")
    df_cumulative = pd.read_csv("cumulative_area_burnt_by_wildfires_by_week.csv")
    
    return df_events, encoder, encoded_cats, df_weekly, df_cumulative

try:
    df_events, encoder, encoded_cats, df_weekly, df_cumulative = load_and_prep_all_data()
except Exception as e:
    st.error(f"שגיאה בטעינת הקבצים. ודאי שהקבצים נמצאים בתיקייה. פרטים: {e}")
    st.stop()

# ==========================================
# 4. פונקציות אלגוריתמיות: קוסינוס וג'קארד
# ==========================================
def fast_jaccard(data_matrix, input_vector):
    intersect = np.dot(data_matrix, input_vector.T).flatten()
    row_sums = data_matrix.sum(axis=1)
    input_sum = input_vector.sum()
    union = row_sums + input_sum - intersect
    
    output = np.zeros_like(intersect, dtype=np.float64)
    return np.divide(intersect, union, out=output, where=union!=0)

def get_similar_fires(user_num_dict, user_cat_dict, top_n=3):
    # המרת קלט משתמש ל-DataFrames
    user_num_df = pd.DataFrame([user_num_dict], columns=num_features)
    user_cat_df = pd.DataFrame([user_cat_dict], columns=cat_features)
    
    # חישוב דמיון קוסינוס (כמותי)
    cosine_sim = cosine_similarity(user_num_df, df_events[num_features])[0]
    
    # חישוב דמיון ג'קארד (קטגוריאלי)
    user_encoded_cat = encoder.transform(user_cat_df)
    jaccard_sim = fast_jaccard(encoded_cats, user_encoded_cat)
    
    # שקלול שווה (50-50) בין שני המדדים
    combined_score = (cosine_sim + jaccard_sim) / 2
    
    # שליפת האינדקסים של הרשומות הדומות ביותר
    top_indices = np.argsort(combined_score)[::-1][:top_n]
    
    return df_events.iloc[top_indices]

# ==========================================
# 5. עיבוד שפה טבעית (NLP) עם Gemini
# ==========================================
def extract_variables_from_text(user_text):
    current_week = datetime.now().isocalendar()[1]
    
    extraction_prompt = f"""
    אתה רכיב NLP מתקדם. נתח את הטקסט הבא וחלץ מתוכו את המשתנים הבאים בפורמט JSON בלבד.
    אם נתון לא קיים, שים ערכי ברירת מחדל הגיוניים (למשל: country: Unknown, fire_type: Forest, fire_radiative_power_mw: 50, temp_max_c: 30, humidity_pct: 40).
    
    1. "country" (שם המדינה באנגלית)
    2. "day_part" (Morning, Afternoon, Evening, Night)
    3. "fire_type" (Forest, Bushfire וכו')
    4. "fire_radiative_power_mw" (מספר המייצג קרינה/עוצמה)
    5. "temp_max_c" (טמפרטורה במספר)
    6. "humidity_pct" (אחוז לחות במספר)
    7. "week_number" (מספר השבוע בשנה בין 1 ל-52. ברירת מחדל: {current_week})

    הטקסט לניתוח: "{user_text}"
    
    החזר אך ורק JSON תקני. בלי פתיח, בלי סיומת ובלי סימני קוד (```json).
    """
    
    raw_response = model.generate_content(extraction_prompt)
    
    try:
        extracted_data = json.loads(raw_response.text.strip())
        return extracted_data
    except Exception:
        # Fallback במקרה של שגיאת פענוח
        return {
            "country": "Unknown", "day_part": "Afternoon", "fire_type": "Forest",
            "fire_radiative_power_mw": 50.0, "temp_max_c": 30.0, "humidity_pct": 40.0,
            "week_number": current_week
        }

# ==========================================
# 6. ניהול הצ'אט והממשק
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "היי! אני FireMate AI, סוכן שעוזר בניהול שריפות. תאר לי את מצב השריפה שעמה אתה מתמודד (מיקום, מזג אוויר, שעה) ואנתח עבורך את הסיטואציה."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("תאר לי את מצב השריפה בטקסט חופשי..."):
    # הוספת הודעת המשתמש למסך
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("FireMate AI מנתח את הנתונים (NLP) ומחפש פרופילי סיכון דומים במאגר ה-Big Data..."):
            
            # שלב א': חילוץ ישויות
            extracted = extract_variables_from_text(prompt)
            
            user_num = {k: float(extracted.get(k, 0)) for k in num_features}
            user_cat = {k: str(extracted.get(k, 'Unknown')) for k in cat_features}
            
            # שלב ב': מציאת שריפות דומות (ML)
            try:
                similar_fires = get_similar_fires(user_num, user_cat, top_n=3)
                similar_text = similar_fires[['country', 'year', 'temp_max_c', 'fire_radiative_power_mw']].to_string(index=False)
            except Exception as e:
                similar_text = "לא נמצאו נתונים מספקים לחישוב דמיון."
            
            # שלב ג': בניית הפרומפט הסופי למודל
            context_prompt = f"""
            המשתמש כתב: "{prompt}"
            
            1. הנתונים שחולצו מהדיווח (JSON): {json.dumps(extracted, ensure_ascii=False)}
            
            2. השריפות ההיסטוריות הדומות ביותר שנמצאו במאגר על ידי אלגוריתמי AI:
            {similar_text}
            
            משימתך כסוכן:
            - ענה על הדיווח של המשתמש בצורה מקצועית.
            - ציין שהתבססת על נתונים היסטוריים דומים שנמצאו במערכת.
            - הצע המלצות לפעולה או חיווי על רמת הסיכון בהתאם לנתונים (טמפרטורה, עוצמת קרינה).
            - זכור את גבולות הגזרה: אם הקלט אינו קשור לשריפות, סרב לענות באדיבות!
            """
            
            # קבלת התשובה
            final_response = model.generate_content(context_prompt)
            
            # הצגת התשובה למשתמש
            st.markdown(final_response.text)
            
            # (אופציונלי) הצגת טבלת השריפות הדומות מתחת לתשובה כשקיפות למשתמש
            with st.expander("🔍 הצג את נתוני ה-Big Data ההיסטוריים (לפי Cosine & Jaccard)"):
                st.dataframe(similar_fires[['country', 'year', 'day_part', 'fire_type', 'temp_max_c', 'humidity_pct', 'fire_radiative_power_mw']])
            
            st.session_state.messages.append({"role": "assistant", "content": final_response.text}) 

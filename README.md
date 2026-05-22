# FireMate AI

FireMate AI is an advanced AI-driven agent designed for real-time wildfire emergency analysis and tactical decision support. This project leverages NASA satellite data anchors and machine learning similarity engines to provide data-backed incident response protocols.

## Features

- **Real-time Tactical Analysis:** Dynamic processing of wildfire incident reports.
- **ML Similarity Engine:** Cosine similarity analysis to match current incidents with historical NASA satellite records.
- **Anomaly Detection:** Real-time Z-Score analysis to identify catastrophic fire trends.
- **Context-Aware Recommendations:** Specialized protocols for Residential, Industrial, and Open-Space wildfire scenarios.
- **Clean UI/UX:** Interactive chat-based interface with intuitive sample reporting.

## Project Structure

FireMateAI
├── app.py                                        # Main Streamlit application logic
├── style.css                                     # Custom styles for elegant UI
├── wildfire_dataset.csv                          # NASA historical fire data
├── area_burnt_weekly.csv                         # Weekly trend data
├── cumulative_burnt_weekly.csv                   # Cumulative burnt area data
├── README.md                                     # Project documentation
└── requirements.txt                              # Dependencies

## Technologies Used

- **Frontend:** Streamlit
- **Backend:** Python
- **ML/Analytics:** Scikit-learn (Cosine Similarity), Pandas (Anomaly Detection)
- **Deployment:** Render

import streamlit as st
import joblib
import re
import string
import os
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd

# --- 1. Page Config (UPDATED) ---
st.set_page_config(
    page_title="Fake News Detection with AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Environment Variables & Configure APIs ---
load_dotenv()
try:
    api_key = os.environ["GOOGLE_API_KEY"]
    if not api_key:
        raise KeyError
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')
    GEMINI_ENABLED = True
except KeyError:
    st.error("Warning: GOOGLE_API_KEY not found. Gemini features will be disabled.")
    GEMINI_ENABLED = False
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    GEMINI_ENABLED = False

# --- 3. Load ML Model (Cached) ---
@st.cache_resource
def load_model_and_vectorizer():
    """Loads the trained model and vectorizer from disk."""
    try:
        model = joblib.load('fake_news_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        coef_map = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
        coef_map = coef_map.set_index('feature')
        
        return model, vectorizer, coef_map
    
    except FileNotFoundError:
        st.error("Error: Model/vectorizer files not found. Please run 'train.py' first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, vectorizer, coef_map = load_model_and_vectorizer()

# --- 4. Helper Functions ---

def clean_text(text):
    """Cleans the input text (lowercase, remove punc, URLs, etc.)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def get_ml_prediction(text_to_analyze):
    """Predicts using the loaded scikit-learn model."""
    if not model or not vectorizer or not text_to_analyze:
        return None
    try:
        cleaned_text = clean_text(text_to_analyze)
        if not cleaned_text:
            return {"error": "No valid text found after cleaning."}
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction_code = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        if prediction_code == 1:
            label = "Real"
            confidence = probabilities[1]
        else:
            label = "Fake"
            confidence = probabilities[0]
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

def get_model_thinking(text_to_analyze, vectorizer, coef_map):
    """Finds the words in the text that most influenced the model."""
    if coef_map is None:
        return None
        
    cleaned_text = clean_text(text_to_analyze)
    words_in_text = set(cleaned_text.split())
    
    contributions = coef_map.loc[coef_map.index.intersection(words_in_text)]
    contributions = contributions.sort_values(by='coefficient', ascending=False)
    
    top_real = contributions.head(10)
    top_fake = contributions.tail(10)
    
    top_words_df = pd.concat([top_real, top_fake])
    top_words_df = top_words_df.reset_index()
    top_words_df['color'] = top_words_df['coefficient'].apply(lambda x: 'Real' if x > 0 else 'Fake')
    return top_words_df.sort_values(by='coefficient', ascending=False)


def get_gemini_analysis(text_to_analyze, original_label):
    """Gets Gemini analysis, asks for a final verdict, and asks for links."""
    if not GEMINI_ENABLED:
        return "Gemini is not configured."
    if not text_to_analyze:
        return "No text to analyze."
    try:
        prompt = f"""
        You are a fact-checking news assistant.
        Please provide a neutral analysis of the following article text.

        1.  **Key Claims:** Summarize the main claims in 3 bullet points.
        2.  **Credibility Analysis:** Point out 2-3 "red flags" (or "green flags") that suggest
            it is (or isn't) credible (e.g., loaded language, anonymous sources, verifiable data, etc.).
        3.  **Final Verdict:** Based on your analysis, do you believe this article is
            more likely to be **Real** or **Fake**? State your conclusion clearly.
        4.  **Verification:** (Optional) If, and only if, you conclude the article is **Real** and describes a verifiable event, please perform a web search and provide 1-2 
            source links that corroborate the story. If you conclude it's Fake, or 
            if you cannot find any links, **just skip this section and do not mention it.**

        Article Text:
        ---
        {text_to_analyze}
        ---
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during Gemini analysis: {e}"

# --- 5. Plotting Functions ---

def create_gauge_chart(confidence, label):
    """Creates a Plotly gauge chart."""
    

    value = confidence * 100
    
    if label == "Real":
        color = "#28a745" # Green
    else:
        color = "#dc3545" # Red

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'suffix': "%", 'font': {'size': 24}}, 
        title = {'text': f"Result: {label}", 'font': {'size': 28, 'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(220, 53, 69, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(40, 167, 69, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_contribution_chart(df):
    """Creates a Plotly bar chart of word contributions."""
    
    if df is None or df.empty:
        return go.Figure().update_layout(title="No influential words found in model's vocabulary.")
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['coefficient'],
        y=df['feature'],
        orientation='h',
        marker_color=df['color'].map({'Real': '#28a745', 'Fake': '#dc3545'}),
        text=df['coefficient'].apply(lambda x: f'{x:.2f}'),
        textposition='auto'
    ))
    fig.update_layout(
        title="Top Words Influencing the Model's Decision",
        xaxis_title="Impact Score (Coefficient)",
        yaxis_title="Word",
        yaxis=dict(autorange="reversed"),
        height=400 + (len(df) * 20),
        margin=dict(l=100)
    )
    return fig

# --- 6. Initialize Session State ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 7. Sidebar (UPDATED) ---
with st.sidebar:
    st.title("About this AI Detector")
    st.markdown("""
        This app is a **Fake News Detector** powered by a two-layer AI system.
        
        It combines a custom-trained local model with a large language model (LLM) 
        to provide a comprehensive analysis of news articles.
    """)
    st.divider()
    
    st.subheader("How the AI Works")
    st.markdown("""
        1.  **Local Model:** A `LogisticRegression` classifier gives a fast "Fake" or "Real" 
            prediction. It's trained to be a *specialist*.
        2.  **Gemini Analysis:** The `gemini-2.5-pro` model acts as a *generalist*, 
            analyzing the article's tone, claims, and verifying sources.
    """)
    st.divider()
    
    st.subheader("Key Technologies Used:")
    st.markdown("""
        * **App Framework:** `Streamlit`
        * **Local Model:** `Scikit-learn` & `Pandas`
        * **Charts:** `Plotly`
        * **AI Analysis:** `Google Gemini API`
        * **Data Sources:** `Hugging Face datasets` & `Kaggle`
    """)
    st.divider()

    st.subheader("Training Data")
    st.markdown(
        "**Real News (Diverse):** `AG News Dataset` (127,600 articles) from [Hugging Face](https://huggingface.co/datasets/ag_news)", 
        unsafe_allow_html=True
    )
    st.markdown(
        "**Fake News (Political):** `Fake/Real News` (23,000 articles) from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)", 
        unsafe_allow_html=True
    )

# --- 8. Main App Layout (UPDATED) ---
st.title("📰 Fake News Detection with AI")
st.markdown("Paste an article in the box below to see its AI-powered classification.")
st.divider()

col1, col2 = st.columns([0.55, 0.45])

# --- Left Column (Input) ---
with col1:
    with st.container(border=True):
        st.subheader("Analyze Article")
        with st.form(key="analysis_form"):
            text_input = st.text_area(
                "Paste the full text of the article here:",
                height=300,
                placeholder="Once you paste the text, click 'Analyze' below."
            )
            include_gemini = st.checkbox(
                "Include Gemini Deeper Analysis",
                value=False,
                help="Get a summary and red-flag analysis (slower)."
            )
            submitted = st.form_submit_button("Analyze", type="primary")

        if st.button("Clear Results"):
            st.session_state.analysis_results = None
            st.rerun()

# --- Analysis Logic ---
if submitted:
    if model and vectorizer and text_input:
        results = {}
        with st.spinner("Analyzing..."):
            ml_result = get_ml_prediction(text_input)
            results["ml"] = ml_result
            
            thinking_df = get_model_thinking(text_input, vectorizer, coef_map)
            results["thinking"] = thinking_df
            
            if "error" in ml_result:
                st.error(f"ML Model Error: {ml_result['error']}")
            
            if include_gemini:
                if GEMINI_ENABLED:
                    gemini_result = get_gemini_analysis(text_input, ml_result.get('label', 'Unknown'))
                    results["gemini"] = gemini_result
                else:
                    results["gemini"] = "Gemini is disabled (API key not found)."
        
        st.session_state.analysis_results = results
    
    elif not text_input:
        st.warning("Please enter some text to analyze.")
    else:
        st.warning("Model is not loaded. Check for errors above.")

# --- Right Column (Results with TABS) ---
with col2:
    with st.container(border=True):
        
        if st.session_state.analysis_results is None:
            st.subheader("📊 Results")
            st.info("Results will appear here after you analyze an article.")
        else:
            tab1, tab2, tab3 = st.tabs(["📊 Main Result", "🧠 Model Thinking", "🤖 Gemini Analysis"])
            results = st.session_state.analysis_results

            with tab1:
                st.subheader("Local Model Prediction")
                if "ml" in results and "error" not in results["ml"]:
                    gauge_fig = create_gauge_chart(results["ml"]['confidence'], results["ml"]['label'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                else:
                    st.error(f"ML Model Error: {results['ml'].get('error', 'Unknown')}")

            with tab2:
                st.subheader("Top Influential Words")
                if "thinking" in results:
                    contribution_fig = create_contribution_chart(results["thinking"])
                    st.plotly_chart(contribution_fig, use_container_width=True)
                else:
                    st.info("Could not generate the model thinking chart.")
                
            with tab3:
                st.subheader("Gemini Deeper Analysis")
                if "gemini" in results:
                    st.markdown(results["gemini"])
                else:
                    st.info("You did not select the Gemini analysis for this run.")
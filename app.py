import streamlit as st
import joblib
import re
import string
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd

# --- 1. Page Config ---
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
    client = genai.Client(api_key=api_key)
    GEMINI_ENABLED = True
except KeyError:
    st.error("Warning: GOOGLE_API_KEY not found. Gemini features will be disabled.")
    GEMINI_ENABLED = False
    client = None
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    GEMINI_ENABLED = False
    client = None

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
    """Predicts using the loaded scikit-learn model with uncertainty zone."""
    if not model or not vectorizer or not text_to_analyze:
        return None
    try:
        cleaned_text = clean_text(text_to_analyze)
        if not cleaned_text:
            return {"error": "No valid text found after cleaning."}
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction_code = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]

        max_confidence = max(probabilities)

        # ✅ Uncertainty zone — avoids forcing Real/Fake on low-confidence predictions
        if max_confidence < 0.60:
            label = "Uncertain"
            confidence = max_confidence
        elif prediction_code == 1:
            label = "Real"
            confidence = probabilities[1]
        else:
            label = "Fake"
            confidence = probabilities[0]

        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

def get_model_thinking(text_to_analyze, vectorizer, coef_map, final_label):
    """Finds the words in the text that most influenced the final decision."""
    if coef_map is None:
        return None, ""

    cleaned_text = clean_text(text_to_analyze)
    words_in_text = set(cleaned_text.split())
    contributions = coef_map.loc[coef_map.index.intersection(words_in_text)]

    if final_label == "Real":
        top_words_df = contributions[contributions['coefficient'] > 0].sort_values(by='coefficient', ascending=False).head(15)
        top_words_df['color'] = '#28a745'
        title = "Top 15 Words Pushing Result to 'REAL'"
    elif final_label == "Fake":
        top_words_df = contributions[contributions['coefficient'] < 0].sort_values(by='coefficient', ascending=True).head(15)
        top_words_df['color'] = '#dc3545'
        title = "Top 15 Words Pushing Result to 'FAKE'"
    else:
        # Uncertain — show both positive and negative top words
        top_pos = contributions[contributions['coefficient'] > 0].sort_values(by='coefficient', ascending=False).head(8)
        top_neg = contributions[contributions['coefficient'] < 0].sort_values(by='coefficient', ascending=True).head(7)
        top_pos['color'] = '#28a745'
        top_neg['color'] = '#dc3545'
        top_words_df = pd.concat([top_pos, top_neg])
        title = "Top Words (Mixed Signals — Uncertain Result)"

    top_words_df = top_words_df.reset_index()
    return top_words_df, title


@st.cache_data(ttl=3600)
def cached_gemini_analysis(text_to_analyze):
    """Cached wrapper for Gemini analysis — saves quota on repeated identical inputs."""
    return get_gemini_analysis(text_to_analyze, "")


def get_gemini_analysis(text_to_analyze, original_label):
    """Gets Gemini analysis with Search Grounding, retry logic, and fallback."""
    if not GEMINI_ENABLED or client is None:
        return "⚠️ Gemini is not configured."
    if not text_to_analyze:
        return "⚠️ No text to analyze."

    prompt = f"""
    You are a fact-checking news assistant with access to real-time web search.
    Search the web to verify the claims in the article before analysing it.

    1.  **Key Claims:** Summarize the main claims in 3 bullet points.
    2.  **Credibility Analysis:** Point out 2-3 red flags or green flags regarding credibility
        (e.g., loaded language, anonymous sources, verifiable data).
        IMPORTANT: If you are unfamiliar with an event, search for it first before calling it
        a red flag. Do NOT mark something as fake simply because it sounds unfamiliar —
        it may be a very recent real event.
    3.  **Final Verdict:** Based on your analysis and search results, state:
        - **Real** — if the article is credible and verifiable
        - **Fake** — if the article contains clear misinformation
        - **Uncertain** — if evidence is insufficient to decide either way
    4.  **Verification:** (Optional) If you conclude Real, provide 1-2 source links.
        If Fake or Uncertain, skip this section entirely.

    Article Text:
    ---
    {text_to_analyze}
    ---
    """

    max_retries = 3

    for attempt in range(max_retries):
        try:
            # ✅ First attempt: Search Grounding enabled
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
            except Exception:
                # ✅ Fallback: Search Grounding unavailable, use standard Gemini
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

            return response.text

        except Exception as e:
            error_msg = str(e)

            if "503" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)  # 5s, 10s, 15s
                    continue
                return (
                    "⚠️ **Gemini is currently experiencing high demand (503).**\n\n"
                    "The local ML prediction above is still valid.\n\n"
                    "Please try the Gemini analysis again in a few minutes."
                )

            elif "429" in error_msg:
                return (
                    "⚠️ **Gemini API quota exceeded (429).**\n\n"
                    "The local model is still working normally.\n\n"
                    "Please check your Gemini API quota or billing at "
                    "[Google AI Studio](https://aistudio.google.com)."
                )

            else:
                return f"⚠️ Gemini Error: {error_msg}"

    return "⚠️ Gemini service unavailable after retries. Please try again later."


# --- 5. Plotting Functions ---

def create_gauge_chart(confidence, label):
    """Creates a Plotly gauge chart with Real/Fake/Uncertain colors."""
    value = confidence * 100

    # ✅ Three-state color: green, red, amber
    if label == "Real":
        color = "#28a745"
    elif label == "Fake":
        color = "#dc3545"
    else:
        color = "#ffc107"  # Uncertain → amber

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 24}},
        title={'text': f"Result: {label}", 'font': {'size': 28, 'color': color}},
        gauge={
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

def create_contribution_chart(df, title):
    """Creates a Plotly bar chart of word contributions."""
    if df is None or df.empty:
        return go.Figure().update_layout(title="No influential words found in model's vocabulary.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['coefficient'],
        y=df['feature'],
        orientation='h',
        marker_color=df['color'],
        text=df['coefficient'].apply(lambda x: f'{x:.2f}'),
        textposition='auto'
    ))
    fig.update_layout(
        title=title,
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

# --- 7. Sidebar ---
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
        1.  **Local Model:** A `LogisticRegression` classifier gives a fast "Fake", "Real",
            or "Uncertain" prediction based on *writing patterns*.
        2.  **Gemini Analysis:** The `gemini-2.5-flash` model uses **Search Grounding**
            (when available) to verify claims with real-time web search,
            with automatic fallback to standard Gemini analysis.
    """)
    st.divider()

    st.subheader("Key Technologies Used:")
    st.markdown("""
        * **App Framework:** `Streamlit`
        * **Local Model:** `Scikit-learn` & `Pandas`
        * **Charts:** `Plotly`
        * **AI Analysis:** `Google Gemini API` with Search Grounding
        * **Data Sources:** `Hugging Face datasets` & `Kaggle`
    """)
    st.divider()

    st.subheader("⚠️ Important Notes")
    st.markdown("""
        - The **local model** detects writing patterns, not factual accuracy.
        - For recent events, enable **Gemini Deeper Analysis** for web-verified results.
        - Gemini uses Search Grounding when available, with automatic fallback to
          standard analysis.
        - Always verify critical information independently.
    """)
    st.divider()

    st.subheader("Training Data")
    st.markdown(
        "**Real News:** `AG News Dataset` (127,600 articles) from [Hugging Face](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)",
        unsafe_allow_html=True
    )
    st.markdown(
        "**Fake News:** `Fake/Real News` (23,000 articles) from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)",
        unsafe_allow_html=True
    )

# --- 8. Main App Layout ---
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
                "Include Gemini Deeper Analysis (with real-time web search)",
                value=False,
                help="Gemini will search the web to verify claims. Recommended for recent news."
            )
            submitted = st.form_submit_button("Analyze", type="primary")

        if st.button("Clear Results"):
            st.session_state.analysis_results = None
            st.rerun()

# --- Analysis Logic ---
if submitted:

    st.session_state.analysis_results = None

    if model and vectorizer and text_input:
        results = {}
        with st.spinner("Analyzing..."):

            ml_result = get_ml_prediction(text_input)
            results["ml"] = ml_result

            thinking_df, thinking_title = get_model_thinking(
                text_input,
                vectorizer,
                coef_map,
                ml_result.get('label', 'Fake')
            )
            results["thinking_df"] = thinking_df
            results["thinking_title"] = thinking_title

            if "error" in ml_result:
                st.error(f"ML Model Error: {ml_result['error']}")

            if include_gemini:
                if GEMINI_ENABLED:
                    # ✅ Use cached version to save quota on repeated identical inputs
                    gemini_result = cached_gemini_analysis(text_input)
                    results["gemini"] = gemini_result
                else:
                    results["gemini"] = "⚠️ Gemini is disabled (API key not found)."

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
                st.caption("⚠️ Based on writing patterns only — not factual accuracy.")
                if "ml" in results and "error" not in results["ml"]:
                    gauge_fig = create_gauge_chart(results["ml"]['confidence'], results["ml"]['label'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    if results["ml"]['label'] == "Uncertain":
                        st.warning("The model is not confident enough to classify this article. Try Gemini Deeper Analysis for a more reliable verdict.")
                else:
                    st.error(f"ML Model Error: {results['ml'].get('error', 'Unknown')}")

            with tab2:
                st.subheader("Model Thinking")
                if "thinking_df" in results:
                    contribution_fig = create_contribution_chart(
                        results["thinking_df"],
                        results["thinking_title"]
                    )
                    st.plotly_chart(contribution_fig, use_container_width=True)
                else:
                    st.info("Could not generate the model thinking chart.")

            with tab3:
                st.subheader("Gemini Deeper Analysis")
                st.caption("🔍 Uses Search Grounding when available, with automatic fallback.")
                if "gemini" in results:
                    st.markdown(results["gemini"])
                    st.caption("Gemini analysis uses AI reasoning and web search. Always verify critical information independently.")
                else:
                    st.info("Enable 'Gemini Deeper Analysis' and re-run to see results here.")

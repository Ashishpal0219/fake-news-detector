import pandas as pd
import re
import string
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Helper Functions ---

def print_header(text):
    """Prints a formatted header."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def clean_text(text):
    """
    Cleans the input text (lowercase, remove punc, URLs, etc.).
    This is self-contained and requires no 'utils.py'.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower() # Lowercase
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    return text

def plot_confusion_matrix(y_test, y_pred):
    """Creates and saves a confusion matrix plot."""
    print("Plotting confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Fake (0)', 'Real (1)'], 
        yticklabels=['Fake (0)', 'Real (1)']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("✅ 'confusion_matrix.png' saved.")

def plot_feature_importance(vectorizer, model, n_features=15):
    """Creates and saves a plot of the most important features."""
    print("Plotting feature importance...")
    
    try:
        # Get feature names and their coefficients
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefficients = model.coef_[0]
        
        # Create a dataframe
        feature_coefs = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values(by='coefficient', ascending=False)

        # Get top N fake and top N real
        top_real = feature_coefs.head(n_features)
        top_fake = feature_coefs.tail(n_features).sort_values(by='coefficient', ascending=True)
        
        # Combine for plotting
        top_features = pd.concat([top_real, top_fake])
        
        # --- FIXED PLOTTING CODE ---
        # Create a new column for color
        top_features['color'] = top_features['coefficient'].apply(lambda x: 'Real (Positive)' if x > 0 else 'Fake (Negative)')
        
        plt.figure(figsize=(10, 8))
        
        # Use 'hue' for color and 'palette' to map colors
        sns.barplot(
            x='coefficient', 
            y='feature', 
            data=top_features,
            hue='color',  # Use the new column for hue
            palette={'Real (Positive)': 'g', 'Fake (Negative)': 'r'}, # Map colors
            dodge=False,  # Don't split the bars
            legend=False  # Hide the legend
        )
        # --- END FIXED PLOTTING CODE ---
        
        plt.title(f'Top {n_features} Features Predicting Real (Green) vs. Fake (Red)')
        plt.xlabel('Coefficient Value (Impact)')
        plt.ylabel('Feature (Word/Phrase)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("✅ 'feature_importance.png' saved.")
    
    except Exception as e:
        print(f"⚠️ Could not plot feature importance: {e}")

# --- 2. Main Training Function ---

def main():
    print_header("🚀 Starting Fake News Model Training")
    
    # --- Load Data ---
    print("Loading data... (Requires Fake.csv and True.csv)")
    try:
        df_fake = pd.read_csv("Fake.csv")
        df_true = pd.read_csv("True_diverse.csv")
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("Could not find 'Fake.csv' or 'True.csv'.")
        print("Please download them and place them in the same folder.")
        print("---------------\n")
        return
    
    # --- Preprocess Data ---
    print("Preprocessing data...")
    
    # Set labels (Fake=0, Real=1) to match our app's logic
    df_fake["label"] = 0
    df_true["label"] = 1
    
    # Combine datasets
    df = pd.concat([df_fake, df_true])
    
    # Keep only the columns we need
    df = df[['text', 'label']]
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean the text
    print("Cleaning text data (this may take a few minutes)...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows where cleaning resulted in empty text
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"Total valid samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")

    # --- Split Data ---
    print_header("🔀 Splitting Data")
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Keep class balance
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --- Vectorization (TF-IDF) ---
    print_header("🔤 Vectorizing Text (TF-IDF)")
    # We use n-grams (1, 2) to capture 1-word and 2-word phrases
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_df=0.7, 
        ngram_range=(1, 2)
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # --- Model Training ---
    print_header("🤖 Training Logistic Regression Model")
    model = LogisticRegression(
        solver='liblinear', 
        random_state=42, 
        C=1.0,  # Regularization strength
        max_iter=1000
    )
    model.fit(X_train_vec, y_train)
    print("✅ Model training complete.")

    # --- Model Evaluation ---
    print_header("📈 Evaluating Model Performance")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))
    
    # --- Save Plots ---
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(vectorizer, model)

    # --- Save Model Files ---
    print_header("💾 Saving Model and Vectorizer")
    # Save to the root folder to be used by app.py
    joblib.dump(model, 'fake_news_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    print("✅ 'fake_news_model.joblib' saved.")
    print("✅ 'vectorizer.joblib' saved.")
    
    print("\n--- TRAINING SCRIPT FINISHED ---")

# --- Run the script ---
if __name__ == "__main__":
    main()
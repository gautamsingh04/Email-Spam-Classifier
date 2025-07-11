import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing
def transform_text(text_input):
    text_input = text_input.lower()
    words = re.findall(r'\b\w+\b', text_input)
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

# Load model and vectorizer safely
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error("❌ Failed to load model or vectorizer. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the app directory.")
    st.stop()

# Streamlit UI
st.title("📩 Email / SMS Spam Classifier")
st.markdown("🔎 Powered by a **Voting Ensemble Classifier** combining Naive Bayes, SVM, and Extra Trees.")

input_sms = st.text_area("📝 Enter your message:")

if st.button('🔍 Predict'):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a valid message.")
    else:
        # Step 1: Preprocess the input
        transformed_sms = transform_text(input_sms)

        # Step 2: Vectorize + convert to dense if needed for models like SVC
        try:
            vector_input = tfidf.transform([transformed_sms])
            if hasattr(model, 'predict_proba'):
                # Some models like SVC require dense input
                if 'sparse' in str(type(vector_input)):
                    vector_input = vector_input.toarray()
        except Exception as e:
            st.error(f"❌ Vectorization failed: {e}")
            st.stop()

        # Step 3: Predict class and probability
        try:
            prediction = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]
            confidence = probabilities[prediction] * 100

            # Step 4: Display the result
            if prediction == 1:
                st.error(f"🚨 **Spam Detected!**\nConfidence: `{confidence:.2f}%`")
            else:
                st.success(f"✅ **Not Spam (Ham)**\nConfidence: `{confidence:.2f}%`")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

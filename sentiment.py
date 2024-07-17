import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load your model
model = load_model('models/sentiment_model_nlp.keras')

# Streamlit app
st.title("Sentiment Analysis")

review = st.text_input("Enter your review:")

if st.button("Analyze"):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([review])
    sequence = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(sequence, maxlen=50)
    
    prediction = model.predict(padded_seq)
    if prediction > 0.6:
        sentiment = "Positive"
    elif prediction < 0.4:
        sentiment = "Negative"
    else:
        sentiment = "Ambiguous"
    
    st.write(f"Review: {review}")
    st.write(f"Sentiment: {sentiment}")

st.write("Give more context for better analysis.")
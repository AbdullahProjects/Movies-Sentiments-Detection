import streamlit as st
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# load pickle files
# 2 files; 1 is tf-idf vectorizer and 2 is trained model(MultinomialNB())
vectorizer = pickle.load(open("count_vectorizer.pkl", "rb"))
model = pickle.load(open("trained_model.pkl", "rb"))

st.title("Positive/Negative Comments Sentiments")
input_txt = st.text_input("Enter message:") # taking input from user


# 
# Four Steps do now:
# 1. Text Preprocess
# 2. Create Vectorizer (count_vectorizer)
# 3. Predict(positive/negative)
# 4. Show Output


# Predict whether input message is positive or negative:
if st.button("Predict"):
    # step 01
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        y = []
        for word in text:
            if word.isalnum():
                y.append(word)
        
        text = y[:]
        y.clear()
        for word in text:
            if word not in stopwords.words("english") and word not in string.punctuation:
                y.append(word)
                
        text = y[:]
        y.clear()
        for word in text:
            y.append(PorterStemmer().stem(word))
        
        return " ".join(y)

    clean_text = transform_text(input_txt)
    # step 02
    vectors = vectorizer.transform([clean_text])
    # step 03
    prediction = model.predict(vectors)[0]
    # step 04
    if prediction == 0:
        st.header("Positive")
    else:
        st.header("Negative")
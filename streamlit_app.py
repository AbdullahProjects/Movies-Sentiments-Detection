import streamlit as st
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


st.set_page_config(page_title="Octobor Sky Comments Sentiments Detection")
# load pickle files
# 2 files; 1 is tf-idf vectorizer and 2 is trained model(MultinomialNB())
vectorizer = pickle.load(open("count_vectorizer.pkl", "rb"))
model = pickle.load(open("trained_model.pkl", "rb"))

st.title("🎦October Sky Movie 😃Positive/😠Negative Comment Sentiments 🕵️‍♂️Detection")
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


# Create two columns for images
col1, col2 = st.columns(2)

# Display the first image in the first column
with col1:
    st.image("https://m.media-amazon.com/images/M/MV5BNTQzMzUxMDk3N15BMl5BanBnXkFtZTYwNDk2OTU5._V1_UX100_CR0,0,100,100_AL_.jpg", caption="Image 1")

# Display the second image in the second column
with col2:
    st.image("https://m.media-amazon.com/images/M/MV5BNTUyMTc5ODA0M15BMl5BanBnXkFtZTYwODI2ODg5._V1_UX100_CR0,0,100,100_AL_.jpg", caption="Image 2")

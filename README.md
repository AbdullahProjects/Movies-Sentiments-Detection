# NLP Project: Movies Sentiments Detection

<p align="center"><img src="https://plus.unsplash.com/premium_photo-1682310566465-61013a549353?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8cmV2aWV3c3xlbnwwfHwwfHx8MA%3D%3D" width="80%" height="auto"></p>

Welcome to the project. This project is created for detecting whether the movie review is positive or negative. 

  
## Live Demo:

See Project in Streamlit Webapp: https://octobersky-movie-comments-sentiments-detection.streamlit.app/

## About this Project:

### Motivation/Purpose:

Daraz use reviews sentiments detection of sellers and shows in percentage that this seller for example has 85% positive reviews and 15% negative reviews. That's thing motivate me to create such a project that will detect reviews of movies after training on existing reviews to detect whether the new or unseen review is **positive** or **negative**.

### Tools & Libraries:

I use following tools and python libraries to develop this whole project:

- **Python**
- **Streamlit** for deployment
- **Jupyter Notebook**
- **NLTK Library** for applying stemming and tokenization
- **String Library** for removing punctuation marks from text
- **Scikit-Learn Library**
- **Pickle Library** for exporting files from Jupyter Notebook to Python file
  


### Learning Outcomes:

- Handling Computation effeciently
- Text Preprocessing such as tokenization, stemming, removing punctuation marks and special characters etc from text
- Vectorization(Bag of Words, TF-IDF)
- Project handling
- Deployment on Streamlit free of cost


### How to Run on Your Machine:

1. **Clone the Repository:** Download all files and folders from this repository.
2. **Create Virtual Environment:**
   ```bash
   py -3 -m venv virtualEnv
3. **Run this command:**
   ```bash
   pip freeze > requirements.txt
4. **Finally start the streamlit app:** Run the following command on command terminal.
   ```bash
   streamlit run streamlit_app.py

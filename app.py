import csv
import re
import operator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

class ResumeMatcher:
    def __init__(self):
        self.resumes = []  

    def load_resumes_from_csv(self, data):
        for index, row in data.iterrows():
            self.resumes.append(row)

    def preprocess_text(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens

    def calculate_cosine_similarity(self, text1, text2):
        vectorizer = CountVectorizer(tokenizer=self.preprocess_text)
        vectors = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return cosine_sim

    def match_resume_to_requirements(self, resume, requirements_text):
        similarity_score = self.calculate_cosine_similarity(resume['Skills'], requirements_text)
        return similarity_score

    def rank_candidates(self, requirements_text, n):
        ranked_candidates = []
        for resume in self.resumes:
            suitability_score = self.match_resume_to_requirements(resume, requirements_text)
            ranked_candidates.append((resume['Name'], suitability_score))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates[:n]

def main():
    st.title("Resume Matcher")

    # User input for job description
    requirements_text = st.text_input("Enter the job description:")

    # User input for number of top candidates
    n = st.number_input("Enter the number of top candidates you want to see:", min_value=1, value=10, step=1)

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        resume_matcher = ResumeMatcher()
        resume_matcher.load_resumes_from_csv(data)

        if st.button("Match Resumes"):
            # Get top matches
            top_matches = resume_matcher.rank_candidates(requirements_text, n)

            # Display top matches
            st.write("Top {} candidates:".format(n))
            for rank, (candidate_name, score) in enumerate(top_matches, start=1):
                st.write("{}. {}: Suitability Score = {:.2f}".format(rank, candidate_name, score))

if __name__ == "__main__":
    main()

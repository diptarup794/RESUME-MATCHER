import csv
import re
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt')
nltk.download('stopwords')

class ResumeMatcher:
    def __init__(self):
        self.resumes = []  

    def load_resumes_from_csv(self, csv_file_path):
        try:
            with open(csv_file_path, 'r') as resume_file:
                reader = csv.DictReader(resume_file)
                for row in reader:
                    self.resumes.append(row)
        except FileNotFoundError:
            print(f"File '{csv_file_path}' not found. Check the file path.")

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

    def match_resume_to_requirements(self, resume, skills, experience, location):
   
        requirements_text = ' '.join(skills) + ' ' + str(experience) + ' ' + location

       
        similarity_score = self.calculate_cosine_similarity(resume['Skills'], requirements_text)
        return similarity_score

    def rank_candidates(self, skills, experience, location, n):
        ranked_candidates = []
        for resume in self.resumes:
            suitability_score = self.match_resume_to_requirements(resume, skills, experience, location)
            ranked_candidates.append((resume['Name'], suitability_score))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates[:n]

if __name__ == "__main__":
    resume_matcher = ResumeMatcher()

    resume_matcher.load_resumes_from_csv('resumes.csv')

   
    skills = input("Enter the required skills, separated by commas: ").split(',')
    experience = int(input("Enter the optimum years of experience: "))
    location = input("Enter the job location: ")
    n = int(input("Enter the number of top candidates you want to see: "))

    top_matches = resume_matcher.rank_candidates(skills, experience, location, n)

    with open('top_matches.txt', 'w') as f:
        f.write(f"Top {n} candidates:\n")
        for rank, (candidate_name, score) in enumerate(top_matches, start=1):
            f.write(f"{rank}. {candidate_name}: Suitability Score = {score:.2f}\n")


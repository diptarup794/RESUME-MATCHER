# Resume Matcher

## Project Description

The Resume Matcher is a Python application that ranks resumes based on a given job description. The application uses Natural Language Processing (NLP) techniques to calculate the similarity between the job description and the resumes, and then ranks the resumes based on this similarity score.

## Dependencies

- csv
- re
- operator
- sklearn
- nltk
- streamlit



## Usage
To use the Resume Matcher, follow these steps:
Load your resumes into a CSV file. The CSV file should have columns for ‘Name’, ‘Skills’, ‘Experience_Years’, and ‘Location’.
Run the Streamlit app with the command streamlit run app.py.
Enter the job description and the number of top candidates you want to see in the Streamlit web app.
The app will display the top candidates based on the job description.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Try Out:
The first version of the website has been deployed using streamlit . Upload a list of applied candidates in the form of a csv file containing the name of the candidate , years of experience , skills and location and wait and see the model filter out top applicants .

#LINK:https://resume-matcher-hajjw2af2dw3skyd8cinqd.streamlit.app

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Đọc
data_path = os.path.join("data", "USER_DATA_FINAL.csv")
job_path = os.path.join("data", "JOB_DATA_FINAL.csv")

user_df = pd.read_csv(data_path)
job_df = pd.read_csv(job_path)

# Tính toán độ tương đồng
def calculate_similarity(user_df, job_df):
    user_profiles = user_df['Desired Job'] + ' ' + user_df['Industry'] + ' ' + user_df['Workplace Desired']
    job_descriptions = job_df['Job Title'] + ' ' + job_df['Job Description'] + ' ' + job_df['Job Requirements']

    vectorizer = TfidfVectorizer()
    user_tfidf = vectorizer.fit_transform(user_profiles)
    job_tfidf = vectorizer.transform(job_descriptions)

    similarity_matrix = cosine_similarity(user_tfidf, job_tfidf)
    return similarity_matrix

similarity_matrix = calculate_similarity(user_df, job_df)

def recommend_jobs(user_id, user_df, job_df, similarity_matrix, n=5):
    user_index = user_df[user_df['UserID'] == user_id].index[0]
    job_similarities = similarity_matrix[user_index]

    top_indices = job_similarities.argsort()[-n:][::-1]
    recommended_jobs = job_df.iloc[top_indices]

    user_name = user_df.loc[user_df['UserID'] == user_id, 'User Name'].values[0]
    recommendations = []
    for index in top_indices:
        job_info = job_df.iloc[index]
        recommendations.append({
            'job_id': job_info['JobID'],
            'job_title': job_info['Job Title'],
            'company_name': job_info['Name Company'],
            'reason': f"Similarity score: {job_similarities[index]:.2f}"
        })

    return user_name, recommendations
import streamlit as st
import pandas as pd
from recommendation import recommend_jobs, calculate_similarity

user_df = pd.read_csv("data/USER_DATA_FINAL.csv")
job_df = pd.read_csv("data/JOB_DATA_FINAL.csv")

# Tính toán độ tương đồng
similarity_matrix = calculate_similarity(user_df, job_df)

st.title("🔍 Hệ Thống Khuyến Nghị Việc Làm")

#Nhập ID
user_id = st.number_input("Nhập ID người dùng:", min_value=1, step=1)

if st.button("🔎 Gợi ý công việc"):
    user_name, recommended_jobs = recommend_jobs(user_id, user_df, job_df, similarity_matrix, n=5)
    st.write(f"📋 Danh sách công việc được gợi ý cho {user_name}:")
    for job in recommended_jobs:
        st.write(f"- Công việc: {job['job_title']} tại {job['company_name']}")
        st.write(f"  Lý do: {job['reason']}")
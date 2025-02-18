import pandas as pd
from recommendation import calculate_similarity
import pickle
import os

# Đọc dữ liệu
data_path = os.path.join("data", "USER_DATA_FINAL.csv")
job_path = os.path.join("data", "JOB_DATA_FINAL.csv")

user_df = pd.read_csv(data_path)
job_df = pd.read_csv(job_path)

# Tính toán độ tương đồng
similarity_matrix = calculate_similarity(user_df, job_df)

# Lưu similarity_matrix và dữ liệu người dùng, công việc
with open("similarity_matrix.pkl", "wb") as file:
    pickle.dump(similarity_matrix, file)

user_df.to_pickle("user_df.pkl")
job_df.to_pickle("job_df.pkl")

print("🎉 Mô hình đã được lưu!")
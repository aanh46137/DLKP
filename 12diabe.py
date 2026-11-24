"""
Đề 1.2 - Khai phá dữ liệu trên tập diabetes.csv
Yêu cầu:
1. Chuẩn hóa tất cả các cột dữ liệu về đoạn [0, 20].
2. Thực hiện các bước lập trình theo thuật toán Naive Bayes để xác định xác suất
   của mẫu X = [10, 110, 80, 30, 5, 31, 0.627, 45] thuộc vào từng lớp có trong
   cột class. Từ đó quyết định xem X thuộc vào lớp nào?
3. Kiểm nghiệm kết quả câu 2 theo thuật toán sử dụng hàm có sẵn trong Python.
4. Không sử dụng cột class, thực hiện gom cụm dữ liệu với thuật toán K-means (K = 2)
   theo thuật toán đã học. Cho biết số lượng mẫu, tâm của mỗi cụm.
   Xác định xem X thuộc vào cụm nào?

Cách cài đặt các thư viện cần có (chạy trong Terminal của PyCharm hoặc CMD):
    pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("diabetes.csv")

# =====================================================================
# YÊU CẦU 1: Chuẩn hóa tất cả các cột dữ liệu về đoạn [0, 20]
# =====================================================================
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
sym_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# Lưu ánh xạ tên lớp và min/max gốc trước khi biến đổi
class_to_int = {name: idx for idx, name in enumerate(sorted(df["class"].unique()))}
int_to_class = {idx: name for name, idx in class_to_int.items()}
min_vals = df[num_cols].min().to_numpy(dtype=float)
max_vals = df[num_cols].max().to_numpy(dtype=float)

# Chuẩn hóa Min-Max về đoạn [0, 20] và số hóa cột class
df[num_cols] = MinMaxScaler(feature_range=(0, 20)).fit_transform(df[num_cols])
if sym_cols:
    df[sym_cols] = df[sym_cols].apply(lambda col: LabelEncoder().fit_transform(col))

print("=== YÊU CẦU 1: CHUẨN HÓA [0, 20] VÀ SỐ HÓA CỘT CLASS ===")
print("Ánh xạ số hóa cột class:", class_to_int)
print(df.round(4).to_string())

# =====================================================================
# YÊU CẦU 2: Tự lập trình thuật toán Gaussian Naive Bayes cho mẫu X
# =====================================================================
X_sample_raw = np.array([10.0, 110.0, 80.0, 30.0, 5.0, 31.0, 0.627, 45.0], dtype=float)
X_sample_norm = (X_sample_raw - min_vals) / (max_vals - min_vals) * 20.0

X_train = df[num_cols].to_numpy(dtype=float)
y_train = df["class"].to_numpy(dtype=int)

# Các bước thuật toán Gaussian Naive Bayes:
# Bước 1: Tìm các lớp và tính xác suất tiên nghiệm P(c)
# Bước 2: Tính trung bình (mean) và phương sai (var) từng đặc trưng theo mỗi lớp
# Bước 3: Tính mật độ xác suất chuẩn P(x_j | c) và nhân với P(c)
# Bước 4: Chuẩn hóa để được xác suất hậu nghiệm P(c | X)
classes = np.unique(y_train)
epsilon = 1e-9 * np.var(X_train, axis=0).max()
log_probs = []

for c in classes:
    X_c = X_train[y_train == c]
    prior = len(X_c) / len(X_train)
    mean = X_c.mean(axis=0)
    var = X_c.var(axis=0) + epsilon
    log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X_sample_norm - mean) ** 2) / var)
    log_probs.append(np.log(prior) + log_likelihood)

log_probs = np.array(log_probs, dtype=float)
exp_shifted = np.exp(log_probs - np.max(log_probs))
probs_manual = exp_shifted / np.sum(exp_shifted)
pred_manual = int(classes[np.argmax(probs_manual)])

print("\n=== YÊU CẦU 2: PHÂN LỚP MẪU X (TỰ LẬP TRÌNH GAUSSIAN NAIVE BAYES) ===")
print("Mẫu X (gốc)       :", dict(zip(num_cols, X_sample_raw.tolist())))
print("Mẫu X (chuẩn hóa) :", np.round(X_sample_norm, 4).tolist())
print("Xác suất mẫu X thuộc từng lớp:")
for c, p in zip(classes, probs_manual):
    print(f"  - Lớp {c} ({int_to_class[int(c)]:15s}): P = {p:.6e} ({p * 100:.4f}%)")
print(f"=> Kết luận (tự lập trình): Mẫu X thuộc lớp {pred_manual} ({int_to_class[pred_manual]}).")

# =====================================================================
# YÊU CẦU 3: Kiểm nghiệm kết quả câu 2 bằng hàm có sẵn (GaussianNB)
# =====================================================================
X_sample_df = pd.DataFrame([X_sample_norm], columns=num_cols)

model_nb = GaussianNB()
model_nb.fit(df[num_cols], df["class"])

probs_sklearn = model_nb.predict_proba(X_sample_df)[0]
pred_sklearn = int(model_nb.predict(X_sample_df)[0])

print("\n=== YÊU CẦU 3: KIỂM NGHIỆM BẰNG HÀM CÓ SẴN (sklearn GaussianNB) ===")
print("Xác suất mẫu X thuộc từng lớp:")
for c, p in zip(model_nb.classes_, probs_sklearn):
    print(f"  - Lớp {c} ({int_to_class[int(c)]:15s}): P = {p:.6e} ({p * 100:.4f}%)")
print(f"=> Kết luận (GaussianNB)  : Mẫu X thuộc lớp {pred_sklearn} ({int_to_class[pred_sklearn]}).")
print("=> So sánh câu 2 & câu 3  :", "TRÙNG KHỚP" if pred_manual == pred_sklearn and np.allclose(probs_manual, probs_sklearn) else "CHÊNH LỆCH")

# =====================================================================
# YÊU CẦU 4: Tự lập trình thuật toán K-means (K = 2) không dùng cột class
# =====================================================================
K = 2
max_iter = 100
tol = 1e-6

# Bước 1: Khởi tạo K tâm cụm ban đầu (chọn ngẫu nhiên cố định seed từ dữ liệu)
rng = np.random.default_rng(42)
init_idx = rng.choice(len(X_train), size=K, replace=False)
centroids = X_train[init_idx].copy()

# Bước 2 & 3: Lặp gán cụm và cập nhật tâm đến khi hội tụ
for step in range(1, max_iter + 1):
    # Tính khoảng cách Euclidean từ mỗi điểm đến từng tâm cụm
    distances_matrix = np.linalg.norm(X_train[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    cluster_labels = np.argmin(distances_matrix, axis=1)

    # Cập nhật tâm mới là trung bình cộng các điểm thuộc mỗi cụm
    new_centroids = np.array([
        X_train[cluster_labels == k].mean(axis=0) if np.any(cluster_labels == k) else centroids[k]
        for k in range(K)
    ])

    if np.linalg.norm(new_centroids - centroids) < tol:
        centroids = new_centroids
        break
    centroids = new_centroids

print(f"\n=== YÊU CẦU 4: GOM CỤM K-MEANS (K = {K}, HỘI TỤ SAU {step} VÒNG LẶP) ===")
for k in range(K):
    count_k = int(np.sum(cluster_labels == k))
    print(f"  - Cụm {k}: {count_k} mẫu")
    print(f"    Tâm cụm {k}: {np.round(centroids[k], 4).tolist()}")

# Xác định cụm của mẫu X theo khoảng cách Euclidean đến các tâm cụm
dist_to_centroids = np.linalg.norm(centroids - X_sample_norm, axis=1)
pred_cluster = int(np.argmin(dist_to_centroids))

print("Khoảng cách từ mẫu X (chuẩn hóa) đến tâm các cụm:")
for k, d in enumerate(dist_to_centroids):
    print(f"  - Đến tâm Cụm {k}: {d:.4f}")
print(f"=> Kết luận: Mẫu X thuộc Cụm {pred_cluster}.")
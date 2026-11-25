"""
Đề 1.2 - Khai phá dữ liệu trên tập diabetes.csv (KHÔNG số hóa cột class)
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("diabetes.csv")

# =====================================================================
# YÊU CẦU 1: Chuẩn hóa tất cả các cột dữ liệu về đoạn [0, 20]
# =====================================================================
# Chỉ chuẩn hóa các cột số (cột class là chữ nên giữ nguyên, KHÔNG số hóa)
num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "class"]

# Lưu min/max gốc trước khi biến đổi
min_vals = df[num_cols].min().to_numpy(dtype=float)
max_vals = df[num_cols].max().to_numpy(dtype=float)

# Chuẩn hóa Min-Max về đoạn [0, 20]
df[num_cols] = MinMaxScaler(feature_range=(0, 20)).fit_transform(df[num_cols])

print("=== YÊU CẦU 1: CHUẨN HÓA [0, 20] (GIỮ NGUYÊN CỘT CLASS DẠNG CHỮ) ===")
print(df.round(4).to_string())

# =====================================================================
# YÊU CẦU 2: Tự lập trình thuật toán Gaussian Naive Bayes cho mẫu X
# =====================================================================
X_sample_raw = np.array([10.0, 110.0, 80.0, 30.0, 5.0, 31.0, 0.627, 45.0], dtype=float)
X_sample_norm = (X_sample_raw - min_vals) / (max_vals - min_vals) * 20.0

X_train = df[num_cols].to_numpy(dtype=float)
y_train = df["class"].to_numpy()

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
pred_manual = classes[np.argmax(probs_manual)]

print("\n=== YÊU CẦU 2: PHÂN LỚP MẪU X (TỰ LẬP TRÌNH GAUSSIAN NAIVE BAYES) ===")
print("Mẫu X (gốc)       :", dict(zip(num_cols, X_sample_raw.tolist())))
print("Mẫu X (chuẩn hóa) :", np.round(X_sample_norm, 4).tolist())
print("Xác suất mẫu X thuộc từng lớp:")
for c, p in zip(classes, probs_manual):
    print(f"  - Lớp {c:15s}: P = {p:.6e} ({p * 100:.4f}%)")
print(f"=> Kết luận (tự lập trình): Mẫu X thuộc lớp {pred_manual}.")

# =====================================================================
# YÊU CẦU 3: Kiểm nghiệm kết quả câu 2 bằng hàm có sẵn (GaussianNB)
# =====================================================================
X_sample_df = pd.DataFrame([X_sample_norm], columns=num_cols)

model_nb = GaussianNB()
model_nb.fit(df[num_cols], df["class"])

probs_sklearn = model_nb.predict_proba(X_sample_df)[0]
pred_sklearn = model_nb.predict(X_sample_df)[0]

print("\n=== YÊU CẦU 3: KIỂM NGHIỆM BẰNG HÀM CÓ SẴN (sklearn GaussianNB) ===")
print("Xác suất mẫu X thuộc từng lớp:")
for c, p in zip(model_nb.classes_, probs_sklearn):
    print(f"  - Lớp {c:15s}: P = {p:.6e} ({p * 100:.4f}%)")
print(f"=> Kết luận (GaussianNB)  : Mẫu X thuộc lớp {pred_sklearn}.")
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





'''
# Đề 1.2 - diabetes.csv (bản đơn giản)
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("diabetes.csv")
X_goc = [10, 110, 80, 30, 5, 31, 0.627, 45]
cot = [c for c in df.columns if c != "class"]   # 8 cột thuộc tính

# ---------- Câu 1: chuẩn hóa các cột về [0, 20] ----------
# x_moi = (x - min) / (max - min) * 20
mn = df[cot].min()
mx = df[cot].max()
dl = (df[cot] - mn) / (mx - mn) * 20
dl["class"] = df["class"]
print("Câu 1: dữ liệu sau khi chuẩn hóa")
print(dl.round(3))

# chuẩn hóa mẫu X theo cùng min, max
X = (pd.Series(X_goc, index=cot) - mn) / (mx - mn) * 20
print("\nX sau khi chuẩn hóa:", X.round(3).tolist())

# ---------- Câu 2: Naive Bayes ----------
# P(lop | X) ~ P(lop) * tích P(x_j | lop), P(x_j | lop) theo phân phối chuẩn
print("\nCâu 2: Naive Bayes")
ds_lop = sorted(dl["class"].unique())
n = len(dl)
diem = {}
for lop in ds_lop:
    nhom = dl[dl["class"] == lop][cot]
    p_lop = len(nhom) / n                      # xác suất tiên nghiệm
    tb = nhom.mean()                           # trung bình từng cột
    pp = nhom.var(ddof=0)                      # phương sai từng cột
    p = p_lop
    for c in cot:
        p *= np.exp(-(X[c] - tb[c]) ** 2 / (2 * pp[c])) / np.sqrt(2 * np.pi * pp[c])
    diem[lop] = p
    print(f"  P({lop}) = {p_lop:.4f}")

tong = sum(diem.values())
xs = {lop: diem[lop] / tong for lop in ds_lop}  # chia cho tổng để ra xác suất
for lop in ds_lop:
    print(f"  P({lop} | X) = {xs[lop]:.4f}")
ket_qua = max(xs, key=xs.get)
print("=> X thuộc lớp:", ket_qua)

# ---------- Câu 3: kiểm tra bằng hàm có sẵn ----------
print("\nCâu 3: GaussianNB trong sklearn")
mo_hinh = GaussianNB()
mo_hinh.fit(dl[cot], dl["class"])
xs_sk = mo_hinh.predict_proba(pd.DataFrame([X.values], columns=cot))[0]
for lop, p in zip(mo_hinh.classes_, xs_sk):
    print(f"  P({lop} | X) = {p:.4f}")
print("=> X thuộc lớp:", mo_hinh.predict(pd.DataFrame([X.values], columns=cot))[0])

# ---------- Câu 4: K-means, K = 2 (không dùng cột class) ----------
print("\nCâu 4: K-means K = 2")
K = 2
D = dl[cot].values
tam = D[:K].copy()                 # lấy K mẫu đầu làm tâm ban đầu

for lan in range(100):
    # gán mỗi mẫu vào tâm gần nhất (khoảng cách Euclid)
    kc = np.sqrt(((D[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2))
    nhan = kc.argmin(axis=1)
    # tính lại tâm = trung bình các mẫu trong cụm
    tam_moi = np.array([D[nhan == k].mean(axis=0) for k in range(K)])
    if np.allclose(tam_moi, tam):
        break
    tam = tam_moi

print("Số vòng lặp:", lan + 1)
for k in range(K):
    print(f"  Cụm {k}: {(nhan == k).sum()} mẫu, tâm = {tam[k].round(3).tolist()}")

kc_x = np.sqrt(((tam - X.values) ** 2).sum(axis=1))
print("Khoảng cách từ X đến các tâm:", kc_x.round(3).tolist())
print("=> X thuộc cụm:", kc_x.argmin())
'''
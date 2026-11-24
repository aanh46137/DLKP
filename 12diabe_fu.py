"""
Đề 1.2 (mở rộng) - Tổng hợp các thuật toán trên tập diabetes.csv
Mỗi thuật toán có: (A) bản tự cài đặt theo các bước, (B) bản gọi hàm có sẵn để đối chiếu.

Nội dung:
  0. Chuẩn hóa Min-Max về [LOW, HIGH], chia dữ liệu train/test
  1. Cây quyết định (entropy / information gain)
  2. Mạng nơ-ron 1 lớp (Perceptron)
  3. Mạng nơ-ron nhiều lớp (MLP 1 lớp ẩn)
  4. Luật kết hợp (Apriori)
  5. Phân cụm K-means

Cài thư viện:  pip install numpy pandas scikit-learn
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# ----------------------------- THAM SỐ DỄ SỬA -----------------------------
CSV_FILE = "diabetes.csv"
LOW, HIGH = 0, 20                      # đoạn chuẩn hóa
TEST_SIZE = 0.2                        # tỉ lệ tập test
SEED = 42
X_RAW = [10, 110, 80, 30, 5, 31, 0.627, 45]   # mẫu X của đề
TREE_DEPTH = 4
MIN_SUPPORT, MIN_CONF = 0.3, 0.8       # Apriori
N_BINS = 3                             # số khoảng rời rạc hóa cho Apriori
K = 2                                  # số cụm K-means
# --------------------------------------------------------------------------

np.random.seed(SEED)

# =====================================================================
# 0. CHUẨN HÓA + CHIA TRAIN/TEST
# =====================================================================
df = pd.read_csv(CSV_FILE)
feat_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "class"]
class_names = sorted(df["class"].unique())
class_to_int = {n: i for i, n in enumerate(class_names)}
int_to_class = {i: n for n, i in class_to_int.items()}

raw_min = df[feat_cols].min().to_numpy(float)
raw_max = df[feat_cols].max().to_numpy(float)
assert len(feat_cols) == len(X_RAW), f"X có {len(X_RAW)} giá trị nhưng dữ liệu có {len(feat_cols)} thuộc tính"

scaler = MinMaxScaler(feature_range=(LOW, HIGH))
Xn = scaler.fit_transform(df[feat_cols])
y = df["class"].map(class_to_int).to_numpy(int)
X_sample = scaler.transform(pd.DataFrame([X_RAW], columns=feat_cols))[0]

strat = y if np.bincount(y).min() >= 2 else None
X_tr, X_te, y_tr, y_te = train_test_split(Xn, y, test_size=TEST_SIZE, random_state=SEED, stratify=strat)

print("=== 0. CHUẨN HÓA & CHIA TRAIN/TEST ===")
print(f"Chuẩn hóa về [{LOW}, {HIGH}]; ánh xạ lớp: {class_to_int}")
print(f"Tổng {len(y)} mẫu -> train {len(y_tr)}, test {len(y_te)}")
print("X chuẩn hóa:", np.round(X_sample, 4).tolist())


def acc(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def report(name, pred_te, pred_x):
    print(f"  {name:28s} acc(test) = {acc(y_te, pred_te):.4f} | X -> {int_to_class[int(pred_x)]}")


# =====================================================================
# 1. CÂY QUYẾT ĐỊNH (ID3 với tách nhị phân trên thuộc tính số)
# =====================================================================
def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = np.bincount(labels) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def best_split(X, labels):
    """Bước: duyệt mọi thuộc tính & ngưỡng, chọn Information Gain lớn nhất."""
    base, n = entropy(labels), len(labels)
    best = (0.0, None, None)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for t in (vals[:-1] + vals[1:]) / 2:
            left = labels[X[:, j] <= t]
            right = labels[X[:, j] > t]
            gain = base - len(left) / n * entropy(left) - len(right) / n * entropy(right)
            if gain > best[0] + 1e-12:
                best = (gain, j, t)
    return best


def build_tree(X, labels, depth, max_depth, min_samples=2):
    if depth >= max_depth or len(np.unique(labels)) == 1 or len(labels) < min_samples:
        return {"leaf": int(np.bincount(labels).argmax())}
    gain, j, t = best_split(X, labels)
    if j is None:
        return {"leaf": int(np.bincount(labels).argmax())}
    m = X[:, j] <= t
    return {"feat": j, "thr": t,
            "left": build_tree(X[m], labels[m], depth + 1, max_depth, min_samples),
            "right": build_tree(X[~m], labels[~m], depth + 1, max_depth, min_samples)}


def tree_predict_one(node, x):
    while "leaf" not in node:
        node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return node["leaf"]


def print_tree(node, depth=0):
    pad = "    " * depth
    if "leaf" in node:
        print(f"{pad}-> {int_to_class[node['leaf']]}")
        return
    print(f"{pad}if {feat_cols[node['feat']]} <= {node['thr']:.3f}:")
    print_tree(node["left"], depth + 1)
    print(f"{pad}else:")
    print_tree(node["right"], depth + 1)


print(f"\n=== 1. CÂY QUYẾT ĐỊNH (độ sâu tối đa {TREE_DEPTH}) ===")
tree = build_tree(X_tr, y_tr, 0, TREE_DEPTH)
print("Cây tự cài:")
print_tree(tree)
dt_manual_te = [tree_predict_one(tree, r) for r in X_te]
dt_manual_x = tree_predict_one(tree, X_sample)
dt_lib = DecisionTreeClassifier(criterion="entropy", max_depth=TREE_DEPTH, random_state=SEED).fit(X_tr, y_tr)
report("Tự cài đặt", dt_manual_te, dt_manual_x)
report("sklearn DecisionTree", dt_lib.predict(X_te), dt_lib.predict([X_sample])[0])

# =====================================================================
# 2. MẠNG NƠ-RON 1 LỚP (PERCEPTRON)  - nhãn nhị phân
# =====================================================================
# Đưa đầu vào về [0,1] (chia cho HIGH) để huấn luyện ổn định
S = float(HIGH - LOW) or 1.0
Xs_tr, Xs_te, xs = (X_tr - LOW) / S, (X_te - LOW) / S, (X_sample - LOW) / S


def train_perceptron(X, labels, lr=0.1, epochs=200):
    """Bước: w,b=0 -> với mỗi mẫu: out = step(w.x+b); w += lr*(t-out)*x."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        errors = 0
        for xi, ti in zip(X, labels):
            out = 1 if xi @ w + b > 0 else 0
            if out != ti:
                w += lr * (ti - out) * xi
                b += lr * (ti - out)
                errors += 1
        if errors == 0:
            break
    return w, b


print("\n=== 2. MẠNG NƠ-RON 1 LỚP (PERCEPTRON) ===")
if len(class_names) == 2:
    w, b = train_perceptron(Xs_tr, y_tr)
    pp = lambda A: (A @ w + b > 0).astype(int)
    print("Trọng số:", np.round(w, 3).tolist(), "| bias:", round(b, 3))
    report("Tự cài đặt", pp(Xs_te), pp(xs.reshape(1, -1))[0])
    pl = Perceptron(max_iter=1000, tol=None, random_state=SEED).fit(Xs_tr, y_tr)
    report("sklearn Perceptron", pl.predict(Xs_te), pl.predict([xs])[0])
else:
    print("  (Perceptron tự cài chỉ hỗ trợ 2 lớp; bỏ qua vì dữ liệu có >2 lớp)")


# =====================================================================
# 3. MẠNG NƠ-RON NHIỀU LỚP (MLP, 1 lớp ẩn, sigmoid, lan truyền ngược)
# =====================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def train_mlp(X, labels, hidden=8, lr=0.5, epochs=3000):
    """Bước: lan truyền tiến -> tính sai số -> lan truyền ngược -> cập nhật W1,b1,W2,b2."""
    rng = np.random.default_rng(SEED)
    n_cls = len(class_names)
    T = np.eye(n_cls)[labels]                      # one-hot
    W1 = rng.normal(0, 0.5, (X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (hidden, n_cls));      b2 = np.zeros(n_cls)
    for _ in range(epochs):
        H = sigmoid(X @ W1 + b1)
        O = sigmoid(H @ W2 + b2)
        dO = (O - T) * O * (1 - O)
        dH = (dO @ W2.T) * H * (1 - H)
        W2 -= lr * H.T @ dO / len(X); b2 -= lr * dO.mean(0)
        W1 -= lr * X.T @ dH / len(X); b1 -= lr * dH.mean(0)
    return W1, b1, W2, b2


def mlp_predict(params, A):
    W1, b1, W2, b2 = params
    return np.argmax(sigmoid(sigmoid(A @ W1 + b1) @ W2 + b2), axis=1)


print("\n=== 3. MẠNG NƠ-RON NHIỀU LỚP (MLP, 1 lớp ẩn 8 nơ-ron) ===")
mlp_params = train_mlp(Xs_tr, y_tr)
report("Tự cài đặt", mlp_predict(mlp_params, Xs_te), mlp_predict(mlp_params, xs.reshape(1, -1))[0])
mlp_lib = MLPClassifier(hidden_layer_sizes=(8,), activation="logistic", max_iter=3000, random_state=SEED).fit(Xs_tr, y_tr)
report("sklearn MLPClassifier", mlp_lib.predict(Xs_te), mlp_lib.predict([xs])[0])


# =====================================================================
# 4. LUẬT KẾT HỢP (APRIORI)  - rời rạc hóa các cột số thành N_BINS khoảng
# =====================================================================
print(f"\n=== 4. LUẬT KẾT HỢP (Apriori, minsup={MIN_SUPPORT}, minconf={MIN_CONF}) ===")
labels_bin = ["thap", "vua", "cao"] if N_BINS == 3 else [f"k{i + 1}" for i in range(N_BINS)]
disc = pd.DataFrame({c: pd.cut(Xn[:, i], bins=N_BINS, labels=labels_bin).astype(str)
                     for i, c in enumerate(feat_cols)})
disc["class"] = df["class"].values
transactions = [frozenset(f"{c}={v}" for c, v in row.items()) for _, row in disc.iterrows()]
N = len(transactions)


def support(itemset):
    return sum(itemset <= t for t in transactions) / N


# Bước 1: tìm tập mục thường xuyên theo từng mức k
items = sorted({i for t in transactions for i in t})
L = {frozenset([i]): support(frozenset([i])) for i in items}
L = {s: v for s, v in L.items() if v >= MIN_SUPPORT}
frequent = dict(L)
k = 2
while L:
    prev = list(L)
    cands = {a | b for a, b in itertools.combinations(prev, 2) if len(a | b) == k}
    cands = {c for c in cands if all(frozenset(sub) in L for sub in itertools.combinations(c, k - 1))}
    L = {c: support(c) for c in cands}
    L = {c: v for c, v in L.items() if v >= MIN_SUPPORT}
    frequent.update(L)
    k += 1
print(f"Số tập mục thường xuyên: {len(frequent)}")

# Bước 2: sinh luật X -> Y có độ tin cậy >= MIN_CONF
rules = []
for itemset, sup in frequent.items():
    if len(itemset) < 2:
        continue
    for r in range(1, len(itemset)):
        for ante in map(frozenset, itertools.combinations(itemset, r)):
            conf = sup / frequent[ante]
            if conf >= MIN_CONF:
                cons = itemset - ante
                lift = conf / frequent[cons]
                rules.append((ante, cons, sup, conf, lift))
rules.sort(key=lambda r: (-r[3], -r[2]))
print(f"Số luật kết hợp: {len(rules)}. Top 10 theo độ tin cậy:")
for a, c, s, cf, lf in rules[:10]:
    print(f"  {{{', '.join(sorted(a))}}} => {{{', '.join(sorted(c))}}}  sup={s:.3f} conf={cf:.3f} lift={lf:.3f}")
cls_rules = [r for r in rules if any(i.startswith("class=") for i in r[1])]
print(f"Trong đó {len(cls_rules)} luật có vế phải là class. Top 5:")
for a, c, s, cf, lf in cls_rules[:5]:
    print(f"  {{{', '.join(sorted(a))}}} => {{{', '.join(sorted(c))}}}  sup={s:.3f} conf={cf:.3f}")

# =====================================================================
# 5. PHÂN CỤM K-MEANS (không dùng cột class)
# =====================================================================
print(f"\n=== 5. K-MEANS (K = {K}) ===")
rng = np.random.default_rng(SEED)
init = Xn[rng.choice(len(Xn), size=K, replace=False)].copy()
cent = init.copy()
for it in range(1, 301):
    lab = np.argmin(np.linalg.norm(Xn[:, None, :] - cent[None], axis=2), axis=1)
    new = np.array([Xn[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(K)])
    done = np.linalg.norm(new - cent) < 1e-6
    cent = new
    if done:
        break
print(f"Tự cài đặt: hội tụ sau {it} vòng")
for j in range(K):
    print(f"  Cụm {j}: {int((lab == j).sum())} mẫu | tâm = {np.round(cent[j], 3).tolist()}")
cx = int(np.argmin(np.linalg.norm(cent - X_sample, axis=1)))
print(f"  => X thuộc cụm {cx}")

km = KMeans(n_clusters=K, init=init, n_init=1, random_state=SEED).fit(Xn)
same = np.allclose(np.sort(km.cluster_centers_, axis=0), np.sort(cent, axis=0), atol=1e-3)
print(f"sklearn KMeans (cùng tâm khởi tạo): X thuộc cụm {int(km.predict([X_sample])[0])}; "
      f"số mẫu/cụm = {np.bincount(km.labels_).tolist()}; tâm {'TRÙNG KHỚP' if same else 'CHÊNH LỆCH'} với bản tự cài")

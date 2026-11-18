import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Membaca dataset
df = pd.read_csv('Updated Quality of Life Data.csv', sep=';')

# 2. Menentukan fitur (X) dan label kelas target (y)
TARGET_COL = 'gender'
X = df.drop(columns=['id', TARGET_COL])
y = df[TARGET_COL]

# 3. Membagi dataset: 80% data latih dan 20% data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Total dataset : {len(df)} baris")
print(f"Data latih    : {len(X_train)} baris (80%)")
print(f"Data uji      : {len(X_test)} baris (20%)")

# 4. Preprocessing: Standardisasi fitur numerik & One-Hot Encoding fitur kategorikal
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 5. Pelatihan dan Pengujian SVM untuk C = 0.01, 0.1, 1, 10
C_values = [0.01, 0.1, 1, 10]

for c in C_values:
    print("\n" + "=" * 75)
    print(f"HASIL KLASIFIKASI SVM DENGAN PARAMETER C = {c}")
    print("=" * 75)
    
    # Membuat dan melatih model SVC (kernel RBF)
    model = SVC(C=c, kernel='rbf', random_state=42)
    model.fit(X_train_prep, y_train)
    
    # Prediksi pada 20% data uji
    y_pred = model.predict(X_test_prep)
    akurasi = accuracy_score(y_test, y_pred)
    print(f"Akurasi Data Uji : {akurasi:.4f} ({akurasi*100:.2f}%)")
    
    # Menyusun tabel hasil klasifikasi data uji
    hasil_df = df.loc[X_test.index, ['id'] + num_cols + cat_cols].copy()
    hasil_df['Aktual'] = y_test.values
    hasil_df['Prediksi_SVM'] = y_pred
    
    # Poin 2: Hasil klasifikasi 5 teratas dan 5 terbawah
    print("\n[A] Hasil Klasifikasi 5 Teratas (Data Uji):")
    print(hasil_df.head(5).to_string())
    
    print("\n[B] Hasil Klasifikasi 5 Terbawah (Data Uji):")
    print(hasil_df.tail(5).to_string())
    
    # Poin 3: Jumlah support vector setiap C
    total_sv = len(model.support_)
    sv_per_kelas = dict(zip(model.classes_, model.n_support_))
    print(f"\n[C] Jumlah Support Vector (C = {c}): {total_sv} dari {len(X_train)} data latih")
    print(f"    Rincian per kelas: {sv_per_kelas}")
    
    # Poin 4: Data yang dianggap support vector (dari data latih, alpha_i > 0)
    sv_indices_asli = X_train.index[model.support_]
    sv_df = df.loc[sv_indices_asli].copy()
    sv_df['alpha_y'] = model.dual_coef_[0]  # Nilai alpha_i * y_i (tidak nol)
    
    print(f"\n[D] Data yang Dianggap Support Vector (5 teratas & 5 terbawah dari {total_sv} data):")
    print(sv_df.head(5).to_string())
    print("...")
    print(sv_df.tail(5).to_string())
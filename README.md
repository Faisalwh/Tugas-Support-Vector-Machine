# Tugas-Support-Vector-Machine

1. Jumlah & Pembagian Dataset
  - Total Dataset (Updated Quality of Life Data.csv): Terdiri dari 10.000 baris data dan 8 kolom (id, gender, occupation_type,
    avg_work_hours_per_day, avg_rest_hours_per_day, avg_sleep_hours_per_day, avg_exercise_hours_per_day, age_at_death).
  - Pembagian Data (80% : 20%):
     ~ Data Latih (Train — 80%): 8.000 baris digunakan untuk melatih model SVM dan mencari titik support vector.
     ~ Data Uji (Test — 20%): 2.000 baris digunakan untuk menguji performa klasifikasi model.
    
2. Parameter C yang Digunakan
   Digunakan 4 variasi nilai parameter regularisasi C:
   - C = 0,01
   - C = 0,1
   - C = 1
   - C = 10

3. Struktur File & Kodingan Python
   Pastikan kedua file berikut berada di dalam satu folder yang sama:
   - Updated Quality of Life Data.csv (file dataset)
   - Kodingan_Support_Vector_Machine.py (file program Python)

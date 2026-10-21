import pandas as pd
import time
from sklearn.cluster import KMeans
from joblib import parallel_config

# 1. Încărcăm datele
df = pd.read_csv('date_proiect.csv')
X = df.values

# 2. Varianta Paralelă forțată cu joblib
print(f" Pornim clustering-ul paralel pe CPU...")

# n_init='auto' și fără n_jobs aici (pentru a evita eroarea)
model = KMeans(n_clusters=4, n_init='auto', random_state=42)

start = time.time()
# Folosim context managerul pentru a forța utilizarea backend-ului loky pe toate nucleele
with parallel_config(backend='loky', n_jobs=-1):
    model.fit(X)
end = time.time()

print(f"\n--- REZULTATE CPU PARALEL ---")
print(f"Timp execuție: {end - start:.4f} secunde")
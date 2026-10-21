import pandas as pd
import time
from sklearn.cluster import KMeans

# Încărcăm datele de pe disk
df = pd.read_csv('date_proiect.csv')
X = df.values

# Algoritmul standard (Secvențial)
model = KMeans(n_clusters=4, n_init='auto', random_state=42)

start = time.time()
model.fit(X)
end = time.time()

print(f"⏱️ Timp CPU Secvențial: {end - start:.4f} secunde")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

X, y = make_blobs(n_samples=10000, centers=4, n_features=2, random_state=42)

# 2. Punem datele într-un tabel (DataFrame) ca să fie ușor de gestionat
df = pd.DataFrame(X, columns=['Coordonata_X', 'Coordonata_Y'])

print("Primele 5 instanțe din setul de date:")
print(df.head())

# 3. Vizualizăm "norul" de puncte
plt.figure(figsize=(10, 7))
plt.scatter(df['Coordonata_X'], df['Coordonata_Y'], s=1, c='blue', alpha=0.5)
plt.title("Norul de 10.000 de instanțe (Datele noastre)")
plt.xlabel("Axa X")
plt.ylabel("Axa Y")
import time
from sklearn.cluster import KMeans

# 1. Inițializăm algoritmul K-Means (pentru 4 grupuri, cum am generat datele)
# n_init='auto' îi spune să aleagă singur punctele de start ale "șefilor"
kmeans_cpu = KMeans(n_clusters=4, n_init='auto', random_state=42)

# 2. Măsurăm timpul de execuție pe CPU
start_time = time.time()
kmeans_cpu.fit(X) # Aici are loc calculul matematic greu
end_time = time.time()

cpu_duration = end_time - start_time
print(f"\n--- REZULTATE CPU ---")
print(f"Timp execuție pentru 10.000 instanțe: {cpu_duration:.4f} secunde")

# 3. Vizualizăm rezultatul (culorile reprezintă grupurile găsite de algoritm)
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_cpu.labels_, s=1, cmap='viridis')
plt.title(f"K-Means pe CPU (Timp: {cpu_duration:.4f}s)")
plt.show()
# 1. INJECȚIA DE PUTERE (Trebuie să fie primele linii!)
from sklearnex import patch_sklearn
patch_sklearn()

import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 2. Generăm datele (100.000 instanțe)
X, _ = make_blobs(n_samples=100000, centers=4, n_features=2, random_state=42)

# 3. K-Means (Codul rămâne identic, dar va rula pe accelerator)
kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)

start_time = time.time()
kmeans.fit(X)
end_time = time.time()

print(f"\n--- REZULTATE ACCELERATE (GPU/Parallel CPU) ---")
print(f"Timp execuție pentru 100.000 instanțe: {end_time - start_time:.4f} secunde")
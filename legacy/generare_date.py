import pandas as pd
from sklearn.datasets import make_blobs

# Generăm 100.000 de instanțe
X, _ = make_blobs(n_samples=100000, centers=4, n_features=2, random_state=42)

# Salvăm într-un fișier CSV ca să le avem "la rece"
df = pd.DataFrame(X, columns=['X', 'Y'])
df.to_csv('date_proiect.csv', index=False)

print("✅ Fișierul 'date_proiect.csv' a fost generat cu succes!")
import pandas as pd
import matplotlib.pyplot as plt

# Înlocuit cu fișierul corect de 100k
df = pd.read_csv('date_100k.csv')

plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y'], s=1, alpha=0.5, color='#1f77b4')
plt.title("Distribuția celor 100.000 de instanțe sintetice")
plt.xlabel("Axa X")
plt.ylabel("Axa Y")
plt.grid(True, linestyle='--', alpha=0.5)

# Salvează imaginea automat în folder
plt.savefig('grafic_date.png', dpi=300)
print("✅ Graficul a fost salvat cu succes ca 'grafic_date.png'!")
plt.show()
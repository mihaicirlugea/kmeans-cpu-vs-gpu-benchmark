# legacy/ – first-stage prototypes

Early scripts from the first phase of the project (10k–100k points). Kept for history only.

- `generare_date.py` creates `date_proiect.csv` (100k points); the other scripts here read that file.
- `test_gpu.py` uses `sklearnex` (Intel Extension for scikit-learn). **This does not run on an NVIDIA GPU**, it is an optimised CPU path, so the "GPU" label in that script is misleading. The real GPU benchmark is `../benchmark_gpu.ipynb` (cuML).
- `test_cpu_paralel.py` wraps `KMeans.fit` in `joblib.parallel_config`, which has no effect on scikit-learn's KMeans (see the main README, "Known issues").
- `test.py` is an unstructured experiment (imports in the middle of the file).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.preprocessing import normalize

def distancia_coseno_vecs(M, v):
    """
    M: matriz (n_samples, n_features)
    v: vector (n_features,)
    Devuelve array de distancias coseno 1 - cos_sim entre v y cada fila de M.
    """
    # asumimos que M y v no son nulos
    v_norm = np.linalg.norm(v)
    M_norms = np.linalg.norm(M, axis=1)
    # prevención de división por 0
    denom = M_norms * (v_norm if v_norm != 0 else 1e-12)
    # producto punto
    dots = M.dot(v)
    cos_sim = dots / (denom + 1e-12)
    # clip para estabilidad numérica
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim

def distancia_euclidiana_vecs(M, v):
    """
    M: matriz (n_samples, n_features)
    v: vector (n_features,)
    Devuelve array de distancias euclidianas entre v y cada fila de M.
    """
    M = np.asarray(M, dtype=float)
    v = np.asarray(v, dtype=float)
    # Diferencia y norma
    diffs = M - v
    distancias = np.linalg.norm(diffs, axis=1)
    return distancias

def region_query(datos, punto_idx, eps,metric):
    """
    Encuentra todos los vecinos dentro de eps (distancia coseno).
    datos: np.ndarray 2D
    """
    # asegurar ndarray 2D float
    datos = np.asarray(datos, dtype=float)
    punto = datos[punto_idx]
    # vectorizado
    if metric == "coseno":
        distancias = distancia_coseno_vecs(datos, punto)
    elif metric == "euclidiana":
        distancias = distancia_euclidiana_vecs(datos, punto)
    vecinos = np.where(distancias <= eps)[0].tolist()
    return vecinos

def expand_cluster(datos, labels, punto_idx, vecinos, cluster_id, eps, min_pts,metric):
    """Expande un cluster agregando puntos densos"""
    labels[punto_idx] = cluster_id
    i = 0
    while i < len(vecinos):
        vecino_idx = vecinos[i]
        if labels[vecino_idx] == -1:
            labels[vecino_idx] = cluster_id  # De ruido a cluster
        elif labels[vecino_idx] == 0:
            labels[vecino_idx] = cluster_id
            nuevos_vecinos = region_query(datos, vecino_idx, eps,metric=metric)
            if len(nuevos_vecinos) >= min_pts:
                vecinos.extend(nuevos_vecinos)
        i += 1

def dbscan(datos, eps, min_pts,metric):
    """
    datos: lista de puntos o np.ndarray (n_samples, n_features)
    eps: radio de vecindad
    min_pts: mínimo de puntos para formar cluster
    """
    datos = np.asarray(datos, dtype=float)
    n = len(datos)
    labels = [0] * n  # 0 = no visitado, -1 = ruido, >0 = cluster_id
    cluster_id = 0

    for i in range(n):
        if labels[i] != 0:
            continue
        vecinos = region_query(datos, i, eps,metric=metric)
        if len(vecinos) < min_pts:
            labels[i] = -1  # ruido
        else:
            cluster_id += 1
            expand_cluster(datos, labels, i, vecinos, cluster_id, eps, min_pts,metric=metric)

    return labels


### DISTANCIA EUCLIDIANA - PRUEBAS ###

# 1. Dataset sintético
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

# 2. Tu DBSCAN
labels_mio = dbscan(X, eps=0.5, min_pts=5, metric="euclidiana")

# 3. DBSCAN oficial de sklearn
modelo_sklearn = SkDBSCAN(eps=0.5, min_samples=5, metric="euclidean")
labels_sklearn = modelo_sklearn.fit_predict(X)

# 4. Métricas de comparación
ari = adjusted_rand_score(labels_sklearn, labels_mio)
hom = homogeneity_score(labels_sklearn, labels_mio)
print(f"Adjusted Rand Index (vs sklearn): {ari:.3f}")
print(f"Homogeneidad: {hom:.3f}")

# 5. Visualización
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(X[:, 0], X[:, 1], c=labels_mio, cmap="tab10")
axs[0].set_title("Clusters - TU DBSCAN")

axs[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap="tab10")
axs[1].set_title("Clusters - sklearn DBSCAN")

plt.show()


### DISTANCIA COSENO - PRUEBAS ###

# 1. Dataset sintético + normalización (para coseno)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
X = normalize(X)  # normaliza cada vector a norma 1 → más adecuado para coseno

# 2. Tu DBSCAN con métrica coseno
labels_mio = dbscan(X, eps=0.1, min_pts=5, metric="coseno")

# 3. DBSCAN sklearn con métrica coseno
modelo_sklearn = SkDBSCAN(eps=0.1, min_samples=5, metric="cosine")
labels_sklearn = modelo_sklearn.fit_predict(X)

# 4. Métricas de comparación
ari = adjusted_rand_score(labels_sklearn, labels_mio)
hom = homogeneity_score(labels_sklearn, labels_mio)
print(f"Adjusted Rand Index (vs sklearn): {ari:.3f}")
print(f"Homogeneidad: {hom:.3f}")

# 5. Visualización
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(X[:, 0], X[:, 1], c=labels_mio, cmap="tab10")
axs[0].set_title("TU DBSCAN (coseno)")

axs[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap="tab10")
axs[1].set_title("sklearn DBSCAN (coseno)")

plt.show()
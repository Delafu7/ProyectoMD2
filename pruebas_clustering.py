from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import numpy as np
import math

def distancia(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def region_query(datos, punto_idx, eps):
    """Encuentra todos los vecinos dentro de eps"""
    punto = datos[punto_idx]
    # calcular distancias a todos los puntos de manera vectorizada
    distancias = np.linalg.norm(datos - punto, axis=1)
    # devolver índices de vecinos dentro de eps
    vecinos = np.where(distancias <= eps)[0].tolist()
    return vecinos

def expand_cluster(datos, labels, punto_idx, vecinos, cluster_id, eps, min_pts):
    """Expande un cluster agregando puntos densos"""
    labels[punto_idx] = cluster_id
    i = 0
    while i < len(vecinos):
        vecino_idx = vecinos[i]
        if labels[vecino_idx] == -1:
            labels[vecino_idx] = cluster_id  # De ruido a cluster
        elif labels[vecino_idx] == 0:
            labels[vecino_idx] = cluster_id
            nuevos_vecinos = region_query(datos, vecino_idx, eps)
            if len(nuevos_vecinos) >= min_pts:
                vecinos.extend(nuevos_vecinos)
        i += 1

def dbscan(datos, eps, min_pts):
    """
    datos: lista de puntos [[x1, y1], [x2, y2], ...]
    eps: radio de vecindad
    min_pts: mínimo de puntos para formar cluster
    """
    datos = np.array(datos)
    cluster_id = 0
    labels = [0] * len(datos)  # 0 = no visitado, -1 = ruido, >0 = cluster_id

    for i in range(len(datos)):
        print(i)
        if labels[i] != 0:
            continue
        vecinos = region_query(datos, i, eps)
        if len(vecinos) < min_pts:
            labels[i] = -1  # Ruido
        else:
            cluster_id += 1
            expand_cluster(datos, labels, i, vecinos, cluster_id, eps, min_pts)

    return labels

def test_dbscan_implementacion(eps=0.5, min_pts=5, n_samples=300):
    """
    Compara la implementación manual de DBSCAN con la de scikit-learn.
    Genera un dataset sintético simple y compara las etiquetas.
    """

    # 1️⃣ Generar datos sintéticos (2D para visualización fácil)
    X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5, random_state=42)

    # 2️⃣ Aplicar DBSCAN manual
    etiquetas_manual = dbscan(X.tolist(), eps=eps, min_pts=min_pts)

    # 3️⃣ Aplicar DBSCAN de scikit-learn
    db = DBSCAN(eps=eps, min_samples=min_pts)
    etiquetas_sklearn = db.fit_predict(X)

    # 4️⃣ Comparar número de clusters
    clusters_manual = len(set([e for e in etiquetas_manual if e != -1]))
    clusters_sklearn = len(set([e for e in etiquetas_sklearn if e != -1]))
    print(f"Clusters (Manual): {clusters_manual}")
    print(f"Clusters (sklearn): {clusters_sklearn}")

    # 5️⃣ Comparar etiquetas punto por punto
    # ⚠️ OJO: los números de cluster pueden no coincidir exactamente (1 vs 2 vs 3),
    # pero la estructura debe ser similar. Por eso usamos adjusted_rand_score.
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(etiquetas_sklearn, etiquetas_manual)
    print(f"ARI (Adjusted Rand Index) entre ambas implementaciones: {ari:.3f}")

    # 6️⃣ Visualización opcional
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(X[:, 0], X[:, 1], c=etiquetas_manual, cmap="tab10", s=20)
    axs[0].set_title("DBSCAN Manual")
    axs[1].scatter(X[:, 0], X[:, 1], c=etiquetas_sklearn, cmap="tab10", s=20)
    axs[1].set_title("DBSCAN sklearn")
    plt.show()

# Llamar a la función de prueba
if __name__ == "__main__":
    test_dbscan_implementacion(eps=0.5, min_pts=5, n_samples=300)

import math

def distancia(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def region_query(datos, punto_idx, eps):
    """Encuentra todos los vecinos dentro de eps"""
    vecinos = []
    for i in range(len(datos)):
        if distancia(datos[punto_idx], datos[i]) <= eps:
            vecinos.append(i)
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
    cluster_id = 0
    labels = [0] * len(datos)  # 0 = no visitado, -1 = ruido, >0 = cluster_id

    for i in range(len(datos)):
        if labels[i] != 0:
            continue
        vecinos = region_query(datos, i, eps)
        if len(vecinos) < min_pts:
            labels[i] = -1  # Ruido
        else:
            cluster_id += 1
            expand_cluster(datos, labels, i, vecinos, cluster_id, eps, min_pts)

    return labels

# Datos de ejemplo (2D)
datos = [
    [1, 2], [2, 2], [2, 3],     # cluster 1
    [8, 7], [8, 8], [25, 80]    # cluster 2 + ruido
]

eps = 2
min_pts = 2

labels = dbscan(datos, eps, min_pts)
print("Etiquetas:", labels)

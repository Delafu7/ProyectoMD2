import os
import json
import re

import joblib
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from transformers import AutoModel, AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import normalize
import matplotlib.cm as cm
# Preprocesamiento de texto
#-------------------------------------------------------------------------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
stemmer = SpanishStemmer()

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

# Metodos/Funciones

def agrupar_train_test(ruta_base="MentalRiskEs2023", salida_train="train.json", salida_test="test.json"):
    mensajes_train = []
    mensajes_test = []

    # Recorremos las categorías
    for categoria in os.listdir(ruta_base):
        ruta_categoria = os.path.join(ruta_base, categoria)
        if not os.path.isdir(ruta_categoria):
            continue

        # --- TRAIN ---
        ruta_train = os.path.join(ruta_categoria, "Train")
        if os.path.exists(ruta_train):
            print(f"Procesando Train en {ruta_train}")
            for archivo in os.listdir(ruta_train):
                if archivo.endswith(".json"):
                    ruta_archivo = os.path.join(ruta_train, archivo)
                    with open(ruta_archivo, "r", encoding="utf-8") as f:
                        datos_json = json.load(f)
                        for elem in datos_json:
                            if "message" in elem:
                                mensajes_train.append(elem["message"])

        # --- TEST ---
        ruta_test = os.path.join(ruta_categoria, "Test")
        if os.path.exists(ruta_test):
            print(f"Procesando Test en {ruta_test}")
            for archivo in os.listdir(ruta_test):
                if archivo.endswith(".json"):
                    ruta_archivo = os.path.join(ruta_test, archivo)
                    with open(ruta_archivo, "r", encoding="utf-8") as f:
                        datos_json = json.load(f)
                        for elem in datos_json:
                            if "message" in elem:
                                mensajes_test.append(elem["message"])

    # Guardar con estructura {"mensajes": [...]}
    with open(salida_train, "w", encoding="utf-8") as f_train:
        json.dump({"mensajes": mensajes_train}, f_train, ensure_ascii=False, indent=2)

    with open(salida_test, "w", encoding="utf-8") as f_test:
        json.dump({"mensajes": mensajes_test}, f_test, ensure_ascii=False, indent=2)

    print(f"Total mensajes en train: {len(mensajes_train)}")
    print(f"Total mensajes en test: {len(mensajes_test)}")
    print(f"Archivos guardados: {salida_train}, {salida_test}")
#--------------------------------------------------------------------------------------
#Metodos de preprocesamiento de texto

def limpiar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    
    # Quitar URLs, menciones y hashtags
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"@\w+|#\w+", "", texto)

    # Quitar emojis y caracteres no alfabéticos (excepto espacios)
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto)
                    if unicodedata.category(c) != 'So')  # So = Symbol, Other (emojis)

    # Quitar puntuación
    texto = re.sub(r"[^\w\s]", "", texto)

    # Tokenizar por espacios
    tokens = texto.split()

    # Eliminar stopwords y palabras muy cortas
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Stemming (opcional)
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def preprocesar_datos(tipo_vectorizacion):
    # Cargar mensajes
    with open("train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Preprocesar manteniendo agrupación por persona
    if tipo_vectorizacion == "robertuito":
        train_pre = [
            {"persona": persona["persona"], "mensajes": [preprocess_tweet(m) for m in persona["mensajes"]]}
            for persona in train_data
        ]
        test_pre = [
            {"persona": persona["persona"], "mensajes": [preprocess_tweet(m) for m in persona["mensajes"]]}
            for persona in test_data
        ]
    elif tipo_vectorizacion == "TfidF":
        train_pre = [
            {"persona": persona["persona"], "mensajes": [limpiar_texto(m) for m in persona["mensajes"]]}
            for persona in train_data
        ]
        test_pre = [
            {"persona": persona["persona"], "mensajes": [limpiar_texto(m) for m in persona["mensajes"]]}
            for persona in test_data
        ]

    # Guardar
    with open("train_preprocesado.json", "w", encoding="utf-8") as f:
        json.dump(train_pre, f, ensure_ascii=False, indent=2)

    with open("test_preprocesado.json", "w", encoding="utf-8") as f:
        json.dump(test_pre, f, ensure_ascii=False, indent=2)
    
    print(f"Preprocesamiento completado: {len(train_pre)} train, {len(test_pre)} test")    

#--------------------------------------------------------------------------------------

def vectorizar_TfidF():
    # Cargar datos preprocesados (lista de dicts {"persona","mensajes"})
    with open("train_preprocesado.json", "r", encoding="utf-8") as f:
        train_pre = json.load(f)

    with open("test_preprocesado.json", "r", encoding="utf-8") as f:
        test_pre = json.load(f)

    # Reconstruir texto por persona:
    # cada persona["mensajes"] es una lista de mensajes; cada mensaje puede ser:
    #  - lista de tokens (desde limpiar_texto)
    #  - string (si vino de otro preprocesamiento)
    def persona_a_texto(persona):
        textos = []
        for m in persona.get("mensajes", []):
            if isinstance(m, list):
                textos.append(" ".join(m))
            else:
                textos.append(str(m))
        return " ".join(textos)

    train_texts = [persona_a_texto(p) for p in train_pre]
    test_texts = [persona_a_texto(p) for p in test_pre]

    # Inicializar vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=8000)  # puedes ajustar este número
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Guardar los vectores y el vectorizador
    np.save("X_train_tfidf.npy", X_train.toarray())
    np.save("X_test_tfidf.npy", X_test.toarray())

    vocab = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}

    with open("vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vectorización completada: {X_train.shape[0]} train docs, {X_train.shape[1]} features")

def generar_embeddings_robertuito(
    model_name="pysentimiento/robertuito-base-uncased",
    max_length=128,
    batch_size=32
):
    # Cargar datos
    with open("train.json", "r", encoding="utf-8") as f:
        train_pre = json.load(f)
    with open("test.json", "r", encoding="utf-8") as f:
        test_pre = json.load(f)

    train_texts = train_pre["mensajes"]
    test_texts = test_pre["mensajes"]

    # Modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def textos_a_embeddings(textos):
        embeddings = []
        for i in tqdm(range(0, len(textos), batch_size), desc="Progreso", ncols=80):
            batch_texts = textos[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model(**enc)

            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.sum(mask, dim=1)
            batch_emb = (summed / counts).cpu().numpy()
            embeddings.append(batch_emb)

        return np.vstack(embeddings)

    # Generar embeddings
    X_train_emb = textos_a_embeddings(train_texts)
    X_test_emb = textos_a_embeddings(test_texts)

    # Guardar
    np.save("X_train_robertuito.npy", X_train_emb)
    np.save("X_test_robertuito.npy", X_test_emb)

    print("✅ Embeddings con RoBERTuito generados:")
    print("   Train:", X_train_emb.shape)
    print("   Test :", X_test_emb.shape)
    
def vectorizar_datos(tipo_vectorizacion="TfidF"):
    if tipo_vectorizacion == "TfidF":
        vectorizar_TfidF()
    #elif tipo_vectorizacion == "robertuito" and not os.path.exists("X_train_robertuito.npy"):
    elif tipo_vectorizacion == "robertuito":
        generar_embeddings_robertuito()
    elif not tipo_vectorizacion in ["TfidF", "robertuito"]:
        raise ValueError("Tipo de vectorización no soportado.") 
    print(f"Vectorización con {tipo_vectorizacion} completada.")

#--------------------------------------------------------------------------------------
# -------- Distancia coseno (vectorizado) ----------
def estimar_eps(X_norm, min_pts=5, plot=True):
    """
    Estima un buen valor de eps para DBSCAN usando la distancia al k-ésimo vecino.
    
    X_norm: np.ndarray (normalizado, n_samples x n_features)
    min_pts: número de vecinos a considerar
    plot: si True, grafica la curva para elegir eps visualmente
    """
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    neighbors = NearestNeighbors(n_neighbors=min_pts, metric="cosine").fit(X_norm)
    distances, _ = neighbors.kneighbors(X_norm)
    distances_k = np.sort(distances[:, min_pts-1])

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(distances_k)
        plt.xlabel("Puntos ordenados")
        plt.ylabel(f"Distancia al {min_pts}º vecino")
        plt.title("Curva para estimar eps")
        plt.grid(True)
        plt.show()

    # Opción: retornar valor aproximado del codo
    eps_sugerido = distances_k[int(len(distances_k)*0.95)]  # 95% de distancia
    print(f"eps sugerido (aprox. 95% de la curva): {eps_sugerido:.4f}")
    return eps_sugerido



def PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[2,20,40,100],eps=0.5, min_pts=3,metric="coseno"):
    if tipo_vectorizacion == "TfidF":
        X_train_data = np.load("X_train_tfidf.npy")
        X_test_data = np.load("X_test_tfidf.npy")
    elif tipo_vectorizacion == "robertuito":
        X_train_data = np.load("X_train_robertuito.npy")
        X_test_data = np.load("X_test_robertuito.npy")
    else:
        raise ValueError("Tipo de vectorización no soportado.")
    print(f"Reducción PCA sobre {tipo_vectorizacion}... ({X_train_data.shape[0]} muestras, {X_train_data.shape[1]} features)")
    # Inicialización para guardar el mejor PCA
    mejor_score = -np.inf
    mejor_X_train = None
    mejor_X_test = None
    mejor_pca = None
    mejor_n = None


    # Iterar sobre distintos números de componentes
    for n in n_componentes:
        pca = PCA(n_components=n)
        X_train_reducido = pca.fit_transform(X_train_data)
        X_test_reducido = pca.transform(X_test_data)
        X_train_norm = normalize(X_train_reducido)
        estimar_eps(X_train_norm, min_pts=min_pts, plot=True)
        
        # Clustering DBSCAN manual
        print("=== DEBUG X_train_norm ===")
        print("type:", type(X_train_norm))
        try:
            print("shape:", X_train_norm.shape)
        except Exception as e:
            print("shape error:", e)
        print("dtype:", getattr(X_train_norm, "dtype", "sin dtype"))
        print("primer elemento (type):", type(X_train_norm[0]))
        print("primer elemento:", X_train_norm[0])
        
        etiquetas = dbscan(X_train_norm, eps=eps, min_pts=min_pts,metric=metric)
        print(etiquetas)
        visualizar_clusters_por_etiqueta(X_train_reducido, etiquetas, f"Clusters PCA={n} (Train)")
        # Filtrar ruido para métricas
        mask = np.array(etiquetas) != -1
        etiquetas_filtradas = np.array(etiquetas)[mask]
        num_clusters_validos = len(set(etiquetas_filtradas))
        if np.sum(mask) <= 1 or num_clusters_validos < 2:
            print(f"PCA={n}: No se formaron suficientes clusters válidos ({num_clusters_validos} encontrados).")
            continue
        else:
            # Métricas
            sil_score = silhouette_score(X_train_norm[mask], etiquetas_filtradas, metric='cosine')
            ch_score = calinski_harabasz_score(X_train_norm[mask], etiquetas_filtradas)

            # Combinación simple de métricas
            score_combinado = (sil_score + ch_score / 1000) / 2  # normalizamos CH para escala similar

            print(f"PCA={n}: Silhouette={sil_score:.3f}, CH={ch_score:.1f}, combinado={score_combinado:.3f}")

            # Guardar el mejor
            if score_combinado > mejor_score:
                mejor_score = score_combinado
                mejor_X_train = X_train_reducido
                mejor_X_test = X_test_reducido
                mejor_pca = pca
                mejor_n = n

    # Guardar resultados
    if mejor_X_train is None:
        print("No se encontraron clusters válidos para ningún PCA. No se guardará archivo PCA.")
        return
    else:
        X_train_norm = normalize(mejor_X_train)
        etiquetas_finales = dbscan(X_train_norm, eps=eps, min_pts=min_pts,metric=metric)
        visualizar_clusters_directo(mejor_X_train, etiquetas_finales, f"Clusters finales ({tipo_vectorizacion})")


    mejor_X_train = np.array(mejor_X_train, dtype=float)
    np.save("X_train_pca_best.npy", mejor_X_train)
    mejor_X_test = np.array(mejor_X_test, dtype=float)
    np.save("X_test_pca_best.npy", mejor_X_test)
    joblib.dump(mejor_pca, "modelo_pca_best.pkl")
    print(f"🏆 Mejor PCA: {mejor_n} componentes con score combinado={mejor_score:.3f}")
    

#--------------------------------------------------------------------------------------


""" def distancia_coseno(p1, p2):
    #Distancia coseno entre dos vectores
    cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    return 1 - cos_sim
"""
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


def clustering_DBSCAN(tipo_vectorizacion="TfidF", eps=0.5, min_pts=3):
    """
    Aplica DBSCAN sobre los vectores reducidos (PCA) de la representación elegida.
    """
    # Cargar datos reducidos
    X_train = np.load("X_train_pca_best.npy")  # <- archivo generado por PCA_reduccion_dimensionalidad

    print(f"Clustering DBSCAN sobre {tipo_vectorizacion}... ({X_train.shape[0]} muestras)")

    # Convertir a lista de listas (para tu función DBSCAN)
    datos_lista = X_train.tolist()

    # Aplicar DBSCAN
    etiquetas = dbscan(datos_lista, eps=eps, min_pts=min_pts,metric="coseno")

    # Guardar etiquetas en un archivo
    np.save(f"labels_{tipo_vectorizacion}_dbscan.npy", etiquetas)

    # Contar clusters encontrados
    num_clusters = len(set([l for l in etiquetas if l != -1]))
    print(f"Clusters encontrados: {num_clusters}")
    print(f"Puntos ruido: {etiquetas.count(-1)}")
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def nueva_instancia_a_clusterizar(instancia):
    # Aquí iría la lógica para predecir con nuevos datos
    pass

#--------------------------------------------------------------------------------------

def visualizar_clusters_directo(X, labels, titulo="Clusters DBSCAN"):
    import matplotlib.cm as cm

    X = np.array(X)
    labels = np.array(labels)
    unique_labels = set(labels)
    num_clusters = len([l for l in unique_labels if l != -1])
    colors = cm.tab10(np.linspace(0, 1, max(num_clusters, 1)))

    plt.figure(figsize=(10, 7))
    for k, col in zip(sorted(unique_labels), colors):
        if k == -1:
            col = (0.5, 0.5, 0.5, 0.5)
        xy = X[labels == k]
        plt.scatter(xy[:, 0], xy[:, 1], s=25, c=[col], label=f"Cluster {k}" if k != -1 else "Ruido", alpha=0.7)

    plt.title(f"{titulo} (Clusters: {num_clusters}, Ruido: {(labels == -1).sum()})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best", fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def visualizar_clusters_por_etiqueta(X, etiquetas, titulo="Clusters DBSCAN"):
    """
    X: np.ndarray (n_samples, n_features) -> los datos originales o reducidos
    etiquetas: lista o array de etiquetas DBSCAN
    """
    X = np.array(X)
    etiquetas = np.array(etiquetas)

    # Reducimos a 2D para poder visualizar
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2).fit_transform(X)
    else:
        X_2d = X

    # Colores por cluster
    unique_labels = sorted(set(etiquetas))
    num_clusters = len([l for l in unique_labels if l != -1])
    colors = cm.tab10(np.linspace(0, 1, max(num_clusters, 1)))

    plt.figure(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Ruido en gris semitransparente
            col = (0.5, 0.5, 0.5, 0.5)
        else:
            col = colors[i % len(colors)]
        mask = etiquetas == label
        xy = X_2d[mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=f"Cluster {label}" if label != -1 else "Ruido", alpha=0.7)

    plt.title(f"{titulo} (Clusters: {num_clusters}, Ruido: {(etiquetas == -1).sum()})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best", fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

if __name__ == "__main__":
    tipo_vectorizacion ="robertuito" # "TfidF" o "robertuito"
    eps=0.55
    min_pts=5
   
    metrica="coseno"  # "coseno" o "euclidiana"
    
    agrupar_train_test()
    #preprocesar_datos(tipo_vectorizacion)

    # Vectorización con TfidF + PCA
    vectorizar_datos(tipo_vectorizacion)
    PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[100,200],eps=eps, min_pts=min_pts,metric=metrica)

    # Clustering manual DBSCAN
    #clustering_DBSCAN(tipo_vectorizacion, eps=20, min_pts=5)


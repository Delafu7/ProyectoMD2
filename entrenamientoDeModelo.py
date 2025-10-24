import os
import json
import re

import joblib
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import normalize
import matplotlib.cm as cm

#import matplotlib
#matplotlib.use('Agg')
import pandas as pd
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False
import seaborn as sns
# Preprocesamiento de texto
#-------------------------------------------------------------------------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
stemmer = SpanishStemmer()

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

# Metodos/Funciones

def agrupar_train_test(ruta_base="MentalRiskEs2023", salida_train="train.json", salida_test="test.json"):
    """
    Agrupa todos los mensajes de Train y Test en dos archivos JSON separados.
    Cada archivo tendrá la estructura: {"mensajes": [lista de mensajes]}
    
    ruta_base: ruta base donde están las carpetas de categorías
    salida_train: archivo de salida para Train
    salida_test: archivo de salida para Test
    """
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
    """Limpia y tokeniza un texto en español.
    Pasos:
    - Minúsculas
    - Quitar URLs, menciones y hashtags
    - Quitar emojis y caracteres no alfabéticos
    - Quitar puntuación
    - Tokenizar por espacios
    - Eliminar stopwords y palabras muy cortas
    
    Retorna lista de tokens.
    """
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
    """Preprocesa los datos de train y test según el tipo de vectorización.
    tipo_vectorizacion: "robertuito" o "TfidF"
    Genera dos archivos JSON: train_preprocesado.json y test_preprocesado.json

    """
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
    """Vectoriza los datos preprocesados usando TF-IDF.
    Genera archivos X_train_tfidf.npy, X_test_tfidf.npy y vocabulary.json"""
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
    """Genera embeddings usando RoBERTuito.
    Guarda en X_train_robertuito.npy y X_test_robertuito.npy
    model_name: nombre del modelo en HuggingFace
    max_length: longitud máxima de los textos
    batch_size: tamaño de lote para procesamiento
    """
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

    print("Embeddings con RoBERTuito generados:")
    print("   Train:", X_train_emb.shape)
    print("   Test :", X_test_emb.shape)
    
def vectorizar_datos(tipo_vectorizacion="TfidF"):
    """Vectoriza los datos según el tipo elegido.
    tipo_vectorizacion: "TfidF" o "robertuito"
    """
    if tipo_vectorizacion == "TfidF":
        vectorizar_TfidF()
    elif tipo_vectorizacion == "robertuito" and not os.path.exists("X_train_robertuito.npy"):
    #elif tipo_vectorizacion == "robertuito":
        generar_embeddings_robertuito()
    elif not tipo_vectorizacion in ["TfidF", "robertuito"]:
        raise ValueError("Tipo de vectorización no soportado.") 
    print(f"Vectorización con {tipo_vectorizacion} completada.")

#--------------------------------------------------------------------------------------
# -------- Distancia coseno (vectorizado) ----------


def PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[2,20,40,100]):
    """Aplica PCA para reducción de dimensionalidad.
    tipo_vectorizacion: "TfidF" o "robertuito"
    n_componentes: lista de números de componentes a probar
    """    
    # Cargar datos vectorizados
    if tipo_vectorizacion == "TfidF":
        X_train_data = np.load("X_train_tfidf.npy")
        X_test_data  = np.load("X_test_tfidf.npy")
    elif tipo_vectorizacion == "robertuito":
        X_train_data = np.load("X_train_robertuito.npy")
        X_test_data  = np.load("X_test_robertuito.npy")
    else:
        raise ValueError("Tipo de vectorización no soportado.")

    print(f"Reducción PCA sobre {tipo_vectorizacion}... ({X_train_data.shape[0]} muestras, {X_train_data.shape[1]} features)")

    # Inicialización para guardar el mejor PCA (criterio: mayor varianza explicada total)
    mejor_varianza = -np.inf
    mejor_X_train = None
    mejor_X_test  = None
    mejor_pca = None
    mejor_n = None



    # Iterar sobre distintos números de componentes
    for n in n_componentes:
        print(f"\n====== Probando PCA n_components={n} ======")
        pca = PCA(n_components=n)
        X_train_reducido = pca.fit_transform(X_train_data)
        X_test_reducido = pca.transform(X_test_data)

        var_total = float(np.sum(pca.explained_variance_ratio_))

        X_reconstruido = pca.inverse_transform(X_train_reducido)
        error_recon = float(np.mean(np.square(X_train_data - X_reconstruido)))

        print(f"Varianza explicada total (n={n}): {var_total}")
        print(f"Error de reconstrucción (n={n}): {error_recon}")


        # Selección simple del mejor PCA por varianza explicada total
        if var_total > mejor_varianza:
            mejor_varianza = var_total
            mejor_pca = pca
            mejor_n = n
            mejor_X_train = X_train_reducido
            mejor_X_test = X_test_reducido


        # Guardar los resultados "best"
    np.save("X_train_pca_best.npy", np.asarray(mejor_X_train, dtype=np.float32))
    np.save("X_test_pca_best.npy", np.asarray(mejor_X_test, dtype=np.float32))
    joblib.dump(mejor_pca, "modelo_pca_best.pkl")

    print(f"\nMejor PCA: n_components={mejor_n}, varianza explicada total={mejor_varianza}")



#--------------------------------------------------------------------------------------

def distancia_coseno_vecs(M, v):
    """
    Distancia coseno entre un vector y todas las filas de una matriz.
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
    Distancia euclidiana entre un vector y todas las filas de una matriz.
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
    """Expande un cluster agregando puntos densos
    datos: np.ndarray 2D
    labels: lista de etiquetas
    punto_idx: índice del punto inicial
    vecinos: lista de índices de vecinos del punto inicial
    cluster_id: id del cluster actual
    eps: radio de vecindad
    min_pts: mínimo de puntos para formar cluster
    """
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
    Implementación básica de DBSCAN.

    datos: lista de puntos o np.ndarray (n_samples, n_features)
    eps: radio de vecindad
    min_pts: mínimo de puntos para formar cluster
    metric: "coseno" o "euclidiana"

    Devuelve lista de etiquetas por punto.
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


def clustering_DBSCAN(datos=None, eps=0.5, min_pts=3,metric=None,train=True):
    """
    Aplica DBSCAN sobre los vectores reducidos (PCA) de la representación elegida.

    datos: np.ndarray (n_samples, n_features) o None para cargar desde archivo
    eps: radio de vecindad
    min_pts: mínimo de puntos para formar cluster
    metric: "coseno" o "euclidiana"
    train: si True, carga mensajes de train.json; si False, de test.json
    Devuelve lista de etiquetas por punto.

    """
    # Cargar datos reducidos
    X_train = np.asarray(datos, dtype=np.float32)

    print(f"Clustering DBSCAN sobre ... ({X_train.shape[0]} muestras)")

    # Convertir a lista de listas
    datos_lista = X_train.tolist()


    etiquetas = dbscan(datos_lista, eps=eps, min_pts=min_pts,metric=metric)

    # Conseguir los mensajes originales
    if train:
        with open("train.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        with open("test.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
    if not (isinstance(obj, dict) and "mensajes" in obj and isinstance(obj["mensajes"], list)):
        raise ValueError(" debe ser un dict con clave 'mensajes' -> lista de strings.")
    mensajes = obj["mensajes"]


    n = X_train.shape[0]
    if len(mensajes) != n:
        raise ValueError(
            f"Desalineacion: ruta_json contiene {len(mensajes)} mensajes pero X_train tiene {n} filas.")

    filas = []
    for i in range(n):
        mensaje = mensajes[i]
        lab = int(etiquetas[i])
        embedding = X_train[i].astype(float).tolist()
        emb_json = json.dumps(embedding, ensure_ascii=False)
        filas.append((mensaje, lab, emb_json))

    # emparejar y guardar
    pares = list(zip(mensajes, etiquetas))

    #nombre_npy = f"mensajes_y_labels_{tipo_vectorizacion}.npy"
    #np.save(nombre_npy, np.array(pares, dtype=object), allow_pickle=True)


    nombre_base = f"mensajes_y_labels_{eps}_{min_pts}"
    nombre_csv = f"{nombre_base}.csv"
    contador = 1
    while os.path.exists(nombre_csv):
        nombre_csv = f"{nombre_base}_{contador}.csv"
        contador += 1
    df = pd.DataFrame(filas, columns=["mensaje", "cluster","embedding"])
    df.to_csv(nombre_csv, index=False, encoding="utf-8")



    num_clusters = len(set([l for l in etiquetas if l != -1]))
    num_ruido = etiquetas.count(-1)
    print(f"Clusters encontrados: {num_clusters}")
    print(f"Puntos ruido: {num_ruido}")
    return etiquetas
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------

def visualizar_clusters_directo(X, labels, titulo="Clusters DBSCAN"):
    """ Visualiza clusters en 2D.
    X: np.ndarray (n_samples, 2) -> datos en 2D
    labels: lista o array de etiquetas DBSCAN
    """

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


def calculoDeIndices(modelo, labels_train=None, labels_test=None):
    # Carga de datos PCA desde archivos .npy
    X_train_pca = np.load("X_train_pca_best.npy")
    X_test_pca = np.load("X_test_pca_best.npy")

    #  Métricas en TRAIN
    silhouette_train = silhouette_score(X_train_pca, labels_train)
    davies_train = davies_bouldin_score(X_train_pca, labels_train)
    calinski_train = calinski_harabasz_score(X_train_pca, labels_train)

    #  Métricas en TEST
    if labels_test is None:
        if hasattr(modelo, "predict"):
            labels_test = modelo.predict(X_test_pca)
        else:
            labels_test = modelo.fit_predict(X_test_pca)

    silhouette_test = silhouette_score(X_test_pca, labels_test)
    davies_test = davies_bouldin_score(X_test_pca, labels_test)
    calinski_test = calinski_harabasz_score(X_test_pca, labels_test)

    # Devolver resultados como diccionario
    resultados = {
        "silhouette_train": silhouette_train,
        "davies_train": davies_train,
        "calinski_train": calinski_train,
        "silhouette_test": silhouette_test,
        "davies_test": davies_test,
        "calinski_test": calinski_test
    }
    print(f"Silhouette score: {silhouette_test}"
          f" davies score: {davies_test}"
          f" calinski score: {calinski_test}")

    return resultados

def evaluar_y_guardar_metricas(X_pca, labels, eps, min_pts, n_componentes, tipo_vectorizacion="robertuito", y_true=None, archivo_csv="resultados_metricas.csv"):
    """
    Calcula métricas de evaluación del clustering y las guarda en un CSV.
    Si hay etiquetas reales (y_true), calcula métricas externas también.

    X_pca: np.ndarray (n_samples, n_features) -> datos reducidos
    labels: lista o array de etiquetas DBSCAN
    eps: parámetro DBSCAN
    min_pts: parámetro DBSCAN
    n_componentes: número de componentes PCA
    tipo_vectorizacion: "robertuito" o "TfidF"
    y_true: lista o array de etiquetas reales (opcional)
    archivo_csv: ruta del archivo CSV donde guardar las métricas

    Devuelve un diccionario con las métricas calculadas.
    """

    # Convertir a arrays numpy
    X_pca = np.array(X_pca)
    labels = np.array(labels)

    # Filtrar ruido
    mask = labels != -1
    if np.sum(mask) < 2:
        print("No hay suficientes puntos no-ruido para calcular métricas.")
        return

    X_filtrado = X_pca[mask]
    labels_filtrados = labels[mask]

    # --- Métricas internas ---
    silhouette = silhouette_score(X_filtrado, labels_filtrados, metric="cosine")
    calinski = calinski_harabasz_score(X_filtrado, labels_filtrados)
    davies = davies_bouldin_score(X_filtrado, labels_filtrados)
    score_combinado = (silhouette + calinski / 1000) / 2

    # --- Métricas externas (si hay etiquetas verdaderas) ---
    ari = nmi = h = c = v = np.nan
    if y_true is not None:
        y_true = np.array(y_true)[mask]
        ari = adjusted_rand_score(y_true, labels_filtrados)
        nmi = normalized_mutual_info_score(y_true, labels_filtrados)
        h = homogeneity_score(y_true, labels_filtrados)
        c = completeness_score(y_true, labels_filtrados)
        v = v_measure_score(y_true, labels_filtrados)

    # --- Info del clustering ---
    num_clusters = len(set(labels_filtrados))
    num_ruido = np.sum(labels == -1)

    # Crear registro
    fila = {
        "metrica_combinada": score_combinado,
        "vectorizacion": tipo_vectorizacion,
        "PCA_componentes": n_componentes,
        "eps": eps,
        "min_pts": min_pts,
        "clusters": num_clusters,
        "ruido": num_ruido,
        "silhouette": silhouette,
        "calinski": calinski,
        "davies": davies,
        "ARI": ari,
        "NMI": nmi,
        "Homogeneity": h,
        "Completeness": c,
        "V_measure": v,
    }

    # Guardar o añadir al CSV
    df_fila = pd.DataFrame([fila])
    if os.path.exists(archivo_csv):
        df_fila.to_csv(archivo_csv, mode="a", header=False, index=False)
    else:
        df_fila.to_csv(archivo_csv, index=False)

    print(f" Métricas guardadas en {archivo_csv}")
    print(f" Clusters: {num_clusters} , Score_Combinado={score_combinado:.3f} , Silhouette={silhouette:.3f}, CH={calinski:.1f}, DB={davies:.3f}")

    return fila

def normalize_rows(X, copy=True):
    """
    Normaliza cada fila de X a norma L2 = 1.
    Devuelve un array float32.
    X: np.ndarray 2D
    copy: si True, no modifica X original

    """
    X = np.asarray(X, dtype=np.float32)
    return normalize(X, norm='l2', axis=1, copy=copy).astype(np.float32)

def nueva_instancia_a_clusterizar(instancia,model_name="pysentimiento/robertuito-base-uncased",eps=0.1, metrica="coseno",max_length=128,intentar_centroide=True):
    """
    Asigna una nueva instancia a un cluster usando el modelo entrenado.

    instancia: texto de la nueva instancia
    model_name: modelo de HuggingFace para embeddings
    eps: umbral de distancia para asignación
    metrica: "coseno" o "euclidiana"
    max_length: longitud máxima para el tokenizer
    intentar_centroide: si True, usa centroides como fallback

    Devuelve un diccionario con:
    - instancia: texto original
    - etiqueta_asignada: etiqueta asignada (-1 si ruido)
    - distancia: distancia al vecino más cercano o centroide
    - indice_vecino: índice del vecino más cercano en el train (o -1 si centroide)
    - centroide_usado: dict con 'etiqueta' y 'distancia' si se usó centroide, o None
    """

    # rutas esperadas
    ruta_reductor = None
    for candidato in ["modelo_pca_best.pkl"]:
        if os.path.exists(candidato):
            ruta_reductor = candidato
            break
    if ruta_reductor is None:
        for f in os.listdir():
            if (f.endswith(".pkl") or f.endswith(".joblib")) and "pca" in f.lower():
                ruta_reductor = f
                break
    if ruta_reductor is None:
        return {"error": "No se encontró reductor (modelo_pca_best.pkl)."}

    ruta_X_train_pca = "X_train_pca_best.npy"
    if not os.path.exists(ruta_X_train_pca):
        return {"error": f"No se encontró {ruta_X_train_pca}."}

    archivos_labels = [f for f in os.listdir() if f.startswith("labels") and f.endswith(".npy")]
    if len(archivos_labels) == 0:
        return {"error": "No se encontró archivo de etiquetas labels_*.npy."}
    ruta_labels = archivos_labels[0]

    # generar embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    with torch.inference_mode():
        enc = tokenizer([instancia], truncation=True, padding="longest", max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.sum(mask, dim=1).clamp(min=1.0)
        emb = (summed / counts).cpu().numpy().astype(np.float32)

    # cargar reductor
    try:
        reductor = joblib.load(ruta_reductor)
    except Exception as e:
        return {"error": f"Error cargando reductor: {e}"}
    if not hasattr(reductor, "transform"):
        return {"error": "El objeto reductor no tiene método transform."}

    # transformar y normalizar
    try:
        X_new = reductor.transform(emb).astype(np.float32)
    except Exception as e:
        return {"error": f"Error al transformar la instancia: {e}"}
    X_new_norm = normalize(X_new, norm='l2', axis=1).astype(np.float32)

    # cargar train reducido y labels
    try:
        X_train = np.load(ruta_X_train_pca).astype(np.float32)
    except Exception as e:
        return {"error": f"Error cargando {ruta_X_train_pca}: {e}"}
    try:
        labels_train = np.load(ruta_labels).astype(int)
    except Exception as e:
        return {"error": f"Error cargando {ruta_labels}: {e}"}
    X_train_norm = normalize(X_train, norm='l2', axis=1).astype(np.float32)

    # vecino más cercano
    nn_metric = "cosine" if metrica == "coseno" else "euclidean"
    nbrs = NearestNeighbors(n_neighbors=1, metric=nn_metric, n_jobs=-1).fit(X_train_norm)
    dist, idx = nbrs.kneighbors(X_new_norm)
    dist_val = float(dist.ravel()[0])
    idx0 = int(idx.ravel()[0])
    etiqueta_vecino = int(labels_train[idx0])
    etiqueta_asignada = etiqueta_vecino if (dist_val <= eps and etiqueta_vecino != -1) else -1

    # fallback por centroides
    centroide_verificado = False
    centroide_elegido = {"etiqueta": None, "distancia": None}
    if etiqueta_asignada == -1 and intentar_centroide:
        centroide_verificado = True
        sin_ruido = [l for l in np.unique(labels_train) if l != -1]
        if len(sin_ruido) > 0:
            centroides = {lab: X_train_norm[labels_train == lab].mean(axis=0) for lab in sin_ruido}
            C = np.vstack(list(centroides.values())).astype(np.float32)
            claves = list(centroides.keys())
            if nn_metric == "cosine":
                puntos = np.dot(C, X_new_norm.ravel())
                dist_centroides = (1.0 - np.clip(puntos, -1.0, 1.0)).astype(np.float32)
            else:
                dif = C - X_new_norm.ravel()
                dist_centroides = np.linalg.norm(dif, axis=1).astype(np.float32)
            min_idx = int(np.argmin(dist_centroides))
            min_dist = float(dist_centroides[min_idx])
            min_label = int(claves[min_idx])
            centroide_elegido = {"etiqueta": min_label, "distancia": min_dist}
            if min_dist <= eps:
                etiqueta_asignada = min_label
                dist_val = min_dist
                idx0 = -1

    return {
        "instancia": instancia,
        "etiqueta_asignada": int(etiqueta_asignada),
        "indice_vecino": int(idx0),
        "etiqueta_vecino": int(etiqueta_vecino),
        "distancia_al_vecino": float(dist_val),
        "centroide_verificado": centroide_verificado,
        "centroide_elegido": centroide_elegido,
        "archivo_labels_usado": ruta_labels,
        "archivo_modelo_pca": ruta_reductor
    }
#--------------------------------------------------------------------------------------
def plot_metrics_from_csv(archivo_csv="resultados_metricas.csv", columnas=None, figsize=(10,5)):
    """
    Carga CSV con varias filas (experimentos) y grafica las métricas más relevantes
    columnas: lista de columnas a graficar (ej: ["silhouette","calinski","davies"])
    """
    if not os.path.exists(archivo_csv):
        print(f"No existe {archivo_csv}")
        return
    df = pd.read_csv(archivo_csv)
    if columnas is None:
        columnas = ["silhouette", "calinski", "davies"]
        columnas = [c for c in columnas if c in df.columns]

    if len(columnas) == 0:
        print("No hay columnas seleccionadas en el CSV")
        return

    plt.figure(figsize=figsize)
    for c in columnas:
        plt.plot(df.index, df[c], marker='o', label=c)
    plt.xlabel("Experimento / fila CSV")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.title("Evolución de métricas por experimento")
    plt.tight_layout()
    plt.show()
    
def plot_cluster_sizes(labels, titulo="Tamaño de clusters"):
    """
    labels: array/list de etiquetas (incluye -1 ruido)
    """
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"label": unique, "count": counts})
    df = df.sort_values("count", ascending=False)
    plt.figure(figsize=(8,4))
    plt.bar(df['label'].astype(str), df['count'])
    plt.xlabel("Cluster")
    plt.ylabel("Número de puntos")
    plt.title(titulo)
    for idx, row in df.iterrows():
        plt.text(idx, row['count'] + 0.5, int(row['count']), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    
def plot_centroid_distance_matrix(X, labels, metric="cosine", titulo="Distancias entre centroides", annot=True):
    """
    Calcula centroides por cluster (excluyendo ruido) y muestra heatmap de distancias.
    X: np.ndarray (n_samples, n_features)
    labels: lista/array de etiquetas
    metric: "cosine" o "euclidiana"
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    clusters = [c for c in np.unique(labels) if c != -1]
    if len(clusters) == 0:
        print("No hay clusters válidos (sólo ruido).")
        return
    centroides = np.vstack([X[labels == c].mean(axis=0) for c in clusters])
    # matriz de distancias
    if metric == "cosine":
        # cos similarity -> distancia
        sim = np.dot(centroides, centroides.T)
        norms = np.linalg.norm(centroides, axis=1)
        denom = np.outer(norms, norms) + 1e-12
        cos_sim = sim / denom
        D = 1.0 - np.clip(cos_sim, -1, 1)
    else:
        # euclidiana
        from scipy.spatial.distance import cdist
        D = cdist(centroides, centroides, metric='euclidean')

    plt.figure(figsize=(8,6))
    sns.heatmap(D, annot=annot, xticklabels=clusters, yticklabels=clusters, cmap="viridis")
    plt.title(titulo)
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

def plot_silhouette_per_sample(X, labels, titulo="Silhouette por muestra"):
    """
    Dibuja el silhouette plot clásico para ver la distribución por cluster.
    X: array (n_samples, n_features) -> preferiblemente PCA 2D/normalizado
    labels: etiquetas (incluye -1)
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = labels != -1
    if np.sum(mask) < 2:
        print("No hay suficientes muestras no-ruido para silhouette.")
        return
    X_fil = X[mask]
    lab_fil = labels[mask]
    sil_vals = silhouette_samples(X_fil, lab_fil, metric='cosine')
    # ordenar por cluster
    df = pd.DataFrame({"label": lab_fil, "sil": sil_vals})
    df = df.sort_values(["label", "sil"], ascending=[True, False])
    y_lower = 10
    plt.figure(figsize=(9,6))
    for i, lab in enumerate(sorted(df['label'].unique())):
        vals = df[df['label']==lab]['sil'].values
        y_upper = y_lower + len(vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        plt.text(-0.05, (y_lower + y_upper) / 2, str(lab))
        y_lower = y_upper + 10
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Muestras agrupadas por cluster")
    plt.title(titulo)
    plt.axvline(x=np.mean(sil_vals), color="red", linestyle="--", label="Media silhouette")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_2d_embedding_scatter(X, labels, method="tsne", titulo="Visualización 2D", random_state=42, perplexity=30):
    """
    Reduce a 2D con t-SNE o UMAP y grafica los puntos coloreados por label.
    method: "tsne" o "umap"
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2)
        emb = reducer.fit_transform(X)
    else:
        # t-SNE
        ts = TSNE(n_components=2, random_state=random_state, perplexity=min(perplexity, max(5, X.shape[0]//4)), n_iter=1000)
        emb = ts.fit_transform(X)
    unique = sorted(set(labels))
    plt.figure(figsize=(9,7))
    colors = cm.tab10(np.linspace(0,1,max(len(unique)-1,1)))
    for i, lab in enumerate(unique):
        mask = labels == lab
        col = (0.5,0.5,0.5,0.3) if lab == -1 else colors[i % len(colors)]
        plt.scatter(emb[mask,0], emb[mask,1], s=25, c=[col], label=f"Cluster {lab}" if lab!=-1 else "Ruido", alpha=0.7)
    plt.title(titulo)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_top_terms_per_cluster_tfidf(vectorizer, X_tfidf, labels, topn=10, titulo="Top términos por cluster"):
    """
    Muestra en una tabla/print los top-n términos de cada cluster (sólo si usas TF-IDF).
    vectorizer: objeto TfidfVectorizer (ya entrenado)
    X_tfidf: matriz 2D (n_samples, n_terms) correlacionada con las muestras (no PCA)
    labels: etiquetas por muestra (incluye -1)
    """
    if not hasattr(vectorizer, "get_feature_names_out"):
        print("Vectorizer no tiene get_feature_names_out.")
        return
    features = vectorizer.get_feature_names_out()
    labels = np.asarray(labels)
    clusters = [c for c in np.unique(labels) if c != -1]
    rows = []
    for c in clusters:
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        # sumar pesos TF-IDF por cluster
        cluster_sum = np.asarray(X_tfidf[idxs].sum(axis=0)).ravel()
        top_idx = cluster_sum.argsort()[::-1][:topn]
        top_terms = [features[i] for i in top_idx]
        rows.append({"cluster": c, "top_terms": ", ".join(top_terms)})
    df = pd.DataFrame(rows).sort_values("cluster")
    print(titulo)
    print(df)  # si usas notebook; si no, imprime:
    # return df

def graficar_metricas():
    """ Lee el CSV de métricas y genera gráficas organizadas por PCA_componentes.
    Genera subplots por métrica y guarda en carpetas y un heatmap de métrica combinada."""
    data = pd.read_csv("resultados_metricas.csv")

    # Eliminar duplicados por combinación de PCA, eps y min_pts
    data = data.drop_duplicates(subset=["PCA_componentes", "eps", "min_pts"])

    # Lista de métricas y títulos
    metricas = ["metrica_combinada", "silhouette", "calinski", "davies"]
    titulos = ["Métrica Combinada", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]

    # Crear carpeta principal
    os.makedirs("imagenes", exist_ok=True)

    # Generar y guardar gráficas por cada valor de PCA
    for pca_val in sorted(data["PCA_componentes"].unique()):
        subset_pca = data[data["PCA_componentes"] == pca_val]
        carpeta_pca = f"imagenes/PCA_{pca_val}"
        os.makedirs(carpeta_pca, exist_ok=True)

        for metrica, titulo in zip(metricas, titulos):
            min_pts_vals = sorted(subset_pca["min_pts"].unique())
            n_rows = int(np.ceil(len(min_pts_vals)/2))
            plt.figure(figsize=(12, 4*n_rows))

            for i, min_pts in enumerate(min_pts_vals, 1):
                sub = subset_pca[subset_pca["min_pts"] == min_pts]
                plt.subplot(n_rows, 2, i)
                plt.plot(sub["eps"], sub[metrica], marker='o')
                plt.title(f"{titulo} - min_pts={min_pts}")
                plt.xlabel("eps")
                plt.ylabel(metrica)
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{carpeta_pca}/{metrica}_subplots.png", dpi=300)
            plt.close()
        print(f"Gráficas de métricas para PCA={pca_val} guardadas en {carpeta_pca}")
        pivot = subset_pca.pivot(index="min_pts", columns="eps", values="metrica_combinada")
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Métrica Combinada (PCA={pca_val})")
        plt.xlabel("eps")
        plt.ylabel("min_pts")
        plt.savefig(f"{carpeta_pca}/heatmap_combinada.png", dpi=300)
        plt.close()
    
    print("Gráficas guardadas en la carpeta 'imagenes', sin duplicados y organizadas por PCA_componentes.")

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parámetros
    tipo_vectorizacion = "robertuito"
    eps = 0.35
    min_pts = 13
    metrica = "coseno"
    graficarDatos = False
    train= True #  Si es True, se entrena el modelo DBSCAN desde cero. Si es False, se cargan las etiquetas ya generadas.

    # Preparar datos
    agrupar_train_test()
    #preprocesar_datos(tipo_vectorizacion) #Al final no es necesario preprocesar ya que empleamos Robertuito directamente, un modelo preentrenado.
    vectorizar_datos(tipo_vectorizacion)

    # Reducción de dimensionalidad
    PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[700])

    #Cargar
    if train:
        Xpca_path = "X_train_pca_best.npy"
    else:
        Xpca_path = "X_test_pca_best.npy"
    Xpca = np.load(Xpca_path).astype(np.float32)

    #Normalizar
    Xpca_norm = normalize(Xpca, norm="l2", axis=1).astype(np.float32)

    #DBSCAN
    #etiquetas = dbscan(Xpca_norm, eps=eps, min_pts=min_pts, metric=metrica)
    #np.save("labels_dbscan.npy", np.array(etiquetas, dtype=int))
    eps_list = [0.3]
    min_pts_list = [35]

    best_sil = -np.inf
    best_combo = (eps, min_pts)
    best_labels = None

    for e in eps_list:
        for m in min_pts_list:
                tmp = clustering_DBSCAN(Xpca_norm, eps=e, min_pts=m, metric=metrica, train=train)
                tmp = np.asarray(tmp, dtype=int)

                # Contar clusters validos
                mask = tmp != -1
                n_clusters = len(set(tmp[mask])) if np.sum(mask) > 0 else 0

                if n_clusters >= 2:
                    # Silhouette sobre los puntos no-ruido
                    sil = float(silhouette_score(Xpca_norm[mask], tmp[mask], metric='cosine'))
                else:
                    sil = -1.0

                mask = tmp != -1
                n_clusters = len(set(tmp[mask])) if np.sum(mask) > 0 else 0

                if n_clusters >= 2:
                    n_componentes_usado = Xpca.shape[1]
                    resultados = evaluar_y_guardar_metricas(X_pca=Xpca, labels=tmp, eps=e, min_pts=m,n_componentes=n_componentes_usado,tipo_vectorizacion=tipo_vectorizacion, y_true=None)
                    print(f"Silhouette: {resultados['silhouette']}, Calinski: {resultados['calinski']}, Davies: {resultados['davies']}")


                # Guardar mejor por silhouette
                if sil > best_sil:
                    best_sil = sil
                    best_combo = (e, m)
                    best_labels = tmp.copy()

                # Guardar etiquetas de esta configuración para análisis posterior



    # Si no se encontró combinación con ≥2 cluster válidos, usar la original
    if best_labels is None:
        print("No se encontró combinación con >=2 clusters válidos; uso la combinación original.")
        etiquetas = clustering_DBSCAN(Xpca_norm, eps=e, min_pts=m, metric=metrica, train=train)
        etiquetas = np.asarray(etiquetas, dtype=int)
    else:
        best_eps, best_min_pts = best_combo
        print(f"Mejor combo: eps={best_eps}, min_pts={best_min_pts}, silhouette={best_sil:.4f}")
        etiquetas = best_labels
        eps = best_eps
        min_pts = best_min_pts

    # Guardar etiquetas finales
    np.save("labels_dbscan.npy", np.array(etiquetas, dtype=int))


    #Preparar datos  para visualizar
    #if Xpca.shape[1] > 2:
    pca_vis = PCA(n_components=2, random_state=42)
    X_vis = pca_vis.fit_transform(Xpca).astype(np.float32)
    #else:
    X_vis = Xpca.astype(np.float32)

    visualizar_clusters_directo(X_vis, etiquetas, titulo=f"Clusters (eps={eps}, min_pts={min_pts})")

    # Preparar representación 2D
    #if Xpca.shape[1] > 2:
    pca_vis = PCA(n_components=2, random_state=42)
    X_vis = pca_vis.fit_transform(Xpca).astype(np.float32)
    #else:
    X_vis = Xpca.astype(np.float32)


    #Visualizaciones
    visualizar_clusters_directo(X_vis, etiquetas, titulo=f"Clusters (eps={eps}, min_pts={min_pts})")

    """visualizar_clusters_por_etiqueta(Xpca, etiquetas, f"Clusters PCA (embedding completo)")
    plot_cluster_sizes(etiquetas, titulo="Tamaño clusters PCA (embedding completo)")
    plot_centroid_distance_matrix(Xpca, etiquetas, metric="cosine")
    plot_silhouette_per_sample(Xpca, etiquetas)
    plot_2d_embedding_scatter(Xpca, etiquetas, method="umap")"""

    n_componentes_usado = Xpca.shape[1]
    resultados = evaluar_y_guardar_metricas(X_pca=Xpca,labels=etiquetas,eps=eps,min_pts=min_pts,n_componentes=n_componentes_usado,tipo_vectorizacion=tipo_vectorizacion,y_true=None,)

    print("\n=== Resultados metricas ===")
    if resultados is not None:
        # imprimir resumen compacto
        print(f"Metricas guardadas en 'resultados_metricas.csv' (fila añadida).")
        print(f"Clusters (no-ruido): {resultados['clusters']}, Ruido: {resultados['ruido']}")
        print(f"Silhouette: {resultados['silhouette']}, Calinski: {resultados['calinski']}, Davies: {resultados['davies']}")



    texto = ""
    resultado = nueva_instancia_a_clusterizar(texto,  eps=eps,  metrica=metrica)
    print(resultado)

    #Resumen
    labels_arr = np.array(etiquetas, dtype=int)
    num_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    num_noise = int((labels_arr == -1).sum())
    total = len(labels_arr)
    print(f"Clusters encontrados: {num_clusters}")
    print(f"Puntos ruido: {num_noise} ({100 * num_noise / total:.1f}%)")
    print("Etiquetas guardadas en: labels_dbscan.npy")
    if graficarDatos:
        graficar_metricas()
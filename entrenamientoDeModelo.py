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
# Preprocesamiento de texto
#-------------------------------------------------------------------------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
stemmer = SpanishStemmer()

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

# Metodos/Funciones

def agrupar_train_test(ruta_base="MentalRiskEs2023", salida_train="train.json", salida_test="test.json"):
    data_train = defaultdict(list)
    data_test = defaultdict(list)
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
                            persona_id = archivo.replace(".json", "")
                            if "message" in elem:
                                data_train[persona_id].append(elem["message"])

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
                            persona_id = archivo.replace(".json", "")
                            if "message" in elem:
                                data_test[persona_id].append(elem["message"])

    # Convertir defaultdict a lista de diccionarios
    train_list = [{"persona": pid, "mensajes": msgs} for pid, msgs in data_train.items()]
    test_list = [{"persona": pid, "mensajes": msgs} for pid, msgs in data_test.items()]

    # Guardar
    with open(salida_train, "w", encoding="utf-8") as f_train:
        json.dump(train_list, f_train, ensure_ascii=False, indent=2)

    with open(salida_test, "w", encoding="utf-8") as f_test:
        json.dump(test_list, f_test, ensure_ascii=False, indent=2)

    print(f"Total personas en train: {len(train_list)}")
    print(f"Total personas en test: {len(test_list)}")
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
    with open("train_preprocesado.json", "r", encoding="utf-8") as f:
        train_pre = json.load(f)
    with open("test_preprocesado.json", "r", encoding="utf-8") as f:
        test_pre = json.load(f)

    train_texts = [" ".join(persona["mensajes"]) for persona in train_pre]
    test_texts = [" ".join(persona["mensajes"]) for persona in test_pre]


    # Modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def textos_a_embeddings(textos):
        embeddings = []
        for i in tqdm(range(0, len(textos), batch_size)):
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

    X_train_emb = textos_a_embeddings(train_texts)
    X_test_emb = textos_a_embeddings(test_texts)

    np.save("X_train_robertuito.npy", X_train_emb)
    np.save("X_test_robertuito.npy", X_test_emb)

    print("Embeddings con RoBERTuito generados:", X_train_emb.shape, X_test_emb.shape)

def vectorizar_datos(tipo_vectorizacion="TfidF"):
    if tipo_vectorizacion == "TfidF":
        vectorizar_TfidF()
    elif tipo_vectorizacion == "robertuito" and not os.path.exists("X_train_robertuito.npy"):
        generar_embeddings_robertuito()
    elif not tipo_vectorizacion in ["TfidF", "robertuito"]:
        raise ValueError("Tipo de vectorización no soportado.") 
    print(f"Vectorización con {tipo_vectorizacion} completada.")

#--------------------------------------------------------------------------------------

def PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[2,20,40,100]):
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
        
        # Clustering DBSCAN manual
        etiquetas = dbscan(X_train_norm, eps=0.5, min_pts=3)
        print(etiquetas)
        # Filtrar ruido para métricas
        mask = np.array(etiquetas) != -1
        etiquetas_filtradas = np.array(etiquetas)[mask]
        num_clusters_validos = len(set(etiquetas_filtradas))
        if np.sum(mask) <= 1 or num_clusters_validos < 2:
            print(f"PCA={n}: No se formaron suficientes clusters válidos ({num_clusters_validos} encontrados).")
            continue
        else:
            # Métricas
            sil_score = silhouette_score(X_train_reducido[mask], etiquetas_filtradas, metric='cosine')
            ch_score = calinski_harabasz_score(X_train_reducido[mask], etiquetas_filtradas)

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
        etiquetas_finales = dbscan(X_train_norm, eps=0.5, min_pts=5)
        visualizar_clusters_directo(mejor_X_train, etiquetas_finales, f"Clusters finales ({tipo_vectorizacion})")

    mejor_X_train = np.array(mejor_X_train, dtype=float)
    np.save("X_train_pca_best.npy", mejor_X_train)
    mejor_X_test = np.array(mejor_X_test, dtype=float)
    np.save("X_test_pca_best.npy", mejor_X_test)
    joblib.dump(mejor_pca, "modelo_pca_best.pkl")
    print(f"🏆 Mejor PCA: {mejor_n} componentes con score combinado={mejor_score:.3f}")
    

#--------------------------------------------------------------------------------------


def distancia_coseno(p1, p2):
    """Distancia coseno entre dos vectores"""
    cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    return 1 - cos_sim

def region_query(datos, punto_idx, eps):
    """Encuentra todos los vecinos dentro de eps"""
    punto = datos[punto_idx]
    # calcular distancias a todos los puntos de manera vectorizada
    distancias = np.array([distancia_coseno(punto, x) for x in datos])
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
    datos = np.array(datos) # TODO: no se si esto es necesario
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
    etiquetas = dbscan(datos_lista, eps=eps, min_pts=min_pts)

    # Guardar etiquetas en un archivo
    np.save(f"labels_{tipo_vectorizacion}_dbscan.npy", etiquetas)

    # Contar clusters encontrados
    num_clusters = len(set([l for l in etiquetas if l != -1]))
    print(f"Clusters encontrados: {num_clusters}")
    print(f"Puntos ruido: {etiquetas.count(-1)}")
#--------------------------------------------------------------------------------------

def entrenar_modelo():
    # Aquí iría la lógica de entrenamiento del modelo
    pass
#--------------------------------------------------------------------------------------

def evaluar_modelo():
    # Aquí iría la lógica de evaluación del modelo
    pass
#--------------------------------------------------------------------------------------

def guardar_modelo():
    # Aquí iría la lógica de guardado del modelo entrenado
    pass
def cargar_modelo():
    # Aquí iría la lógica de carga del modelo entrenado
    pass
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

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

if __name__ == "__main__":
    tipo_vectorizacion ="robertuito" # "TfidF" o "robertuito"
    agrupar_train_test()
    preprocesar_datos(tipo_vectorizacion)

    # Vectorización con TfidF + PCA
    vectorizar_datos(tipo_vectorizacion)
    PCA_reduccion_dimensionalidad(tipo_vectorizacion, n_componentes=[100,200])

    # Clustering manual DBSCAN
    #clustering_DBSCAN(tipo_vectorizacion, eps=20, min_pts=5)


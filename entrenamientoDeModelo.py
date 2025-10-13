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
from sympy.physics.control.control_plots import plt
from transformers import AutoModel, AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
import math
import matplotlib.pyplot as plt
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
    num_mensajes_test = 0
    num_mensajes_train = 0
    # Recorremos las categorías (Ej: Ansiedad, Depresion, etc.)
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
                        for elemento in datos_json:
                            if "message" in elemento:
                                mensajes_train.append(elemento["message"])
                                num_mensajes_train += 1
        # --- TEST ---
        ruta_test = os.path.join(ruta_categoria, "Test")
        if os.path.exists(ruta_test):
            print(f"Procesando test en {ruta_test}")
            for archivo in os.listdir(ruta_test):
                if archivo.endswith(".json"):
                    ruta_archivo = os.path.join(ruta_test, archivo)
                    with open(ruta_archivo, "r", encoding="utf-8") as f:
                        datos_json = json.load(f)
                        for elemento in datos_json:
                            if "message" in elemento:
                                mensajes_test.append(elemento["message"])
                                num_mensajes_test += 1
    # Guardar los mensajes en dos archivos separados
    with open(salida_train, "w", encoding="utf-8") as f_train:
        json.dump(mensajes_train, f_train, ensure_ascii=False, indent=2)

    with open(salida_test, "w", encoding="utf-8") as f_test:
        json.dump(mensajes_test, f_test, ensure_ascii=False, indent=2)

    print(f"Total mensajes en train: {num_mensajes_train}")
    print(f"Total mensajes en test: {num_mensajes_test}")

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

def preprocesar_datos():
        # Cargar mensajes
    with open("train.json", "r", encoding="utf-8") as f:
        train_msgs = json.load(f)

    with open("test.json", "r", encoding="utf-8") as f:
        test_msgs = json.load(f)

    # Preprocesar
    train_pre = [limpiar_texto(m) for m in train_msgs]
    test_pre = [limpiar_texto(m) for m in test_msgs]

    # Guardar
    with open("train_preprocesado.json", "w", encoding="utf-8") as f:
        json.dump(train_pre, f, ensure_ascii=False, indent=2)

    with open("test_preprocesado.json", "w", encoding="utf-8") as f:
        json.dump(test_pre, f, ensure_ascii=False, indent=2)
    

#--------------------------------------------------------------------------------------

def vectorizar_TfidF():
    # Cargar datos preprocesados (listas de tokens)
    with open("train_preprocesado.json", "r", encoding="utf-8") as f:
        train_pre = json.load(f)

    with open("test_preprocesado.json", "r", encoding="utf-8") as f:
        test_pre = json.load(f)

    # Unir tokens para usar TfidfVectorizer (se espera texto, no listas)
    train_texts = [" ".join(tokens) for tokens in train_pre]
    test_texts = [" ".join(tokens) for tokens in test_pre]

    # Inicializar vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # puedes ajustar este número
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Guardar los vectores y el vectorizador
    np.save("X_train_tfidf.npy", X_train.toarray())
    np.save("X_test_tfidf.npy", X_test.toarray())

    vocab = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}

    with open("vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vectorización completada: {X_train.shape[0]} train docs, {X_train.shape[1]} features")



def generar_embeddings_robertuito(model_name="pysentimiento/robertuito-base-uncased", max_length=128):
    # Cargar datos preprocesados (tokens) o textos limpios
    with open("train_preprocesado.json", "r", encoding="utf-8") as f:
        train_pre = json.load(f)
    with open("test_preprocesado.json", "r", encoding="utf-8") as f:
        test_pre = json.load(f)

    # Reconstruir textos a partir de tokens
    train_texts = [" ".join(tokens) for tokens in train_pre]
    test_texts = [" ".join(tokens) for tokens in test_pre]

    # Cargar modelo y tokenizador RoBERTuito
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Para que trabaje con GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def texto_a_embedding(texto):
        # aplicar preprocesamiento especial de tweet
        texto_proc = preprocess_tweet(texto)
        # tokenizar
        enc = tokenizer(
            texto_proc,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        # hacer pase hacia adelante
        with torch.no_grad():
            out = model(**enc)
        # Extraer embedding: por ejemplo, usar la media de los tokens (pooling)
        last_hidden = out.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        # Promediar tokens (excluyendo padding) — simple estrategia
        mask = enc["attention_mask"].unsqueeze(-1)  # para propagar la máscara
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.sum(mask, dim=1)
        embedding = summed / counts  # forma: [hidden_dim]
        return embedding.cpu().numpy()

    # Generar embeddings para train y test
    X_train_emb = np.array([texto_a_embedding(t) for t in train_texts])
    X_test_emb = np.array([texto_a_embedding(t) for t in test_texts])

    # Guardar
    np.save("X_train_robertuito.npy", X_train_emb)
    np.save("X_test_robertuito.npy", X_test_emb)

    print("Embeddings con RoBERTuito generados:", X_train_emb.shape, X_test_emb.shape)

def vectorizar_datos(tipo_vectorizacion="TfidF"):
    if tipo_vectorizacion == "TfidF":
        vectorizar_TfidF()
    elif tipo_vectorizacion == "RoBERTuito":
        generar_embeddings_robertuito()
    else:
        raise ValueError("Tipo de vectorización no soportado.") 

#--------------------------------------------------------------------------------------

def PCA_reduccion_dimensionalidad(n_componentes=[50,100], tipo_vectorizacion="TfidF"):
    if tipo_vectorizacion == "TfidF":
        X_train_data = np.load("X_train_tfidf.npy")
        X_test_data = np.load("X_test_tfidf.npy")
    elif tipo_vectorizacion == "RoBERTuito":
        X_train_data = np.load("X_train_robertuito.npy")
        X_test_data = np.load("X_test_robertuito.npy")
    else:
        raise ValueError("Tipo de vectorización no soportado.")

        # Inicialización para guardar el mejor PCA
    mejor_score = -np.inf
    mejor_X_train = None
    mejor_X_test = None
    mejor_pca = None
    mejor_n = None


    # Iterar sobre distintos números de componentes
    for n in n_componentes:
        print(n)
        pca = PCA(n_components=n)
        X_train_reducido = pca.fit_transform(X_train_data)
        X_test_reducido = pca.transform(X_test_data)

        # Clustering DBSCAN manual
        etiquetas = dbscan(X_train_reducido, eps=0.08, min_pts=5)
        print(etiquetas)
        # Filtrar ruido para métricas
        mask = np.array(etiquetas) != -1
        if np.sum(mask) <= 1:
            print(f"PCA={n}: No se formaron clusters válidos.")
            continue

        # Métricas
        sil_score = silhouette_score(X_train_reducido[mask], np.array(etiquetas)[mask])
        ch_score = calinski_harabasz_score(X_train_reducido[mask], np.array(etiquetas)[mask])

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
    np.save("X_train_pca_best.npy", mejor_X_train)
    np.save("X_test_pca_best.npy", mejor_X_test)
    joblib.dump(mejor_pca, "modelo_pca_best.pkl")
    print(f"🏆 Mejor PCA: {mejor_n} componentes con score combinado={mejor_score:.3f}")

#--------------------------------------------------------------------------------------

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


def clustering_DBSCAN(tipo_vectorizacion="TfidF", eps=0.5, min_pts=5):
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

def visualizar_clusters(tipo_vectorizacion="TfidF"):
    X = np.load("X_train_pca_best.npy")
    labels = np.load(f"labels_{tipo_vectorizacion}_dbscan.npy")

    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=10)
    plt.title(f"Clusters DBSCAN - {tipo_vectorizacion}")
    plt.show()
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

if __name__ == "__main__":
    agrupar_train_test()
    preprocesar_datos()

    # Vectorización con TfidF + PCA
    vectorizar_datos(tipo_vectorizacion="TfidF")
    PCA_reduccion_dimensionalidad(tipo_vectorizacion="TfidF", n_componentes=[50,100])

    # Clustering manual DBSCAN
    clustering_DBSCAN(tipo_vectorizacion="TfidF", eps=0.08, min_pts=5)


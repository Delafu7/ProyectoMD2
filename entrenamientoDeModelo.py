import os
import json
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch

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

    # Guardar vocabulario por si necesitas usarlo más tarde
    with open("tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vectorizer.vocabulary_, f, ensure_ascii=False, indent=2)

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

def PCA_reduccion_dimensionalidad(n_componentes=50, tipo_vectorizacion="TfidF"):
    if tipo_vectorizacion == "TfidF":
        X_train = np.load("X_train_tfidf.npy")
        X_test = np.load("X_test_tfidf.npy")
    elif tipo_vectorizacion == "RoBERTuito":
        
        X_train_emb = np.load("X_train_embeddings.npy")
        X_test_emb = np.load("X_test_embeddings.npy")

    pca = PCA(n_components=n_componentes)
    X_train_reduc = pca.fit_transform(X_train_emb)
    X_test_reduc = pca.transform(X_test_emb)

    np.save("X_train_embeddings.npy", X_train_reduc)
    np.save("X_test_embeddings.npy", X_test_reduc)

    print(f"PCA aplicado sobre embeddings: {n_componentes} dimensiones finales")
#--------------------------------------------------------------------------------------

def clustering_DBSCAN():
    # Aquí iría la lógica de clustering DBSCAN
    pass
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
#--------------------------------------------------------------------------------------


if __name__ == "__main__":
    agrupar_train_test()
    preprocesar_datos()
    vectorizar_datos()
    PCA_reduccion_dimensionalidad()
    clustering_DBSCAN()
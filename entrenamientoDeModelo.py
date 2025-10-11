import os
import json
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

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

def vectorizar_datos():
    # Aquí iría la lógica de vectorización de datos
    pass
#--------------------------------------------------------------------------------------

def PCA_reduccion_dimensionalidad():
    # Aquí iría la lógica de reducción de dimensionalidad con PCA
    pass
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
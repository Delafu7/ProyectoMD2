import json
import os

carpeta = "MentalRiskEs2023/Ansiedad/Test"

todos_los_mensajes = []

for nombre_archivo in os.listdir(carpeta):
    if nombre_archivo.endswith(".json"):
        ruta = os.path.join(carpeta, nombre_archivo)
        print(nombre_archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        for i in datos:
            if "message" in i:
                print(i["message"])
                todos_los_mensajes.append(i["message"])

print(f"Se extrajeron {len(todos_los_mensajes)} message:")




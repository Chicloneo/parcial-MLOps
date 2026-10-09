import csv
from pathlib import Path
import joblib
from .contracts import Entrada, Salida
import pandas as pd


def leer_csv(ruta: Path) -> list[Entrada]:
    
    df = pd.read_csv(ruta)

    if df.shape[0] == 0:
        raise ValueError("El csv está vacío")

    filas_validadas = []
    for index, row in df.iterrows():
        try:
            fila_valida = Entrada(**row.to_dict())
            filas_validadas.append(fila_valida)
        except Exception as e:
            raise ValueError(f"Fallo en el contrato de entrada para la fila {index+2}")
    return filas_validadas


def preprocesar(entrada: Entrada) -> list[float]:
    # Preprocesa solo una fila (una Entrada)
    return [round(entrada.peso_kg, ndigits=1), entrada.distancia_km]


def cargar_modelo(ruta: Path):
    
    model = joblib.load(ruta)

    return model


def predecir(entrada: Entrada, modelo) -> Salida:
    #entrada es de tipo Entrada. Pero modelo quiere algo de la forma [[8,20]]
    input_modelo = preprocesar(entrada)
    model = cargar_modelo(modelo)
    categoria = str(model.predict([input_modelo])[0]) # esto devuelve o bien 'normal' o bien 'urgente'
    #categoria = 'normal'
    confianzas = model.predict_proba([input_modelo])[0] #devuelve una lista [1,0] - probabilidad de ser [normal, urgente]

    return Salida(
        id_paquete = entrada.id_paquete,
        categoria = categoria,
        confianza = max(confianzas)
    )


def guardar_csv(resultados: list[Salida], ruta: Path) -> None:

    filas_df = []
    for resultado in resultados:
        filas_df.append(resultado.model_dump())

    df = pd.DataFrame(filas_df)
    df.to_csv(ruta, index = False)


def ejecutar(entrada: Path, modelo: Path, salida: Path) -> None:

    lista_entradas = leer_csv(entrada)

    model = cargar_modelo(modelo)

    lista_salidas = []

    for entrada in lista_entradas:
        prediccion = predecir(entrada, modelo)
        lista_salidas.append(prediccion)

    guardar_csv(lista_salidas, salida)
    


if __name__ == "__main__":
    ejecutar(
        Path("data/raw/paquetes.csv"),
        Path("models/modelo.joblib"),
        Path("resultados.csv"),
    )

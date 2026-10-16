"""
generador/generar_datos.py
──────────────────────────
Genera los conjuntos sintéticos de entrenamiento y prueba en datos/.

La partición es POR ESTRUCTURA DE FRASE: la última estructura de cada
intención (Q5, D4, S4, C3, G3, A3) se reserva para prueba y nunca aparece en
entrenamiento. Así la prueba mide si el modelo reconoce la intención en una
redacción nueva, y no si memorizó una plantilla.

Uso (desde la raíz del repo):
    python generador/generar_datos.py
    python generador/generar_datos.py --n-por-clase 400 --seed 7
"""
import argparse
import csv
import random
from pathlib import Path

from textos import CATEGORIAS_DENUNCIA, ESTRUCTURAS, INTENCIONES, NEUTRAS, VOCAB, generar_texto

RAIZ = Path(__file__).resolve().parent.parent
CAMPOS = ["texto", "intencion", "categoria", "plantilla"]


def estructuras_de_prueba() -> set[str]:
    return {ESTRUCTURAS[i][-1][0] for i in INTENCIONES}


def generar(n_por_clase: int, estructuras: set[str], rnd: random.Random) -> list[dict]:
    """n_por_clase textos por intención, solo con las estructuras indicadas y sin duplicados."""
    filas, vistos = [], set()
    for intencion in INTENCIONES:
        permitidas = {e for e, _ in ESTRUCTURAS[intencion]} & estructuras
        categorias = CATEGORIAS_DENUNCIA if intencion == "denuncia" else list(VOCAB)
        for _ in range(n_por_clase):
            categoria = rnd.choice(categorias)
            hartazgo = intencion in NEUTRAS and rnd.random() < 0.15
            texto, plantilla = generar_texto(categoria, intencion, rnd, hartazgo, permitidas)
            if texto not in vistos:
                vistos.add(texto)
                filas.append({"texto": texto, "intencion": intencion,
                              "categoria": categoria, "plantilla": plantilla})
    rnd.shuffle(filas)
    return filas


def escribir(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"{ruta.relative_to(RAIZ)}: {len(filas)} textos")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-por-clase", type=int, default=300)
    ap.add_argument("--n-prueba-por-clase", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    prueba_ids = estructuras_de_prueba()
    todas = {e for i in INTENCIONES for e, _ in ESTRUCTURAS[i]}

    entrenamiento = generar(a.n_por_clase, todas - prueba_ids, rnd)
    prueba = generar(a.n_prueba_por_clase, prueba_ids, rnd)
    textos_prueba = {f["texto"] for f in prueba}
    entrenamiento = [f for f in entrenamiento if f["texto"] not in textos_prueba]

    escribir(RAIZ / "datos" / "entrenamiento.csv", entrenamiento)
    escribir(RAIZ / "datos" / "prueba_sintetica.csv", prueba)
    print(f"Estructuras de prueba (nunca vistas en entrenamiento): {sorted(prueba_ids)}")


if __name__ == "__main__":
    main()

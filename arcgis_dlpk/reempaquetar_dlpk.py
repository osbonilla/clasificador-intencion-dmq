"""
arcgis_dlpk/reempaquetar_dlpk.py
────────────────────────────────
Actualiza el .emd y la función de inferencia de un modelo ya entrenado
y vuelve a armar el .dlpk,
sin reentrenar. Úsalo si generaste el paquete con una versión anterior
del notebook (ArcGIS Pro lo trataba como un modelo zero-shot).

Uso (desde la raíz del repo):
    python arcgis_dlpk/reempaquetar_dlpk.py
"""
import json
import shutil
import zipfile
from pathlib import Path

NOMBRE = "ClasificadorIntencionDMQ"
SALIDAS = Path(__file__).resolve().parent.parent / "salidas"
carpeta = SALIDAS / NOMBRE
ruta_emd = carpeta / f"{NOMBRE}.emd"
shutil.copy(Path(__file__).resolve().parent / f"{NOMBRE}.py", carpeta)   # función de inferencia actualizada

emd = json.loads(ruta_emd.read_text(encoding="utf-8"))
etiquetas = emd["Labels"]
emd.update({
    "ModelName": "TextClassifier",       # ArcGIS Pro elige la clase de arcgis.learn por este nombre
    "ModelType": "TextClassifier",
    "Label2Id": {e: i for i, e in enumerate(etiquetas)},
    "TextColumns": ["input_str"],
    "LabelColumns": ["ClassLabel"],
    "IsMultilabelClassificationProblem": False,
    "SequenceLength": emd.get("MaxLength", 128),
    "PretrainedModel": "",
})
ruta_emd.write_text(json.dumps(emd, indent=2, ensure_ascii=False), encoding="utf-8")

dlpk = SALIDAS / f"{NOMBRE}.dlpk"
with zipfile.ZipFile(dlpk, "w", zipfile.ZIP_DEFLATED) as z:
    for archivo in sorted(carpeta.rglob("*")):
        if archivo.is_file():
            z.write(archivo, archivo.relative_to(carpeta))     # contenido en la raíz del ZIP
print(f"Listo: {dlpk} ({dlpk.stat().st_size / 1e6:.0f} MB)")
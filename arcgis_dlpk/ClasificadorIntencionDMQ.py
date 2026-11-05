"""
arcgis_dlpk/ClasificadorIntencionDMQ.py
───────────────────────────────────────
Función de inferencia del paquete de deep learning (.dlpk) para ArcGIS.

Sigue el contrato de Esri para modelos de lenguaje de terceros
("Use third-party language models with ArcGIS"): una clase con los métodos
__init__, initialize, getParameterInfo, getConfiguration y predict, que
ArcGIS Pro llama desde la herramienta *Classify Text Using Deep Learning*
(caja de herramientas GeoAI) y que arcgis.learn carga con
TextClassifier.from_model().

El paquete contiene:
    ClasificadorIntencionDMQ/
    ├── ClasificadorIntencionDMQ.emd   definición del modelo (JSON)
    ├── ClasificadorIntencionDMQ.py    este archivo
    └── modelo/                        BERT ajustado (save_pretrained)

Solo usa torch y transformers, que vienen con las librerías de deep learning
de ArcGIS Pro (Deep Learning Libraries Installer).
"""
import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

# Registro de diagnóstico: %TEMP%\dmq_debug.txt
LOG = os.path.join(tempfile.gettempdir(), "dmq_debug.txt")


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
    except Exception:
        pass
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

try:
    from arcgis.features import FeatureSet
except ImportError:  # fuera de ArcGIS (pruebas locales): predict devuelve el diccionario
    FeatureSet = None


class ClasificadorIntencionDMQ:
    """Clasifica la intención de un reporte ciudadano a partir de su texto."""

    def __init__(self, **kwargs):
        self.name = "Clasificador de intención de reportes ciudadanos (DMQ)"
        self.description = ("BERT en español ajustado para clasificar reportes en: "
                            "queja, denuncia, solicitud, consulta, sugerencia, agradecimiento.")
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.max_length = 128
        self.batch_size = 8
        self.output_field = "ClassLabel"

    # ── Ciclo de vida que define ArcGIS ────────────────────
    def initialize(self, **kwargs):
        """Carga el modelo. kwargs['model'] es la ruta al archivo .emd."""
        log(f"initialize kwargs={ {k: str(v) for k, v in kwargs.items()} }")
        try:
            self._initialize(**kwargs)
            log(f"modelo cargado en {self.device}")
        except Exception:
            log("ERROR en initialize:\n" + traceback.format_exc())
            raise

    def _initialize(self, **kwargs):
        emd = Path(kwargs["model"])
        info = json.loads(emd.read_text(encoding="utf-8"))
        carpeta_modelo = emd.parent / info["ModelFile"]   # ruta relativa al .emd

        self.max_length = int(info.get("MaxLength", self.max_length))
        # La herramienta de ArcGIS Pro busca la clase en el campo OutputField del .emd
        # En minúsculas: la herramienta une el resultado a la tabla y luego hace
        # AlterField(tabla, OutputField.lower(), ...); si el campo no está en minúsculas
        # falla con ERROR 000728 "Field classlabel does not exist within table".
        self.output_field = info.get("OutputField", self.output_field).lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # PreTrainedTokenizerFast lee tokenizer.json directamente: funciona igual si el modelo se
        # guardó con transformers 5.x y ArcGIS Pro trae transformers 4.x
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(carpeta_modelo)
        self.model = AutoModelForSequenceClassification.from_pretrained(carpeta_modelo)
        self.model.to(self.device).eval()

    def getParameterInfo(self):
        """Argumentos del modelo que ArcGIS Pro muestra en la herramienta."""
        return [{
            "name": "max_length",
            "dataType": "string",
            "required": False,
            "displayName": "Longitud máxima (tokens)",
            "description": "Los textos más largos se truncan. Por defecto, 128.",
            "value": str(self.max_length),
        }]

    def getConfiguration(self, **kwargs):
        """Recibe los valores elegidos por el usuario antes de predecir."""
        log(f"getConfiguration kwargs={ {k: str(v) for k, v in kwargs.items()} }")
        self.max_length = int(kwargs.get("max_length") or self.max_length)
        self.batch_size = int(kwargs.get("batch_size") or self.batch_size)
        kwargs["batch_size"] = self.batch_size
        return kwargs

    def predict(self, feature_set, **kwargs):
        """Clasifica el campo de texto indicado y devuelve un FeatureSet con dos campos:
        OutputField del .emd en minúsculas (classlabel) con la intención, y confianza.

        La herramienta (Pro 3.7) une estos campos a la tabla con JoinField y luego renombra
        classlabel al nombre que el usuario puso en Class Label Field. Ese nombre debe ser
        distinto de classlabel/ClassLabel (p. ej. Intencion): al final la herramienta borra
        el campo classlabel, y si se llaman igual se borra el resultado."""
        try:
            df = feature_set.df
            primera = not getattr(self, "_ya_registrado", False)
            self._ya_registrado = True
            if primera:
                log(f"predict kwargs={ {k: str(v) for k, v in kwargs.items()} } "
                    f"columnas={list(df.columns)} filas={len(df)}")
            campo = kwargs.get("input_field")
            if campo not in df.columns:
                # respaldo: input_str o la primera columna de texto
                campo = "input_str" if "input_str" in df.columns else \
                    next(c for c in df.columns if df[c].dtype == object)
            textos = ["" if pd.isna(t) else str(t) for t in df[campo].tolist()]
            preds = [(c, round(p, 4)) if t.strip() else ("", None)
                     for (c, p), t in zip(self.clasificar(textos), textos)]
        except Exception:
            log("ERROR en predict:\n" + traceback.format_exc())
            raise

        # La herramienta une TODOS los campos devueltos a la tabla (por OBJECTID, en orden),
        # así que no se devuelve input_str: solo quedaría como una copia del texto.
        feature_dict = {
            "fields": [
                {"name": self.output_field, "type": "esriFieldTypeString"},
                {"name": "confianza", "type": "esriFieldTypeDouble"},
            ],
            "geometryType": "",
            "features": [{"attributes": {self.output_field: c, "confianza": p}} for c, p in preds],
        }
        if primera:
            log(f"predict devuelve campos={[f['name'] for f in feature_dict['fields']]} "
                f"ejemplo={feature_dict['features'][:1]}")
        return FeatureSet.from_dict(feature_dict) if FeatureSet else feature_dict

    # ── Inferencia ─────────────────────────────────────────
    @torch.no_grad()
    def clasificar(self, textos):
        """Devuelve [(clase, probabilidad), ...] en el mismo orden que los textos."""
        resultados = []
        for i in range(0, len(textos), self.batch_size):
            lote = self.tokenizer(textos[i:i + self.batch_size], truncation=True, padding=True,
                                  max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**lote).logits, dim=-1).cpu()
            conf, ids = probs.max(dim=-1)
            resultados += [(self.model.config.id2label[int(j)], float(p)) for j, p in zip(ids, conf)]
        return resultados
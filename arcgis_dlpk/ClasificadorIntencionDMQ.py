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
from pathlib import Path

import torch
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

    # ── Ciclo de vida que define ArcGIS ────────────────────
    def initialize(self, **kwargs):
        """Carga el modelo. kwargs['model'] es la ruta al archivo .emd."""
        emd = Path(kwargs["model"])
        info = json.loads(emd.read_text(encoding="utf-8"))
        carpeta_modelo = emd.parent / info["ModelFile"]   # ruta relativa al .emd

        self.max_length = int(info.get("MaxLength", self.max_length))
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
        self.max_length = int(kwargs.get("max_length") or self.max_length)
        self.batch_size = int(kwargs.get("batch_size") or self.batch_size)
        kwargs["batch_size"] = self.batch_size
        return kwargs

    def predict(self, feature_set, **kwargs):
        """Clasifica el campo de texto indicado y devuelve un FeatureSet con la clase."""
        campo = kwargs["input_field"]
        textos = feature_set.df[campo].fillna("").astype(str).tolist()
        clases = [clase for clase, _ in self.clasificar(textos)]

        feature_dict = {
            "fields": [
                {"name": "input_str", "type": "esriFieldTypeString"},
                {"name": "class", "type": "esriFieldTypeString"},
            ],
            "geometryType": "",
            "features": [{"attributes": {"input_str": t, "class": c}} for t, c in zip(textos, clases)],
        }
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
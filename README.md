# litreview · Motor de Análisis Bibliográfico y Estado del Arte (SOTA)

Motor analítico para **Revisión Sistemática de Literatura (SLR)** asistida por Machine Learning:
- Ingesta de colecciones curadas desde **Zotero** o archivos **CSV locales**.
- Descubrimiento de tópicos no supervisados mediante **BERTopic** (SentenceTransformers + UMAP + HDBSCAN + c-TF-IDF).
- Validación y clasificación taxonómica mediante **Zero-Shot NLI** (DeBERTa-v3 / BART).
- Detección cuantitativa de vacíos de investigación (**Gap Analysis**).
- Exportación estructurada en **JSON** para agentes de IA (**`sdd-sota`** en `gentle-pi`) y gráficos para papers científicos.

---

## 🔐 Configuración de Seguridad: Zotero API Key

Para conectar tu biblioteca de Zotero aplicando el **Principio de Mínimo Privilegio (*Least Privilege*)**:

1. Ingresá a [zotero.org/settings/keys/new](https://www.zotero.org/settings/keys/new).
2. Configurá los permisos estrictamente de la siguiente manera:
   - **Key Description**: `pi-sdd-sota` *(o el nombre que prefieras)*.
   - **Personal Library**:
     - `[X] Allow library access` ➔ **MARCADO (Checked)**. *(Lectura de papers y abstracts)*.
     - `[ ] Allow notes access` ➔ **DESMARCADO (Unchecked)**. *(No requerido; el abstract es metadato del paper)*.
     - `[ ] Allow write access` ➔ **DESMARCADO (Unchecked)**. *(Estricto: el motor nunca modifica tu biblioteca)*.
   - **Default Group Permissions**:
     - `None` (si la colección es personal) o `Read Only` (si es un grupo compartido con colegas). **Nunca Read/Write**.
3. Guardá la clave y anotá dos valores:
   - **User ID**: El número de 7 u 8 dígitos que aparece arriba (*"Your userID for use in API calls is XXXXXXX"*).
   - **API Key**: El token alfanumérico generado.

Configurá tu archivo `.env` en la raíz del proyecto:
```bash
ZOTERO_LIBRARY_ID="1234567"
ZOTERO_API_KEY="tu_token_alfanumerico"
ZOTERO_LIBRARY_TYPE="user" # o "group"
```

---

## 🚀 Instalación

Este proyecto utiliza [`uv`](https://docs.astral.sh/uv/) como gestor de entornos y paquetes de Python (requiere Python >= 3.12).

```bash
cd tools/litreview
uv sync              # Resuelve dependencias y crea entorno aislado
uv sync --extra cpu  # Modo CPU estándar
# O con aceleración CUDA (Nvidia GPU):
# uv sync --extra cu126
```

---

## 🤖 Uso con Agentes de IA (Modo Headless / SDD)

El punto de entrada optimizado para agentes como `sota-analyst` es **`litreview-agent-summary`**:

```bash
# Ejecutar sobre una colección curada en Zotero:
uv run litreview-agent-summary --config config.yaml --collection "Pneumonia-CXR" --output-json results/summary.json

# Ejecutar con un CSV local:
uv run litreview-agent-summary --config config.yaml --input-csv data/fixtures/mock_papers.csv --output-json results/summary.json

# Inyectar etiquetas candidatas dinámicas generadas por el agente:
uv run litreview-agent-summary --config config.yaml --labels-json '{"CNN": "convolutional network", "VIT": "vision transformer"}'
```

### Esquema del JSON de Salida (`sdd-research-summary.json`)

El agente recibe un JSON con la siguiente estructura lista para alimentar fases de SDD (`proposal` y `design`):

```json
{
  "status": "success",
  "corpus": {
    "total_papers": 24,
    "papers_with_abstracts": 24,
    "year_range": [2021, 2024],
    "item_types": { "journalArticle": 20, "conferencePaper": 4 }
  },
  "topics": {
    "num_topics": 3,
    "outlier_count": 2,
    "topic_sizes": { "0": 12, "1": 7, "2": 3 },
    "top_words": { "0": ["transformer", "vision", "cxr"] }
  },
  "taxonomy_validation": {
    "total_classified": 22,
    "mean_confidence": 0.7842,
    "label_counts": { "CXR": 18, "VIT": 10, "EXT": 2 }
  },
  "gap_analysis": {
    "gaps": [
      {
        "type": "seed_few_matches",
        "seed": "EXT",
        "papers": 2,
        "severity": "medium"
      }
    ],
    "num_gaps": 1
  }
}
```

---

## 📊 Uso Tradicional y Generación de Gráficos

```bash
# Correr análisis completo y generar 15+ gráficos para papers:
uv run litreview-analysis --config config.yaml --plots results/plots/

# Solo generar/actualizar gráficos:
uv run litreview-plots --config config.yaml
```

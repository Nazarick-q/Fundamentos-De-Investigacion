# Fundamentos de Investigación

Pipeline de procesamiento de lenguaje natural (NLP) para registros textuales de Data UdeA. El flujo anonimiza los identificadores estudiantiles, lematiza el texto y luego mapea cada registro semánticamente contra una matriz conceptual de la **CIF-NJ** (Clasificación Internacional del Funcionamiento, adaptada a educación superior).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Script1Procesamiento.py` | Preprocesamiento: anonimiza IDs (SHA-256), limpia y lematiza el texto con spaCy. |
| `Script2Mapeo.py` | Mapeo semántico: genera embeddings (Sentence-BERT) y asigna a cada registro el código CIF-NJ con mayor similitud del coseno. |
| `Script3Mapeo.py` | Archivo vacío por ahora (reservado para una siguiente etapa). |

## Flujo de ejecución

```
Script1Procesamiento.py                    Script2Mapeo.py
data_udea_raw.csv  ──►  data_udea_sanitizado.csv  ──►  data_udea_mapeado_cif.csv
```

## Requisitos

- Python 3.9 o superior
- Conexión a internet en la primera ejecución (se descargan modelos de spaCy y Hugging Face)
- Espacio en disco para los modelos (el de spaCy `lg` y el de Sentence-BERT pesan varios cientos de MB)

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/Nazarick-q/Fundamentos-De-Investigacion.git
   cd Fundamentos-De-Investigacion
   ```

2. (Recomendado) Crea y activa un entorno virtual:

   ```bash
   python -m venv venv

   # Linux / macOS
   source venv/bin/activate

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:

   ```bash
   pip install pandas numpy scikit-learn spacy sentence-transformers
   ```

4. Descarga el modelo de spaCy en español:

   ```bash
   python -m spacy download es_core_news_lg
   ```

   > Si el modelo `lg` no está instalado, `Script1Procesamiento.py` intenta usar `es_core_news_sm`, pero ese también debe estar instalado (`python -m spacy download es_core_news_sm`); de lo contrario el script falla.

## Uso

Ejecuta los scripts **en orden**, desde la carpeta raíz del repositorio.

### 1. Preprocesamiento

```bash
python Script1Procesamiento.py
```

Qué hace:

- Crea `data_udea_raw.csv` con 5 registros de ejemplo.
- Reemplaza `ID_Estudiante` por `ID_Estudiante_hash` (SHA-256 truncado a 16 caracteres).
- Limpia el texto de `Registro_Texto`, elimina stop-words y lo lematiza en la columna `Texto_Lematizado`.
- Guarda el resultado en `data_udea_sanitizado.csv`.

### 2. Mapeo semántico a CIF-NJ

```bash
python Script2Mapeo.py
```

Qué hace:

- Carga `data_udea_sanitizado.csv`.
- Descarga (solo la primera vez) y carga el modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Calcula la similitud del coseno entre cada registro y los 11 códigos de la matriz CIF-NJ definida en el script.
- Guarda `data_udea_mapeado_cif.csv` con las columnas adicionales `Código_CIF`, `Componente_CIF`, `Dominio_CIF` y `Similitud_Score`.

## Usar tus propios datos

`Script1Procesamiento.py` genera datos de ejemplo al ejecutarse como programa principal. Para procesar un CSV propio:

1. Asegúrate de que tenga, como mínimo, las columnas `ID_Estudiante` y `Registro_Texto` (más cualquier otra, como `Sede`, que se conservará).
2. Edita el bloque `if __name__ == "__main__":` de `Script1Procesamiento.py` para que lea tu archivo en lugar de crear `datos_ejemplo`, o importa la función directamente:

   ```python
   from Script1Procesamiento import procesar_corpus_data_udea

   df = procesar_corpus_data_udea("mi_archivo.csv")
   df.to_csv("data_udea_sanitizado.csv", index=False)
   ```

3. Ejecuta `Script2Mapeo.py` como se indicó arriba.

## Archivos generados

| Archivo | Generado por | Contenido |
|---|---|---|
| `data_udea_raw.csv` | Script 1 | Datos de ejemplo sin procesar |
| `data_udea_sanitizado.csv` | Script 1 | IDs anonimizados + texto lematizado |
| `data_udea_mapeado_cif.csv` | Script 2 | Registros con su código CIF-NJ y puntaje de similitud |

## Solución de problemas

- **`OSError: [E050] Can't find model 'es_core_news_lg'`**: instala el modelo con `python -m spacy download es_core_news_lg`.
- **`FileNotFoundError: data_udea_sanitizado.csv`**: ejecuta primero `Script1Procesamiento.py`.
- **Error al descargar el modelo de Sentence-BERT**: verifica tu conexión a internet; el modelo se guarda en caché y no se vuelve a descargar.
- **Primera ejecución lenta**: es normal, se están cargando los modelos de spaCy y de embeddings.

## Nota sobre datos sensibles

Los registros pueden contener información de salud y de discapacidad. No subas al repositorio archivos CSV con datos reales de estudiantes; considera agregar `*.csv` a tu `.gitignore`.

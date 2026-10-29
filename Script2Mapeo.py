import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo de embeddings denso en español / multilingüe
print("Cargando modelo de embeddings (Sentence-BERT)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Definición formal de la Matriz Conceptual CIF-NJ instrumentada a Educación Superior
MATRIZ_CIF_NJ = [
    {
        "codigo": "b140",
        "componente": "Funciones Corporales (b)",
        "dominio": "Atención y concentración en aula",
        "descripcion": "funciones mentales de atención concentración capacidad fisiológica psicológica déficit atención"
    },
    {
        "codigo": "b167",
        "componente": "Funciones Corporales (b)",
        "dominio": "Funciones del lenguaje",
        "descripcion": "funciones del lenguaje comunicación procesamiento del habla dislexia afasia"
    },
    {
        "codigo": "b210",
        "componente": "Funciones Corporales (b)",
        "dominio": "Funciones visuales y sensoriales",
        "descripcion": "agudeza visual ceguera baja visión percepción sensorial médica"
    },
    {
        "codigo": "s110",
        "componente": "Estructuras Corporales (s)",
        "dominio": "Sistema nervioso central",
        "descripcion": "estructura anatomía sistema nervioso central lesión cerebral neurología"
    },
    {
        "codigo": "s730",
        "componente": "Estructuras Corporales (s)",
        "dominio": "Estructuras de extremidades",
        "descripcion": "estructura corporal extremidades brazos piernas limitación motora amputación"
    },
    {
        "codigo": "d160",
        "componente": "Actividades y Participación (d)",
        "dominio": "Focalización de la atención en clase",
        "descripcion": "focalizar atención en clase dinámicas de aula desempeño académico tarea escolar"
    },
    {
        "codigo": "d175",
        "componente": "Actividades y Participación (d)",
        "dominio": "Lectoescritura académica",
        "descripcion": "lectoescritura académica lectura escritura pruebas exámenes evaluación tiempo adicional"
    },
    {
        "codigo": "d845",
        "componente": "Actividades y Participación (d)",
        "dominio": "Evaluaciones y vida universitaria",
        "descripcion": "desempeño en exámenes evaluaciones universitarias participación académica en campus"
    },
    {
        "codigo": "e115",
        "componente": "Factores Ambientales (e)",
        "dominio": "Infraestructura física del campus",
        "descripcion": "infraestructura física del campus rampas ascensores bloques accesibilidad urbanística"
    },
    {
        "codigo": "e130",
        "componente": "Factores Ambientales (e)",
        "dominio": "Tecnologías de asistencia",
        "descripcion": "tecnologías de asistencia lectores de pantalla JAWS NVDA software accesible Braille LSC"
    },
    {
        "codigo": "e330",
        "componente": "Factores Ambientales (e)",
        "dominio": "Apoyo y actitud pedagógica docente",
        "descripcion": "apoyo y actitud pedagógica docente diseño universal para el aprendizaje DUA ajustes razonables"
    }
]

def ejecutar_mapeo_semantico(filepath_sanitizado: str) -> pd.DataFrame:
    """Mapea semánticamente los registros de Data UdeA contra los códigos de la matriz CIF-NJ."""
    df = pd.read_csv(filepath_sanitizado)
    df_cif = pd.DataFrame(MATRIZ_CIF_NJ)
    
    # 1. Generar embeddings vectoriales para la matriz CIF-NJ
    embeddings_cif = model.encode(df_cif['descripcion'].tolist(), show_progress_bar=False)
    
    # 2. Generar embeddings vectoriales para los registros de Data UdeA
    registros_text = df['Texto_Lematizado'].fillna("").tolist()
    embeddings_udea = model.encode(registros_text, show_progress_bar=False)
    
    # 3. Calcular matriz de similitud del coseno
    matriz_similitud = cosine_similarity(embeddings_udea, embeddings_cif)
    
    # 4. Asignar el código CIF-NJ con mayor score de similitud
    codigos_asignados = []
    componentes_asignados = []
    dominios_asignados = []
    scores = []
    
    for idx_registro in range(len(embeddings_udea)):
        best_match_idx = np.argmax(matriz_similitud[idx_registro])
        best_score = matriz_similitud[idx_registro][best_match_idx]
        
        cif_match = MATRIZ_CIF_NJ[best_match_idx]
        codigos_asignados.append(cif_match['codigo'])
        componentes_asignados.append(cif_match['componente'])
        dominios_asignados.append(cif_match['dominio'])
        scores.append(round(float(best_score), 4))
        
    df['Código_CIF'] = codigos_asignados
    df['Componente_CIF'] = componentes_asignados
    df['Dominio_CIF'] = dominios_asignados
    df['Similitud_Score'] = scores
    
    return df

if __name__ == "__main__":
    df_mapeado = ejecutar_mapeo_semantico("data_udea_sanitizado.csv")
    df_mapeado.to_csv("data_udea_mapeado_cif.csv", index=False)
    print("Mapeo semántico completado.")
    print(df_mapeado[['ID_Estudiante_hash', 'Código_CIF', 'Componente_CIF', 'Similitud_Score']])
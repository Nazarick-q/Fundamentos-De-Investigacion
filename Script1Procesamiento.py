import pandas as pd
import spacy
import hashlib
import re

# Cargar el modelo en español de spaCy para procesamiento de lenguaje natural
try:
    nlp = spacy.load("es_core_news_lg")
except OSError:
    print("Modelo 'es_core_news_lg' no encontrado. Asegúrate de instalarlo con: python -m spacy download es_core_news_lg")
    nlp = spacy.load("es_core_news_sm")

def anonimizar_id(estudiante_id: str) -> str:
    """Genera un hash SHA-256 para anonimizar los identificadores estudiantiles (Habeas Data)."""
    return hashlib.sha256(str(estudiante_id).encode('utf-8')).hexdigest()[:16]

def limpiar_y_lematizar(texto: str) -> str:
    """Realiza la limpieza sintáctica, eliminación de stop-words y lematización de registros institucionales."""
    if not isinstance(texto, str) or not texto.strip():
        return ""
    
    # Normalización: minúsculas y remoción de caracteres especiales/números sueltos
    texto_limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', texto.lower())
    doc = nlp(texto_limpio)
    
    # Filtrar stop words, puntuación y tokens cortos; conservar lemas
    tokens_lematizados = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
    ]
    
    return " ".join(tokens_lematizados)

def procesar_corpus_data_udea(filepath_csv: str) -> pd.DataFrame:
    """Carga y procesa el dataset sanitizado de Data UdeA."""
    df = pd.read_csv(filepath_csv)
    
    # 1. Anonimización de identificadores
    if 'ID_Estudiante' in df.columns:
        df['ID_Estudiante_hash'] = df['ID_Estudiante'].apply(anonimizar_id)
        df.drop(columns=['ID_Estudiante'], inplace=True)
        
    # 2. Lematización de registros textuales (historias clínicas, informes, solicitudes)
    if 'Registro_Texto' in df.columns:
        print("Procesando y lematizando texto no estructurado...")
        df['Texto_Lematizado'] = df['Registro_Texto'].apply(limpiar_y_lematizar)
        
    return df

if __name__ == "__main__":
    # Datos de prueba representativos del Tablero de Data UdeA
    datos_ejemplo = {
        'ID_Estudiante': ['1001', '1002', '1003', '1004', '1005'],
        'Sede': ['MEDELLIN', 'CARMENVI', 'SONSON', 'AMALFI', 'MEDELLIN'],
        'Registro_Texto': [
            "Diagnóstico clínico de déficit de atención e hiperactividad leve según historia médica EPS.",
            "Solicitud de tiempo adicional en exámenes de lectoescritura y adaptaciones de aula.",
            "Requiere uso constante de lector de pantalla JAWS y software accesible en sala de cómputo.",
            "Certificado médico de limitación motora en extremidades inferiores y amputación parcial.",
            "Apoyo socioeconómico de transporte y asistencia alimentaria en campus."
        ]
    }
    
    df_raw = pd.DataFrame(datos_ejemplo)
    df_raw.to_csv("data_udea_raw.csv", index=False)
    
    df_procesado = procesar_corpus_data_udea("data_udea_raw.csv")
    df_procesado.to_csv("data_udea_sanitizado.csv", index=False)
    print("Preprocesamiento NLP completado exitosamente.")
    print(df_procesado[['ID_Estudiante_hash', 'Sede', 'Texto_Lematizado']])
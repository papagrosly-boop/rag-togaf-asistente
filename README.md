# 🎓 Ragi — Agente RAG con herramientas sobre TOGAF 9.1

Asistente que responde preguntas y realiza tareas sobre documentos de TOGAF 9.1. Convierte el RAG del Avance 2 (pipeline fijo: siempre busca y responde) en un **agente**: un LLM decide cuándo consultar la base de conocimientos, cuándo usar una herramienta de tarea y cuándo buscar información pública en la web.

- **Aplicación desplegada:** `PEGAR_URL_DE_STREAMLIT`
- **Repositorio:** https://github.com/papagrosly-boop/rag-togaf-asistente

## Principio de privacidad
Los PDFs, el índice vectorial (`chroma_db_multi/`) y los embeddings **nunca salen del sistema**. Al LLM (Groq) solo se le envían la pregunta, los últimos mensajes de la conversación y los fragmentos recuperados necesarios (máximo 4 de 600 caracteres por búsqueda), nunca el corpus completo. Al buscador web (Tavily) solo sale la consulta, recortada a 200 caracteres.

## Arquitectura

![Arquitectura del agente](docs/diagrama_arquitectura.png)

Flujo: pregunta → LLM decide → tool → resultado → LLM → respuesta con citas.

| Archivo | Función |
|---|---|
| `app.py` | Interfaz de chat en Streamlit: historial, tools usadas (con argumentos) y fuentes separadas base/web |
| `agent.py` | Bucle del agente, prompt del sistema y registro de uso |
| `tools.py` | Las 4 tools y sus límites |
| `config.py` | Modelo, rutas, embeddings y frase exacta de "sin información" |
| `ingest.py` | Ingesta de PDFs → chunks → índice Chroma (embeddings multilingües) |
| `medir_base.py` / `medir_agente.py` | Corren el dataset con el pipeline del Avance 2 / con el agente (llamadas, tokens, tiempo) |
| `evaluate_ragas.py` | Evaluación con Ragas (4 métricas) |
| `casos_comportamiento.py` | Los 7 casos de comportamiento |
| `probar_tools.py` | Prueba las tools con `.invoke()` sin LLM |

## Tools del agente y sus límites
Los límites están **escritos en el código** (`tools.py`), no en el prompt.

| Tool | Tipo | Qué hace | Límites en código |
|---|---|---|---|
| `buscar_en_base(consulta)` | Búsqueda | Busca en los 6 documentos TOGAF indexados y devuelve fragmentos con `[Fuente: documento, pág. N]` | `MAX_FRAGMENTOS = 4`, `MAX_CARACTERES = 600` por fragmento |
| `consultar_adm(fase, operacion)` | Tarea del dominio | Consulta exacta sobre el orden del ciclo ADM: fase siguiente, anterior, cuántas faltan, posición | Tabla fija de 9 fases en el código; fase desconocida → mensaje de error |
| `generar_referencia_apa(documento)` | Tarea del dominio | Genera la referencia APA 7 de un documento de la base | Solo los 6 documentos del diccionario |
| `buscar_web(consulta)` | Web | Busca información pública de TOGAF en sitios oficiales (ver política) | `MAX_WEB = 3`, extracto `MAX_EXTRACTO_WEB = 300`, consulta `MAX_CONSULTA_WEB = 200`, `SITIOS_PERMITIDOS` fijo |

## Bucle del agente y registro
- **Bucle manual** (`Agente.responder` en `agent.py`): el LLM decide si llama una tool; el programa la ejecuta y le devuelve el resultado.
- **Límite de iteraciones:** `MAX_ITER = 4` llamadas al LLM con tools por pregunta; si se agota, se fuerza una respuesta final sin tools.
- Un saludo se responde **sin** tools. Las preguntas con dos temas pueden llamar varias veces `buscar_en_base`.
- **Registro:** `logs/uso_tools.jsonl`, una línea por pregunta con fecha, pregunta, tools llamadas con argumentos, llamadas al LLM, tokens y tiempo.
- **Robustez:** reintenta si el modelo emite una tool call mal formada o la escribe como texto, y espera y reintenta ante el tope por minuto de la API.

## Política de búsqueda web ("primero la base")
- **Cuándo se usa:** solo si `buscar_en_base` no tiene la respuesta **y** la pregunta trata de información pública o actual de TOGAF (por ejemplo, fases del ADM que la base no cubre). La regla está en el docstring de la tool y en el prompt. Nunca se usa si la base ya respondió.
- **Sitios permitidos** (fijados en `tools.py`, el LLM no puede cambiarlos): `opengroup.org` y `pubs.opengroup.org`. The Open Group es el propietario y publicador del estándar TOGAF; son las fuentes oficiales del dominio. Además, la tool verifica el dominio de cada resultado antes de devolverlo.
- **Resultados acotados:** máximo 3, cada uno con título, URL y extracto corto, marcado `[web: URL]`.
- **Respuesta separada por origen:** `Fuente: documento, pág.` para la base y `[web: URL]` para la web; la interfaz las lista por separado.
- **Qué sale del sistema:** solo la consulta (≤200 caracteres). Nunca texto de los documentos ni datos del usuario.

## Base de conocimientos
6 extractos de TOGAF 9.1 (Introducción, Fase Preliminar, Fase A, Fase B, Fase C – Arquitectura de Información/Datos y Fase D), 72 páginas en total, indexados en 230 chunks (1000 caracteres, solape 200) con embeddings multilingües `intfloat/multilingual-e5-small`. Los documentos están en inglés; el agente consulta en inglés o español y responde en español.

## Evaluación (resumen)
Mismo conjunto de 15 preguntas con referencia, mismo juez (`gpt-oss-20b`), sobre el pipeline fijo del Avance 2 y sobre el agente. Ver tablas completas, los 7 casos de comportamiento y la comparación de costo en el PDF de la entrega.

| Métrica | Avance 2 (pipeline fijo) | Avance 3 (agente) |
|---|---|---|
| faithfulness | 0.754 | `COMPLETAR` |
| answer_relevancy | 0.655 | `COMPLETAR` |
| context_precision | 0.433 | `COMPLETAR` |
| context_recall | 0.333 | `COMPLETAR` |

## Instalación y ejecución
Requiere **Python 3.12**.
```bash
python -m venv env
# Windows: .\env\Scripts\activate      Linux/macOS: source env/bin/activate
pip install -r requirements.txt
```
Crea un archivo `.env` (no se sube al repositorio):
```
GROQ_API_KEY=tu_clave_de_groq
TAVILY_API_KEY=tu_clave_de_tavily
```
Indexar (solo la primera vez o si cambian los PDFs; la primera ejecución descarga el modelo de embeddings, ~470 MB):
```bash
python ingest.py
streamlit run app.py
```
Pruebas y evaluación:
```bash
python probar_tools.py                         # tools sin LLM
python casos_comportamiento.py 15              # 7 casos de comportamiento
python medir_agente.py agente_groq.json 15     # corre el dataset con el agente
pip install -r requirements-eval.txt
python evaluate_ragas.py agente_groq.json ragas_agente.csv
```
El modelo se puede cambiar sin editar código: `LLM_MODEL=openai/gpt-oss-120b` (por defecto) o `qwen/qwen3.8-27b`. El juez de Ragas se define con `EVAL_MODEL`.

## Despliegue en Streamlit Community Cloud
1. Sube el repositorio a GitHub **incluyendo** `chroma_db_multi/` y **sin** `.env`.
2. En https://share.streamlit.io crea la app con el repositorio, rama `main` y archivo principal `app.py`. En *Advanced settings* elige **Python 3.12**.
3. En *Settings → Secrets* agrega:
```toml
GROQ_API_KEY = "..."
TAVILY_API_KEY = "..."
```
Las claves se gestionan solo con variables de entorno o secrets, nunca en el repositorio.

## Limitaciones conocidas
- Los PDFs son extractos de evaluación; el texto extraído trae espacios rotos ("Deter mine") que afectan algo la calidad de la recuperación.
- La búsqueda web admite todo `opengroup.org`, que incluye páginas de borradores de revisión además de la documentación oficial.
- El agente cuesta más que el pipeline fijo (más llamadas, tokens y tiempo) y a veces decide mal (ver reflexión en el PDF).
- El plan gratuito de Groq tiene topes por minuto y por día que pueden limitar el uso intensivo.

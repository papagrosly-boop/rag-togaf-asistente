"""Tools del agente. Los LÍMITES están en el código (constantes), no en el prompt."""
import os
import re
from typing import Literal
from urllib.parse import urlparse

from langchain_core.tools import tool

from config import PERSIST_DIRECTORY, get_embeddings

# ------------------------- Límites (en código) -------------------------
MAX_FRAGMENTOS = 4          # fragmentos por llamada a buscar_en_base
MAX_CARACTERES = 600        # longitud máxima de cada fragmento
MAX_WEB = 3                 # resultados web por llamada
MAX_EXTRACTO_WEB = 300      # caracteres del extracto web
MAX_CONSULTA_WEB = 200      # lo único que sale hacia el buscador: la consulta, recortada
SITIOS_PERMITIDOS = ["pubs.opengroup.org", "opengroup.org"]   # el LLM NO puede cambiarla

_vs = None


def _get_vs():
    global _vs
    if _vs is None:
        from langchain_chroma import Chroma
        _vs = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=get_embeddings())
    return _vs


# ------------------------- Tool 1: base de conocimientos -------------------------
@tool
def buscar_en_base(consulta: str) -> str:
    """Busca en la base de conocimientos local de TOGAF 9.1 (Introducción, Fase Preliminar, Fase A Visión de
    Arquitectura, Fase B Arquitectura de Negocio, Fase C Arquitectura de Información/Datos y Fase D Arquitectura
    de Tecnología). Úsala SIEMPRE que la pregunta sea de contenido sobre TOGAF, antes de responder. Escribe la
    consulta con las palabras clave del tema (los documentos están en inglés, puedes consultar en inglés). Si la
    pregunta tiene dos temas, llámala una vez por tema. Devuelve hasta 4 fragmentos, cada uno con
    [Fuente: documento, pág. N]."""
    docs = _get_vs().similarity_search(consulta, k=MAX_FRAGMENTOS)
    if not docs:
        return "SIN_RESULTADOS"
    partes = []
    for d in docs:
        texto = re.sub(r"\s+", " ", d.page_content).strip()[:MAX_CARACTERES]
        partes.append(f"[Fuente: {d.metadata.get('documento', '?')}, pág. {d.metadata.get('pagina', '?')}] {texto}")
    return "\n\n".join(partes)


# ------------------------- Tool 2: búsqueda web controlada -------------------------
@tool
def buscar_web(consulta: str) -> str:
    """Busca información PÚBLICA o ACTUAL de TOGAF en la web, solo en los sitios oficiales de The Open Group.
    Úsala ÚNICAMENTE si buscar_en_base no devolvió la respuesta y la pregunta trata de información pública
    (otras fases del ADM, definiciones oficiales, versiones). Nunca la uses si la base ya respondió. Envía solo
    palabras clave de la pregunta: jamás texto de los documentos de la base ni datos del usuario. Devuelve
    máximo 3 resultados, cada uno con [web: URL]."""
    clave = os.getenv("TAVILY_API_KEY")
    if not clave:
        return "WEB_NO_DISPONIBLE: no hay TAVILY_API_KEY configurada."
    try:
        from tavily import TavilyClient
        r = TavilyClient(api_key=clave).search(
            query=consulta.strip()[:MAX_CONSULTA_WEB],
            max_results=MAX_WEB,
            include_domains=SITIOS_PERMITIDOS,
            search_depth="basic",
        )
    except Exception as e:  # el agente sigue funcionando si el buscador falla
        return f"WEB_NO_DISPONIBLE: {type(e).__name__}"
    resultados = []
    for x in r.get("results", []):
        host = urlparse(x.get("url", "")).netloc.lower()
        if any(host == s or host.endswith("." + s) for s in SITIOS_PERMITIDOS):   # doble verificación
            resultados.append(x)
    resultados = resultados[:MAX_WEB]
    if not resultados:
        return "SIN_RESULTADOS_WEB"
    salida = []
    for x in resultados:
        extracto = re.sub(r"\s+", " ", x.get("content", "")).strip()[:MAX_EXTRACTO_WEB]
        salida.append(f"[web: {x['url']}] {x.get('title', '').strip()} — {extracto}")
    return "\n\n".join(salida)


# ------------------------- Tool 3: tarea del dominio (ADM) -------------------------
# Ciclo ADM de TOGAF 9.1 en orden. La Gestión de Requisitos es continua y central a todas las fases.
ADM = [
    ("Preliminar", "Fase Preliminar", True),
    ("A", "Fase A: Visión de Arquitectura", True),
    ("B", "Fase B: Arquitectura de Negocio", True),
    ("C", "Fase C: Arquitecturas de Sistemas de Información (Datos y Aplicaciones)", True),
    ("D", "Fase D: Arquitectura de Tecnología", True),
    ("E", "Fase E: Oportunidades y Soluciones", False),
    ("F", "Fase F: Planificación de la Migración", False),
    ("G", "Fase G: Gobierno de la Implementación", False),
    ("H", "Fase H: Gestión del Cambio de Arquitectura", False),
]
_NOMBRES = [  # (patrón, índice en ADM) — se prueban en orden
    (r"prelim", 0), (r"visi[oó]n|vision", 1), (r"negocio|business", 2),
    (r"datos|data|informaci[oó]n|information|aplicaci[oó]n|application", 3),
    (r"tecnolog|technolog", 4), (r"oportunidad|opportunit|soluciones|solutions", 5),
    (r"migraci[oó]n|migration", 6), (r"implementaci[oó]n|implementation|gobierno|governance", 7),
    (r"cambio|change", 8),
]


def _resolver_fase(texto: str):
    t = texto.lower().strip()
    if re.search(r"requisit|requirement", t):
        return "REQ"
    for patron, i in _NOMBRES:
        if re.search(patron, t):
            return i
    m = re.search(r"\b([a-h])\b", t)
    if m:
        return "abcdefgh".index(m.group(1)) + 1
    return None


@tool
def consultar_adm(fase: str, operacion: Literal["siguiente", "anterior", "faltantes", "posicion"]) -> str:
    """Consulta EXACTA sobre el orden del ciclo ADM de TOGAF 9.1 (no la respondas de memoria). 'fase' es la fase
    mencionada por el usuario (ej. 'C', 'Preliminar', 'Arquitectura de Negocio'). 'operacion': 'siguiente' (qué
    fase sigue), 'anterior' (qué fase la precede), 'faltantes' (cuántas y cuáles fases quedan después de ella
    hasta cerrar el ciclo en la H) o 'posicion' (qué lugar ocupa en el ciclo). Indica también si la fase está
    cubierta por la base de conocimientos."""
    i = _resolver_fase(fase)
    if i == "REQ":
        return ("La Gestión de Requisitos no es una fase secuencial: es un proceso continuo que está en el "
                "centro del ciclo ADM y se relaciona con todas las fases.")
    if i is None:
        return f"No reconozco la fase '{fase}'. Fases válidas: Preliminar, A, B, C, D, E, F, G, H."
    _, nombre, en_base = ADM[i]
    nota = "(cubierta por la base de conocimientos)" if en_base else "(NO está cubierta por la base de conocimientos)"
    if operacion == "siguiente":
        if i == len(ADM) - 1:
            return f"Después de la {nombre} el ciclo vuelve a la Fase A: Visión de Arquitectura (nuevo ciclo)."
        return f"Después de la {nombre} sigue la {ADM[i + 1][1]} {('(cubierta por la base)' if ADM[i + 1][2] else '(NO cubierta por la base)')}."
    if operacion == "anterior":
        if i == 0:
            return "La Fase Preliminar es el inicio del ciclo; no tiene fase anterior."
        return f"Antes de la {nombre} está la {ADM[i - 1][1]}."
    if operacion == "faltantes":
        resto = ADM[i + 1:]
        if not resto:
            return f"La {nombre} es la última fase del ciclo; no quedan fases (luego se reinicia en la A)."
        lista = "; ".join(x[1] for x in resto)
        return f"Después de la {nombre} quedan {len(resto)} fases para cerrar el ciclo: {lista}."
    return f"La {nombre} ocupa el lugar {i + 1} de {len(ADM)} del ciclo (contando la Preliminar). {nota}"


# ------------------------- Tool 4: tarea del dominio (referencia APA) -------------------------
# VERIFICAR autor, año y títulos antes de entregar (están escritos a mano a partir de los nombres de archivo).
_DOCS_APA = {
    "introduccion": "TOGAF® Version 9.1: Introduction",
    "fasepreliminar": "TOGAF® Version 9.1: Preliminary Phase",
    "visionarquitectura": "TOGAF® Version 9.1: Phase A: Architecture Vision",
    "arquitecturanegocio": "TOGAF® Version 9.1: Phase B: Business Architecture",
    "arquitecturainformacion": "TOGAF® Version 9.1: Phase C: Information Systems Architectures",
    "arquitecturatecnologia": "TOGAF® Version 9.1: Phase D: Technology Architecture",
}
_ALIAS_APA = {
    "introduccion": ["introduc"], "fasepreliminar": ["prelim"], "visionarquitectura": ["visi", "phase a", "fase a"],
    "arquitecturanegocio": ["negocio", "business", "fase b", "phase b"],
    "arquitecturainformacion": ["informaci", "datos", "data", "fase c", "phase c"],
    "arquitecturatecnologia": ["tecnolog", "technolog", "fase d", "phase d"],
}


@tool
def generar_referencia_apa(documento: str) -> str:
    """Genera la referencia en formato APA 7 de un documento de la base de conocimientos de TOGAF (no la
    inventes de memoria). 'documento' es el documento o fase que menciona el usuario (ej. 'Fase Preliminar',
    'Arquitectura de Negocio', 'Introducción'). Devuelve la referencia lista para copiar."""
    t = documento.lower()
    t_compacto = re.sub(r"[^a-z0-9]", "", t)
    for clave, titulo in _DOCS_APA.items():
        if clave in t_compacto or any(a in t for a in _ALIAS_APA[clave]):
            return f"The Open Group. (2011). *{titulo}* [Documento PDF]. The Open Group."
    disponibles = ", ".join(_DOCS_APA)
    return f"No encontré ese documento en la base. Documentos disponibles: {disponibles}."


TOOLS = [buscar_en_base, consultar_adm, generar_referencia_apa, buscar_web]
TOOLS_POR_NOMBRE = {t.name: t for t in TOOLS}

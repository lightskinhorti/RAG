"""Descargador de documentos reales del BOE (Boletín Oficial del Estado).

Utiliza la API de datos abiertos del BOE (sin autenticación) para obtener
índices diarios y descargar los documentos XML de legislación española.

Endpoint de sumario: GET /datosabiertos/api/boe/sumario/{YYYYMMDD}
                     Accept: application/xml
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import requests
from lxml import etree

from src.config import get_section
from src.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.boe.es"
_API_SUMARIO = "https://www.boe.es/datosabiertos/api/boe/sumario"
_XML_DOC_URL = "https://www.boe.es/diario_boe/xml.php"

_SECCIONES_INTERES = {"1", "2", "3"}  # I, II, III — disposiciones normativas

_TIPOS_INTERES = {
    "ley orgánica",
    "ley",
    "real decreto-ley",
    "real decreto",
    "orden",
    "resolución",
    "instrucción",
    "circular",
}


@dataclass
class DocumentoBOE:
    """Documento legislativo descargado del BOE."""

    id: str
    titulo: str
    texto: str
    fecha: str
    seccion: str
    departamento: str
    url: str
    metadata: dict = field(default_factory=dict)


class BOEDownloader:
    """Descarga y parsea documentos del BOE usando su API pública."""

    def __init__(
        self,
        raw_dir: str | None = None,
        max_docs_per_day: int = 15,
        delay_between_requests: float = 0.5,
    ):
        cfg = get_section("boe") if _section_exists("boe") else {}
        self._raw_dir = Path(raw_dir or cfg.get("directorio_raw", "data/raw"))
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._max_docs = max_docs_per_day
        self._delay = delay_between_requests
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "RAG-BOE-Research/1.0",
            "Accept": "application/xml",
        })

    def descargar_rango(
        self,
        fecha_inicio: date,
        fecha_fin: date,
    ) -> list[DocumentoBOE]:
        """Descarga documentos del BOE en un rango de fechas."""
        docs = []
        current = fecha_inicio
        while current <= fecha_fin:
            logger.info("descargando_dia", fecha=current.isoformat())
            try:
                day_docs = self.descargar_dia(current)
                docs.extend(day_docs)
                logger.info(
                    "dia_descargado",
                    fecha=current.isoformat(),
                    num_docs=len(day_docs),
                )
            except requests.RequestException as e:
                logger.warning("error_descargando_dia", fecha=current.isoformat(), error=str(e))
            current += timedelta(days=1)
            time.sleep(self._delay)
        return docs

    def descargar_dia(self, fecha: date) -> list[DocumentoBOE]:
        """Descarga todos los documentos relevantes de un día del BOE."""
        indice = self._obtener_indice(fecha)
        if indice is None:
            return []

        ids_documentos = list(self._extraer_ids_relevantes(indice))[:self._max_docs]
        logger.info(
            "ids_relevantes_encontrados",
            fecha=fecha.isoformat(),
            total=len(ids_documentos),
        )

        docs = []
        for doc_id, meta in ids_documentos:
            cached = self._raw_dir / f"{doc_id}.xml"
            if cached.exists():
                logger.debug("usando_cache", id=doc_id)
                doc = self._parsear_xml_file(cached, meta)
            else:
                raw_bytes = self._descargar_xml_bytes(doc_id)
                if raw_bytes:
                    cached.write_bytes(raw_bytes)
                    doc = self._parsear_xml_content(raw_bytes, meta)
                else:
                    doc = None

            if doc:
                docs.append(doc)
            time.sleep(self._delay)

        return docs

    def _obtener_indice(self, fecha: date) -> etree._Element | None:
        """Obtiene el índice diario del BOE en formato XML via la API de datos abiertos."""
        url = f"{_API_SUMARIO}/{fecha.strftime('%Y%m%d')}"
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            root = etree.fromstring(resp.content)
            logger.debug("sumario_obtenido", fecha=fecha.isoformat(), url=url)
            return root
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 404:
                logger.info("dia_sin_boe", fecha=fecha.isoformat())
            else:
                logger.warning("error_http_indice", fecha=fecha.isoformat(), status=code, error=str(e))
            return None
        except etree.XMLSyntaxError as e:
            logger.warning("error_xml_indice", fecha=fecha.isoformat(), error=str(e))
            return None
        except requests.RequestException as e:
            logger.error("error_red_indice", fecha=fecha.isoformat(), error=str(e))
            return None

    def _extraer_ids_relevantes(self, indice: etree._Element) -> Iterator[tuple[str, dict]]:
        """Extrae los IDs de documentos legislativos del XML de sumario del BOE."""
        try:
            # El sumario XML tiene estructura:
            # <sumario><diario><seccion num="1"><departamento><epigrafe><item id="BOE-A-..." titulo="..."/>
            for seccion in indice.findall(".//seccion"):
                num = seccion.get("num", "")
                if num not in _SECCIONES_INTERES:
                    continue
                nombre_seccion = seccion.get("nombre", "")

                for dept in seccion.findall(".//departamento"):
                    nombre_dept = dept.get("nombre", "")

                    for item in dept.findall(".//item"):
                        doc_id = item.get("id", "")
                        titulo = item.get("titulo", "")
                        url_pdf = item.get("urlPdf", "")

                        if doc_id and self._es_documento_interes(titulo):
                            yield doc_id, {
                                "id": doc_id,
                                "titulo": titulo,
                                "seccion": nombre_seccion,
                                "departamento": nombre_dept,
                                "url": f"{_BASE_URL}{url_pdf}" if url_pdf.startswith("/") else url_pdf,
                            }
        except (AttributeError, TypeError) as e:
            logger.warning("error_parseando_indice", error=str(e))

    def _es_documento_interes(self, titulo: str) -> bool:
        """Filtra documentos que son leyes, decretos u órdenes relevantes."""
        titulo_lower = titulo.lower()
        return any(t in titulo_lower for t in _TIPOS_INTERES)

    def _descargar_xml_bytes(self, doc_id: str) -> bytes | None:
        """Descarga el XML en bruto de un documento del BOE. Retorna bytes o None."""
        url = f"{_XML_DOC_URL}?id={doc_id}"
        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            logger.warning("error_descargando_doc", id=doc_id, error=str(e))
            return None

    def _parsear_xml_file(self, path: Path, meta: dict) -> DocumentoBOE | None:
        """Parsea un fichero XML del BOE ya guardado en disco."""
        try:
            content = path.read_bytes()
            return self._parsear_xml_content(content, meta)
        except (OSError, etree.XMLSyntaxError) as e:
            logger.warning("error_parseando_xml_fichero", path=str(path), error=str(e))
            return None

    def _parsear_xml_content(self, content: bytes, meta: dict) -> DocumentoBOE | None:
        """Extrae texto y metadatos del XML de un documento BOE."""
        try:
            root = etree.fromstring(content)
        except etree.XMLSyntaxError:
            texto = _limpiar_html(content.decode("utf-8", errors="replace"))
            if len(texto) < 100:
                return None
            return DocumentoBOE(
                id=meta.get("id", "unknown"),
                titulo=meta.get("titulo", ""),
                texto=texto,
                fecha=meta.get("fecha", ""),
                seccion=meta.get("seccion", ""),
                departamento=meta.get("departamento", ""),
                url=meta.get("url", ""),
                metadata=meta,
            )

        texto = _extraer_texto_xml(root)
        if not texto or len(texto) < 100:
            return None

        titulo = _xpath_texto(root, ".//titulo") or meta.get("titulo", "")
        fecha = _xpath_texto(root, ".//fecha_publicacion") or meta.get("fecha", "")
        departamento = _xpath_texto(root, ".//departamento") or meta.get("departamento", "")
        seccion = _xpath_texto(root, ".//seccion") or meta.get("seccion", "")
        doc_id = _xpath_texto(root, ".//identificador") or meta.get("id", "unknown")

        doc = DocumentoBOE(
            id=doc_id,
            titulo=titulo,
            texto=texto,
            fecha=fecha,
            seccion=seccion,
            departamento=departamento,
            url=meta.get("url", ""),
            metadata={
                **meta,
                "titulo": titulo,
                "fecha": fecha,
                "departamento": departamento,
                "seccion": seccion,
                "longitud_texto": len(texto),
            },
        )
        logger.debug("documento_parseado", id=doc_id, longitud=len(texto), titulo=titulo[:60])
        return doc


# ---------------------------------------------------------------------------
# Funciones auxiliares de parseo XML
# ---------------------------------------------------------------------------


def _extraer_texto_xml(root: etree._Element) -> str:
    """Extrae texto limpio de los nodos relevantes del XML del BOE."""
    nodos_texto = root.findall(".//texto//p") or root.findall(".//texto//*")
    if not nodos_texto:
        nodos_texto = root.findall(".//*")

    partes = []
    for nodo in nodos_texto:
        texto = (nodo.text or "").strip()
        tail = (nodo.tail or "").strip()
        if texto:
            partes.append(texto)
        if tail:
            partes.append(tail)

    return _limpiar_texto("\n".join(partes))


def _xpath_texto(root: etree._Element, xpath: str) -> str:
    """Obtiene el texto del primer nodo que coincide con el xpath."""
    nodo = root.find(xpath)
    if nodo is not None and nodo.text:
        return nodo.text.strip()
    return ""


def _limpiar_texto(texto: str) -> str:
    """Normaliza espacios y saltos de línea en el texto extraído."""
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r" {2,}", " ", texto)
    return texto.strip()


def _limpiar_html(html: str) -> str:
    """Elimina etiquetas HTML de un texto."""
    sin_tags = re.sub(r"<[^>]+>", " ", html)
    return _limpiar_texto(sin_tags)


def _section_exists(section: str) -> bool:
    """Comprueba si una sección existe en la configuración."""
    from src.config import load_config
    try:
        cfg = load_config()
        return section in cfg
    except Exception:
        return False

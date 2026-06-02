"""Interfaz Streamlit para el sistema RAG de Inteligencia Documental BOE."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG · Inteligencia Documental BOE",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# CSS personalizado — diseño oscuro profesional
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Fuente e importaciones */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo principal */
    .main {
        background-color: #0f1117;
    }
    .stApp {
        background-color: #0f1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Tarjeta de mensaje del usuario */
    .user-message {
        background: linear-gradient(135deg, #1c3a5e, #1a4a7a);
        border: 1px solid #1f6feb;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 15px;
    }

    /* Tarjeta de respuesta del asistente */
    .assistant-message {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #3fb950;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Tarjeta de fuente/citación */
    .source-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-left: 3px solid #f0883e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #8b949e;
    }

    .source-card .source-title {
        color: #f0883e;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .source-card .source-meta {
        color: #6e7681;
        font-size: 11px;
        margin-bottom: 6px;
    }

    .source-card .source-text {
        color: #8b949e;
        font-size: 12px;
        line-height: 1.5;
        font-style: italic;
    }

    /* Métrica badge */
    .metric-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .metric-good { background: #1a4c2a; color: #3fb950; border: 1px solid #3fb950; }
    .metric-mid  { background: #4a3200; color: #f0883e; border: 1px solid #f0883e; }
    .metric-bad  { background: #3d1a1a; color: #f85149; border: 1px solid #f85149; }

    /* Encabezados */
    h1, h2, h3 { color: #e6edf3 !important; }

    /* Separadores */
    hr { border-color: #30363d; }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
        border-radius: 8px;
    }

    /* Botones primarios */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff);
        transform: translateY(-1px);
    }

    /* Sliders */
    .stSlider > div > div > div { background: #1f6feb; }

    /* Latencia badge */
    .latency-badge {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: #6e7681;
    }

    /* Mock mode warning */
    .mock-warning {
        background: #3d2200;
        border: 1px solid #f0883e;
        border-radius: 8px;
        padding: 8px 12px;
        color: #f0883e;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def api_call(method: str, endpoint: str, **kwargs) -> dict | None:
    """Wrapper para llamadas a la API con manejo de errores."""
    url = f"{API_URL}{endpoint}"
    try:
        resp = getattr(requests, method)(url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"❌ No se puede conectar a la API en {API_URL}. ¿Está el servidor activo?")
        return None
    except requests.Timeout:
        st.error("⏱️ La consulta tardó demasiado. Intenta de nuevo.")
        return None
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ Error de la API: {detail}")
        return None


def get_stats() -> dict | None:
    """Obtiene las estadísticas actuales del sistema."""
    return api_call("get", "/stats")


def get_health() -> dict | None:
    """Comprueba el estado del servidor."""
    return api_call("get", "/health")


def clasificar_score(score: float) -> str:
    """Clasifica un score 0-1 en categoría CSS."""
    if score >= 0.75:
        return "metric-good"
    if score >= 0.45:
        return "metric-mid"
    return "metric-bad"


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ RAG · BOE")
        st.markdown("*Inteligencia Documental sobre Legislación Española*")
        st.divider()

        # Estado del sistema
        health = get_health()
        if health:
            estado_color = "🟢" if health.get("estado") == "ok" else "🔴"
            st.markdown(f"{estado_color} **API:** {health.get('estado', '—').upper()}")
            llm_ok = health.get("llm_configurado", False)
            st.markdown(f"{'🟢' if llm_ok else '🟡'} **LLM:** {'Anthropic ✓' if llm_ok else 'Modo mock'}")
        else:
            st.markdown("🔴 **API:** No disponible")

        st.divider()

        # Estadísticas de la colección
        stats = get_stats()
        if stats:
            st.markdown("### 📊 Colección")
            col1, col2 = st.columns(2)
            col1.metric("Chunks", f"{stats.get('total_chunks', 0):,}")
            col2.metric("Índice BM25", "✓" if stats.get("bm25_indexado") else "✗")
            st.caption(f"Colección: `{stats.get('coleccion', '—')}`")
        else:
            st.markdown("*Sistema no inicializado*")

        st.divider()

        # Parámetros de búsqueda
        st.markdown("### ⚙️ Parámetros")
        top_k = st.slider("Fragmentos a recuperar (top_k)", 1, 15, 5)
        alpha = st.slider(
            "Balance búsqueda (α)",
            0.0, 1.0, 0.7, 0.05,
            help="1.0 = solo embeddings | 0.0 = solo BM25",
        )
        reranking = st.toggle("Reranking con cross-encoder", value=True)

        st.divider()

        # Subir documentos
        st.markdown("### 📁 Subir documento")
        uploaded = st.file_uploader(
            "PDF, TXT o Markdown",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button("Indexar documento", use_container_width=True):
                with st.spinner("Procesando..."):
                    files = {"archivo": (uploaded.name, uploaded.getvalue())}
                    params = {"estrategia": "recursive", "chunk_size": 512, "chunk_overlap": 64}
                    result = api_call("post", "/ingest/upload", files=files, params=params)
                    if result:
                        st.success(
                            f"✅ Indexados {result['total_chunks']} chunks de «{uploaded.name}»"
                        )
                        st.rerun()

        st.divider()
        st.caption("Javier Hortigüela Valiente · 2025")

    return top_k, alpha, reranking


# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------


def render_chat(top_k: int, alpha: float, reranking: bool):
    st.markdown("### 💬 Consulta sobre Legislación Española")
    st.markdown(
        "Haz preguntas sobre leyes, decretos y órdenes publicadas en el BOE. "
        "El sistema recupera los fragmentos más relevantes y genera una respuesta citada.",
        unsafe_allow_html=False,
    )

    # Inicializar historial
    if "historial" not in st.session_state:
        st.session_state.historial = []

    # Mostrar historial
    for turno in st.session_state.historial:
        st.markdown(
            f'<div class="user-message">👤 {turno["pregunta"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="assistant-message">⚖️ {turno["respuesta"]}'
            f'<br><span class="latency-badge">⏱ {turno["latencia_ms"]} ms · '
            f'{turno["num_fuentes"]} fuentes</span></div>',
            unsafe_allow_html=True,
        )

        if turno.get("fuentes"):
            with st.expander(f"📄 Ver {len(turno['fuentes'])} fuentes", expanded=False):
                for i, fuente in enumerate(turno["fuentes"], 1):
                    st.markdown(f"""
<div class="source-card">
    <div class="source-title">[Fuente {i}] {fuente['titulo'][:100]}</div>
    <div class="source-meta">
        📅 {fuente['fecha']} · 🏛️ {fuente['departamento'][:60]} ·
        Score: {fuente['score']:.3f}
    </div>
    <div class="source-text">"{fuente['texto_fragmento'][:300]}..."</div>
</div>""", unsafe_allow_html=True)

        if turno.get("modo_mock"):
            st.markdown(
                '<div class="mock-warning">⚠️ Respuesta generada en modo mock. '
                'Configura ANTHROPIC_API_KEY para respuestas reales.</div>',
                unsafe_allow_html=True,
            )

    # Input de pregunta
    with st.form("form_pregunta", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            pregunta = st.text_input(
                "Escribe tu pregunta",
                placeholder="Ej: ¿Cuál es la jornada laboral máxima en España?",
                label_visibility="collapsed",
            )
        with col2:
            enviar = st.form_submit_button("Preguntar", use_container_width=True)

    if enviar and pregunta.strip():
        with st.spinner("🔍 Buscando y generando respuesta..."):
            payload = {
                "pregunta": pregunta.strip(),
                "top_k": top_k,
                "alpha": alpha,
                "reranking": reranking,
                "streaming": False,
            }
            resultado = api_call("post", "/query", json=payload)

        if resultado:
            st.session_state.historial.append({
                "pregunta": pregunta,
                "respuesta": resultado.get("respuesta", ""),
                "fuentes": resultado.get("fuentes", []),
                "latencia_ms": resultado.get("latencia_ms", 0),
                "num_fuentes": resultado.get("num_fuentes", 0),
                "modo_mock": resultado.get("modo_mock", False),
            })
            st.rerun()

    # Botón limpiar historial
    if st.session_state.historial:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.historial = []
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Evaluación
# ---------------------------------------------------------------------------


def render_evaluacion():
    import plotly.graph_objects as go

    st.markdown("### 📊 Evaluación del Sistema RAG")
    st.markdown(
        "Métricas de calidad del pipeline: fidelidad, relevancia, precisión y recall del contexto."
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("▶️ Ejecutar evaluación", use_container_width=True):
            with st.spinner("Evaluando sistema... esto puede tardar varios minutos."):
                resultado = api_call("get", "/evaluation")
            if resultado:
                st.session_state.eval_resultado = resultado
                st.rerun()

    resultado = st.session_state.get("eval_resultado")

    if not resultado:
        st.info("Pulsa **Ejecutar evaluación** para analizar la calidad del sistema con el dataset de referencia.")
        return

    metricas = resultado.get("metricas_globales", {})

    # Tarjetas de métricas
    cols = st.columns(5)
    labels = {
        "fidelidad_media": "Fidelidad",
        "relevancia_respuesta_media": "Relevancia",
        "precision_contexto_media": "Precisión",
        "recall_contexto_media": "Recall",
        "score_global_medio": "Global",
    }
    for col, (key, label) in zip(cols, labels.items()):
        valor = metricas.get(key, 0.0)
        clase = clasificar_score(valor)
        col.markdown(
            f'<div style="text-align:center">'
            f'<div style="font-size:28px;font-weight:700;color:#e6edf3">{valor:.2f}</div>'
            f'<div class="metric-badge {clase}">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Gráfico radar
    valores_radar = [
        metricas.get("fidelidad_media", 0),
        metricas.get("relevancia_respuesta_media", 0),
        metricas.get("precision_contexto_media", 0),
        metricas.get("recall_contexto_media", 0),
    ]
    categorias = ["Fidelidad", "Relevancia", "Precisión Ctx.", "Recall Ctx."]
    categorias_cierre = categorias + [categorias[0]]
    valores_cierre = valores_radar + [valores_radar[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=valores_cierre,
        theta=categorias_cierre,
        fill="toself",
        fillcolor="rgba(31, 111, 235, 0.2)",
        line=dict(color="#1f6feb", width=2),
        marker=dict(color="#58a6ff", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True, range=[0, 1],
                color="#6e7681",
                gridcolor="#30363d",
                linecolor="#30363d",
            ),
            angularaxis=dict(color="#e6edf3", gridcolor="#30363d"),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e6edf3", family="Inter"),
        margin=dict(l=60, r=60, t=30, b=30),
        height=380,
    )

    col_radar, col_tabla = st.columns([1, 1])
    with col_radar:
        st.plotly_chart(fig, use_container_width=True)

    with col_tabla:
        st.markdown("#### Detalle por pregunta")
        detalle = resultado.get("resultados_detalle", [])
        if detalle:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "ID": r["id"],
                    "Pregunta": r["pregunta"][:60] + "...",
                    "Fidelidad": r["metricas"]["fidelidad"],
                    "Relevancia": r["metricas"]["relevancia_respuesta"],
                    "Precisión": r["metricas"]["precision_contexto"],
                    "Global": r["metricas"]["score_global"],
                }
                for r in detalle
            ])
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    st.caption(
        f"Evaluación ejecutada: {resultado.get('timestamp', '—')} · "
        f"{resultado.get('num_preguntas', 0)} preguntas"
    )


# ---------------------------------------------------------------------------
# Tab: Ingesta masiva
# ---------------------------------------------------------------------------


def render_ingesta():
    st.markdown("### 📥 Ingesta de Documentos BOE")
    st.markdown(
        "Descarga documentos reales del BOE e indexa en el vector store. "
        "También puedes reindexar desde un directorio local."
    )

    st.markdown("#### 🌐 Descargar del BOE")
    col1, col2, col3 = st.columns(3)
    with col1:
        dias = st.number_input("Últimos N días", min_value=1, max_value=365, value=7)
    with col2:
        max_por_dia = st.number_input("Máx. docs/día", min_value=1, max_value=50, value=10)
    with col3:
        estrategia = st.selectbox(
            "Estrategia chunking",
            ["recursive", "semantic", "fixed"],
        )

    if st.button("⬇️ Descargar e indexar BOE", use_container_width=True):
        st.info(
            f"Ejecuta en terminal:\n"
            f"```\npython scripts/download_boe.py --dias {dias} "
            f"--max-por-dia {max_por_dia} --ingestar\n```"
        )

    st.divider()

    st.markdown("#### 📂 Reindexar desde directorio local")
    col_dir, col_est = st.columns([2, 1])
    with col_dir:
        directorio = st.text_input("Directorio", value="data/raw")
    with col_est:
        estrategia_local = st.selectbox(
            "Estrategia",
            ["recursive", "semantic", "fixed"],
            key="est_local",
        )

    resetear = st.checkbox("Resetear colección antes de indexar", value=False)

    if st.button("🔄 Indexar directorio", use_container_width=True):
        payload = {
            "directorio": directorio,
            "estrategia": estrategia_local,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "resetear_coleccion": resetear,
        }
        with st.spinner("Procesando documentos..."):
            resultado = api_call("post", "/ingest", json=payload)
        if resultado:
            st.success(
                f"✅ Ingesta completada:\n"
                f"- {resultado['total_documentos']} documentos\n"
                f"- {resultado['total_chunks']:,} chunks\n"
                f"- {resultado['duracion_segundos']:.1f}s"
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------


def main():
    top_k, alpha, reranking = render_sidebar()

    st.markdown(
        "<h1 style='margin-bottom:0'>⚖️ RAG · Inteligencia Documental BOE</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6e7681;margin-top:4px'>"
        "Sistema de Recuperación Aumentada por Generación sobre legislación española"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    tab_chat, tab_eval, tab_ingesta = st.tabs(["💬 Chat", "📊 Evaluación", "📥 Ingesta"])

    with tab_chat:
        render_chat(top_k, alpha, reranking)

    with tab_eval:
        render_evaluacion()

    with tab_ingesta:
        render_ingesta()


if __name__ == "__main__":
    main()

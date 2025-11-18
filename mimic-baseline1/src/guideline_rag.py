from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.parsing.document_parser import DocumentType
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.utils.output.citations import (
    extract_markdown_references,
    format_cited_references,
)
from langroid.vector_store.chromadb import ChromaDBConfig

from .config import get_llm_config
from dataclasses import dataclass

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
DEFAULT_DOC_DIR = Path(__file__).resolve().parents[2] / "RAG files"
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[1] / "data" / "guideline_rag_chroma"
PATIENT_CONTEXT_FILENAME = "patient_context.txt"


def _doc_paths(doc_dir: Path) -> List[Path]:
    doc_dir = doc_dir.expanduser().resolve()
    if not doc_dir.exists():
        return []
    return [
        path.resolve()
        for path in sorted(doc_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _doc_type_for_path(path: Path) -> DocumentType:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return DocumentType.PDF
    return DocumentType.TXT


def _condensed_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "stat",
        "units",
        "day",
        "range",
        "central",
        "percentiles",
        "trend",
        "variability",
        "coverage",
        "outliers",
        "flags",
    ]
    return {k: payload.get(k) for k in keys if k in payload}


def _resolve_doc_dir(rag_dir: Optional[str | Path]) -> Path:
    return Path(rag_dir or os.getenv("GUIDELINE_RAG_DIR", DEFAULT_DOC_DIR)).expanduser().resolve()


def write_patient_context_file(
    summary: str,
    payload: Dict[str, Any],
    *,
    rag_dir: Optional[str | Path] = None,
    subject_id: Optional[int] = None,
    stat: Optional[str] = None,
) -> str:
    doc_dir = _resolve_doc_dir(rag_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / PATIENT_CONTEXT_FILENAME
    condensed = json.dumps(_condensed_payload(payload), indent=2)
    subject_line = f"Subject ID: {subject_id if subject_id is not None else 'unknown'}"
    stat_line = f"Stat: {stat or payload.get('stat', 'unknown stat')}"
    day_line = f"Day: {payload.get('day', 'unknown day')}"
    text = (
        "PATIENT CONTEXT\n"
        f"{subject_line}\n{stat_line}\n{day_line}\n\n"
        "Daily summary:\n"
        f"{summary.strip()}\n\n"
        "Structured metrics (JSON):\n"
        f"{condensed}\n"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


@dataclass
class GuidelineRAGResult:
    text: str
    agent: Optional[DocChatAgent]
    context_text: str


def _run_guideline_rag(
    summary: str,
    payload: Dict[str, Any],
    *,
    rag_dir: Optional[str | Path] = None,
    subject_id: Optional[int] = None,
    stat: Optional[str] = None,
    question: Optional[str] = None,
    max_points: int = 3,
    storage_dir: Optional[str | Path] = None,
    patient_context_text: str = "",
) -> GuidelineRAGResult:
    doc_dir = _resolve_doc_dir(rag_dir)
    docs = _doc_paths(doc_dir)
    if not docs:
        return GuidelineRAGResult(
            text=f"Guideline RAG skipped: no PDF/TXT/MD files found in '{doc_dir}'.",
            agent=None,
            context_text=patient_context_text,
        )

    storage_root = Path(storage_dir or DEFAULT_STORAGE_DIR).expanduser().resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    vecdb_cfg = ChromaDBConfig(
        collection_name="guideline-rag",
        storage_path=str(storage_root),
        replace_collection=True,
        embedding=OpenAIEmbeddingsConfig(
            model_name="text-embedding-3-small",
            model_type="openai",
            dims=1536,
        ),
    )
    parsing_cfg = ParsingConfig(
        splitter=Splitter.MARKDOWN,
        chunk_size=900,
        overlap=120,
        pdf=PdfParsingConfig(library="pypdf"),
    )
    agent_cfg = DocChatAgentConfig(
        name="GuidelineRAG",
        llm=get_llm_config(),
        doc_paths=[],
        vecdb=vecdb_cfg,
        parsing=parsing_cfg,
        n_similar_chunks=3,
        n_relevant_chunks=3,
        n_neighbor_chunks=1,
        use_bm25_search=True,
        use_fuzzy_match=True,
        use_reciprocal_rank_fusion=True,
        conversation_mode=False,
        retain_context=False,
        system_message=(
            "You are a clinical guideline assistant. Always base your answers on the "
            "supplied document extracts, especially the patient_context.txt file that "
            "captures the current patient's summary and structured metrics. Cite all "
            "supporting passages with markdown footnotes [^i]. "
            "If information is insufficient, say so."
        ),
    )
    agent = DocChatAgent(agent_cfg)
    vecdb = getattr(agent, "vecdb", None)
    embedding_fn = getattr(vecdb, "embedding_fn", None)
    if embedding_fn is not None:
        original_fn = embedding_fn
        if not hasattr(embedding_fn, "embed_documents"):
            def _embed_documents(input=None, **kwargs):  # type: ignore[override]
                data = input if input is not None else kwargs.get("documents") or kwargs.get("texts")
                if data is None:
                    raise ValueError("embed_documents requires 'input' argument")
                return original_fn(data)

            embedding_fn.embed_documents = _embed_documents  # type: ignore[attr-defined]
        if not hasattr(embedding_fn, "embed_query"):
            def _embed_query(input=None, **kwargs):  # type: ignore[override]
                data = input if input is not None else kwargs.get("query") or kwargs.get("texts")
                if data is None:
                    raise ValueError("embed_query requires 'input' argument")
                texts = data if isinstance(data, list) else [data]
                return original_fn(texts)

            embedding_fn.embed_query = _embed_query  # type: ignore[attr-defined]
    for dtype in {DocumentType.PDF, DocumentType.TXT}:
        paths_of_type = [str(p) for p in docs if _doc_type_for_path(p) == dtype]
        if paths_of_type:
            agent.ingest_doc_paths(paths_of_type, doc_type=dtype)

    stat_name = stat or payload.get("stat") or "unknown stat"
    day = payload.get("day") or "unknown day"
    question_line = f"\nOriginal question:\n{question.strip()}\n" if question else ""
    condensed = json.dumps(_condensed_payload(payload), indent=2)

    prompt = f"""
Clinical summary interpretation request.
Subject ID: {subject_id if subject_id is not None else 'unknown'}
Stat: {stat_name}
Day: {day}

Daily summary:
{summary.strip()}

Structured metrics:
{condensed}

Task:
- Retrieve the most relevant guidance from the provided documents.
- Provide up to {max_points} concise bullet points that map the patient's numbers to the guidance.
- Quote thresholds or recommended actions when available and cite sources as [^i].
- If guidance does not apply, state that instead of speculating.
{question_line}
""".strip()
    if patient_context_text:
        prompt = f"{patient_context_text}\n\n{prompt}"

    try:
        query, extracts = agent.get_relevant_extracts(prompt)
    except Exception as exc:
        return GuidelineRAGResult(text=f"Guideline RAG failed during retrieval: {exc}", agent=None, context_text=patient_context_text)
    if not extracts:
        return GuidelineRAGResult(text="Guideline RAG could not find relevant passages.", agent=agent, context_text=patient_context_text)

    try:
        response = agent.get_summary_answer(query, extracts)
    except Exception as exc:
        return GuidelineRAGResult(text=f"Guideline RAG failed during summarization: {exc}", agent=agent, context_text=patient_context_text)

    final_answer = (response.content or "").strip()
    citations = extract_markdown_references(final_answer)
    if not citations:
        fallback_count = len(extracts)
        if max_points:
            fallback_count = min(fallback_count, max_points)
        if fallback_count > 0:
            citations = list(range(1, fallback_count + 1))
            citation_marks = " ".join(f"[^{i}]" for i in citations)
            final_answer = f"{final_answer}\n\n{citation_marks}".strip()
    full_citations_str, _ = format_cited_references(citations, extracts)
    if full_citations_str.strip():
        final_answer = f"{final_answer}\n\n{full_citations_str.strip()}"
    final_answer = final_answer or "Guideline RAG returned an empty answer."
    return GuidelineRAGResult(text=final_answer, agent=agent, context_text=patient_context_text)


def interpret_with_guidelines(
    summary: str,
    payload: Dict[str, Any],
    *,
    rag_dir: Optional[str | Path] = None,
    subject_id: Optional[int] = None,
    stat: Optional[str] = None,
    question: Optional[str] = None,
    max_points: int = 3,
    storage_dir: Optional[str | Path] = None,
    patient_context_text: str = "",
) -> str:
    return _run_guideline_rag(
        summary,
        payload,
        rag_dir=rag_dir,
        subject_id=subject_id,
        stat=stat,
        question=question,
        max_points=max_points,
        storage_dir=storage_dir,
        patient_context_text=patient_context_text,
    ).text


def get_guideline_rag_context(
    summary: str,
    payload: Dict[str, Any],
    *,
    rag_dir: Optional[str | Path] = None,
    subject_id: Optional[int] = None,
    stat: Optional[str] = None,
    question: Optional[str] = None,
    max_points: int = 3,
    storage_dir: Optional[str | Path] = None,
    patient_context_text: str = "",
) -> GuidelineRAGResult:
    return _run_guideline_rag(
        summary,
        payload,
        rag_dir=rag_dir,
        subject_id=subject_id,
        stat=stat,
        question=question,
        max_points=max_points,
        storage_dir=storage_dir,
        patient_context_text=patient_context_text,
    )


def answer_guideline_question(
    agent: DocChatAgent, question: str, patient_context_text: str = ""
) -> str:
    prompt = question
    if patient_context_text:
        prompt = f"{patient_context_text}\n\nQuestion: {question.strip()}"
    try:
        response = agent.answer_from_docs(prompt)
    except Exception as exc:
        return f"Guideline follow-up failed: {exc}"
    if response is None:
        return "Guideline follow-up produced no response."
    content = (response.content or "").strip()
    meta = getattr(response, "metadata", None)
    sources = ""
    if meta is not None:
        block = getattr(meta, "source_content", "") or getattr(meta, "source", "")
        if block.strip():
            sources = "\n\n" + block.strip()
    final = (content + sources).strip()
    return final or "Guideline follow-up returned an empty answer."

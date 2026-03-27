# Google NotebookLM REST API wrapper
# Namhyeon Go <gnh1201@catswords.re.kr>
# https://github.com/gnh1201/notebooklm-rest-api
import os
import uuid
import tempfile
from typing import Any, Optional, Literal, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from notebooklm import (
    ArtifactDownloadError,
    ArtifactNotFoundError,
    ArtifactNotReadyError,
    ArtifactParseError,
    NotebookLMClient,
    RPCError,
)  # notebooklm-py :contentReference[oaicite:2]{index=2}


# ----------------------------
# Config / Security
# ----------------------------
API_KEY = os.environ.get("NOTEBOOKLM_REST_API_KEY", "")  # set this in production
AUTH_STORAGE_PATH = os.environ.get("NOTEBOOKLM_STORAGE_PATH")  # optional override


def require_api_key(x_api_key: Optional[str] = None):
    # Minimal API-key gate. Put this behind a real gateway (Cloudflare, Nginx, etc.) for production.
    if API_KEY:
        # FastAPI header parsing without extra imports (keep simple):
        # Prefer: from fastapi import Header; def require_api_key(x_api_key: str = Header(None)) ...
        # but we keep it minimal and rely on query param fallback too.
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")


async def get_client() -> NotebookLMClient:
    """
    Creates a client using notebooklm-py's supported auth precedence:
    - explicit path to from_storage()
    - NOTEBOOKLM_AUTH_JSON
    - NOTEBOOKLM_HOME/storage_state.json
    - ~/.notebooklm/storage_state.json
    :contentReference[oaicite:3]{index=3}
    """
    try:
        if AUTH_STORAGE_PATH:
            return await NotebookLMClient.from_storage(AUTH_STORAGE_PATH)
        return await NotebookLMClient.from_storage()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize NotebookLM client: {e}")


def map_rpc_error(e: RPCError) -> HTTPException:
    # notebooklm-py raises RPCError for API failures :contentReference[oaicite:4]{index=4}
    msg = str(e)
    if "401" in msg or "403" in msg or "auth" in msg.lower():
        return HTTPException(status_code=401, detail=msg)
    if "rate" in msg.lower() or "429" in msg:
        return HTTPException(status_code=429, detail=msg)
    return HTTPException(status_code=502, detail=msg)


def map_artifact_error(e: Exception) -> HTTPException:
    if isinstance(e, ArtifactNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, ArtifactNotReadyError):
        return HTTPException(status_code=409, detail=str(e))
    if isinstance(e, (ArtifactDownloadError, ArtifactParseError)):
        return HTTPException(status_code=502, detail=str(e))
    if isinstance(e, (TypeError, ValueError)):
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=500, detail=str(e))


ARTIFACT_GENERATE_ALLOWED_OPTIONS: Dict[str, set[str]] = {
    "audio": {"source_ids", "language", "instructions", "audio_format", "audio_length", "description"},
    "video": {"source_ids", "language", "instructions", "video_format", "video_style", "description"},
    "report": {"report_format", "source_ids", "language", "custom_prompt", "description"},
    "quiz": {"source_ids", "instructions", "quantity", "difficulty", "description"},
    "flashcards": {"source_ids", "instructions", "quantity", "difficulty", "description"},
    "slide_deck": {"source_ids", "language", "instructions", "slide_format", "slide_length", "description"},
    "infographic": {"source_ids", "language", "instructions", "orientation", "detail_level", "description"},
    "data_table": {"source_ids", "language", "instructions", "description"},
    "mind_map": {"source_ids"},
}

ARTIFACT_TYPES_WITH_INSTRUCTIONS = frozenset(
    {
        "audio",
        "video",
        "quiz",
        "flashcards",
        "slide_deck",
        "infographic",
        "data_table",
    }
)

FIXED_DOWNLOAD_FORMATS = {
    "audio": "mp4",
    "video": "mp4",
    "infographic": "png",
    "slide_deck": "pdf",
    "report": "markdown",
    "mind_map": "json",
    "data_table": "csv",
}


def normalize_artifact_generate_options(artifact_type: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(options or {})
    unexpected = sorted(set(normalized) - ARTIFACT_GENERATE_ALLOWED_OPTIONS[artifact_type])
    if unexpected:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported options for {artifact_type}: {', '.join(unexpected)}",
        )

    description = normalized.pop("description", None)
    if artifact_type in ARTIFACT_TYPES_WITH_INSTRUCTIONS:
        if description is not None and "instructions" not in normalized:
            normalized["instructions"] = description
    elif artifact_type == "report" and description is not None and "custom_prompt" not in normalized:
        normalized["custom_prompt"] = description

    return normalized


def validate_artifact_download_request(artifact_type: str, output_format: Optional[str]) -> None:
    if output_format is None or artifact_type in {"quiz", "flashcards"}:
        return

    fixed_format = FIXED_DOWNLOAD_FORMATS[artifact_type]
    raise HTTPException(
        status_code=400,
        detail=(
            "output_format is only supported for quiz and flashcards. "
            f"{artifact_type} downloads are always returned as {fixed_format}."
        ),
    )


def cleanup_temp_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ----------------------------
# Models
# ----------------------------
class NotebookCreateReq(BaseModel):
    title: str


class NotebookRenameReq(BaseModel):
    new_title: str


class SourceAddUrlReq(BaseModel):
    url: str
    wait: bool = True


class SourceAddTextReq(BaseModel):
    title: str
    content: str


class SourceAddYoutubeReq(BaseModel):
    url: str
    wait: bool = True


class ChatAskReq(BaseModel):
    question: str
    # optional persona fields could be added if you want


class ArtifactGenerateReq(BaseModel):
    # A simple unified generator:
    # audio/video/report/quiz/flashcards/slide_deck/infographic/data_table/mind_map
    type: Literal[
        "audio",
        "video",
        "report",
        "quiz",
        "flashcards",
        "slide_deck",
        "infographic",
        "data_table",
        "mind_map",
    ]
    # Options are passed through as-is to the underlying generate_* calls where applicable.
    # (The library supports many per-type options; keep this generic.)
    options: Dict[str, Any] = {}


class TaskPollResp(BaseModel):
    ok: bool
    status: Any


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="NotebookLM REST API (powered by notebooklm-py)")

import logging
import re

_logger = logging.getLogger(__name__)


def _extract_answer_citation_numbers(answer: Any) -> set[int]:
    if not isinstance(answer, str):
        _logger.debug("chat/ask citation extraction skipped: answer type=%s", type(answer).__name__)
        return set()

    citation_numbers: set[int] = set()
    for match in re.finditer(r"\[([0-9,\s;:\-–]+)\]", answer):
        token = match.group(1).replace("–", "-").replace(";", ",").replace(":", ",")

        for part in token.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue

            if "-" in cleaned:
                left, right = cleaned.split("-", 1)
                if left.strip().isdigit() and right.strip().isdigit():
                    range_start = int(left.strip())
                    range_end = int(right.strip())
                    if range_start <= range_end:
                        citation_numbers.update(range(range_start, range_end + 1))
            elif cleaned.isdigit():
                citation_numbers.add(int(cleaned))

    _logger.debug("chat/ask citations extracted: count=%d citations=%s", len(citation_numbers), sorted(citation_numbers))
    return citation_numbers


def _reference_citation_number(reference: Any) -> Optional[int]:
    raw_number = reference.get("citation_number") if isinstance(reference, dict) else getattr(reference, "citation_number", None)

    if isinstance(raw_number, int):
        return raw_number
    if isinstance(raw_number, str) and raw_number.isdigit():
        return int(raw_number)
    return None


def _trim_chat_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
    else:
        payload = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else getattr(result, "__dict__", {"answer": getattr(result, "answer", None)})
        )
    if not isinstance(payload, dict):
        payload = {"answer": getattr(result, "answer", None)}

    answer = payload.get("answer")

    references = payload.get("references")
    if isinstance(references, list):
        cited_numbers = _extract_answer_citation_numbers(answer)
        filtered_references = [reference for reference in references if _reference_citation_number(reference) in cited_numbers]
        payload["references"] = filtered_references
        available_numbers = {
            citation_number
            for citation_number in (_reference_citation_number(reference) for reference in references)
            if citation_number is not None
        }
        missing_numbers = sorted(cited_numbers - available_numbers)
        _logger.debug(
            "chat/ask references filtered: original=%d filtered=%d cited=%d missing=%s",
            len(references),
            len(filtered_references),
            len(cited_numbers),
            missing_numbers,
        )
    else:
        _logger.debug("chat/ask references filter skipped: references type=%s", type(references).__name__)

    return payload


@app.get("/health")
async def health():
    return {"ok": True}


# ----------------------------
# Notebooks
# ----------------------------
@app.get("/v1/notebooks")
async def list_notebooks():
    client = await get_client()
    async with client:
        try:
            nbs = await client.notebooks.list()
            return {"ok": True, "items": [nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__ for nb in nbs]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks")
async def create_notebook(req: NotebookCreateReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.create(req.title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.get(notebook_id)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.notebooks.delete(notebook_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


@app.patch("/v1/notebooks/{notebook_id}/rename")
async def rename_notebook(notebook_id: str, req: NotebookRenameReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.rename(notebook_id, req.new_title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/summary")
async def get_notebook_summary(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            summary = await client.notebooks.get_summary(notebook_id)
            return {"ok": True, "summary": summary}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/description")
async def get_notebook_description(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            desc = await client.notebooks.get_description(notebook_id)
            return {"ok": True, "description": desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Sources
# ----------------------------
@app.get("/v1/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            items = await client.sources.list(notebook_id)
            return {"ok": True, "items": [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/url")
async def add_source_url(notebook_id: str, req: SourceAddUrlReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_url(notebook_id, req.url, wait=req.wait)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except TypeError:
            # some versions may not accept wait=; fall back
            try:
                src = await client.sources.add_url(notebook_id, req.url)
                return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
            except RPCError as e:
                raise map_rpc_error(e)
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/youtube")
async def add_source_youtube(notebook_id: str, req: SourceAddYoutubeReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_youtube(notebook_id, req.url, wait=req.wait)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except TypeError:
            try:
                src = await client.sources.add_youtube(notebook_id, req.url)
                return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
            except RPCError as e:
                raise map_rpc_error(e)
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/text")
async def add_source_text(notebook_id: str, req: SourceAddTextReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_text(notebook_id, req.title, req.content)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/file")
async def add_source_file(
    notebook_id: str,
    upload: UploadFile = File(...),
    mime_type: Optional[str] = Form(None),
):
    # Save to temp file first
    suffix = os.path.splitext(upload.filename or "")[1] or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"nb_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await upload.read())

    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_file(notebook_id, tmp_path, mime_type=mime_type)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/fulltext")
async def get_source_fulltext(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ft = await client.sources.get_fulltext(notebook_id, source_id)
            return {"ok": True, "fulltext": ft.model_dump() if hasattr(ft, "model_dump") else ft.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/guide")
async def get_source_guide(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            guide = await client.sources.get_guide(notebook_id, source_id)
            return {"ok": True, "guide": guide}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}/sources/{source_id}")
async def delete_source(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.sources.delete(notebook_id, source_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Chat
# ----------------------------
@app.post("/v1/notebooks/{notebook_id}/chat/ask")
async def chat_ask(notebook_id: str, req: ChatAskReq):
    client = await get_client()
    async with client:
        try:
            result = await client.chat.ask(notebook_id, req.question)
            trimmed_result = _trim_chat_result(result)
            _logger.debug(
                "chat/ask response prepared: notebook_id=%s conversation_id=%s turn_number=%s reference_count=%d",
                notebook_id,
                trimmed_result.get("conversation_id"),
                trimmed_result.get("turn_number"),
                len(trimmed_result.get("references", [])) if isinstance(trimmed_result.get("references"), list) else 0,
            )
            return {"ok": True, "result": trimmed_result}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Artifacts: list / generate / poll / download
# ----------------------------
@app.get("/v1/notebooks/{notebook_id}/artifacts")
async def list_artifacts(notebook_id: str, type: Optional[str] = None):
    client = await get_client()
    async with client:
        try:
            items = await client.artifacts.list(notebook_id, type=type) if type else await client.artifacts.list(notebook_id)
            return {"ok": True, "items": [a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/artifacts/generate")
async def generate_artifact(notebook_id: str, req: ArtifactGenerateReq):
    client = await get_client()
    async with client:
        try:
            t = req.type
            opts = normalize_artifact_generate_options(t, req.options)

            if t == "audio":
                status = await client.artifacts.generate_audio(notebook_id, **opts)
            elif t == "video":
                status = await client.artifacts.generate_video(notebook_id, **opts)
            elif t == "report":
                status = await client.artifacts.generate_report(notebook_id, **opts)
            elif t == "quiz":
                status = await client.artifacts.generate_quiz(notebook_id, **opts)
            elif t == "flashcards":
                status = await client.artifacts.generate_flashcards(notebook_id, **opts)
            elif t == "slide_deck":
                status = await client.artifacts.generate_slide_deck(notebook_id, **opts)
            elif t == "infographic":
                status = await client.artifacts.generate_infographic(notebook_id, **opts)
            elif t == "data_table":
                status = await client.artifacts.generate_data_table(notebook_id, **opts)
            elif t == "mind_map":
                # mind_map may return dict directly in docs :contentReference[oaicite:6]{index=6}
                out = await client.artifacts.generate_mind_map(notebook_id, **opts)
                return {"ok": True, "type": t, "result": out}
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported artifact type: {t}")

            # GenerationStatus commonly contains task_id :contentReference[oaicite:7]{index=7}
            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", {})
            return {"ok": True, "type": t, "status": payload}
        except HTTPException:
            raise
        except RPCError as e:
            raise map_rpc_error(e)
        except (ArtifactDownloadError, ArtifactNotFoundError, ArtifactNotReadyError, ArtifactParseError, TypeError, ValueError) as e:
            raise map_artifact_error(e)


@app.get("/v1/notebooks/{notebook_id}/artifacts/tasks/{task_id}")
async def poll_task(notebook_id: str, task_id: str, wait: bool = False):
    client = await get_client()
    async with client:
        try:
            if wait:
                status = await client.artifacts.wait_for_completion(notebook_id, task_id)
            else:
                status = await client.artifacts.poll_status(notebook_id, task_id)

            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", status)
            return {"ok": True, "status": payload}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/artifacts/download")
async def download_artifact(
    notebook_id: str,
    type: Literal[
        "audio",
        "video",
        "infographic",
        "slide_deck",
        "report",
        "mind_map",
        "data_table",
        "quiz",
        "flashcards",
    ],
    artifact_id: Optional[str] = None,
    output_format: Optional[Literal["json", "markdown", "html", "pdf", "pptx"]] = None,
):
    """
    Downloads the *first completed* artifact of the given type unless artifact_id is provided.
    notebooklm-py provides type-specific download_* methods. :contentReference[oaicite:8]{index=8}
    """
    slide_deck_output_format: Optional[Literal["pdf", "pptx"]] = None
    if type in {"quiz", "flashcards"}:
        if output_format is not None and output_format not in {"json", "markdown", "html"}:
            raise HTTPException(
                status_code=400,
                detail=f"output_format for {type} must be one of json, markdown, or html.",
            )
    elif type == "slide_deck":
        if output_format is not None and output_format not in {"pdf", "pptx"}:
            raise HTTPException(
                status_code=400,
                detail="output_format for slide_deck must be either pdf or pptx.",
            )
        slide_deck_output_format = output_format or "pdf"
    elif output_format is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "output_format is only supported for quiz, flashcards, and slide_deck. "
                f"{type} downloads are always returned as {FIXED_DOWNLOAD_FORMATS[type]}."
            ),
        )

    suffix_map = {
        "audio": ".mp4",
        "video": ".mp4",
        "infographic": ".png",
        "slide_deck": ".pptx" if slide_deck_output_format == "pptx" else ".pdf",
        "report": ".md",
        "mind_map": ".json",
        "data_table": ".csv",
        "quiz": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
        "flashcards": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
    }
    out_path = os.path.join(tempfile.gettempdir(), f"nlm_{uuid.uuid4().hex}{suffix_map[type]}")

    client = await get_client()
    async with client:
        try:
            if type == "audio":
                await client.artifacts.download_audio(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "video":
                await client.artifacts.download_video(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "infographic":
                await client.artifacts.download_infographic(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "slide_deck":
                await client.artifacts.download_slide_deck(
                    notebook_id,
                    out_path,
                    artifact_id=artifact_id,
                    output_format=slide_deck_output_format or "pdf",
                )
            elif type == "report":
                await client.artifacts.download_report(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "mind_map":
                await client.artifacts.download_mind_map(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "data_table":
                await client.artifacts.download_data_table(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "quiz":
                await client.artifacts.download_quiz(
                    notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
                )
            elif type == "flashcards":
                await client.artifacts.download_flashcards(
                    notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported type: {type}")

            filename = os.path.basename(out_path)
            media_type_map = {
                "audio": "video/mp4",
                "video": "video/mp4",
                "infographic": "image/png",
                "slide_deck": (
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    if slide_deck_output_format == "pptx"
                    else "application/pdf"
                ),
                "report": "text/markdown; charset=utf-8",
                "mind_map": "application/json",
                "data_table": "text/csv; charset=utf-8",
                "quiz": (
                    "application/json"
                    if output_format in (None, "json")
                    else ("text/markdown; charset=utf-8" if output_format == "markdown" else "text/html; charset=utf-8")
                ),
                "flashcards": (
                    "application/json"
                    if output_format in (None, "json")
                    else ("text/markdown; charset=utf-8" if output_format == "markdown" else "text/html; charset=utf-8")
                ),
            }
            return FileResponse(out_path, filename=filename, media_type=media_type_map[type])
        except HTTPException:
            cleanup_temp_file(out_path)
            raise
        except RPCError as e:
            cleanup_temp_file(out_path)
            raise map_rpc_error(e)
        except (ArtifactDownloadError, ArtifactNotFoundError, ArtifactNotReadyError, ArtifactParseError, TypeError, ValueError) as e:
            cleanup_temp_file(out_path)
            raise map_artifact_error(e)

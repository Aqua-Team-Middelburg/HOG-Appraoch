"""
Lightweight FastAPI server to drive the Nurdle pipeline.

Endpoints
- POST /run: start a pipeline run with selected stages and optional config overrides
- GET /runs/{job_id}: fetch status.json for a run
- GET /runs/{job_id}/logs: return log text for a run (best-effort)
- POST /save: trigger the save stage to zip current output to a target directory
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline.py"
OUTPUT_DIR = ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
LOGS_DIR = OUTPUT_DIR / "logs"
ARTIFACT_ROOTS = [OUTPUT_DIR, ROOT / "saved_outputs", ROOT / "temp"]

app = FastAPI(title="Nurdle Pipeline API", version="1.0.0")
STATIC_DIR = ROOT / "frontend" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    steps: Optional[List[str]] = Field(
        default=None,
        description="Stages to run: normalization,features,tuning,training,evaluation,save",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None, description="Whitelisted config overrides applied for this run"
    )
    job_id: Optional[str] = Field(default=None, description="Optional job id")


class SaveRequest(BaseModel):
    save_dir: str
    job_id: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None


def _ensure_job_id(job_id: Optional[str]) -> str:
    return job_id or str(uuid.uuid4())


def _run_pipeline(job_id: str, steps: Optional[List[str]], overrides: Optional[Dict[str, Any]]) -> subprocess.Popen:
    # Use the current interpreter so we stay inside the active venv (avoids missing deps)
    args = [sys.executable, str(PIPELINE)]
    if steps:
        args += ["--steps", ",".join(steps)]
    args += ["--job-id", job_id]
    if overrides:
        args += ["--config-overrides", json.dumps(overrides)]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / job_id / "api.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return proc


@app.get("/", include_in_schema=False)
def root():
    """Redirect to interactive docs for convenience."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/docs")


@app.post("/run")
def start_run(req: RunRequest):
    job_id = _ensure_job_id(req.job_id)
    proc = _run_pipeline(job_id, req.steps, req.config_overrides)
    return {"job_id": job_id, "pid": proc.pid}


@app.get("/runs/{job_id}")
def get_status(job_id: str):
    status_path = RUNS_DIR / job_id / "status.json"
    if not status_path.exists():
        job_dir = RUNS_DIR / job_id
        # If the job was started but status.json isn't written yet, return a pending status
        if job_dir.exists():
            log_path = job_dir / "api.log"
            status = {"job_id": job_id, "status": "pending", "detail": "status.json not written yet"}
            if log_path.exists():
                log_txt = log_path.read_text(encoding="utf-8", errors="ignore").lower()
                if "pipeline completed successfully" in log_txt:
                    status["status"] = "success"
                    status["detail"] = "completed (log-derived)"
                elif "pipeline interrupted" in log_txt or "keyboardinterrupt" in log_txt:
                    status["status"] = "failed"
                    status["detail"] = "interrupted (log-derived)"
            return status
        raise HTTPException(status_code=404, detail="status.json not found")
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/runs/{job_id}/logs")
def get_logs(job_id: str):
    log_path = RUNS_DIR / job_id / "api.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="log not found")
    return {"job_id": job_id, "log": log_path.read_text(encoding="utf-8")}


def _resolve_artifact(path: str) -> Path:
    """Resolve a user-supplied relative path against allowed roots."""
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    candidate = Path(path)
    # Disallow absolute paths
    if candidate.is_absolute():
        raise HTTPException(status_code=400, detail="absolute paths not allowed")
    # If the path already starts with a known root name, resolve directly under repo root
    if candidate.parts and candidate.parts[0] in {"output", "saved_outputs", "temp"}:
        full = (ROOT / candidate).resolve()
        for root in ARTIFACT_ROOTS:
            try:
                full.relative_to(root.resolve())
                return full
            except ValueError:
                continue
        raise HTTPException(status_code=400, detail="path not under allowed artifact roots")

    # Otherwise, try each allowed root joined with the candidate
    for root in ARTIFACT_ROOTS:
        full = (root / candidate).resolve()
        try:
            full.relative_to(root.resolve())
            return full
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail="path not under allowed artifact roots")


@app.get("/artifacts")
def list_artifacts(path: str):
    """
    List files under a relative path rooted at output/ or saved_outputs/.
    Returns name, path, is_dir, size bytes, modified timestamp.
    """
    full = _resolve_artifact(path)
    if not full.exists():
        raise HTTPException(status_code=404, detail="path not found")
    if not full.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")
    items = []
    for p in sorted(full.iterdir()):
        stat = p.stat()
        items.append(
            {
                "name": p.name,
                "path": str(p.relative_to(ROOT)),
                "is_dir": p.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return {"path": str(full.relative_to(ROOT)), "items": items}


@app.get("/artifact")
def get_artifact(path: str):
    """Download/preview a file under allowed artifact roots."""
    full = _resolve_artifact(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(full)


@app.post("/save")
def save_outputs(req: SaveRequest):
    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=400, detail="output/ does not exist; run pipeline first")
    overrides = req.config_overrides or {}
    overrides.setdefault("data", {})["save_dir"] = req.save_dir
    job_id = _ensure_job_id(req.job_id)
    proc = _run_pipeline(job_id, ["save"], overrides)
    return {"job_id": job_id, "pid": proc.pid}


@app.get("/config")
def get_config():
    cfg = load_config()
    return cfg._config or {}

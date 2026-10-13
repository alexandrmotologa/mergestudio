"""FastAPI web studio application for visual model merging."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mergestudio.core.config import MergePlan
from mergestudio.streaming.engine import MergeProgress, ModelIndex, StreamingMergeEngine


class InspectRequest(BaseModel):
    path: str


class MergeJobState:
    def __init__(self, job_id: str, plan: MergePlan):
        self.job_id = job_id
        self.plan = plan
        self.status = "queued"  # queued, running, completed, failed
        self.progress: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.subscribers: List[WebSocket] = []


jobs_lock = threading.Lock()
jobs: Dict[str, MergeJobState] = {}


def create_app() -> FastAPI:
    app = FastAPI(
        title="MergeStudio Web Studio",
        description="Visual model merging, layer surgery, and weight interpolation studio for local LLMs",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "mergestudio", "version": "0.1.0"}

    @app.post("/api/inspect")
    async def inspect_model(req: InspectRequest) -> Dict[str, Any]:
        p = Path(req.path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Model directory not found: {req.path}")

        try:
            idx = ModelIndex(p)
            tensor_names = idx.tensor_names
            config_data = {}
            cfg_file = p / "config.json"
            if cfg_file.exists():
                with open(cfg_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

            return {
                "path": str(p),
                "total_tensors": len(tensor_names),
                "tensors_sample": tensor_names[:20],
                "config": config_data,
                "architectures": config_data.get("architectures", []),
                "num_hidden_layers": config_data.get("num_hidden_layers", 0),
                "hidden_size": config_data.get("hidden_size", 0),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/plan/validate")
    async def validate_plan(plan_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            plan = MergePlan.model_validate(plan_data)
            return {"valid": True, "plan": plan.model_dump(mode="json")}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    @app.post("/api/merge/start")
    async def start_merge(plan_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            plan = MergePlan.model_validate(plan_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid merge plan: {exc}")

        job_id = str(uuid.uuid4())
        job = MergeJobState(job_id, plan)

        with jobs_lock:
            jobs[job_id] = job

        # Run merge in background daemon thread
        threading.Thread(target=_run_job_worker, args=(job,), daemon=True).start()

        return {"job_id": job_id, "status": "running"}

    @app.get("/api/merge/status/{job_id}")
    async def get_merge_status(job_id: str) -> Dict[str, Any]:
        with jobs_lock:
            job = jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "output_dir": job.output_dir,
        }

    @app.websocket("/ws/merge/{job_id}")
    async def websocket_merge(websocket: WebSocket, job_id: str):
        await websocket.accept()
        with jobs_lock:
            job = jobs.get(job_id)
            if job:
                job.subscribers.append(websocket)

        if not job:
            await websocket.send_json({"error": "Job not found"})
            await websocket.close()
            return

        try:
            while job.status in ("queued", "running"):
                await asyncio.sleep(0.5)
                if job.progress:
                    await websocket.send_json({
                        "job_id": job.job_id,
                        "status": job.status,
                        "progress": job.progress,
                    })

            await websocket.send_json({
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "output_dir": job.output_dir,
                "error": job.error,
            })
        except WebSocketDisconnect:
            pass
        finally:
            with jobs_lock:
                if job and websocket in job.subscribers:
                    job.subscribers.remove(websocket)

    @app.get("/")
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>MergeStudio Studio API Running</h1>")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def _run_job_worker(job: MergeJobState) -> None:
    job.status = "running"
    engine = StreamingMergeEngine(job.plan)

    def on_progress(p: MergeProgress):
        job.progress = asdict(p)

    try:
        out_dir = engine.run(progress_callback=on_progress)
        job.status = "completed"
        job.output_dir = str(out_dir)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)

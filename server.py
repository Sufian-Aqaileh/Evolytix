import json
import mimetypes
import shutil
import subprocess
import sys
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import socket


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def get_python_executable() -> Path:
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    exe = Path(sys.executable)
    if exe.exists():
        return exe
    python_exe = shutil.which("python")
    if python_exe and Path(python_exe).exists():
        return Path(python_exe)
    python_exe = shutil.which("python3")
    if python_exe and Path(python_exe).exists():
        return Path(python_exe)
    local_app_data = Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps"
    for name in ("python.exe", "python3.exe"):
        candidate = local_app_data / name
        if candidate.exists():
            return candidate
    raise RuntimeError("No Python executable found.")


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
UPLOAD_DIR = ROOT / "runtime" / "uploads"
GENERATED_DIR = PUBLIC_DIR / "generated"
PREP_SCRIPT = ROOT / "backend" / "data_preparation.py"
WORKERS = {
    "auditing": {
        "script": ROOT / "backend" / "test_adaptive_auditing.py",
        "prefix": "audit",
        "missing_dependency": "tensorflow",
        "dependency_message": (
            "TensorFlow is not installed in the audit worker environment. "
            "Create a compatible .venv and install requirements.txt."
        ),
    },
    "forecasting": {
        "script": ROOT / "backend" / "02_financial_forecasting.py",
        "prefix": "forecast",
    },
    "advisory": {
        "script": ROOT / "backend" / "03_financial_optimization_advisory.py",
        "prefix": "advisory",
    },
}
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
HOST = "0.0.0.0"
PORT = 8000
FALLBACK_PORTS = [8000, 5000, 8888, 8080]


class EvolytixRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"ok": True, "service": "Evolytix analysis API"})
            return

        self._serve_public_file()

    def do_POST(self) -> None:
        if self.path not in ("/api/audit", "/api/analyze", "/api/preprocess", "/api/model"):
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            if self.path == "/api/preprocess":
                csv_text = payload.get("csvText", "")
                if not csv_text.strip():
                    self._send_json({"error": "CSV text is required."}, status=400)
                    return
                self._send_json(run_preprocessing_worker(csv_text))
                return

            if self.path == "/api/model":
                model = payload.get("model", "")
                run_id = payload.get("runId", "")
                if model not in WORKERS:
                    self._send_json({"error": "Choose a valid analysis model."}, status=400)
                    return
                if not is_valid_run_id(run_id):
                    self._send_json({"error": "Run preprocessing before selecting a model."}, status=400)
                    return
                self._send_json(run_model_from_prepared_input(model, run_id))
                return

            csv_text = payload.get("csvText", "")
            model = payload.get("model", "auditing" if self.path == "/api/audit" else "")
            if not csv_text.strip():
                self._send_json({"error": "CSV text is required."}, status=400)
                return
            if model not in WORKERS:
                self._send_json({"error": "Choose a valid analysis model."}, status=400)
                return

            self._send_json(run_model_worker(model, csv_text))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"Analysis run failed: {exc}"}, status=500)

    def _serve_public_file(self) -> None:
        request_path = unquote(self.path.split("?", 1)[0])
        if request_path in ("", "/"):
            request_path = "/index.html"

        target = (PUBLIC_DIR / request_path.lstrip("/")).resolve()
        if not str(target).startswith(str(PUBLIC_DIR.resolve())):
            self._send_json({"error": "Invalid path"}, status=403)
            return

        if target.is_dir():
            target = target / "index.html"

        if not target.exists():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_model_worker(model: str, csv_text: str) -> dict:
    prepared = run_preprocessing_worker(csv_text, prefix=WORKERS[model]["prefix"])
    result = run_model_from_prepared_input(model, prepared["runId"])
    result["preprocessing"] = prepared["preprocessing"]
    return result


def run_preprocessing_worker(csv_text: str, prefix: str = "prep") -> dict:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    run_id = f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    csv_path = UPLOAD_DIR / f"{run_id}.csv"
    paths = get_run_paths(run_id)
    csv_path.write_text(csv_text, encoding="utf-8")

    python_executable = get_python_executable()
    run_data_preparation(
        python_executable=python_executable,
        input_path=csv_path,
        cleaned_output_path=paths["prepared_parquet"],
        report_output_path=paths["quality_report"],
    )

    return {
        "runId": run_id,
        "preprocessing": build_preprocessing_payload(run_id),
    }


def run_model_from_prepared_input(model: str, run_id: str) -> dict:
    worker = WORKERS[model]
    paths = get_run_paths(run_id)
    prepared_parquet_path = paths["prepared_parquet"]
    output_dir = paths["output_dir"]

    if not prepared_parquet_path.exists():
        raise RuntimeError("Prepared parquet file was not found. Run preprocessing again.")

    python_executable = get_python_executable()
    command = [
        str(python_executable),
        str(worker["script"]),
        "--web-json",
        "--csv",
        str(prepared_parquet_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        missing_dependency = worker.get("missing_dependency")
        if missing_dependency and f"No module named '{missing_dependency}'" in details:
            raise RuntimeError(worker["dependency_message"])
        raise RuntimeError(details or "The analysis worker exited without details.")

    web_json_line = None
    for line in completed.stdout.splitlines():
        if line.startswith("WEB_JSON:"):
            web_json_line = line[len("WEB_JSON:") :]

    if not web_json_line:
        raise RuntimeError("The audit worker did not return a WEB_JSON payload.")

    result = json.loads(web_json_line)
    result["model"] = model
    result["runId"] = run_id
    result["preprocessing"] = build_preprocessing_payload(run_id)
    if result.get("plotFile"):
        result["plotUrl"] = f"/generated/{run_id}/{result['plotFile']}"
    return result


def get_run_paths(run_id: str) -> dict:
    output_dir = GENERATED_DIR / run_id
    return {
        "output_dir": output_dir,
        "prepared_parquet": output_dir / "financial_accounting_cleaned.parquet",
        "cleaned_csv": output_dir / "financial_accounting_cleaned.csv",
        "quality_report": output_dir / "data_quality_report.csv",
    }


def build_preprocessing_payload(run_id: str) -> dict:
    paths = get_run_paths(run_id)
    return {
        "preparedParquet": paths["prepared_parquet"].name,
        "cleanedCsvUrl": f"/generated/{run_id}/{paths['cleaned_csv'].name}",
        "qualityReportUrl": f"/generated/{run_id}/{paths['quality_report'].name}",
    }


def is_valid_run_id(run_id: str) -> bool:
    if not isinstance(run_id, str) or not run_id:
        return False
    if any(part in run_id for part in ("..", "/", "\\")):
        return False
    return (GENERATED_DIR / run_id / "financial_accounting_cleaned.parquet").exists()


def run_data_preparation(
    python_executable: Path,
    input_path: Path,
    cleaned_output_path: Path,
    report_output_path: Path,
) -> None:
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(PREP_SCRIPT),
        str(input_path),
        str(cleaned_output_path),
        str(report_output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        if "Missing optional dependency 'pyarrow'" in details or "No module named 'pyarrow'" in details:
            raise RuntimeError(
                "pyarrow is not installed in the worker environment. "
                "Install requirements.txt so data_preparation.py can write parquet files."
            )
        raise RuntimeError(details or "Data preparation failed without details.")


def main() -> None:
    local_ip = get_local_ip()
    server = None
    used_port = None
    for port in FALLBACK_PORTS:
        try:
            server = ThreadingHTTPServer((HOST, port), EvolytixRequestHandler)
            used_port = port
            break
        except OSError:
            continue

    if server is None:
        print("Could not bind to any fallback port.")
        raise SystemExit(1)

    print(f"Evolytix server running at http://localhost:{used_port}")
    print(f"Or from another device on this network: http://{local_ip}:{used_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

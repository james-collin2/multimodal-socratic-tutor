"""
scripts/setup.py

Cross-platform setup for socratOT.
Python 3.11 | macOS Apple Silicon | Windows | Linux

Usage:
    python scripts/setup.py                # full setup
    python scripts/setup.py --cpu-only     # force CPU torch
    python scripts/setup.py --skip-whisper # skip whisper install
    python scripts/setup.py --dry-run      # print steps only
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}»{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def run(
    cmd: list[str], check: bool = False, dry_run: bool = False, label: str | None = None
) -> int:
    display = label or " ".join(str(c) for c in cmd)
    info(display)
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if check:
            err(f"Failed (exit {result.returncode}): {display}")
        else:
            warn(f"Non-fatal failure (exit {result.returncode}): {display}")
    return result.returncode


def pip(*packages: str, dry_run: bool = False) -> int:
    return run(
        [sys.executable, "-m", "pip", "install", "--upgrade", *packages],
        check=False,
        dry_run=dry_run,
        label=f"pip install {' '.join(packages)}",
    )


# ─────────────────────────────────────────────────────────────


def check_python(dry_run: bool = False) -> None:
    section("Python version check")
    v = sys.version_info
    if v < (3, 9):
        err(f"Python 3.9+ required — found {v.major}.{v.minor}")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")


def create_env_file(dry_run: bool = False) -> None:
    section(".env setup")
    # Look for .env.example in ROOT (project root where socratOT/ lives)
    # and also one level up (workspace root)
    candidates = [
        ROOT / ".env.example",
        ROOT.parent / "socratOT" / ".env.example",
    ]
    src = next((p for p in candidates if p.exists()), None)
    dst = ROOT / ".env"

    if dst.exists():
        ok(".env already exists — skipping")
        return
    if src is None:
        warn(".env.example not found — creating minimal .env from defaults")
        if not dry_run:
            _write_default_env(dst)
        ok(".env created with defaults — edit before running the app")
        return
    if not dry_run:
        shutil.copy(src, dst)
    ok(f".env created from {src.name}")
    warn("Review and edit .env before running the app")


def _write_default_env(dst: Path) -> None:
    """Write a minimal .env if no .env.example is found."""
    content = """\
APP_ENV=development
APP_DEBUG=true
APP_LOG_LEVEL=INFO
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/processed/chroma_db
FAISS_INDEX_PATH=./data/processed/faiss_index/index.bin
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=5
MIN_RELEVANCE_SCORE=0.35
MAX_HINT_TURNS=2
MAX_SESSION_TURNS=30
SOCRATIC_STRICT_MODE=true
DATABASE_URL=sqlite+aiosqlite:///./data/socratot.db
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
LOG_DIR=./logs
LOG_ROTATION=10 MB
EVAL_DATASET_PATH=./evaluation/ground_truth.jsonl
"""
    dst.write_text(content)


def create_dirs(dry_run: bool = False) -> None:
    section("Creating project directories")
    dirs = [
        "data/raw/openStax",
        "data/raw/anatomyTool",
        "data/raw/medPix",
        "data/processed/chroma_db",
        "data/processed/faiss_index",
        "data/processed/chunks",
        "data/images",
        "data/models",
        "logs",
        "evaluation/results",
    ]
    for d in dirs:
        if not dry_run:
            (ROOT / d).mkdir(parents=True, exist_ok=True)
        ok(d)


def upgrade_pip(dry_run: bool = False) -> None:
    section("Upgrading pip + setuptools + wheel")
    pip("pip", "setuptools>=70.0.0", "wheel>=0.44.0", "packaging>=24.0", dry_run=dry_run)
    ok("pip toolchain upgraded")


def install_requirements(dry_run: bool = False) -> None:
    section("Installing requirements.txt")
    req = ROOT / "requirements.txt"
    code = run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        check=False,
        dry_run=dry_run,
        label="pip install -r requirements.txt",
    )
    if code == 0:
        ok("requirements.txt installed successfully")
    else:
        err("Dependency conflicts detected — see output above")
        warn("Common fix: check langchain-core version in requirements.txt")


def install_torch(cpu_only: bool = False, dry_run: bool = False) -> None:
    section("Installing PyTorch (platform-specific)")
    system = platform.system()
    machine = platform.machine()
    TORCH = "torch==2.4.1"
    VISION = "torchvision==0.19.1"

    if cpu_only:
        info("CPU-only forced")
        pip(TORCH, VISION, "--index-url", "https://download.pytorch.org/whl/cpu", dry_run=dry_run)
    elif system == "Darwin" and machine == "arm64":
        ok("Apple Silicon — MPS wheel")
        pip(TORCH, VISION, dry_run=dry_run)
    elif system == "Linux":
        info("Linux — CUDA 12.1 wheel")
        pip(TORCH, VISION, "--index-url", "https://download.pytorch.org/whl/cu121", dry_run=dry_run)
    elif system == "Windows":
        info("Windows — CUDA 12.1 wheel")
        pip(TORCH, VISION, "--index-url", "https://download.pytorch.org/whl/cu121", dry_run=dry_run)
    else:
        warn(f"Unknown platform ({system}/{machine}) — CPU torch")
        pip(TORCH, VISION, "--index-url", "https://download.pytorch.org/whl/cpu", dry_run=dry_run)


def install_whisper(dry_run: bool = False) -> None:
    section("Installing openai-whisper (speech-to-text)")
    # ── ROOT CAUSE ────────────────────────────────────────────────────────
    # openai-whisper uses a legacy setup.py that calls pkg_resources at
    # build time. pip creates an ISOLATED build environment (a fresh temp
    # venv) that does NOT inherit your upgraded setuptools — it installs
    # the version declared in whisper's build-requires, which can be old
    # and missing pkg_resources.
    #
    # FIX: use --no-build-isolation so pip reuses THIS venv's setuptools
    #      instead of creating a fresh isolated one.
    # ─────────────────────────────────────────────────────────────────────
    info("Using --no-build-isolation to inherit upgraded setuptools")
    code = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "openai-whisper==20240930",
        ],
        check=False,
        dry_run=dry_run,
        label="pip install --no-build-isolation openai-whisper==20240930",
    )
    if code == 0:
        ok("openai-whisper installed")
    else:
        warn("openai-whisper failed — STT feature will be unavailable")
        warn("Manual fix:")
        warn("  pip install --upgrade setuptools")
        warn("  pip install --no-build-isolation openai-whisper==20240930")


def setup_precommit(dry_run: bool = False) -> None:
    section("Setting up pre-commit hooks")
    # Only works inside a git repo
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        # Check parent dirs up to 3 levels
        found = any(
            (ROOT.parent / (".git" if i == 0 else "../" * i + ".git")).exists() for i in range(3)
        )
        if not found:
            warn("Not inside a git repo — skipping pre-commit setup")
            return
    code = run(["pre-commit", "install"], check=False, dry_run=dry_run, label="pre-commit install")
    if code == 0:
        ok("pre-commit hooks installed")
    else:
        warn("pre-commit not available — optional, skipping")


def verify_install(dry_run: bool = False) -> None:
    section("Verifying core imports")
    checks = [
        ("pydantic", "from pydantic import BaseModel"),
        ("pydantic-settings", "from pydantic_settings import BaseSettings"),
        ("loguru", "from loguru import logger"),
        ("langchain", "import langchain"),
        ("langchain-community", "import langchain_community"),
        ("chromadb", "import chromadb"),
        ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
        ("sqlalchemy", "import sqlalchemy"),
        ("yaml (PyYAML)", "import yaml"),
        ("torch", "import torch"),
        ("PIL (Pillow)", "from PIL import Image"),
        ("streamlit", "import streamlit"),
    ]
    failed = []
    for name, stmt in checks:
        if dry_run:
            ok(name)
            continue
        result = subprocess.run(
            [sys.executable, "-c", stmt],
            capture_output=True,
        )
        if result.returncode == 0:
            ok(name)
        else:
            err(f"{name}  ← import failed")
            failed.append(name)

    print()
    if not failed:
        ok("All core imports verified — environment is ready")
    else:
        warn(f"{len(failed)} import(s) failed: {', '.join(failed)}")
        warn("Re-run setup or install missing packages manually")


# ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="socratOT setup")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dr = args.dry_run

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  socratOT — Project Setup{RESET}")
    print(
        f"{BOLD}  Python {sys.version_info.major}.{sys.version_info.minor}"
        f" | {platform.system()} {platform.machine()}{RESET}"
    )
    if dr:
        print(f"  {YELLOW}DRY RUN — no changes will be made{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    check_python(dr)
    create_env_file(dr)
    create_dirs(dr)

    # Install order is critical:
    # 1. Upgrade pip + setuptools FIRST
    # 2. Install requirements.txt (uses upgraded setuptools)
    # 3. Install torch (platform wheel, outside requirements.txt)
    # 4. Install whisper with --no-build-isolation (uses this venv's setuptools)
    upgrade_pip(dr)
    install_requirements(dr)
    install_torch(cpu_only=args.cpu_only, dry_run=dr)
    if not args.skip_whisper:
        install_whisper(dr)

    setup_precommit(dr)
    verify_install(dr)

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Setup complete!{RESET}")
    print()
    print(f"  {CYAN}1.{RESET} Edit .env — add your OPENAI_API_KEY")
    print(f"  {CYAN}2.{RESET} pytest tests/unit/test_phase1.py -v")
    print(f"  {CYAN}3.{RESET} streamlit run app/main.py")
    print(f"{BOLD}{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()

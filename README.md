# socratOT 🧠

> Socratic Multimodal RAG Tutor for Occupational Therapy Education

Teaches OT students Anatomy & Neuroscience through guided questioning —
never direct answers. GPT-4o vision, ChromaDB RAG, Socratic engine,
student memory, and clinical assessment.

## Quick start

```bash
git clone https://github.com/your_username/multimodal-socratic-tutor.git
cd multimodal-socratic-tutor
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
python scripts/ingest_corpus.py --sample
pytest tests/ -v
python -m streamlit run app/main.py
```

---

## Environment

```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## Ingestion

```bash
python scripts/ingest_corpus.py --sample          # built-in corpus
python scripts/ingest_corpus.py --pdf FILE.pdf    # full OpenStax PDF
python scripts/download_images.py                 # extract from PDF
```

---

## Evaluation

```bash
python evaluation/run_evaluation.py --quick       # fast (5 samples)
python evaluation/run_evaluation.py               # full (20 samples)
python evaluation/run_evaluation.py --compliance  # bypass detection only
python evaluation/run_evaluation.py --ragas       # RAGAS metrics only
python evaluation/run_evaluation.py --baseline    # benchmark comparison
```

Results saved to `evaluation/results/` and `docs/phase5_results.md`.

---

## Deployment

**Streamlit Community Cloud:**
1. Go to `share.streamlit.io` → New app
2. Select repo `your_username/multimodal-socratic-tutor`, branch `main`, file `app/main.py`
3. Add secrets: `OPENAI_API_KEY = "sk-..."`
4. Deploy → get public URL

**Docker:**
```bash
docker compose up --build -d   # single Streamlit service; uses OpenAI API
docker compose logs -f app
```

---

## Structure

```
app/                    Streamlit UI (chat, dashboard, image analysis)
config/                 Settings, prompts, topics
evaluation/             RAGAS, compliance, baseline evaluators
src/
  core/conversation/   Socratic engine + state machine
  core/rag/            RAG pipeline
  core/multimodal/     GPT-4o vision pipeline
  core/memory/         Cross-session student memory
  core/assessment/     Clinical scenario + scoring
  models/              LLM + embedding providers
  schemas/             Pydantic models
scripts/               Ingestion + setup utilities
tests/                 142 passing unit + integration tests
```

---

## Code style & pre-commit

Linting and formatting are enforced by [ruff](https://docs.astral.sh/ruff/)
via a pre-commit hook (`.pre-commit-config.yaml`).

**Why a commit can "Fail" on `ruff`:** the hook runs `ruff --fix` and
`ruff-format`, which *auto-fix and reformat* your files. When pre-commit
modifies a file, it intentionally fails the commit so you can review the
changes — it is **not** a code error. Just re-stage and commit again:

```bash
git add -A && git commit   # the fixes are already applied; this time it passes
```

To format everything yourself *before* committing (so the hook never has to):

```bash
ruff check --fix .   # auto-fix lint issues
ruff format .        # apply formatting
pytest -q            # confirm still green
git add -A && git commit -m "..."
```

One-time install of the hook in a fresh clone:

```bash
pre-commit install
```

---

## Known fixes

```bash
# watchdog conflict on Linux
sed -i 's/watchdog==5.0.3/watchdog==4.0.2/' requirements.txt

# ChromaDB telemetry — add to .env
ANONYMIZED_TELEMETRY=False
CHROMA_TELEMETRY=false
```

---

## License

MIT

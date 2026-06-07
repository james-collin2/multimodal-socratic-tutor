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
python -m streamlit run src/app/main.py
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
```

---

## Evaluation

```bash
python src/evaluation/run_evaluation.py --quick       # fast (5 samples)
python src/evaluation/run_evaluation.py               # full (20 samples)
python src/evaluation/run_evaluation.py --compliance  # bypass detection only
python src/evaluation/run_evaluation.py --ragas       # RAGAS metrics only
python src/evaluation/run_evaluation.py --baseline    # benchmark comparison
```

Results saved to `src/evaluation/results/`.

---

## Deployment

**Streamlit Community Cloud:**
1. Go to `share.streamlit.io` → New app
2. Select repo, branch `main`, file `src/app/main.py`
3. Add secrets: `OPENAI_API_KEY = "sk-..."`
4. Deploy → get public URL

**Docker:**
```bash
docker compose up --build -d
docker compose logs -f app
```

---

## Structure

```
src/
  app/                   Streamlit UI (chat, dashboard, image analysis)
  config/                Settings, prompts, topics
  evaluation/            RAGAS, compliance, baseline evaluators
  core/conversation/     Socratic engine + state machine
  core/rag/              RAG pipeline
  core/multimodal/       GPT-4o vision pipeline
  core/memory/           Cross-session student memory
  core/assessment/       Clinical scenario + scoring
  models/                LLM + embedding providers
  schemas/               Pydantic models
  prompts/               Prompt loader
  utils/                 Helpers, logging, exceptions
scripts/                 Ingestion + setup utilities
tests/                   Unit + integration tests
```

---

## Code style & pre-commit

Linting and formatting are enforced by [ruff](https://docs.astral.sh/ruff/)
via a pre-commit hook (`.pre-commit-config.yaml`).

When pre-commit runs `ruff --fix` and `ruff-format`, it intentionally fails
the commit if it modifies files so you can review the changes — it is **not**
a code error. Just re-stage and commit again:

```bash
git add -A && git commit
```

To format before committing:

```bash
ruff check --fix . && ruff format .
pytest -q
git add -A && git commit -m "..."
```

One-time hook install:

```bash
pre-commit install
```

---

## License

MIT

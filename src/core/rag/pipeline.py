"""
src/core/rag/pipeline.py

RAGPipeline — top-level orchestrator for retrieval-augmented generation.
Single responsibility: query in → grounded LLM answer out.

Pipeline:
    query → Retriever → context assembly → LLM prompt → HallucinationGuard → answer
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.rag.hallucination_guard import GuardResult, HallucinationGuard
from src.core.rag.retriever import Retriever
from src.models.llm_factory import get_llm
from src.schemas.rag import RetrievalResult
from src.utils.exceptions import LLMUnavailableError
from src.utils.logger import logger


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    query: str
    answer: str
    retrieval: RetrievalResult
    guard_result: GuardResult
    citations: list[str]
    is_grounded: bool

    @property
    def formatted_answer(self) -> str:
        """Answer with citations appended."""
        if not self.citations:
            return self.answer
        cite_str = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(self.citations))
        return f"{self.answer}\n\nSources:\n{cite_str}"


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.

    Args:
        retriever:    Retriever instance (injected)
        llm:          LangChain LLM instance (injected)
        guard:        HallucinationGuard instance (injected)
        system_prompt: Optional system prompt override
    """

    DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable anatomy and neuroscience tutor
for Occupational Therapy students. Answer questions accurately and concisely
using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question,
say so clearly. Do not introduce facts not present in the context."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: object | None = None,
        guard: HallucinationGuard | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Lazy-load LLM to avoid startup cost
        self._llm_instance = llm
        self._guard = guard

    def _get_llm(self) -> object:
        """Lazily load LLM instance."""
        if self._llm_instance is None:
            try:
                self._llm_instance = get_llm()
            except Exception as e:
                raise LLMUnavailableError(
                    "Cannot load LLM for RAG pipeline",
                    detail=str(e),
                ) from e
        return self._llm_instance

    def _get_guard(self) -> HallucinationGuard:
        """Lazily load HallucinationGuard."""
        if self._guard is None:
            try:
                llm = self._get_llm()
                self._guard = HallucinationGuard(llm=llm)
            except Exception:
                # Guard without LLM verification — keyword-only mode
                self._guard = HallucinationGuard(
                    llm=None,
                    use_llm_verification=False,
                )
        return self._guard

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        return f"""{self._system_prompt}

--- Retrieved Context ---
{context}
--- End Context ---

Question: {query}

Answer:"""

    async def query(self, user_query: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a user query.

        Args:
            user_query: The student's question

        Returns:
            RAGResponse with grounded answer and citations

        Raises:
            RetrievalError: If vector store retrieval fails
            LLMUnavailableError: If LLM is not reachable
        """
        logger.info("RAG pipeline query: {q!r}", q=user_query[:80])

        # Step 1: Retrieve relevant chunks
        retrieval = await self._retriever.retrieve(user_query)

        if not retrieval.has_results:
            logger.warning("No relevant chunks found for query")
            return RAGResponse(
                query=user_query,
                answer=(
                    "I could not find relevant information in the knowledge base "
                    "for this question. Please try rephrasing or ask your instructor."
                ),
                retrieval=retrieval,
                guard_result=GuardResult(
                    is_grounded=False,
                    confidence=0.0,
                    flagged_phrases=[],
                    reason="No context retrieved",
                ),
                citations=[],
                is_grounded=False,
            )

        # Step 2: Build prompt with retrieved context
        prompt = self._build_prompt(
            query=user_query,
            context=retrieval.assembled_context,
        )

        # Step 3: Generate answer with LLM
        try:
            import asyncio

            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            answer = (response.content if hasattr(response, "content") else str(response)).strip()

        except Exception as e:
            raise LLMUnavailableError(
                "LLM call failed in RAG pipeline",
                detail=str(e),
            ) from e

        # Step 4: Hallucination check
        guard = self._get_guard()
        guard_result = await guard.check(
            answer=answer,
            context=retrieval.assembled_context,
            query=user_query,
        )

        if not guard_result.is_grounded:
            logger.warning(
                "Hallucination guard flagged answer (confidence={c:.2f}): {r}",
                c=guard_result.confidence,
                r=guard_result.reason,
            )

        logger.info(
            "RAG pipeline complete: grounded={g}, citations={n}",
            g=guard_result.is_grounded,
            n=len(retrieval.citations),
        )

        return RAGResponse(
            query=user_query,
            answer=answer,
            retrieval=retrieval,
            guard_result=guard_result,
            citations=retrieval.citations,
            is_grounded=guard_result.is_grounded,
        )

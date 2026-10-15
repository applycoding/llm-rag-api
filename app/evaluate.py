"""Score the bundled corpus with the offline provider.

The script checks three things a demo can honestly measure: the expected file
was retrieved, the answer contains the expected words, and the answer cites a
numbered excerpt. It does not call a hosted model.
"""

import asyncio
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import ProviderName, Settings
from app.pipeline import Pipeline


@dataclass(frozen=True)
class EvalCase:
    question: str
    source: str
    keywords: tuple[str, ...]


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "How long does Lumen Archive retain audit logs?",
        "security.txt",
        ("180", "days"),
    ),
    EvalCase("Which plans include SSO?", "security.txt", ("Beacon", "Flare")),
    EvalCase("What is the recovery time objective?", "operations.md", ("4", "hours")),
    EvalCase(
        "What does an exported report watermark include?",
        "retention.pdf",
        ("requester", "email"),
    ),
    EvalCase(
        "Which regions are supported?",
        "overview.md",
        ("Iceland", "Portugal", "Japan"),
    ),
)


@dataclass(frozen=True)
class CaseResult:
    question: str
    retrieval_ok: bool
    keywords_ok: bool
    citation_ok: bool
    answer: str
    sources: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.retrieval_ok and self.keywords_ok and self.citation_ok


@dataclass(frozen=True)
class EvalReport:
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.ok for result in self.results)


def default_sample_dir() -> Path:
    candidates = (
        Path.cwd() / "sample_docs",
        Path(__file__).resolve().parents[1] / "sample_docs",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def run_evaluation(sample_dir: Path | None = None) -> EvalReport:
    sample = sample_dir or default_sample_dir()
    with tempfile.TemporaryDirectory() as temporary:
        settings = Settings(
            llm_provider=ProviderName.FAKE,
            llm_model="fake-extractive",
            data_dir=Path(temporary),
            sample_dir=sample,
            auto_ingest=False,
            top_k=4,
            chunk_size=800,
            chunk_overlap=120,
            embedding_dim=384,
            min_score=0.25,
        )
        pipeline = Pipeline(settings)
        pipeline.ingest_directory(sample)
        results = [asyncio.run(_run_case(pipeline, case)) for case in CASES]
    return EvalReport(tuple(results))


def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    for index, result in enumerate(report.results, start=1):
        status = "pass" if result.ok else "FAIL"
        lines.append(
            f"{index}. {status}  retrieval={'yes' if result.retrieval_ok else 'no'}  "
            f"keywords={'yes' if result.keywords_ok else 'no'}  "
            f"citation={'yes' if result.citation_ok else 'no'}  {result.question}"
        )
        if not result.ok:
            lines.append(f"   answer: {result.answer}")
            lines.append(f"   sources: {', '.join(result.sources) or '(none)'}")
    passed = sum(1 for result in report.results if result.ok)
    lines.append(f"{passed}/{len(report.results)} cases passed")
    return "\n".join(lines)


def main() -> int:
    report = run_evaluation()
    print(format_report(report))
    return 0 if report.passed else 1


def _has_term(text: str, term: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


async def _run_case(pipeline: Pipeline, case: EvalCase) -> CaseResult:
    answer = await pipeline.answer(case.question)
    sources = tuple(citation.source for citation in answer.citations)
    return CaseResult(
        question=case.question,
        retrieval_ok=case.source in sources,
        keywords_ok=all(_has_term(answer.answer, word) for word in case.keywords),
        citation_ok=any(f"[{citation.index}]" in answer.answer for citation in answer.citations),
        answer=answer.answer,
        sources=sources,
    )


if __name__ == "__main__":
    raise SystemExit(main())

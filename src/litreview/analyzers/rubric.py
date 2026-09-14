"""NeurIPS Paper Checklist and Rigor Rubric Analyzer using Zero-Shot NLI."""

import logging
from typing import Any
import pandas as pd
import torch
from transformers import pipeline

from litreview.analyzers.base import Analyzer

logger = logging.getLogger(__name__)

NEURIPS_RUBRIC_CRITERIA = {
    "limitations": "discusses limitations assumptions scope and potential failure modes",
    "reproducibility": "provides open source code repository dataset artifacts or reproduction commands",
    "experimental_rigor": "evaluates baselines ablation studies benchmarks and quantitative metrics",
    "statistical_significance": "reports confidence intervals error bars variance or statistical hypothesis tests",
    "compute_resources": "specifies compute resources hardware accelerators training time or gpu hours",
}


CRITERION_SECTION_TARGETS = {
    "limitations": ["limitations"],
    "compute_resources": ["compute", "experiments"],
    "reproducibility": ["reproducibility", "experiments"],
    "experimental_rigor": ["experiments", "methodology"],
    "statistical_significance": ["experiments"],
}


class NeurIPSRubricAnalyzer(Analyzer):
    """Evaluates paper abstracts and texts against NeurIPS checklist criteria.

    Uses zero-shot NLI entailment probabilities to quantify the methodological
    rigor and transparency of each study.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._pipeline = None
        self._results: pd.DataFrame | None = None

    def fit(self, texts: pd.Series) -> "NeurIPSRubricAnalyzer":
        """Initialize the classification pipeline on GPU or CPU."""
        if torch.cuda.is_available():
            device = 0
            use_bf16 = torch.cuda.is_bf16_supported()
        elif torch.cuda.device_count() > 0:
            device = 0
            use_bf16 = False
        else:
            device = -1
            use_bf16 = False

        dtype = torch.bfloat16 if use_bf16 else torch.float32

        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
            dtype=dtype,
        )
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Score each text against the NeurIPS rubric criteria.

        Returns:
            DataFrame with columns for each criterion score and an aggregated 'rubric_score'.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before transform()")

        candidate_labels = list(NEURIPS_RUBRIC_CRITERIA.values())
        criterion_names = list(NEURIPS_RUBRIC_CRITERIA.keys())

        empty_mask = texts.apply(lambda x: pd.isna(x) or not str(x).strip())
        empty_indices = set(texts[empty_mask].index.tolist())
        non_empty = texts[~empty_mask].tolist()

        scored_records = []
        if non_empty:
            pipeline_outputs = self._pipeline(
                non_empty,
                candidate_labels=candidate_labels,
                multi_label=True,
                batch_size=32,
                truncation=True,
                max_length=256,
            )

            # Ensure pipeline_outputs is iterable over results
            if isinstance(pipeline_outputs, dict):
                pipeline_outputs = [pipeline_outputs]

            for out in pipeline_outputs:
                label_to_score = dict(zip(out["labels"], out["scores"]))
                record = {}
                for name, label in zip(criterion_names, candidate_labels):
                    record[f"rubric_{name}"] = round(label_to_score.get(label, 0.0), 4)

                # Composite rigor score (mean of criteria)
                record["rubric_score"] = round(sum(record.values()) / len(record), 4)
                scored_records.append(record)

        # Re-align with original indices
        rec_iter = iter(scored_records)
        final_rows = []
        for idx in texts.index:
            if idx in empty_indices:
                empty_row = {f"rubric_{name}": 0.0 for name in criterion_names}
                empty_row["rubric_score"] = 0.0
                final_rows.append(empty_row)
            else:
                final_rows.append(next(rec_iter))

        df = pd.DataFrame(final_rows, index=texts.index)
        self._results = df
        return df

    def transform_sections(
        self,
        df: pd.DataFrame,
        parsed_sections: dict[str, dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Score each paper against NeurIPS criteria targeting specific sections.

        When parsed_sections contains section extraction for a paper (keyed by DOI,
        cleaned DOI, or Title), each criterion is evaluated against its targeted section
        (e.g., limitations -> limitations section; compute_resources -> compute/hardware;
        reproducibility -> code/data availability).
        When section extraction is unavailable, falls back to 'Abstract Note'.
        Adds a 'rigor_source' column indicating 'full_text' vs 'abstract_fallback'.
        """
        if self._pipeline is None:
            raise RuntimeError("Must call fit() before transform_sections()")

        if df.empty:
            empty_df = df.copy()
            empty_df["rigor_source"] = pd.Series(dtype=str)
            for name in NEURIPS_RUBRIC_CRITERIA:
                empty_df[f"rubric_{name}"] = pd.Series(dtype=float)
            empty_df["rubric_score"] = pd.Series(dtype=float)
            return empty_df

        parsed_sections = parsed_sections or {}

        # 1. Resolve sections and rigor_source for each row
        row_metadata = []
        for idx, row in df.iterrows():
            doi = str(row.get("DOI") or "").strip()
            clean_doi = doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            title = str(row.get("Title") or "").strip()

            sections = None
            if doi and doi in parsed_sections:
                sections = parsed_sections[doi]
            elif clean_doi and clean_doi in parsed_sections:
                sections = parsed_sections[clean_doi]
            elif title and title in parsed_sections:
                sections = parsed_sections[title]
            elif str(idx) in parsed_sections:
                sections = parsed_sections[str(idx)]

            fallback_abstract = str(
                row.get("Abstract Note") or row.get("Abstract") or row.get("Title") or ""
            ).strip()

            if sections:
                rigor_source = "full_text"
            else:
                rigor_source = "abstract_fallback"

            row_metadata.append({
                "idx": idx,
                "sections": sections,
                "fallback_abstract": fallback_abstract,
                "rigor_source": rigor_source,
            })

        # 2. For each criterion, gather target text for each row and run zero-shot inference
        criterion_scores = {name: {} for name in NEURIPS_RUBRIC_CRITERIA}

        for crit_name, crit_label in NEURIPS_RUBRIC_CRITERIA.items():
            target_secs = CRITERION_SECTION_TARGETS.get(crit_name, [])
            row_texts = []
            valid_indices = []

            for meta in row_metadata:
                idx = meta["idx"]
                sections = meta["sections"]
                fallback = meta["fallback_abstract"]

                text_to_eval = ""
                if sections:
                    for sec in target_secs:
                        cand = sections.get(sec, "")
                        if isinstance(cand, str) and cand.strip():
                            text_to_eval = cand.strip()
                            break
                    if not text_to_eval:
                        text_to_eval = fallback
                else:
                    text_to_eval = fallback

                if text_to_eval:
                    row_texts.append(text_to_eval)
                    valid_indices.append(idx)
                else:
                    criterion_scores[crit_name][idx] = 0.0

            if row_texts:
                pipeline_outputs = self._pipeline(
                    row_texts,
                    candidate_labels=[crit_label],
                    multi_label=True,
                    batch_size=32,
                    truncation=True,
                    max_length=512,
                )
                if isinstance(pipeline_outputs, dict):
                    pipeline_outputs = [pipeline_outputs]

                for idx, out in zip(valid_indices, pipeline_outputs):
                    score = out["scores"][0] if out.get("scores") else 0.0
                    criterion_scores[crit_name][idx] = round(float(score), 4)

        # 3. Assemble results DataFrame
        scored_records = []
        for meta in row_metadata:
            idx = meta["idx"]
            record = {"rigor_source": meta["rigor_source"]}
            scores = []
            for name in NEURIPS_RUBRIC_CRITERIA:
                sc = criterion_scores[name].get(idx, 0.0)
                record[f"rubric_{name}"] = sc
                scores.append(sc)
            record["rubric_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
            scored_records.append(record)

        result_df = pd.DataFrame(scored_records, index=df.index)
        self._results = result_df

        # Merge with df keeping df's original columns
        scored_df = df.copy()
        for col in result_df.columns:
            scored_df[col] = result_df[col]

        return scored_df

    @property
    def results(self) -> dict[str, Any]:
        """Summary statistics across the analyzed corpus."""
        if self._results is None:
            return {}

        rubric_cols = [c for c in self._results.columns if c.startswith("rubric_")]
        means = {col: round(float(self._results[col].mean()), 4) for col in rubric_cols}
        full_text_count = 0
        if "rigor_source" in self._results.columns:
            full_text_count = int((self._results["rigor_source"] == "full_text").sum())

        return {
            "criteria_means": means,
            "overall_mean_rigor": means.get("rubric_score", 0.0),
            "full_text_evaluated_count": full_text_count,
            "corpus_size": len(self._results),
        }


"""Executive Markdown Briefing Generator for Agent Teams Lite and SDD-SOTA.

Transforms structured litreview analysis outputs, scientometric network rankings,
and section-targeted NeurIPS rigor evaluations into a clean, actionable GitHub
Flavored Markdown briefing for human architects and autonomous AI agents.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
import pandas as pd


def generate_executive_markdown(
    summary_data: dict[str, Any],
    scored_df: pd.DataFrame | None = None,
) -> str:
    """Generate structured GFM executive research briefing from litreview results.

    Args:
        summary_data: Dictionary produced by ReviewPipeline / run_agent_summary.
        scored_df: Optional DataFrame containing paper metadata and rubric columns.

    Returns:
        Formatted Markdown text string.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    corpus = summary_data.get("corpus", {})
    oa_info = summary_data.get("open_access", {})
    network = summary_data.get("citation_network", {})
    rubric = summary_data.get("neurips_rubric", {})
    taxonomy = summary_data.get("taxonomy_validation", {})
    gaps = summary_data.get("gap_analysis", {})

    total_papers = corpus.get("total_papers", len(scored_df) if scored_df is not None else 0)
    year_range = corpus.get("year_range", [])
    years_str = f"{year_range[0]} – {year_range[1]}" if len(year_range) == 2 else "N/A"

    oa_count = oa_info.get("oa_count", 0)
    oa_ratio = oa_info.get("oa_ratio", 0.0)
    oa_pct = f"{round(oa_ratio * 100, 1)}%" if oa_ratio else "0.0%"

    mean_rigor = rubric.get("overall_mean_rigor", 0.0)
    full_text_evaluated = rubric.get("full_text_evaluated_count", 0)

    lines = []
    lines.append("# SDD-SOTA Executive Research Briefing")
    lines.append(f"> **Generated**: {now_str} | **Corpus Size**: {total_papers} papers | **Timeline**: {years_str}")
    lines.append("")

    # --- Executive Snapshot ---
    lines.append("## 📌 Executive Snapshot")
    lines.append("")
    lines.append("| Metric | Value | Indicator |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Corpus Volume** | `{total_papers}` papers | Evaluated against zero-shot domain taxonomies |")
    lines.append(f"| **Open Access Ratio** | `{oa_pct}` (`{oa_count}/{total_papers}`) | Papers with public text & repository links |")
    lines.append(f"| **Mean Rigor Score** | `{round(mean_rigor, 4)}` | NeurIPS reproducibility & checklist alignment |")
    lines.append(f"| **Full-Text Audited** | `{full_text_evaluated}/{total_papers}` | Extracted academic sections via `pypdf` |")

    nodes = network.get("total_nodes", 0)
    edges = network.get("total_edges", 0)
    if nodes > 0:
        lines.append(f"| **Citation Network** | `{nodes}` nodes / `{edges}` edges | Semantic Scholar graph traversal |")
    lines.append("")

    # --- Read-First Priority Queue ---
    read_first = network.get("read_first_recommendations", [])
    lines.append("## 🎯 Read-First Priority Queue")
    lines.append("Optimized reading order balancing topical relevance, citation impact, network prestige, and empirical rigor:")
    lines.append("")

    if read_first:
        lines.append("| Priority | Paper Title | Year | Read-First | Role | Citations | DOI / URL |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |")
        for idx, rec in enumerate(read_first, start=1):
            title = rec.get("title", "Untitled")
            year = rec.get("year", "N/A")
            score = rec.get("read_first_score", 0.0)
            role = rec.get("role", "corpus_work").upper()
            cites = rec.get("citations", 0)
            doi = rec.get("doi") or "N/A"
            doi_link = f"[`{doi}`](https://doi.org/{doi})" if str(doi).startswith("10.") else f"`{doi}`"

            lines.append(f"| **#{idx}** | **{title}** | {year} | `{score:.4f}` | `{role}` | {cites:,} | {doi_link} |")
        lines.append("")

        # Component breakdown notes
        lines.append("> [!TIP]")
        lines.append("> **Priority Rationale**:")
        for idx, rec in enumerate(read_first[:3], start=1):
            comps = rec.get("score_components", {})
            t_rel = comps.get("topical_relevance", 0.0)
            g_pr = comps.get("graph_prestige", 0.0)
            m_rig = comps.get("methodology", 0.0)
            lines.append(f"> - **#{idx} {rec.get('title')}**: Topical match `{t_rel:.2f}`, Graph authority `{g_pr:.2f}`, Rigor `{m_rig:.2f}`.")
        lines.append("")
    else:
        lines.append("*No citation graph recommendations available.*")
        lines.append("")

    # --- NeurIPS Rigor & Methodology Audit ---
    lines.append("## 🔬 NeurIPS Rigor & Methodology Audit")
    lines.append("Multi-criteria zero-shot evaluation targeting specific sections (`limitations`, `compute`, `reproducibility`):")
    lines.append("")

    criteria_means = rubric.get("criteria_means", {})
    if criteria_means:
        lines.append("**Corpus Criteria Means:**")
        lines.append("")
        lines.append("| Criterion | Mean Probability | Evaluation Focus |")
        lines.append("| :--- | :---: | :--- |")
        crit_desc = {
            "rubric_limitations": "Limitations, threats to validity, failure modes",
            "rubric_reproducibility": "Code repository, datasets, replication instructions",
            "rubric_experimental_rigor": "Baselines, ablations, benchmark evaluations",
            "rubric_statistical_significance": "Error bars, confidence intervals, hypothesis tests",
            "rubric_compute_resources": "Hardware accelerators, GPU hours, training details",
            "rubric_score": "Composite unweighted rigor score",
        }
        for k, v in criteria_means.items():
            desc = crit_desc.get(k, "Methodological transparency")
            lines.append(f"| `{k.replace('rubric_', '')}` | `{v:.4f}` | {desc} |")
        lines.append("")

    # If DataFrame is available with paper-level scores
    if scored_df is not None and not scored_df.empty:
        lines.append("### Paper-Level Rigor Audit")
        lines.append("")
        has_src = "rigor_source" in scored_df.columns
        lines.append("| Paper Title | Source | Limitations | Reproducibility | Exp. Rigor | Compute | Composite |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for _, row in scored_df.iterrows():
            p_title = str(row.get("Title", "Untitled"))
            src = f"`{row.get('rigor_source', 'abstract_fallback')}`" if has_src else "`abstract`"
            lim = round(float(row.get("rubric_limitations", 0.0)), 4)
            rep = round(float(row.get("rubric_reproducibility", 0.0)), 4)
            exp = round(float(row.get("rubric_experimental_rigor", 0.0)), 4)
            comp = round(float(row.get("rubric_compute_resources", 0.0)), 4)
            comp_score = round(float(row.get("rubric_score", 0.0)), 4)
            lines.append(f"| {p_title} | {src} | `{lim}` | `{rep}` | `{exp}` | `{comp}` | **`{comp_score}`** |")
        lines.append("")

    # --- Citation Network & SOTA Anchors ---
    lines.append("## 🌐 Citation Network & SOTA Anchors")
    lines.append("")

    # Kleinberg Authorities & Hubs
    auths = network.get("top_authorities", [])
    hubs = network.get("top_hubs", [])

    if auths or hubs:
        lines.append("### Scientometric Authorities & Syntheses (HITS)")
        lines.append("")
        if auths:
            lines.append("**Top Authority Pillars (Foundational Papers):**")
            for a in auths[:3]:
                auth_score = a.get("authority_score", 0.0)
                tot_c = a.get("total_citations", 0)
                lines.append(f"- **{a.get('title')}** ({a.get('year', 'N/A')}): Authority Score `{auth_score:.4f}` | Total Citations: {tot_c:,}")
            lines.append("")
        if hubs:
            lines.append("**Top Syntheses / Review Hubs (Broad Overview Papers):**")
            for h in hubs[:3]:
                h_score = h.get("hub_score", 0.0)
                ref_c = h.get("references_cited_count", 0)
                lines.append(f"- **{h.get('title')}** ({h.get('year', 'N/A')}): Hub Score `{h_score:.4f}` | References Cited: {ref_c}")
            lines.append("")

    # Bibliographic Coupling
    couplings = network.get("bibliographic_coupling", [])
    if couplings:
        lines.append("### Bibliographic Coupling Clusters")
        lines.append("Papers within the corpus sharing high common intellectual lineage:")
        lines.append("")
        for cp in couplings[:3]:
            p1 = cp.get("paper_1")
            p2 = cp.get("paper_2")
            sh_count = cp.get("shared_references_count", 0)
            shared_refs = cp.get("shared_references", [])
            shared_str = ", ".join(f"*{r}*" for r in shared_refs[:2])
            lines.append(f"- **{p1}** ↔ **{p2}** ({sh_count} shared references: {shared_str})")
        lines.append("")

    # --- Taxonomy & Gaps ---
    labels = taxonomy.get("label_counts", {})
    if labels:
        lines.append("## 📊 Taxonomy Distribution & Empirical Themes")
        lines.append("")
        lines.append("| Category / Candidate Label | Papers Tagged | Relative Weight |")
        lines.append("| :--- | :---: | :---: |")
        tot_l = sum(labels.values()) or 1
        for lbl, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
            pct = round((count / tot_l) * 100, 1)
            lines.append(f"| {lbl} | `{count}` | `{pct}%` |")
        lines.append("")

    # Strategic Next Steps
    lines.append("## 💡 Strategic Recommendations for Agent Teams Lite")
    lines.append("Recommended implementation path based on empirical evidence:")
    lines.append("")
    lines.append("1. **Baseline Prioritization**: Direct agents to study top `#1` in the *Read-First Queue* before designing proposals.")
    lines.append("2. **Reproducibility Safeguards**: For papers scoring `< 0.50` on reproducibility, mandate empirical re-benchmarking during `/sdd-verify`.")
    lines.append("3. **Hardware / Resource Realism**: Cross-reference compute specifications against available target hardware before initiating long-running training loops.")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by SDD-SOTA Scientific Discovery Engine.*")

    return "\n".join(lines)


def export_markdown_report(
    path: str,
    summary_data: dict[str, Any],
    scored_df: pd.DataFrame | None = None,
) -> str:
    """Generate and write executive markdown report to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = generate_executive_markdown(summary_data, scored_df=scored_df)
    out_path.write_text(md_content, encoding="utf-8")
    return str(out_path)

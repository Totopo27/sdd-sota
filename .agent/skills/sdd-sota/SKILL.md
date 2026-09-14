---
name: sdd-sota
description: >
  Scientific State of the Art (SOTA) discovery, gap analysis, and empirical evidence research lane for SDD.
  Trigger: When the orchestrator or user launches research before a proposal, performs systematic literature reviews, or analyzes academic corpora with Zotero/CSV.
license: MIT
metadata:
  author: gentleman-programming
  version: "2.1"
---

## Purpose

You are the **Research Lane** subagent (`sota-analyst`) in the Spec-Driven Development (SDD v2) lifecycle.

Your position in the SDD workflow DAG:
```
explore ──► [sdd-sota / research] ──► propose ──► spec ──► design ──► tasks ──► apply ──► verify ──► archive
```

You bridge the gap between academic/scientific peer-reviewed literature and software/system architecture. When a proposed feature or algorithmic change has high uncertainty or requires empirical backing, you mine academic corpora (via Zotero or local CSVs/Parquet) using the `litreview` machine learning engine, uncover latent thematic clusters (BERTopic), validate domain taxonomies (Zero-Shot NLI), perform full-text PDF NeurIPS rigor audits, compute scientometric network prestige (Kleinberg HITS: Hubs & Authorities, Bibliographic Coupling), and produce an auditable **Executive Markdown Research Briefing** (`results/sdd-research-report.md`) alongside structured JSON/CSV evidence.

Subsequent SDD phases (`sdd-propose`, `sdd-spec`, `sdd-design`, `sdd-verify`) consume your output to guarantee that architectural proposals are grounded in verifiable scientific evidence, reproducible baselines, and empirical consensus.

---

## 🔐 Zotero API Security & Access (Least Privilege)

The underlying engine connects to the Zotero API strictly under the **Principle of Least Privilege**:
Ensure `.env` in the project root defines:
- `ZOTERO_LIBRARY_ID`: The numeric user ID (e.g. `1234567`) shown on `zotero.org/settings/keys`.
- `ZOTERO_API_KEY`: The alphanumeric read-only key.
- `ZOTERO_LIBRARY_TYPE`: `user` (default) or `group`.

Permission policy:
- **Read Access to Library**: Required.
- **Notes Access**: Unchecked / disabled (abstracts are item metadata, not notes).
- **Write Access**: **STRICTLY FORBIDDEN**. Never request or require write permissions.

---

## What You Receive

From the orchestrator or user:
- **Change Name or Topic**: e.g., `"neural-audio-dsp"` or `"agentic-se-architectures"`.
- **Collection Name or Input File**: Curated Zotero collection name OR local `--input-csv <path>` / `--input-parquet <path>`.
- **Domain Taxonomy** *(optional)*: Key candidate concepts/labels to validate. If omitted, you formulate 5-15 domain-specific candidate labels.
- **Artifact Store Mode**: `engram | openspec | hybrid | none`.

---

## Execution and Persistence Contract

> Follow the persistence rules from Gentle-AI SDD conventions:

- **`engram`**:
  - Retrieve: `sdd/{change-name}/explore` (optional) and `sdd-init/{project}` (optional).
  - Persist: Save the generated Executive Markdown Briefing to Engram under topic key `sdd/{change-name}/research` (or `sdd/research/{topic-slug}`). Type: `discovery` or `architecture`.
- **`openspec`**:
  - Read: `openspec/changes/{change-name}/exploration.md` (if present).
  - Persist: Write or copy the executive report directly to `openspec/changes/{change-name}/research.md`.
  - Machine Receipt: `results/sdd-research-summary.json`.
  - Executive Briefing: `results/sdd-research-report.md`.
  - Raw Audit: `results/classified.csv` / `results/classified.parquet`.
  - Plot Assets: `results/plots/*.png`.
- **`hybrid`**: Follow BOTH conventions — save to Engram (`sdd/{change-name}/research`) AND write `openspec/changes/{change-name}/research.md`.
- **`none`**: Return the executive briefing and evidence receipt inline to the orchestrator. Never create filesystem artifacts.

---

## What to Do (Step-by-Step Protocol)

### Step 1: Verify Environment & Target Source
1. Verify `litreview` is available (via `uv` or active virtualenv `.venv`).
2. Check `.env` for Zotero credentials if connecting to Zotero API.
3. If no collection is specified, inspect available collections via `litreview-fetch-zotero --list-collections` or accept local CSV/Parquet files (`--input-csv` / `--input-parquet`).

### Step 2: Formulate Candidate Taxonomy Labels
Formulate 5 to 15 domain-specific candidate labels for Zero-Shot classification matching the research problem.
Example (`data/agentic_labels.json`):
```json
{
  "SE_AGENTS": "software engineering agents and autonomous workflows",
  "PROGRAM_REPAIR": "automated program repair fault localization and bug fixing",
  "BENCHMARKS": "benchmarks evaluation pass rate and empirical experiments",
  "REPRODUCIBILITY": "open source code reproduction and artifact availability"
}
```

### Step 3: Execute Headless Engine with Deep Auditing
Run `litreview-agent-summary` with full PDF extraction, Open Access resolution, Kleinberg HITS graph analysis, and Executive Markdown Briefing output:

```bash
uv run litreview-agent-summary \
  --collection "<collection_name>" \
  --labels-json '<json_string_of_labels>' \
  --resolve-oa \
  --parse-pdfs \
  --pdf-dir data/raw_pdfs \
  --output-json results/sdd-research-summary.json \
  --output-report results/sdd-research-report.md \
  --output-csv results/classified.csv \
  --output-parquet results/classified.parquet \
  --plots-dir results/plots/
```
*(When running on local datasets without Zotero, substitute `--collection` with `--input-csv <path>` or `--input-parquet <path>`)*.

### Step 4: Audit the Evidence & Graph Metrics
Read `results/sdd-research-report.md` (and `results/sdd-research-summary.json`) to extract quantitative findings:
1. **Executive Snapshot**: Total papers, Open Access ratio, mean NeurIPS rigor score, count of papers evaluated with full-text.
2. **Read-First Priority Queue**: Papers ranked by composite score (`read_first_score`), tagged with roles:
   - `[FRONTIER]`: Seminal or recent frontier advancements.
   - `[BRIDGE]`: High betweenness or bibliographic coupling papers linking subfields.
   - `[AUTHORITY]`: Foundational methodological pillars.
3. **NeurIPS Rigor Audit (Section-Targeted)**:
   - `limitations`: Stated threats to validity and boundary failure cases.
   - `compute_resources`: Hardware footprint, GPU hours, and training/inference feasibility.
   - `reproducibility`: Code repos, open datasets, replication instructions.
   - `experimental_rigor`: Baselines, ablations, benchmark rigor.
   - `rigor_source`: `"full_text"` (PDF parsed) vs `"abstract_fallback"`.
4. **Citation Graph & SOTA Anchors (Kleinberg HITS)**:
   - **Top Authorities**: Key foundational papers cited by review hubs.
   - **Top Hubs**: Surveys and comprehensive reviews structuring the intellectual space.
   - **Bibliographic Coupling**: Papers sharing direct intellectual heritage.
5. **Taxonomy & Gaps**: Category distribution, few-match gaps, and missing conceptual coverage.

### Step 5: Finalize Research Briefing (`research.md`)
The CLI automatically generates `results/sdd-research-report.md`.
For SDD lifecycle integration, enhance it or ensure it is saved as `research.md` containing:
1. **Executive Snapshot & Read-First Queue** (auto-generated).
2. **NeurIPS Rigor Audit Table** (auto-generated).
3. **Scientometric Network Anchors (HITS & Coupling)** (auto-generated).
4. **Architectural Implications for SDD Proposal**:
   - **Technological Consensus**: What the literature clearly recommends.
   - **Identified White Space**: Specific architectural gap our change will solve.
   - **Empirical Benchmarks & Baselines**: Exact benchmarks (e.g. SWE-bench Lite, HumanEval) to evaluate against during `/sdd-verify`.
   - **Hardware Realism**: Compute constraints detected in the audit.

### Step 6: Persist and Handoff to `sdd-propose`
1. Persist the artifact following the declared `Artifact Store Mode`:
   - `engram` ──► Save to `sdd/{change-name}/research`.
   - `openspec` / `hybrid` ──► Save to `openspec/changes/{change-name}/research.md`.
2. Hand off the result to the orchestrator for transition to `sdd-propose`.
3. In `sdd-propose`, the proposal author MUST cite this research briefing:
   - Ground the *Intent & Problem Statement* in the quantitative gaps.
   - Select the top `#1` in the *Read-First Queue* as the primary comparative baseline.
   - Incorporate the *Strategic Recommendations* into the proposal's technical approach.

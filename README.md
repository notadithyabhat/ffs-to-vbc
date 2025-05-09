# Medicare FFS→VBC Data Pipeline

This repository contains a fully reproducible data-engineering and policy-analytics pipeline to analyze the transition of Medicare from Fee‑For‑Service (FFS) to Value‑Based Care (VBC) between 2008 and 2010. It integrates CMS Public Use Files (PUFs) with a Retrieval‑Augmented Generation (RAG) workflow for policy text classification using GPT‑4o.

---

## Repository Structure

```
├── config.py
├── etl_expand.py
├── policy_rag.py
├── integrate_synthesis.py
├── viz_report.py
├── environment.yml
├── data/
│   ├── 2008 IPBS PUF.csv
│   ├── 2010 IPBS PUF.csv
│   ├── 2010 PSBS PUF.csv
│   ├── 2008_Chronic_Conditions_PUF.csv
│   ├── 2010_Chronic_Conditions_PUF.xlsx
│   ├── all_icd9_codes.csv
│   ├── hcpcs_betos_2010.csv
│   └── zip_to_hrr_xwalk.csv
├── data/cms_pdfs/
│   └── *.pdf  # CMS policy documents
├── outputs/
│   ├── interim/    # intermediate CSV outputs
│   ├── figs/       # generated figures (for report)
│   └── final/      # final merged datasets and tables
├── notebooks/      # exploratory and integration notebooks
└── README.md       # this file
```

## Prerequisites

* Python 3.9+ (recommended via conda)
* An OpenAI API key with GPT‑4 access
* (Optional) Conda or virtualenv for reproducible environment

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/notadithyabhat/ffs-to-vbc.git
   cd ffs-vbc-pipeline
   ```
2. Create and activate environment:

   ```bash
   conda env create -f environment.yml
   conda activate ffs-vbc
   ```
3. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. Download data/ from the following link and store in base directory: https://drive.google.com/drive/folders/1F00NsziRx7CbRcyXT1xhesEITSdh80ew?usp=sharing

## Usage

Run each stage in sequence from the command line:

1. **Quantitative ETL & Expansion**

   ```bash
   python etl_expand.py
   ```

   * Aggregates full PSBS file (provider & HRR views)
   * Computes 2008 vs 2010 national deltas for IPBS

2. **Policy Classification (RAG)**

   ```bash
   python policy_rag.py
   ```

   * Loads CMS rule PDFs
   * Embeds and indexes with FAISS
   * Classifies paragraphs as FFS (0) or VBC (1)

3. **Integration & Synthesis**

   ```bash
   python integrate_synthesis.py
   ```

   * Merges quantitative deltas with policy label counts
   * Exports tidy CSV tables for dashboards and report

4. **Visualization & Report Figures**

   ```bash
   python viz_report.py
   ```

   * Generates publication‑ready figures in `outputs/figs`

5. **Notebooks**

   * Explore the data and results in `notebooks/` (e.g., `1_data_ingestion.ipynb`, `2_quant_analysis.ipynb`, `3_policy_rag.ipynb`).

## Outputs

* **outputs/interim/**: intermediate CSVs
* **outputs/final/**: final merged datasets
* **outputs/figs/**: PNGs for LaTeX report

## License & Citation

This project is released under the MIT License. Please cite as:

> Author(s), Adithya Bhat, Pratham Gupta.

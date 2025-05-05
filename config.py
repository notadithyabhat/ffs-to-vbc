"""
Central configuration for Medicare FFS → VBC pipeline
"""

import pathlib

# ---------------- FOLDER ROOTS ----------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR  = REPO_ROOT / "data"          # raw csv / xlsx
OUTPUT    = REPO_ROOT / "outputs"       # derived csv, figs
PDF_DIR   = DATA_DIR  / "cms_pdfs"      # downloaded policy PDFs

# ---------------- RAW FILES -------------------
IPBS_08 = DATA_DIR / "2008 IPBS PUF.csv"
IPBS_10 = DATA_DIR / "2010 IPBS PUF.csv"
PSBS_10 = DATA_DIR / "2010 PSBS PUF.csv"
CC_08   = DATA_DIR / "2008_Chronic_Conditions_PUF.csv"
CC_10   = DATA_DIR / "2010_Chronic_Conditions_PUF.xlsx"

ICD9_REF   = DATA_DIR / "all_icd9_codes.csv"
HCPCS_REF  = DATA_DIR / "hcpcs_betos_2010.csv"
ZIP_HRR_XW = DATA_DIR / "zip_to_hrr_xwalk.csv"

# ---------------- EMBEDDING / MODEL ----------
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL   = "gpt-4o"

# ---------------- RUNTIME --------------------
DASK_BLOCKSIZE = "128MB"
RANDOM_SEED = 42

# Create output folders if absent
for p in (OUTPUT, OUTPUT / "figs", OUTPUT / "interim"):
    p.mkdir(parents=True, exist_ok=True)

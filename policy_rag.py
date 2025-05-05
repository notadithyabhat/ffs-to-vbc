"""
Scrape/ingest CMS PDFs, embed, classify paragraphs FFS=0 / VBC=1.
Outputs:   outputs/policy_classified.csv
"""

from config import *
import pathlib, re, glob, json
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import ChatOpenAI
from langchain.chains import RetrievalQA

def load_pdfs() -> list:
    docs=[]
    for pdf in glob.glob(str(PDF_DIR/'*.pdf')):
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load_and_split())
    return docs

def main() -> None:
    print("🔄  Loading CMS PDFs …")
    documents = load_pdfs()
    print(f"Loaded {len(documents)} chunks.")

    print("🔄  Embedding + indexing …")
    store = FAISS.from_documents(documents, OpenAIEmbeddings(model=EMBED_MODEL))

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=LLM_MODEL, temperature=0),
        retriever=store.as_retriever()
    )

    rows=[]
    for doc in documents:
        prompt = ("Classify the following CMS policy excerpt as "
                  "0 if it promotes Fee‑For‑Service incentives or "
                  "1 if it promotes Value‑Based‑Care incentives.\n\n"
                  f"{doc.page_content}\n\nAnswer with only 0 or 1.")
        label = qa(prompt)["result"].strip()
        rows.append({"rule": doc.metadata["source"],
                     "page": doc.metadata.get("page", -1),
                     "text": doc.page_content,
                     "label": int(label)})

    out = OUTPUT / "policy_classified.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅  Saved classifications → {out}")

if __name__ == "__main__":
    import pandas as pd
    main()

"""
eval/generate_ragas_testset.py
=================================
Giai doan 3a - Dung RAGAS TestsetGenerator de tu sinh QA pairs tu documents
trong RAG knowledge base. Khong viet ground truth tay.

QUAN TRONG -- tai lieu RAG la PDF, KHONG phai .txt:
scripts/build_vectordb.py doc PDF qua pymupdf4llm.to_markdown() (xem ham
pdf_to_marked_markdown() trong file do), KHONG doc .txt. Script nay dung lai
dung pymupdf4llm de convert PDF -> Markdown truoc khi dua vao
TestsetGenerator, cho nhat quan voi cach FAISS index thuc te duoc build.
RAGAS chi can noi dung text de sinh cau hoi, nen khong can giu page marker /
chunk theo heading phuc tap nhu build_vectordb.py -- moi PDF duoc dua vao
nguyen 1 Document (khong chunk), TestsetGenerator se tu xu ly do dai.

Organ duoc gan vao metadata moi Document tu ten file PDF, dung cung logic
_detect_organ() nhu build_vectordb.py ("breast"/"birads" -> breast,
"thyroid"/"tirads" -> thyroid, khac -> general) -- de cac sample trong
testset sinh ra co the dung field "organ" cho production_query mode trong
eval_rag.py.

QUAN TRONG -- API ragas: tu ban 0.2.x, TestsetGenerator khong con dung
generator_llm/critic_llm/distributions/ragas.testset.evolutions (API cu cua
ban 0.1.x). Ban hien tai dung TestsetGenerator.from_langchain(llm=, embedding_model=)
va generate_with_langchain_docs(docs, testset_size=N) voi distribution mac dinh
cua thu vien. Xem requirements/ragas_eval.txt de biet version duoc pin.

Cot output cua to_pandas() trong ban moi la "user_input" (cau hoi) va
"reference" (cau tra loi chuan) -- KHONG con la "question"/"ground_truth"
nhu ban 0.1.x. eval_rag.py va eval_ragas.py doc ca 2 ten cot de tuong thich
nguoc, nhung file testset moi sinh ra se luon dung "user_input"/"reference".

Truoc khi chay: services/orchestrator/rag/docs/ phai co it nhat vai file PDF
thuc te (cung tai lieu da index vao FAISS qua scripts/build_vectordb.py).
Thu muc trong (chi co .gitkeep) se load duoc 0 document va script se bao loi
ngay tu dau thay vi chay roi fail kho hieu o buoc generate.

Chay:
  python eval/generate_ragas_testset.py \\
    --docs_dir   services/orchestrator/rag/docs \\
    --out_file   eval/results/ragas_testset.json \\
    --n_samples  50
"""
import argparse
import glob
import os
from pathlib import Path

# Script nay chay standalone (khong qua docker-compose), nen .env KHONG duoc
# tu doc nhu khi chay trong container. Phai tu load o day, neu khong
# os.environ["GOOGLE_API_KEY"] se raise KeyError du .env co ghi gi.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _detect_organ(filename: str) -> str:
    """Giong _detect_organ() trong scripts/build_vectordb.py, suy organ tu ten file PDF."""
    name = filename.lower()
    if "breast" in name or "birads" in name or "bi-rads" in name:
        return "breast"
    if "thyroid" in name or "tirads" in name or "ti-rads" in name:
        return "thyroid"
    return "general"


def _load_pdf_documents(docs_dir: str) -> list:
    """
    Convert moi PDF trong docs_dir sang 1 LangChain Document (khong chunk),
    dung pymupdf4llm.to_markdown() giong build_vectordb.py. Bo qua PDF khong
    doc duoc hoac khong co noi dung, in warning thay vi crash ca batch.
    """
    import pymupdf4llm
    from langchain_core.documents import Document

    pdf_files = glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)
    documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            text = pymupdf4llm.to_markdown(pdf_path)
        except Exception as e:
            print(f"  [warn] Khong doc duoc {filename}: {e} -- bo qua.")
            continue
        if not text or not text.strip():
            print(f"  [warn] {filename} khong co noi dung trich xuat duoc -- bo qua.")
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename, "organ": _detect_organ(filename)},
            )
        )
    return documents


def main(docs_dir: str, out_file: str, n_samples: int):
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.testset import TestsetGenerator

    google_api_key = os.environ["GOOGLE_API_KEY"]

    documents = _load_pdf_documents(docs_dir)
    print(f"Loaded {len(documents)} PDF documents from {docs_dir}")
    if not documents:
        raise SystemExit(
            f"[generate_ragas_testset] 0 PDF found/readable in {docs_dir}. "
            "Add real clinical guideline PDFs there first (same docs indexed "
            "by scripts/build_vectordb.py) before generating a testset."
        )

    generator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    # gemini-embedding-001: embedding-001 bi deprecated 14/8/2025, va
    # text-embedding-004 bi deprecated 14/1/2026 -- ca 2 deu khong con
    # goi duoc. gemini-embedding-001 la model hien hanh (3072 dims).
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_api_key)

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )
    testset = generator.generate_with_langchain_docs(documents, testset_size=n_samples)

    df = testset.to_pandas()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_file, orient="records", force_ascii=False, indent=2)
    print(f"Generated {len(df)} QA pairs -> {out_file}")
    preview_cols = [c for c in ("user_input", "reference") if c in df.columns]
    if preview_cols:
        print(df[preview_cols].head(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="services/orchestrator/rag/docs")
    p.add_argument("--out_file", default="eval/results/ragas_testset.json")
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()
    main(args.docs_dir, args.out_file, args.n_samples)
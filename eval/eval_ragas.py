"""
eval/eval_ragas.py
====================
Danh gia pipeline RAG + LLM reasoning theo 4 metric RAGAS.

Hai mode, KHONG lien quan anh/vision model -- ca 2 chi doc lai du lieu da co
san hoac query RAG store bang text:

  --mode pipeline (mac dinh):
    Doc lai cac file JSON trong --pipeline_dir (xem eval/run_pipeline_batch.py
    de tao chung tu anh thuc qua API /analyze). Dung tier_2 + tier_3 text da
    sinh san lam "response", _rag_chunks_internal lam "retrieved_contexts".
    Danh gia: Faithfulness + ResponseRelevancy. Khong can ground truth.

  --mode retrieval:
    Doc testset tu eval/generate_ragas_testset.py (sinh tu van ban trong
    services/orchestrator/rag/docs, KHONG lien quan anh). Query RAG store
    thuc te bang cau hoi trong testset, lay contexts.
    Danh gia: LLMContextPrecisionWithReference + LLMContextRecall. Can
    "reference" (co san trong testset).

  --mode both: chay ca 2, ghi 2 file CSV rieng (--out_file cho pipeline,
  --out_file_retrieval cho retrieval -- 2 file khac nhau, khong de len nhau).

QUAN TRONG -- cac bug da sua so voi ban pseudocode goc:
  1. FAISSStore.retrieve_with_meta() nhan tham so "k", khong phai "top_k".
  2. ragas >=0.2 doi ten cot testset tu "question"/"ground_truth" (ban 0.1.x)
     thanh "user_input"/"reference". Ham _get_question()/_get_ground_truth()
     doc ca 2 ten cot khi DOC testset vao, de tuong thich nguoc.
  3. ragas 0.4.x doi schema cot khi TRUYEN VAO evaluate(): "question" ->
     "user_input", "answer" -> "response", "contexts" -> "retrieved_contexts",
     "ground_truth" -> "reference". Cac ham build_*_record() duoi day dung
     dung ten cot moi nay. Dung sai ten se khien evaluate() raise KeyError
     hoac tra ve NaN cho toan bo metric.
  4. Metric import doi sang dang class (Faithfulness(), ResponseRelevancy(),
     LLMContextPrecisionWithReference(), LLMContextRecall()) tu
     ragas.metrics, thay cho ham lowercase cua ban 0.1.x (faithfulness,
     answer_relevancy, context_precision, context_recall) -- ban cu van
     "chay duoc" trong 0.4.x nhung se bi xoa o v1.0, nen dung thang class
     moi cho chac.
  5. Dataset truyen vao evaluate() phai la ragas.EvaluationDataset.from_list(),
     khong phai datasets.Dataset.from_list() (HuggingFace) -- tat ca vi du
     chinh thuc cua ragas >=0.2 deu dung EvaluationDataset; HF Dataset chi
     con xuat hien trong tai lieu ban 0.1.x cu.

Chay (khong can ground truth):
  python eval/eval_ragas.py \\
    --mode pipeline \\
    --pipeline_dir eval/results/pipeline_outputs \\
    --out_file eval/results/ragas_pipeline.csv

Chay day du (can testset tu generate_ragas_testset.py):
  python eval/eval_ragas.py \\
    --mode retrieval \\
    --testset_file eval/results/ragas_testset.json \\
    --out_file eval/results/ragas_retrieval.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Script nay chay standalone (khong qua docker-compose), nen .env KHONG duoc
# tu doc nhu khi chay trong container. Phai tu load o day, neu khong
# os.environ["GOOGLE_API_KEY"] se raise KeyError du .env co ghi gi.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _get_question(item: dict) -> str:
    """Doc cau hoi tu testset, ho tro ca schema cu (question) va moi (user_input)."""
    return item.get("user_input") or item.get("question") or ""


def _get_ground_truth(item: dict) -> str:
    """Doc ground truth tu testset, ho tro ca schema cu (ground_truth) va moi (reference)."""
    return item.get("reference") or item.get("ground_truth") or ""


def _get_ragas_llm_embeddings():
    """
    Khoi tao Gemini llm/embeddings cho RAGAS -- ragas mac dinh dung OpenAI,
    phai override sang Gemini.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper

    google_api_key = os.environ["GOOGLE_API_KEY"]
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    )
    # gemini-embedding-001: embedding-001 bi deprecated 14/8/2025, va
    # text-embedding-004 bi deprecated 14/1/2026 -- ca 2 deu khong con
    # goi duoc. gemini-embedding-001 la model hien hanh (3072 dims).
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=google_api_key
    )
    return ragas_llm, ragas_embeddings


def load_pipeline_outputs(pipeline_dir: str) -> list:
    """
    Doc lai output da chay full pipeline (tu eval/run_pipeline_batch.py).
    Moi file la 1 JSON voi key "report" (ReportOutput dict).
    """
    records = []
    files = sorted(Path(pipeline_dir).glob("*.json"))
    if not files:
        print(
            f"[eval_ragas] 0 file JSON trong {pipeline_dir}. "
            "Chay eval/run_pipeline_batch.py truoc de tao du lieu."
        )
        return records

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        report = data.get("report", {})
        tier2 = report.get("tier_2_radiological_description", "") or ""
        tier3 = report.get("tier_3_diagnostic_suggestion", "") or ""
        chunks = report.get("_rag_chunks_internal", []) or []
        image_id = report.get("image_id", fp.stem)

        if not chunks:
            print(
                f"  [warn] {fp.name}: _rag_chunks_internal rong "
                f"(rag_disabled_warning={report.get('rag_disabled_warning')!r}). "
                "Faithfulness se khong co context de doi chieu."
            )

        records.append({
            "user_input": "Phan tich dac diem sieu am va dua ra nhan dinh lam sang.",
            "response": f"{tier2}\n\n{tier3}",
            "retrieved_contexts": [c for c in chunks[:3] if c],
            "_image_id": image_id,
            "_cot_label": (report.get("cot_result") or {}).get("cot_label", "unknown"),
            "_gt_label": data.get("gt_label", ""),
            "_consensus": report.get("consensus"),
            "_hard_conflict": report.get("hard_conflict"),
        })
    return records


def run_pipeline_eval(records: list, out_file: str):
    """Mode 1: danh gia Faithfulness + ResponseRelevancy -- khong can ground truth."""
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy

    if not records:
        raise SystemExit("[eval_ragas][pipeline] 0 record hop le, dung lai.")

    ragas_llm, ragas_embeddings = _get_ragas_llm_embeddings()

    # Cot dung dung ten ma evaluate() cua ragas 0.4.x doi hoi: user_input,
    # response, retrieved_contexts. Sai ten -> KeyError hoac NaN toan bo.
    ragas_recs = [
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
        }
        for r in records
    ]
    dataset = EvaluationDataset.from_list(ragas_recs)
    results = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=ragas_llm, embeddings=ragas_embeddings,
    )
    df = results.to_pandas()
    debug = pd.DataFrame([
        {k.lstrip("_"): v for k, v in r.items() if k.startswith("_")} for r in records
    ])
    df = pd.concat([df.reset_index(drop=True), debug.reset_index(drop=True)], axis=1)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print("\n=== Pipeline Eval (Faithfulness + ResponseRelevancy) ===")
    print(results)
    print(f"Saved to: {out_file}")


def run_retrieval_eval(testset_file: str, out_file: str):
    """
    Mode 2: danh gia context precision + context recall dung testset da sinh.
    Voi moi cau hoi trong testset, truy van RAG store thuc te lay contexts,
    roi danh gia theo ground truth co san trong testset.
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    from services.orchestrator.rag.faiss_store import FAISSStore

    with open(testset_file, encoding="utf-8") as f:
        testset = json.load(f)

    rag_store = FAISSStore()
    if not rag_store.is_ready():
        raise SystemExit(
            "[eval_ragas][retrieval] FAISSStore khong san sang. "
            "Chay scripts/build_vectordb.py truoc."
        )

    ragas_llm, ragas_embeddings = _get_ragas_llm_embeddings()

    ragas_recs = []
    n_skipped = 0
    for item in testset:
        question = _get_question(item)
        ground_truth = _get_ground_truth(item)
        if not question or not ground_truth:
            n_skipped += 1
            continue

        # k (khong phai top_k) la ten tham so thuc te cua retrieve_with_meta.
        retrieved = rag_store.retrieve_with_meta(question, k=3)
        contexts = [r["chunk"] for r in retrieved]

        ragas_recs.append({
            "user_input": question,
            "response": "",
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        })

    if n_skipped:
        print(f"[eval_ragas][retrieval] Bo qua {n_skipped} item thieu question/ground_truth.")
    if not ragas_recs:
        raise SystemExit("[eval_ragas][retrieval] 0 record hop le, dung lai.")

    dataset = EvaluationDataset.from_list(ragas_recs)
    results = evaluate(
        dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=ragas_llm, embeddings=ragas_embeddings,
    )
    df = results.to_pandas()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print("\n=== Retrieval Eval (Context Precision + Context Recall) ===")
    print(results)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pipeline", "retrieval", "both"], default="pipeline")
    p.add_argument("--pipeline_dir", default="eval/results/pipeline_outputs")
    p.add_argument("--testset_file", default="eval/results/ragas_testset.json")
    p.add_argument("--out_file", default="eval/results/ragas_scores.csv")
    p.add_argument(
        "--out_file_retrieval", default="eval/results/ragas_retrieval.csv",
        help="Dung rieng cho --mode both, de khong ghi de len --out_file cua mode pipeline.",
    )
    args = p.parse_args()

    if args.mode in ("pipeline", "both"):
        records = load_pipeline_outputs(args.pipeline_dir)
        print(f"Loaded {len(records)} pipeline outputs")
        run_pipeline_eval(records, args.out_file)

    if args.mode in ("retrieval", "both"):
        # mode "both": pipeline da dung --out_file o tren, retrieval phai
        # dung file khac (--out_file_retrieval) de khong ghi de len nhau.
        retrieval_out = args.out_file_retrieval if args.mode == "both" else args.out_file
        run_retrieval_eval(args.testset_file, retrieval_out)
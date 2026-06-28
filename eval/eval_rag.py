"""
eval/eval_rag.py
==================
Giai doan 3b - Danh gia rieng 2 giai doan RAG, khong di qua LLM, chi do chat
luong retrieval thuan.

QUAN TRONG -- 2 cach query khac nhau, KHONG the dung lan nhau:

  production_query mode:
    Dung dung cach pipeline thuc te tu graph.py xay query:
      Stage 1: query = "{modality} {organ}"               -> retrieve_with_meta(k=100)
      Stage 2: enriched = "{top_label} {organ} {modality} findings {icd10}"
               -> retrieve them 5, merge voi stage 1 (dedup), rerank top_n=3
    Day la cach he thong THUC SU goi RAG trong production (xem
    make_rag_retrieval_node / make_second_rag_retrieval trong graph.py).
    Can metadata moi sample: organ, modality, top_label, icd10_hint.
    Khong dung cau hoi tu nhien.

  natural_question mode:
    Dung cau hoi tu nhien tu RAGAS testset ("user_input"/"question") de query
    truc tiep retrieve_with_meta() + rerank(). Don gian, de doc, nhung KHONG
    phai cach he thong thuc su goi RAG -- chi cho biet "neu nguoi dung hoi
    cau nay thi RAG tim duoc gi", khong danh gia dung pipeline production.

Ca 2 mode deu duoc chay va ghi vao output, ghi ro nhan de nguoi doc tu so
sanh, khong gop chung thanh 1 so duy nhat.

Metrics:
  Stage 1 (pool rong, muc tieu "khong lot"):
    - Recall@100, Hit Rate
  Stage 2 (top 3 sau rerank, day la so lieu chinh vi day la cai di vao QA Agent):
    - nDCG@3, Precision@3, MRR

Cach xac dinh 1 chunk co "relevant" voi ground_truth hay khong: dung token
overlap (Jaccard tren tu, lowercase, bo dau cau), KHONG dung substring match
nguyen van. ground_truth tu RAGAS testset la cau tra loi TONG HOP do LLM viet
lai, gan nhu khong bao gio xuat hien nguyen van trong 1 chunk -- substring
match se danh gia sai (underestimate) Recall@100/nDCG@3 mot cach co he thong.
Threshold mac dinh 0.15, chinh qua --relevance_threshold neu can nhay/chat
hon. Day van la xap xi (khong phai LLM-judge nhu RAGAS context_recall), nhung
on dinh hon nhieu so voi substring.

Input testset JSON sinh boi eval/generate_ragas_testset.py. Doc ca 2 schema
cot ("user_input"/"reference" cua ragas >=0.2, hoac "question"/"ground_truth"
cua ban cu) de tuong thich nguoc. Cho production_query mode, testset item can
them cac key metadata: "organ", "modality" (vd: "breast", "ultrasound");
"top_label" va "icd10_hint" la tuy chon, mac dinh rong neu thieu.

QUAN TRONG: FAISSStore.retrieve_with_meta() nhan tham so "k" (khong phai
"top_k") -- xem services/orchestrator/rag/faiss_store.py.

Chay:
  python eval/eval_rag.py \\
    --testset_file  eval/results/ragas_testset.json \\
    --out_file      eval/results/rag_retrieval.json \\
    --top_stage1    100 \\
    --top_stage2    3 \\
    --mode          both \\
    --relevance_threshold 0.15
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Script nay chay standalone (khong qua docker-compose), nen .env KHONG duoc
# tu doc nhu khi chay trong container. FAISSStore doc FAISS_INDEX_PATH qua
# os.getenv() -- neu nguoi dung doi gia tri nay trong .env, can load_dotenv()
# o day de script dung dung file index, khong roi ve default code.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopword toi thieu (tieng Anh, vi tai lieu lam sang/RAGAS thuong la tieng
# Anh) -- chi loai cac tu chuc nang qua pho bien, khong co y dinh day du.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "by", "as", "at", "from", "this", "that",
    "it", "be", "has", "have", "had",
}


def _tokenize(text: str) -> set:
    """Lowercase, bo dau cau, tach tu, loai stopword toi thieu."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _relevance_score(ground_truth: str, chunk: str) -> float:
    """
    Jaccard overlap giua token cua ground_truth va token cua chunk.
    0.0 neu 1 trong 2 rong. Day la xap xi nhanh, khong phai LLM-judge --
    dung de xep hang tuong doi giua cac chunk, khong dung de ket luan tuyet doi.
    """
    gt_tokens = _tokenize(ground_truth)
    chunk_tokens = _tokenize(chunk)
    if not gt_tokens or not chunk_tokens:
        return 0.0
    intersection = gt_tokens & chunk_tokens
    union = gt_tokens | chunk_tokens
    return len(intersection) / len(union) if union else 0.0


def _get_question(item: dict) -> str:
    return item.get("user_input") or item.get("question") or ""


def _get_ground_truth(item: dict) -> str:
    return item.get("reference") or item.get("ground_truth") or ""


def ndcg_at_k(relevances: list, k: int) -> float:
    """Tinh nDCG@k. relevances[i] = relevance score cua item o rank i+1."""
    relevances = relevances[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(sum(r > 0 for r in relevances), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevances: list) -> float:
    """Mean Reciprocal Rank -- vi tri relevant chunk dau tien."""
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _score_pair(
    stage1_results: list, stage2_results: list, ground_truth: str, threshold: float,
) -> dict:
    """
    Tinh metrics cho 1 query, dung ca ket qua stage1 va stage2 da co san.
    1 chunk duoc tinh la relevant (rel=1) neu Jaccard overlap voi ground_truth
    >= threshold -- xem _relevance_score().
    """
    stage1_chunks = [r["chunk"] for r in stage1_results]
    stage1_rels = [1 if _relevance_score(ground_truth, c) >= threshold else 0 for c in stage1_chunks]

    stage2_chunks = [r["chunk"] for r in stage2_results]
    stage2_rels = [1 if _relevance_score(ground_truth, c) >= threshold else 0 for c in stage2_chunks]

    return {
        "stage1_hit": 1 if any(stage1_rels) else 0,
        "stage2_ndcg3": ndcg_at_k(stage2_rels, k=3),
        "stage2_precision3": sum(stage2_rels) / min(3, len(stage2_rels)) if stage2_rels else 0.0,
        "stage2_mrr": mrr(stage2_rels),
    }


def _aggregate(per_query: list) -> dict:
    if not per_query:
        return None
    return {
        "n_queries": len(per_query),
        "stage1": {
            "recall_at_100": float(np.mean([q["stage1_hit"] for q in per_query])),
            "hit_rate": float(np.mean([q["stage1_hit"] for q in per_query])),
            "note": "Muc tieu chinh: khong de lot relevant chunk ra ngoai top 100",
        },
        "stage2": {
            "ndcg_at_3": float(np.mean([q["stage2_ndcg3"] for q in per_query])),
            "precision_at_3": float(np.mean([q["stage2_precision3"] for q in per_query])),
            "mrr": float(np.mean([q["stage2_mrr"] for q in per_query])),
            "note": "Day la so lieu chinh -- top 3 nay di vao QA Agent",
        },
    }


def run_production_query_eval(store, testset: list, top1: int, top2: int, threshold: float) -> dict:
    """
    Eval theo dung cach graph.py goi RAG: query Stage 1 = "{modality} {organ}",
    Stage 2 enriched = "{top_label} {organ} {modality} findings {icd10}",
    retrieve them 5, merge voi stage1 (dedup theo text), rerank top_n.
    Can testset item co key "organ" va "modality"; "top_label"/"icd10_hint" tuy chon.
    """
    per_query = []
    n_skipped = 0
    for item in testset:
        ground_truth = _get_ground_truth(item)
        organ = item.get("organ")
        modality = item.get("modality", "ultrasound")
        if not ground_truth or not organ:
            n_skipped += 1
            continue

        query = f"{modality} {organ}".strip()
        stage1_results = store.retrieve_with_meta(query, k=top1, organ_filter=organ)

        top_label = item.get("top_label", "")
        icd10 = item.get("icd10_hint", "")
        enriched_query = f"{top_label} {organ} {modality} findings {icd10}".strip()
        meta2 = store.retrieve_with_meta(enriched_query, k=5, organ_filter=organ)

        existing_texts = {r["chunk"] for r in stage1_results}
        combined = list(stage1_results)
        for m in meta2:
            if m["chunk"] not in existing_texts:
                combined.append(m)
                existing_texts.add(m["chunk"])

        stage2_results = store.rerank(enriched_query, combined, top_n=top2)
        per_query.append(_score_pair(stage1_results, stage2_results, ground_truth, threshold))

    if n_skipped:
        print(f"[eval_rag][production_query] Bo qua {n_skipped} item thieu ground_truth/organ.")
    return _aggregate(per_query)


def run_natural_question_eval(store, testset: list, top1: int, top2: int, threshold: float) -> dict:
    """
    Eval theo cau hoi tu nhien trong testset, KHONG phai cach production goi RAG.
    Chi cho biet retriever lam gi voi 1 cau hoi tu do, tham khao, khong thay
    the cho production_query mode.
    """
    per_query = []
    n_skipped = 0
    for item in testset:
        question = _get_question(item)
        ground_truth = _get_ground_truth(item)
        if not question or not ground_truth:
            n_skipped += 1
            continue

        organ = item.get("organ")
        stage1_results = store.retrieve_with_meta(question, k=top1, organ_filter=organ)
        stage2_results = store.rerank(question, stage1_results, top_n=top2)
        per_query.append(_score_pair(stage1_results, stage2_results, ground_truth, threshold))

    if n_skipped:
        print(f"[eval_rag][natural_question] Bo qua {n_skipped} item thieu question/ground_truth.")
    return _aggregate(per_query)


def run_rag_eval(
    testset_file: str, out_file: str, top1: int, top2: int, mode: str, threshold: float,
):
    from services.orchestrator.rag.faiss_store import FAISSStore

    with open(testset_file, encoding="utf-8") as f:
        testset = json.load(f)

    store = FAISSStore()
    if not store.is_ready():
        raise SystemExit(
            "[eval_rag] FAISSStore khong san sang (chua co index.faiss / chunks.pkl). "
            "Chay scripts/build_vectordb.py truoc."
        )

    results = {"relevance_threshold": threshold}
    if mode in ("production_query", "both"):
        prod = run_production_query_eval(store, testset, top1, top2, threshold)
        if prod is None:
            print("[eval_rag] production_query: 0 sample hop le (can key 'organ' trong testset).")
        else:
            results["production_query"] = prod

    if mode in ("natural_question", "both"):
        nat = run_natural_question_eval(store, testset, top1, top2, threshold)
        if nat is None:
            print("[eval_rag] natural_question: 0 sample hop le.")
        else:
            results["natural_question"] = nat

    if len(results) <= 1:
        raise SystemExit("[eval_rag] Khong co mode nao tra ve ket qua hop le.")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Giai doan 3b - RAG retrieval eval")
    p.add_argument("--testset_file", default="eval/results/ragas_testset.json")
    p.add_argument("--out_file", default="eval/results/rag_retrieval.json")
    p.add_argument("--top_stage1", type=int, default=100)
    p.add_argument("--top_stage2", type=int, default=3)
    p.add_argument(
        "--mode", choices=["production_query", "natural_question", "both"], default="both",
    )
    p.add_argument(
        "--relevance_threshold", type=float, default=0.15,
        help="Jaccard token overlap toi thieu de tinh 1 chunk la relevant voi ground_truth.",
    )
    args = p.parse_args()
    run_rag_eval(
        args.testset_file, args.out_file, args.top_stage1, args.top_stage2,
        args.mode, args.relevance_threshold,
    )
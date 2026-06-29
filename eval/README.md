# Quy trinh danh gia (Evaluation Pipeline)

README nay liet ke tat ca buoc danh gia he thong, theo dung thu tu nen chay.
Moi buoc ghi ro: script nao, can gi truoc, lenh chay, va output nam o dau.

Quy uoc chung: tat ca lenh chay tu thu muc root cua repo (`Med_QA_Platform/`),
khong phai tu trong `eval/`.

> **Windows (PowerShell):** Thay ky tu `\` cuoi dong bang `` ` `` (backtick).
> Hoac viet toan bo lenh tren 1 dong, bo het cac `\` va xuong dong.

```
Tong quan thu tu:

  Giai doan 1: eval_router.py, eval_vision.py        (CNN thuan, offline, khong can LLM)
        |
  Giai doan 2: eval_cot.py voi LLM_BACKEND=google     (baseline Gemini)
        |
        +--> Giai doan 2.5: fine-tune Qwen
        |       eval_cot.py voi LLM_BACKEND=remote (Qwen BASE, truoc train) -> coso
        |       eval_cot.py voi LLM_BACKEND=remote (Qwen FINE-TUNED, sau train)
        |
  Giai doan 3: RAG + RAGAS (doc lap voi Giai doan 2/2.5)
        eval_rag.py, generate_ragas_testset.py, eval_ragas.py, run_pipeline_batch.py
        |
  Giai doan 4: QA Agent (/chat) -- eval_qa.py
        Phu thuoc run_pipeline_batch.py (Giai doan 3d) de co image_id trong
        context cache. Phai chay trong cung 1 lan docker compose up, trong
        vong TTL (mac dinh 3600s). Khong dung U2-Bench (xem giai thich o
        Giai doan 4 ben duoi).
```

---

## 0. Chuan bi chung

```bash
pip install -r requirements/orchestrator.txt --break-system-packages
pip install -r requirements/vision.txt --break-system-packages
pip install -r requirements/ragas_eval.txt --break-system-packages   # chi can cho Giai doan 3
```

File `.env` o root repo (copy tu `.env.example`). Cac script o Giai doan 1
khong doc `.env` (load checkpoint truc tiep qua `--ckpt`), nhung Giai doan
2/2.5/3 deu can:

```
LLM_BACKEND=google
GOOGLE_API_KEY=<api key cua ban>
GOOGLE_MODEL=gemini-2.5-flash
```

---

## Giai doan 1 -- Router + Vision CNN

Khong can Docker, khong can LLM/API key. Load checkpoint truc tiep, chay tren
CPU hoac GPU local.

### 1a. Router (EfficientNet-B0 -- phan loai breast/thyroid/OOD)

Can du lieu: `data/router/test_router/{us_breast,us_thyroid,ood}/`

```bash
python eval/eval_router.py \
  --data_dir data/router/test_router \
  --ckpt     models/checkpoints/router_effnet_b0.pth \
  --out_dir  eval/results/router
```

Output: `eval/results/router/router_eval.json`
(accuracy/F1/precision/recall 2 lop, AUROC + FPR@95TPR cho OOD, confusion
matrix, thoi gian inference).

### 1b. Vision CNN (UNet_MTL EfficientNet-B4 -- segmentation + classification)

Can du lieu: `data/busi/test_busi/` va `data/tn3k/test_tn3k/` (xem cau truc
chi tiet trong docstring dau file `eval/eval_vision.py`).

```bash
python eval/eval_vision.py \
  --busi_dir     data/busi/test_busi \
  --tn3k_dir     data/tn3k/test_tn3k \
  --busi_ckpt    models/checkpoints/mtl_effnet_fc_conv_breast.pt \
  --thyroid_ckpt models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \
  --out_dir      eval/results/vision
```

Output: `eval/results/vision/vision_eval.json`
(Dice/IoU segmentation, confusion matrix + macro-F1 classification cho ca
BUSI va TN3K, parameters/FLOPs, inference time).

Day la nen tang -- neu router/vision CNN co van de (accuracy thap, OOD
khong tach duoc), moi buoc sau (CoT, RAG) deu bi anh huong vi chung dung
output cua 2 model nay lam input.

---

## Giai doan 2 -- CoT baseline (Gemini)

Chay offline (khong can Docker), nhung CAN Gemini API key (`LLM_BACKEND=google`
trong `.env`). Goi truc tiep cac ham noi bo (vision model, knowledge mapper,
visual_interpreter) + LLM client -- khong qua HTTP giua cac service.

```bash
python eval/eval_cot.py \
  --busi_dir     data/busi/test_busi \
  --tn3k_dir     data/tn3k/test_tn3k \
  --busi_ckpt    models/checkpoints/mtl_effnet_fc_conv_breast.pt \
  --thyroid_ckpt models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \
  --out_dir      eval/results/cot_gemini \
  --rate_limit   10 \
  --resume
```

Luon dung `--resume`: moi sample duoc ghi checkpoint ngay khi xong, neu script
bi ngat giua duong (rate-limit 429, mat mang) chay lai voi `--resume` se bo
qua sample da co, khong ton quota goi lai.

Output: `eval/results/cot_gemini/cot_eval_summary.json` va `cot_eval_records.json`
(Precision/Recall/F1 macro per-class so voi ground truth, Cohen's Kappa so
voi nhan CNN, ty le parse-failure, audit case thyroid bi gan nham "normal").

Day la **baseline** -- so voi ket qua o Giai doan 2.5 de biet fine-tune Qwen
co thuc su cai thien hay khong.

---

## Giai doan 2.5 -- Fine-tune Qwen (tuy chon, ton thoi gian + chi phi GPU)

Lam buoc nay neu muon thay Gemini bang mot model nho hon, re hon, tu host
duoc (Qwen-3B). Bo qua buoc nay neu chi can dung Gemini.

### 2.5a. Sinh training data tu Gemini teacher

```bash
python scripts/generate_finetune_data.py \
  --busi_train_dir data/busi/train_busi \
  --tn3k_train_dir data/tn3k/train_tn3k \
  --busi_ckpt      models/checkpoints/mtl_effnet_fc_conv_breast.pt \
  --thyroid_ckpt   models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \
  --out_file       scripts/finetune_data/cot_training.jsonl \
  --resume
```

Dung `train_busi`/`train_tn3k` (KHONG dung `test_busi`/`test_tn3k` -- 2 tap
test danh rieng cho Giai doan 1/2/2.5c, dung lam training data se lam sai
lech ket qua eval sau nay). Nen thu `--max_busi 5 --max_tn3k 5` truoc de
kiem tra khong loi, roi moi chay full.

### 2.5b. Eval Qwen BASE (truoc fine-tune) -- lam co so so sanh

Deploy Qwen goc (chua fine-tune) len vLLM TRUOC khi train, de co mot mau so
sanh "Qwen truoc fine-tune" doc lap voi Gemini:

```bash
docker build -t cot-vllm -f docker/inference/Dockerfile .
docker run --gpus all -p 8000:8000 \
  -e HF_MODEL_REPO=Qwen/Qwen2.5-3B-Instruct \
  cot-vllm
```

`.env`:
```
LLM_BACKEND=remote
REMOTE_INFERENCE_URL=https://xxxx.runpod.net
REMOTE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
```

```bash
python eval/eval_cot.py \
  --busi_dir data/busi/test_busi --tn3k_dir data/tn3k/test_tn3k \
  --busi_ckpt models/checkpoints/mtl_effnet_fc_conv_breast.pt \
  --thyroid_ckpt models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \
  --out_dir eval/results/cot_qwen_base --resume
```

Khong co buoc nay se khong tach duoc 2 cau hoi khac nhau: "Qwen co thua
Gemini khong" va "fine-tune co cai thien Qwen khong".

### 2.5c. Fine-tune tren Colab/Kaggle

Mo `scripts/finetune_cot_colab.ipynb`, upload `cot_training.jsonl` (Buoc
2.5a) len Google Drive/Kaggle Dataset, chay Cell 1 den 9 theo thu tu. Cell 6
(eval BEFORE) PHAI chay truoc Cell 7 (train). Cell 9 merge LoRA va export
model -- upload thu muc `*_merged` len HuggingFace Hub (Private repo).

Luu y: Cell 6/8 chi danh gia tren 20 sample con trong noi bo cua notebook,
khong phai toan bo test set thuc -- chi dung de kiem tra nhanh truoc khi ton
cong deploy. Ket qua chinh thuc lay tu Buoc 2.5d ben duoi.

### 2.5d. Eval Qwen FINE-TUNED (sau train) tren toan bo test set thuc

Deploy lai model da merge (thay cho Qwen base o Buoc 2.5b):

```bash
docker run --gpus all -p 8000:8000 \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -e HF_MODEL_REPO=<your-hf-repo> \
  -e HF_TOKEN=<token> \
  -e VLLM_API_KEY=<optional-secret> \
  cot-vllm
```

`.env`:
```
REMOTE_MODEL_NAME=<your-hf-repo>
REMOTE_INFERENCE_TOKEN=<phai khop VLLM_API_KEY tren>
```

```bash
python eval/eval_cot.py \
  --busi_dir data/busi/test_busi --tn3k_dir data/tn3k/test_tn3k \
  --busi_ckpt models/checkpoints/mtl_effnet_fc_conv_breast.pt \
  --thyroid_ckpt models/checkpoints/mtl_effnet_fc_conv_thyroid.pt \
  --out_dir eval/results/cot_qwen_finetuned --resume
```

### 2.5e. So sanh 3 mau

So `cot_eval_summary.json` cua ca 3 thu muc:

| | `cot_gemini/` | `cot_qwen_base/` | `cot_qwen_finetuned/` |
|---|---|---|---|
| Macro-F1 vs ground truth | | | |
| Cohen's Kappa vs CNN | | | |
| Ty le parse-failure | | | |

Ca 3 phai chay tren cung `--busi_dir`/`--tn3k_dir`/checkpoint vision model de
so sanh cong bang. Thu tu chay A/B/C khong quan trong (doc lap nhau).

---

## Giai doan 3 -- RAG + RAGAS

Doc lap voi Giai doan 2/2.5 -- co the lam truoc, sau, hoac song song.

### 3a. Chuan bi tai lieu + build vector DB

`services/orchestrator/rag/docs/` dang trong (chi co `.gitkeep`). Tu them
tai lieu lam sang thuc te (.pdf) vao day truoc, roi:

```bash
python scripts/build_vectordb.py
```

### 3b. Sinh testset tu dong bang RAGAS

```bash
python eval/generate_ragas_testset.py \
  --docs_dir  services/orchestrator/rag/docs \
  --out_file  eval/results/ragas_testset.json \
  --n_samples 50
```

### 3c. Eval retrieval thuan (khong can anh, khong can Docker stack)

```bash
python eval/eval_rag.py \
  --testset_file eval/results/ragas_testset.json \
  --out_file     eval/results/rag_retrieval.json \
  --mode         both
```
(`both` = chay ca 2 cach query: `production_query` dung dung cach he thong
thuc te goi RAG, va `natural_question` dung cau hoi tu nhien tu testset --
2 so khac nhau, khong gop chung.)

```bash
python eval/eval_ragas.py \
  --mode         retrieval \
  --testset_file eval/results/ragas_testset.json \
  --out_file     eval/results/ragas_retrieval.csv
```

### 3d. Eval faithfulness tren full pipeline (can anh thuc + Docker stack)

```bash
docker compose up -d   # router, vision, knowledge, orchestrator phai chay

python eval/run_pipeline_batch.py \
  --image_dir data/busi/test_busi/malignant \
  --organ_hint breast \
  --api_url   http://localhost:8000

python eval/eval_ragas.py \
  --mode        pipeline \
  --pipeline_dir eval/results/pipeline_outputs \
  --out_file     eval/results/ragas_pipeline.csv
```

Luu y: `--organ_hint`/`--modality_hint` cua `/analyze` deu nhan `'breast' |
'thyroid' | None` -- khong phai ten modality kieu `'ultrasound'`. Thuong chi
can dien `--organ_hint`, de `--modality_hint` trong la du.

---

## Giai doan 4 -- QA Agent (/chat endpoint)

KHONG dung U2-Bench: U2-Bench danh gia LVLM tu nhin anh tho de tra loi, con
`/chat` cua he thong nay khong nhan anh nua (chi nhan `image_id` + `message`
+ `history`, dung lai context da cache tu `/analyze`). Format cua U2-Bench
khong khop kien truc nay -- xem chi tiet trong docstring dau file `eval_qa.py`.

Day la danh gia **rieng cho `/chat`** (chatbot tra loi cau hoi follow-up),
KHONG phai danh gia lai report Tier 2/3 ban dau -- viec do thuoc ve
`eval_ragas.py --mode pipeline` o Giai doan 3d.

QUAN TRONG -- TTL: `_context_cache` la in-memory theo tung process
orchestrator, TTL mac dinh 3600s (`CHAT_CONTEXT_TTL` trong `.env`). Phai
chay `eval_qa.py` ngay sau `run_pipeline_batch.py`, trong cung 1 lan
`docker compose up`, khong cach qua xa ve thoi gian -- neu khong `/chat` se
tra 404 vi `image_id` khong con trong cache.

```bash
docker compose up -d   # neu chua chay tu Giai doan 3d

python eval/run_pipeline_batch.py \
  --image_dir data/busi/test_busi/malignant \
  --organ_hint breast \
  --api_url   http://localhost:8000

python eval/eval_qa.py \
  --pipeline_dir eval/results/pipeline_outputs \
  --api_url      http://localhost:8000 \
  --out_file     eval/results/qa_eval.json \
  --n_dynamic_questions 2
```

Nguon cau hoi la hybrid: 1 bo cau hoi co dinh (`FIXED_QUESTIONS` trong
`eval_qa.py`, on dinh giua cac lan chay) + N cau hoi Gemini tu sinh rieng cho
tung report (`--n_dynamic_questions`, bat duoc cau hoi dac thu cho tung case).

G-Eval (LLM-judge la Gemini) cham moi cau tra loi tren 5 tieu chi, 0-5 diem:
Faithfulness (bam sat report/RAG context, khong tu bia), Relevance (dung
trong tam cau hoi), Safety (giu dung khuyen nghi xac nhan voi bac si, khong
chan doan xac dinh), Consistency (khong mau thuan voi so lieu Tier 1), va
Clarity (de hieu voi nguoi khong chuyen mon).

Output: `eval/results/qa_eval.json` (diem trung binh moi tieu chi +
chi tiet tung cau hoi/tra loi/diem).

---

## Tom tat vi tri output

```
eval/results/
  router/router_eval.json
  vision/vision_eval.json
  cot_gemini/{cot_eval_summary.json, cot_eval_records.json}
  cot_qwen_base/{...}                  (Giai doan 2.5b)
  cot_qwen_finetuned/{...}             (Giai doan 2.5d)
  ragas_testset.json                   (Giai doan 3b)
  rag_retrieval.json                   (Giai doan 3c, eval_rag.py)
  ragas_retrieval.csv                  (Giai doan 3c, eval_ragas.py --mode retrieval)
  pipeline_outputs/*.json              (Giai doan 3d, run_pipeline_batch.py)
  ragas_pipeline.csv                   (Giai doan 3d, eval_ragas.py --mode pipeline)
  qa_eval.json                         (Giai doan 4, eval_qa.py)
```
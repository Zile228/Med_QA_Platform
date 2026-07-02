"""
eval/generate_ragas_testset.py
=================================
Giai doan 3a - Dung RAGAS TestsetGenerator de tu sinh QA pairs tu documents
trong RAG knowledge base. Khong viet ground truth tay.

QUAN TRONG -- tai lieu RAG la PDF, KHONG phai .txt:
scripts/build_vectordb.py doc PDF qua pymupdf4llm.to_markdown() (xem ham
pdf_to_marked_markdown() trong file do), KHONG doc .txt. Script nay dung lai
CUNG MOT pipeline tien xu ly voi build_vectordb.py -- khong tu chunk rieng
nua -- de testset phan anh dung granularity va noi dung thuc su nam trong
FAISS index:

    pdf_to_marked_markdown()  (page markers <!--page:N-->, co timeout/subprocess)
        -> split_into_sections()   (tach theo heading Markdown H1-H4)
        -> _is_reference_heading() (bo section "References"/"Bibliography")
        -> chunk_section()         (cat theo token cua CHINH embedder that,
                                     khong phai tiktoken -- xem phan duoi)
        -> _looks_like_reference_chunk() (bo chunk citation-list con sot lai
                                     trong 1 section noi dung binh thuong)

4 ham + 2 regex-filter tren deu IMPORT TRUC TIEP tu scripts/build_vectordb.py
(khong copy lai) de 2 pipeline khong bao gio lech pha nhau.

Sau khi co list Document da chunk dung chuan production, moi dua vao RAGAS
qua generate_with_chunks().

TAI SAO PHAI DUNG EMBEDDER TOKENIZER THAT (khong con dung tiktoken cl100k_base):
build_vectordb.py dem token bang embedder.tokenizer (tokenizer cua chinh
model embedding se encode cac chunk nay, xem chunk_section() trong
build_vectordb.py). tiktoken cl100k_base la tokenizer cua OpenAI, cho ra
token count KHAC (thuong it token hon cho cung 1 doan text y khoa) --
dung no de cat chunk_tokens=200 se tao ra chunk DAI HON thuc te so voi
chunk nam trong FAISS index, khien testset khong con phan anh dung
granularity retrieval that. Script nay gio load cung 1 embedding model
(--model, mac dinh trung voi build_vectordb.py) chi de lay .tokenizer,
KHONG dung no de encode/tao embedding (viec do van do GoogleGenerativeAIEmbeddings/
OpenAIEmbeddings cua RAGAS dam nhiem o buoc generate).

Ly do phai chunk truoc (khong de RAGAS tu chunk):
RAGAS HeadlineSplitter noi bo loop rat cham khi gap Document khong lo khong co
heading (dac biet la fallback fitz). Script dung generate_with_chunks() de RAGAS
biet day la pre-chunked data va dung default_transforms_for_prechunked(), bo qua
hoan toan HeadlinesExtractor + HeadlineSplitter.

FALLBACK fitz khi gap loi ONNX int32/int64 HOAC khi pdf_to_marked_markdown()
timeout/loi:
pymupdf_layout (layout model ONNX ben trong pymupdf4llm) co the throw
"[ONNXRuntimeError]: Unexpected input data type. Actual: (tensor(int32)),
expected: (tensor(int64))" tren mot so CPU/OS/version combo. Day KHONG phai
loi cua file PDF -- file van doc duoc bang fitz thuan. pdf_to_marked_markdown()
(import tu build_vectordb.py) DA TU XU LY loi ONNX nay o BEN TRONG worker
subprocess cua no (retry voi use_layout(False)) va co timeout cung
(PDF_TIMEOUT_SECONDS) de 1 PDF treo khong lam ket ca batch.

Script nay them 1 lop fallback fitz THUAN o NGOAI, chi kich hoat khi
pdf_to_marked_markdown() van tra ve rong sau ca 2 lan thu noi bo (vi du:
timeout, hoac loi khac ONNX). fitz.get_text() KHONG tao heading Markdown
(khong co dong "## ..."), nen voi PDF roi vao nhanh nay:
  - split_into_sections() se KHONG tach duoc section nao (toan bo van ban
    nam trong 1 section duy nhat, heading=None)
  - _is_reference_heading() do do KHONG BAO GIO khop (khong co heading de
    so khop) -- toan bo section "References" cua PDF do se KHONG bi loai
    o tang heading-level nua
  - Chi con _looks_like_reference_chunk() (tang chunk-level, dua tren mat
    do citation-marker/multi-author/journal-cite trong tung chunk rieng le)
    con hieu luc loc reference cho PDF nay
Day la suy giam co chu y, KHONG phai bug: van tot hon truoc day (0% duoc
loc), va script IN CANH BAO RO RANG moi khi 1 PDF roi vao nhanh nay, kem ten
file, de nguoi doc log biet ngay heading-based filtering da mat hieu luc
cho PDF do va co the kiem tra thu cong neu can.

Chi index cac PDF khop ALLOWED_PDF_FILENAMES (import truc tiep tu
scripts/build_vectordb.py, khong copy lai) -- cung allow-list dang dung de
build FAISS index, nen testset sinh ra luon phan anh dung tap tai lieu thuc
te trong vectordb. Organ duoc gan vao metadata moi Document tu ten file PDF
qua _detect_organ() (cung import tu build_vectordb.py) de cac sample trong
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

LLM sinh testset (Gemini hoac OpenAI) duoc chon qua env RAGAS_LLM_BACKEND
(mac dinh: theo LLM_BACKEND, fallback "google" neu ca hai khong set).
Model cu the doc tu GOOGLE_MODEL (Google) hoac OPENAI_MODEL (OpenAI) trong
.env -- khong con hard-code "gemini-2.5-flash" nhu phien ban truoc.

RATE LIMITING -- free tier Gemini (10-15 RPM):
RAGAS dung async va fire nhieu request cung luc trong HeadlinesExtractor va
QA generation, de bi 429 "quota exceeded" tren free tier. Script dung
langchain_core.rate_limiters.InMemoryRateLimiter de throttle o tang Langchain
truoc khi request di ra ngoai. Mac dinh --rpm 8 (an toan cho free tier 10 RPM).
Tang len --rpm 30 neu dung Gemini paid, --rpm 60 neu dung OpenAI.

OPENAI O-SERIES (o1/o3/o4) -- temperature fix:
RAGAS noi bo force temperature=0.01 vao moi LLM call. OpenAI o-series models
(o1, o1-mini, o3, o3-mini, o4-mini, ...) chi chap nhan temperature=1 va tra
ve 400 "Unsupported value: temperature does not support 0.01". Script tu dong
phat hien o-series qua ten model va dung subclass override _get_request_payload
de strip temperature khoi call-time kwargs truoc khi gui request.

Chay:
  # Free tier Gemini (mac dinh 8 RPM):
  python eval/generate_ragas_testset.py \\
    --docs_dir   services/orchestrator/rag/docs \\
    --out_file   eval/results/ragas_testset.json \\
    --n_samples  50

  # Paid tier hoac OpenAI (tang RPM):
  python eval/generate_ragas_testset.py \\
    --docs_dir   services/orchestrator/rag/docs \\
    --out_file   eval/results/ragas_testset.json \\
    --n_samples  50 --rpm 30

  # Dung embedding model khac voi mac dinh build_vectordb.py (vi du dang
  # thu nghiem 1 checkpoint moi va muon testset khop tokenizer cua no):
  python eval/generate_ragas_testset.py \\
    --model models/checkpoints/embedding_model_finetuned_v2
"""
import argparse
import asyncio
import glob
import os
import random
import re
import sys
from pathlib import Path

# RAGAS goi asyncio.run() nhieu lan TUAN TU (moi transform pipeline mot lan:
# SummaryExtractor, EmbeddingExtractor, ThemesExtractor, NERExtractor...).
# asyncio.run() luon dong loop khi coroutine xong -- dieu nay dung cho CA
# ProactorEventLoop lan SelectorEventLoop, nen doi policy KHONG tu no giai
# quyet duoc "RuntimeError: Event loop is closed". Nguyen nhan that: neu
# httpx.AsyncClient duoc tao MOT LAN roi tai su dung qua nhieu asyncio.run(),
# client se giu tham chieu toi transport gan voi loop da bi dong tu lan
# truoc. Fix that su nam o _get_testset_llm_embeddings() (KHONG truyen
# http_async_client tuong minh, de SDK lazy-tao client dung trong tung loop).
# Selector van duoc giu lai o day vi no on dinh hon Proactor cho cac tac vu
# I/O khac tren Windows (vi du subprocess cua thu vien khac dung sau nay),
# nhung ban than no khong phai fix cho loi Event loop is closed noi tren.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dung chung pipeline tien xu ly voi scripts/build_vectordb.py, thay vi duy
# tri 2 ban sao co the lech pha nhau:
#   - ALLOWED_PDF_FILENAMES, _detect_organ: chon PDF + gan organ
#   - pdf_to_marked_markdown: PDF -> markdown co <!--page:N--> marker,
#     chay trong subprocess rieng voi timeout, tu retry use_layout(False)
#     khi gap loi ONNX int32/int64
#   - split_into_sections: tach markdown thanh cac section theo heading H1-H4
#   - _is_reference_heading: loai section co heading la "References"/...
#   - chunk_section: cat 1 section thanh cac chunk theo token (dung
#     embedder.tokenizer THAT, khong phai tiktoken), strip page-marker/
#     picture-text-marker/<br> khoi chunk text
#   - _looks_like_reference_chunk: loai chunk MOSTLY la citation list con
#     sot lai trong 1 section noi dung binh thuong
# parents[1] la project root vi file nay nam o eval/, giong pattern da dung
# trong eval/eval_rag.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_vectordb import (
    ALLOWED_PDF_FILENAMES,
    _detect_organ,
    _is_reference_heading,
    _looks_like_reference_chunk,
    chunk_section,
    pdf_to_marked_markdown,
    split_into_sections,
)

# Script nay chay standalone, .env khong tu doc nhu trong container.
# Phai load o day, neu khong cac bien GOOGLE_API_KEY/OPENAI_API_KEY se rong.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _fitz_fallback_text(pdf_path: str) -> str:
    """
    Fallback NGOAI CUNG: doc toan bo text tu PDF bang fitz thuan tuy
    (khong qua ONNX layout model, khong qua ca hai lan retry noi bo cua
    pdf_to_marked_markdown()). Chi duoc goi khi pdf_to_marked_markdown()
    da tra ve rong (vi du: PDF_TIMEOUT_SECONDS timeout, hoac 1 loi khac
    voi loi ONNX int32/int64 ma retry noi bo khong xu ly duoc).

    CANH BAO QUAN TRONG: fitz.get_text("text") KHONG tao heading Markdown
    ("## ..."). Voi PDF doc qua nhanh nay, split_into_sections() o buoc
    sau se KHONG the tach section nao ca -- toan bo noi dung PDF roi vao
    dung 1 section voi heading=None. He qua truc tiep:
      - _is_reference_heading() khong co gi de so khop -> section
        "References"/"Bibliography" cua PDF nay se KHONG bi loai o tang
        heading-level nua.
      - Chi con _looks_like_reference_chunk() (tang chunk-level, dua vao
        mat do citation-marker trong tung chunk rieng le) tiep tuc loc
        duoc phan nao, nhung se BO SOT nhieu citation hon so voi PDF doc
        thanh cong qua pymupdf4llm.

    Moi trang duoc noi vao chuoi ket qua voi page marker <!--page:N--> de
    giu nhat quan voi format cua pdf_to_marked_markdown() trong
    build_vectordb.py (can cho _extract_page_range-style logic o cac noi
    khac dung chung PAGE_MARKER_RE).
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"<!--page:{i}-->\n{text}")
    doc.close()
    return "\n\n".join(parts)


def _pdf_to_markdown_with_fallback(pdf_path: str) -> tuple:
    """
    Goi pdf_to_marked_markdown() (co timeout + retry ONNX noi bo, xem
    build_vectordb.py). Neu no van tra ve rong, thu fallback fitz thuan
    o day.

    Tra ve (markdown_text, used_fitz_fallback: bool). Cai thu 2 duoc dung
    o cap tren de in canh bao ro rang khi heading-based reference filtering
    mat hieu luc cho file nay (xem docstring _fitz_fallback_text).
    """
    filename = os.path.basename(pdf_path)
    text = pdf_to_marked_markdown(pdf_path)
    if text.strip():
        return text, False

    print(f"  [warn] pdf_to_marked_markdown() khong doc duoc {filename} "
          f"(timeout hoac loi) -- thu fallback fitz thuan...")
    try:
        text = _fitz_fallback_text(pdf_path)
    except Exception as e2:
        print(f"  [warn] Fallback fitz cung that bai voi {filename}: {e2} -- bo qua.")
        return "", False

    if not text.strip():
        print(f"  [warn] Fallback fitz: {filename} khong co noi dung -- bo qua.")
        return "", False

    print(
        f"  [info] Fallback fitz thanh cong: {filename}. "
        f"[CANH BAO] PDF nay KHONG co heading Markdown tu fitz thuan, nen "
        f"split_into_sections() se coi toan bo la 1 section (heading=None) "
        f"va bo loc heading-level cho section References/Bibliography SE "
        f"KHONG hoat dong voi file nay -- chi con bo loc chunk-level "
        f"(_looks_like_reference_chunk) con hieu luc. Kiem tra thu cong "
        f"testset sinh ra tu {filename} neu can chac chan khong con sot "
        f"citation list."
    )
    return text, True


# Cap so chunk de kiem soat so LLM call trong SummaryExtractor.
# RAGAS goi 1 LLM request cho moi chunk (SummaryExtractor) truoc khi generate.
# Voi PDF lon (vi du sach giao khoa Breast Ultrasound nhieu tram trang),
# so chunk co the len toi hang nghin. Cap per-file + total giu runtime o
# muc chap nhan duoc:
#   300 chunks x (1/30 RPM) = 10 phut cho SummaryExtractor o 30 RPM
#   300 chunks x (1/60 RPM) = 5 phut o 60 RPM
# Sampling ngau nhien (random.sample) trong moi file de tranh lay toan bo
# phan dau ma bo qua nua sau cua tai lieu.
_MAX_CHUNKS_PER_FILE = 50    # sample toi da 50 chunks moi PDF
_MAX_CHUNKS_TOTAL = 300      # cap tong so chunks dua vao RAGAS

# Thong so chunk mac dinh giong build_vectordb.py: 200 tokens moi chunk,
# 30 overlap -- dung lam gia tri --chunk_tokens/--overlap_tokens mac dinh
# de testset phan anh dung granularity FAISS index thuc te (tru khi nguoi
# dung chu dong doi bang CLI flag).
_CHUNK_TOKENS = 200
_CHUNK_OVERLAP_TOKENS = 30

# RAGAS multi-hop synthesizer chen prefix nay vao dau moi context trong
# reference_contexts; chunk goc khong co prefix nay nen phai cat truoc khi
# so khop organ (xem _organ_from_ref_contexts trong main()).
_HOP_PREFIX_RE = re.compile(r"^<\d+-hop>\s*")


def _pdf_to_documents(
    pdf_path: str,
    embedder,
    chunk_tokens: int,
    overlap_tokens: int,
    max_per_file: int,
) -> list:
    """
    Chuyen 1 PDF thanh list LangChain Document, dung DUNG pipeline tien xu
    ly cua build_vectordb.py (page-marker markdown -> tach heading -> loc
    section References -> chunk theo token cua embedder that -> loc chunk
    citation-list con sot) thay vi tu chunk phang toan bo van ban.

    Neu so chunk hop le vuot max_per_file, lay mau ngau nhien (random.sample)
    phan bo deu tren toan bo tai lieu thay vi chi lay phan dau -- tranh bias
    theo trang, giong logic cap cu.

    Metadata moi Document gom:
      - source: ten file PDF (dung "source" de tuong thich nguoc voi cho
        nao trong pipeline dang doc key nay)
      - organ: tu _detect_organ(filename), giong build_vectordb.py
      - section_heading, page_number: giu lai tu metadata that cua
        build_vectordb.py (huu ich khi debug testset, khong bat buoc RAGAS
        dung toi)
    """
    filename = os.path.basename(pdf_path)
    organ = _detect_organ(filename)

    marked_markdown, used_fitz_fallback = _pdf_to_markdown_with_fallback(pdf_path)
    if not marked_markdown.strip():
        return []

    sections = split_into_sections(marked_markdown)

    from langchain_core.documents import Document

    docs = []
    n_sections_dropped = 0
    n_chunks_dropped = 0
    for section in sections:
        if _is_reference_heading(section["heading"]):
            n_sections_dropped += 1
            continue

        section_chunks = chunk_section(section["text"], embedder, chunk_tokens, overlap_tokens)
        for _raw_chunk, clean_text in section_chunks:
            if _looks_like_reference_chunk(clean_text):
                n_chunks_dropped += 1
                continue
            docs.append(
                Document(
                    page_content=clean_text,
                    metadata={
                        "source": filename,
                        "organ": organ,
                        "section_heading": section["heading"],
                    },
                )
            )

    if n_sections_dropped or n_chunks_dropped:
        print(
            f"    [ref-filter] {filename}: dropped {n_sections_dropped} "
            f"reference-heading section(s), {n_chunks_dropped} citation-like "
            f"chunk(s)"
            + (" [gioi han: chi chunk-level filter co hieu luc do fallback fitz]"
               if used_fitz_fallback else "")
        )

    if len(docs) > max_per_file:
        docs = random.sample(docs, max_per_file)

    return docs


def _load_pdf_documents(
    docs_dir: str,
    embedder,
    chunk_tokens: int = _CHUNK_TOKENS,
    overlap_tokens: int = _CHUNK_OVERLAP_TOKENS,
    max_chunks_per_file: int = _MAX_CHUNKS_PER_FILE,
    max_chunks_total: int = _MAX_CHUNKS_TOTAL,
) -> list:
    """
    Doc toan bo PDF trong docs_dir khop ALLOWED_PDF_FILENAMES (xem
    scripts/build_vectordb.py), tra ve list LangChain Document da chunk
    dung PHIEN BAN GIONG HET pipeline production (heading-split + loc
    reference 2 tang + token-chunk bang embedder that), co cap so luong
    de kiem soat runtime cua RAGAS SummaryExtractor.

    Dung chung allow-list voi scripts/build_vectordb.py de testset sinh ra
    tu dung cung tap PDF da duoc index vao FAISS -- PDF dang bang tra cuu ma
    benh (ICD-10) khong nam trong allow-list vi khong phai van ban lam sang
    dien giai, xem ly do chi tiet trong docstring ALLOWED_PDF_FILENAMES.

    Ly do cap max_chunks_*:
    SummaryExtractor cua RAGAS goi 1 LLM request cho moi chunk. max_chunks_total
    gioi han tong so LLM call du so PDF trong allow-list la bao nhieu, giu
    runtime o muc du doan duoc (vi du 300 chunks x 2s/req ~ 10-15 phut o 30 RPM).

    Ly do chunking truoc (thay vi de RAGAS tu chunk):
    RAGAS HeadlineSplitter noi bo loop rat lau voi doc khong co heading ro rang
    (dac biet fallback fitz). generate_with_chunks() dung
    default_transforms_for_prechunked() bo qua hoan toan HeadlinesExtractor +
    HeadlineSplitter.
    """
    all_found = sorted(glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True))
    allowed_lower = {n.lower() for n in ALLOWED_PDF_FILENAMES}
    pdf_files = [p for p in all_found if os.path.basename(p).lower() in allowed_lower]
    for p in sorted(set(all_found) - set(pdf_files)):
        print(f"  [SKIP] {os.path.basename(p)} not in ALLOWED_PDF_FILENAMES -- not used for testset.")

    all_docs: list = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")
        try:
            docs = _pdf_to_documents(
                pdf_path, embedder, chunk_tokens, overlap_tokens, max_chunks_per_file
            )
        except Exception as e:
            print(f"  [warn] Unexpected error processing {filename}: {e} -- bo qua.")
            continue

        if not docs:
            print(f"  [warn] {filename} sau khi chunk khong co doan nao -- bo qua.")
            continue

        all_docs.extend(docs)
        organ = _detect_organ(filename)
        print(f"    -> {len(docs)} chunks (organ={organ})")

    # Cap tong: shuffle de phan bo deu cac file, sau do slice
    if len(all_docs) > max_chunks_total:
        random.shuffle(all_docs)
        all_docs = all_docs[:max_chunks_total]
        print(f"  [info] Cap tong: giu {max_chunks_total} chunks (shuffle ngau nhien)")

    return all_docs


def _load_embedder(model_path: str):
    """
    Load CUNG mot embedding model (kien truc + tokenizer) voi
    build_vectordb.py, chi de lay .tokenizer dung cho chunk_section().

    Day KHONG phai buoc tao embedding cho RAGAS -- embedding cho RAGAS
    (dung de xay knowledge graph / chon cau hoi) van do
    GoogleGenerativeAIEmbeddings hoac OpenAIEmbeddings dam nhiem rieng o
    _get_testset_llm_embeddings(). Load model o day chi nham muc dich
    dung DUNG tokenizer ma production dung de quyet dinh do dai 200-token
    cua moi chunk, thay vi tiktoken cl100k_base (tokenizer khac, cho token
    count khac, se lam testset khong con phan anh dung granularity chunk
    thuc te trong FAISS index).
    """
    from sentence_transformers import SentenceTransformer

    print(f"[embedder] Loading tokenizer tu {model_path} (dung de chunk giong build_vectordb.py)...")
    return SentenceTransformer(model_path)


def _get_testset_llm_embeddings(rpm: int):
    """
    Khoi tao langchain LLM + embeddings cho TestsetGenerator, co kem rate limiter.

    Tra ve tuple (langchain_llm, embeddings, bypass_temperature):
      - langchain_llm: ChatOpenAI hoac ChatGoogleGenerativeAI thuan tuy (chua wrap).
      - embeddings: OpenAIEmbeddings hoac GoogleGenerativeAIEmbeddings.
      - bypass_temperature: True khi backend la OpenAI.

    Tai sao bypass_temperature=True voi OpenAI?
    RAGAS noi bo hard-code temperature=0.01 vao moi LLM call qua
    LangchainLLMWrapper.agenerate_text(). Nhieu model OpenAI moi (gpt-5,
    o1, o3, o4, ...) chi chap nhan temperature=1 (default) va tra ve
    400 "Unsupported value: temperature does not support 0.01". Day la
    dac tinh cua tung model, khong the detect qua ten (gpt-5 khong phai
    o-series nhung cung bi).

    Giai phap dung: khi goi LangchainLLMWrapper(llm, bypass_temperature=True),
    RAGAS khong set temperature vao langchain_llm.temperature truoc moi call
    (ragas/llms/base.py dong 291). Model dung temperature mac dinh cua no (= 1).
    Cach nay future-proof hon detect ten model.

    Google Gemini khong co restriction nay nen bypass_temperature=False.

    rpm: gioi han request/phut. RAGAS chay async va fire nhieu request cung
    luc, de bi 429 voi free-tier. InMemoryRateLimiter inject vao LLM object
    de throttle truoc khi request ra ngoai. max_bucket_size=1 loai bo burst.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter

    rps = rpm / 60.0
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.5,
        max_bucket_size=1,  # khong burst
    )
    print(f"[rate-limit] {rpm} RPM ({rps:.3f} RPS) -- gap khoang {60/rpm:.1f}s giua moi request LLM")

    backend = (os.getenv("RAGAS_LLM_BACKEND") or os.getenv("LLM_BACKEND") or "google").lower()

    if backend == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[llm] OpenAI model: {openai_model} (bypass_temperature=True)")
        # KHONG truyen http_async_client=httpx.AsyncClient() tuong minh o day.
        #
        # RAGAS goi asyncio.run() nhieu lan TUAN TU, moi lan cho 1 cum
        # transform (SummaryExtractor, EmbeddingExtractor, ThemesExtractor,
        # NERExtractor...). asyncio.run() luon DONG loop khi coroutine xong,
        # bat ke loop la Proactor hay Selector -- doi loop policy (ban truoc)
        # khong sua duoc van de nay.
        #
        # Neu tao httpx.AsyncClient() MOT LAN ben ngoai (nhu ban cu), object
        # do song xuyen suot ca process, con loop no dung lan dau thi da bi
        # asyncio.run() dong ngay sau lan goi dau tien. Lan goi thu 2 tai su
        # dung client cu -> client giu tham chieu toi transport gan voi loop
        # da dong -> "RuntimeError: Event loop is closed" (xay ra o buoc
        # aclose() khi httpcore don dep connection, thay vi luc gui request).
        #
        # AsyncOpenAI (SDK goc, ChatOpenAI dung ben trong) tu lazy-tao
        # httpx.AsyncClient rieng NEU khong truyen http_client/http_async_client.
        # Client lazy nay duoc tao dung luc request dau tien chay, tuc la BEN
        # TRONG coroutine dang chay tren loop hien tai -- khong bi "mang theo"
        # tu loop cu sang loop moi. Bo tham so nay di la cach don gian nhat
        # de moi asyncio.run() co client rieng, khop dung voi loop cua no.
        #
        # NHUNG: tren thuc te (xac nhan qua traceback that tren Windows +
        # openai SDK ban moi), ChatOpenAI/AsyncOpenAI van cache 1
        # httpx.AsyncClient o CAP INSTANCE (self.root_async_client), khong
        # lazy-tao lai moi request nhu gia dinh o tren -- nen client tao ra
        # o transform dau tien (vi du EmbeddingExtractor/ThemesExtractor)
        # van bi "mang sang" asyncio.run() cua transform SAU (NERExtractor),
        # noi loop cu da dong -> aclose() nem "RuntimeError: Event loop is
        # closed" khi httpcore don connection, loi nay bi wrap thanh
        # openai.APIConnectionError roi RAGAS in ra "Task failed with
        # APIConnectionError: Connection error." Khong truyen
        # http_async_client KHONG du de tranh van de nay tren moi SDK.
        #
        # max_retries=6 (langchain_openai truyen xuong OpenAI SDK, dung
        # exponential backoff noi bo cua SDK) la lop phong thu thuc te: khi
        # 1 request chet vi client cu gan voi loop da dong, SDK retry se
        # thu lai va thuong thanh cong ngay o lan ke tiep (luc do da dang
        # chay on dinh trong loop hien tai). Day KHONG phai fix goc cho bug
        # cache-client noi tren, nhung tranh duoc viec 1 loi thoang qua lam
        # sap ca lan generate testset (thuong chay hang chuc phut).
        generator_llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=1,           # gia tri mac dinh; bypass_temperature se skip viec RAGAS ghi de
            rate_limiter=rate_limiter,
            max_retries=6,           # chong "Event loop is closed" / connection error thoang qua, xem giai thich tren
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
            max_retries=6,
        )
        return generator_llm, embeddings, True   # bypass_temperature=True

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    google_api_key = os.environ["GOOGLE_API_KEY"]
    # Doc model tu env GOOGLE_MODEL, fallback "gemini-2.0-flash-lite" neu khong set.
    # Truoc day hard-code "gemini-2.5-flash" nen bo qua GOOGLE_MODEL trong .env.
    google_model = os.getenv("GOOGLE_MODEL") or "gemini-2.0-flash-lite"
    print(f"[llm] Google model: {google_model}")
    generator_llm = ChatGoogleGenerativeAI(
        model=google_model,
        google_api_key=google_api_key,
        rate_limiter=rate_limiter,
    )
    # gemini-embedding-001: embedding-001 va text-embedding-004 da deprecated,
    # day la model embedding hien hanh cua Gemini (3072 dims).
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=google_api_key
    )
    return generator_llm, embeddings, False   # bypass_temperature=False


def main(docs_dir: str, out_file: str, n_samples: int, rpm: int,
         model_path: str,
         chunk_tokens: int = _CHUNK_TOKENS,
         overlap_tokens: int = _CHUNK_OVERLAP_TOKENS,
         max_chunks_per_file: int = _MAX_CHUNKS_PER_FILE,
         max_chunks_total: int = _MAX_CHUNKS_TOTAL):
    import warnings
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph

    embedder = _load_embedder(model_path)

    chunks = _load_pdf_documents(
        docs_dir,
        embedder,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        max_chunks_per_file=max_chunks_per_file,
        max_chunks_total=max_chunks_total,
    )
    print(f"[chunks] {len(chunks)} chunks dua vao RAGAS (cap: {max_chunks_per_file}/file, {max_chunks_total} tong)")
    if not chunks:
        raise SystemExit(
            f"[generate_ragas_testset] 0 chunks from {docs_dir}. "
            "Add real clinical guideline PDFs there first (same docs indexed "
            "by scripts/build_vectordb.py) before generating a testset."
        )

    langchain_llm, embeddings, bypass_temperature = _get_testset_llm_embeddings(rpm=rpm)

    # Khong dung TestsetGenerator.from_langchain() vi no hard-code
    # LangchainLLMWrapper(llm) khong co bypass_temperature -- RAGAS se force
    # temperature=0.01 va gay 400 voi OpenAI gpt-5/o-series/model moi.
    # Tu tao wrapper voi bypass_temperature dung, roi construct TestsetGenerator
    # truc tiep (dataclass, nhan llm/embedding_model/knowledge_graph).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # LangchainLLMWrapper deprecated warning
        llm_wrapper = LangchainLLMWrapper(langchain_llm, bypass_temperature=bypass_temperature)
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings)

    generator = TestsetGenerator(
        llm=llm_wrapper,
        embedding_model=emb_wrapper,
        knowledge_graph=KnowledgeGraph(),
    )
    # generate_with_chunks() dung default_transforms_for_prechunked() -- bo qua
    # HeadlinesExtractor va HeadlineSplitter, tranh loop cham voi doc lon.
    testset = generator.generate_with_chunks(chunks, testset_size=n_samples)

    df = testset.to_pandas()

    # RAGAS to_pandas() / to_list() chi serialize SingleTurnSample.model_dump() +
    # synthesizer_name -- khong co metadata/source/organ cua Document goc.
    # Map nguoc: xay dict chunk_text -> organ tu danh sach chunks da co (moi
    # chunk.metadata["organ"] duoc set truoc khi dua vao RAGAS). Moi sample trong
    # testset co "reference_contexts" la list van ban chunk goc RAGAS chon lam
    # nguon -- lay organ cua chunk dau tien tra ve, fallback "general" neu rong.
    if "organ" not in df.columns:
        chunk_organ: dict = {
            doc.page_content: doc.metadata.get("organ", "general")
            for doc in chunks
        }

        # Cac synthesizer multi-hop (single_hop_specific_query_synthesizer,
        # multi_hop_*) chen prefix "<N-hop>\n\n" vao dau moi context trong
        # reference_contexts. Prefix nay khong ton tai trong chunk goc, nen
        # phai bi cat truoc khi so khop, neu khong ca exact match lan
        # partial-prefix match ben duoi deu truot va moi mau co prefix nay
        # se bi gan nham organ="general".
        def _strip_hop_prefix(ctx: str) -> str:
            return _HOP_PREFIX_RE.sub("", ctx)

        def _organ_from_ref_contexts(ref_ctxs) -> str:
            if not isinstance(ref_ctxs, list):
                return "general"
            cleaned = [
                _strip_hop_prefix(ctx) for ctx in ref_ctxs if isinstance(ctx, str)
            ]
            for ctx in cleaned:
                if ctx in chunk_organ:
                    return chunk_organ[ctx]
            # Fallback: partial match 120 ky tu dau de xu ly RAGAS trim text
            for ctx in cleaned:
                prefix = ctx[:120]
                for chunk_text, organ in chunk_organ.items():
                    if chunk_text.startswith(prefix) or prefix in chunk_text:
                        return organ
            return "general"

        ref_col = df["reference_contexts"] if "reference_contexts" in df.columns else None
        df["organ"] = (ref_col if ref_col is not None else [None] * len(df)).apply(
            _organ_from_ref_contexts
        )
    if "modality" not in df.columns:
        df["modality"] = "ultrasound"

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
    p.add_argument(
        "--model",
        default="models/checkpoints/embedding_model_finetuned_final",
        help=(
            "Embedding model dung de lay tokenizer khi chunk (--chunk_tokens/"
            "--overlap_tokens duoc dem bang tokenizer cua model nay, giong "
            "het build_vectordb.py --model). Mac dinh trung voi build_vectordb.py "
            "de granularity chunk khop chinh xac voi FAISS index thuc te. "
            "KHONG dung de tao embedding cho RAGAS -- chi de lay .tokenizer."
        ),
    )
    p.add_argument("--chunk_tokens", type=int, default=_CHUNK_TOKENS,
                    help=f"So token moi chunk (mac dinh {_CHUNK_TOKENS}, giong build_vectordb.py).")
    p.add_argument("--overlap_tokens", type=int, default=_CHUNK_OVERLAP_TOKENS,
                    help=f"So token overlap giua 2 chunk lien tiep (mac dinh {_CHUNK_OVERLAP_TOKENS}).")
    p.add_argument(
        "--rpm",
        type=int,
        default=8,
        help=(
            "Gioi han request/phut gui den LLM API. Mac dinh 8 RPM an toan cho "
            "Gemini free-tier (quota 10-15 RPM). Tang len neu dung paid tier "
            "(vi du --rpm 60 voi OpenAI, --rpm 30 voi Gemini paid)."
        ),
    )
    p.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=_MAX_CHUNKS_PER_FILE,
        help=(
            f"So chunk toi da lay tu moi PDF (sample ngau nhien neu vuot). "
            f"Mac dinh {_MAX_CHUNKS_PER_FILE}. "
            "Tang len de testset da dang hon nhung lau hon."
        ),
    )
    p.add_argument(
        "--max_chunks_total",
        type=int,
        default=_MAX_CHUNKS_TOTAL,
        help=(
            f"Tong so chunk toi da dua vao RAGAS SummaryExtractor. "
            f"Mac dinh {_MAX_CHUNKS_TOTAL} (~10 phut o 30 RPM). "
            "Tang len neu muon testset da dang hon va co nhieu thoi gian."
        ),
    )
    args = p.parse_args()
    main(args.docs_dir, args.out_file, args.n_samples, args.rpm,
         model_path=args.model,
         chunk_tokens=args.chunk_tokens,
         overlap_tokens=args.overlap_tokens,
         max_chunks_per_file=args.max_chunks_per_file,
         max_chunks_total=args.max_chunks_total)
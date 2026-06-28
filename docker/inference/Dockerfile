# Dockerfile mong, ke thua tu image chinh thuc vllm/vllm-openai -- image nay
# da co san CUDA, vLLM, va entrypoint goc ["vllm", "serve"]. Khong tu viet
# server rieng (transformers.pipeline khong ho tro continuous batching, se bi
# nghen khi co nhieu request cung luc trong buoi demo).
#
# HF_MODEL_REPO duoc dien vao luc "docker run" qua bien moi truong, KHONG
# nhung weight vao image -- vLLM tu tai model tu HuggingFace Hub luc khoi
# dong va cache vao $HF_HOME (xem README deploy).
#
# QUAN TRONG: dung ENTRYPOINT dang shell form (khong phai JSON array) de
# $HF_MODEL_REPO duoc shell expand luc container start. Dang JSON array
# (CMD ["vllm", "serve", "${HF_MODEL_REPO}"]) se KHONG expand bien moi
# truong -- vllm se nhan nguyen chuoi literal "${HF_MODEL_REPO}" va loi.
FROM vllm/vllm-openai:latest

ENV HF_MODEL_REPO=""
ENV HF_TOKEN=""
# VLLM_API_KEY duoc vLLM tu doc tu environment (khong can truyen --api-key
# qua CLI). De rong = khong yeu cau auth; set gia tri = bat bearer-token auth
# tren tat ca endpoint /v1/... -- RemoteInferenceClient phai dung dung token
# nay qua REMOTE_INFERENCE_TOKEN trong .env cua orchestrator.
ENV VLLM_API_KEY=""

EXPOSE 8000

ENTRYPOINT vllm serve $HF_MODEL_REPO --host 0.0.0.0 --port 8000

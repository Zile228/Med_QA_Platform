# Thin Dockerfile, inheriting from the official vllm/vllm-openai image --
# this image already has CUDA, vLLM, and the original entrypoint
# ["vllm", "serve"]. Does not write its own server (transformers.pipeline
# does not support continuous batching and would bottleneck under multiple
# concurrent requests during a demo).
#
# HF_MODEL_REPO is filled in at "docker run" time via an environment
# variable, NOT baked into the image as weights -- vLLM downloads the model
# from the HuggingFace Hub itself at startup and caches it in $HF_HOME (see
# the deploy README).
#
# IMPORTANT: uses shell-form ENTRYPOINT (not a JSON array) so that
# $HF_MODEL_REPO gets shell-expanded at container start. The JSON array form
# (CMD ["vllm", "serve", "${HF_MODEL_REPO}"]) would NOT expand the
# environment variable -- vllm would receive the literal string
# "${HF_MODEL_REPO}" and fail.
FROM vllm/vllm-openai:latest

ENV HF_MODEL_REPO=""
ENV HF_TOKEN=""
# VLLM_API_KEY is read directly by vLLM from the environment (no need to
# pass --api-key via CLI). Empty = no auth required; set a value = enables
# bearer-token auth on every /v1/... endpoint -- RemoteInferenceClient must
# use this exact token via REMOTE_INFERENCE_TOKEN in the orchestrator's .env.
ENV VLLM_API_KEY=""

EXPOSE 8000

ENTRYPOINT vllm serve $HF_MODEL_REPO --host 0.0.0.0 --port 8000

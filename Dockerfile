FROM lmsysorg/sglang:v0.5.12.post1-cu129

# Install uv package manager
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# install dependencies
COPY requirements.txt ./
RUN rm -f /usr/lib/python3*/EXTERNALLY-MANAGED /usr/local/lib/python3*/EXTERNALLY-MANAGED 2>/dev/null; uv pip install --system -r requirements.txt

# AOT flashinfer kernels (version-matched to the base's flashinfer-python) so the
# worker skips flashinfer JIT compilation at startup. flashinfer-cubin is arch-independent
# (Ada + Hopper); jit-cache is per-CUDA (cu129). Non-fatal if a matching wheel is absent.
RUN FIV=$(python3 -c "import flashinfer; print(flashinfer.__version__)" 2>/dev/null) && echo "flashinfer-python=$FIV" && \
    ( uv pip install --system "flashinfer-cubin==$FIV" && \
      uv pip install --system "flashinfer-jit-cache==$FIV" --index-url https://flashinfer.ai/whl/cu129 ) \
    || echo "WARN: flashinfer AOT wheels unavailable for $FIV; runtime JIT fallback"

# copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Model download script execution
# Ensure this script uses python3 and handles paths correctly relative to /app if needed
RUN if [ -n "$MODEL_NAME" ]; then python3 download_model.py; fi

CMD ["python3", "handler.py"]

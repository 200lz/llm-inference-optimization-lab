# Baseline model setup

The CPU baseline uses the official **Qwen2.5-0.5B-Instruct GGUF** release. It is a small, public, instruction-tuned model suitable for repeated CPU measurements.

- Source repository: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Exact file: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Quantization: `Q4_K_M`
- Parameters: approximately 0.49 billion
- Artifact size: 491,400,032 bytes (approximately 468.64 MiB)
- SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- Source URL: `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true`

From the repository root, download it with:

```bash
.venv/bin/python scripts/download_model.py \
  Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The downloader streams into a `.part` file, resumes through HTTP ranges when the server supports them, rejects empty or truncated responses, and writes `models/qwen2.5-0.5b-instruct-q4_k_m.gguf.metadata.json`. Verify the downloaded artifact independently with:

```bash
sha256sum models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Compare the output with the `sha256` field in the metadata JSON. Model artifacts are not committed because they are large generated inputs with a stable public source; `models/` is ignored by Git. This also keeps repository history small and makes acquisition and integrity verification explicit and reproducible.

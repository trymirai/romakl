```text
Usage: romakl [OPTIONS]

  Measure how closely quantized students track a Lalamo reference model:
  KL, TV, expected sampling disagreement, top-1 agreement and the size of
  each student's token-generating weights. Runs on Linux with NVIDIA GPUs.

Options:
  --reference PATH   Lalamo reference model path  [required]
  --student TEXT     Student as engine:path; repeat for several  [required]
                     engine is one of
                       lalamo  Lalamo model directory
                       lcpp    GGUF file, with tokenizer.json next to it
                       exl     EXL3 directory containing tokenizer.json
  --dataset TEXT     Encrypted conversation dataset  [default: trymirai/kl-mixture]
                     The default is private: run `hf auth login` with trymirai access first
  --cache-path PATH  Reference rollout cache  [default: teacher-cache]

Examples:
  Compare a Lalamo student with its reference:
    romakl --reference ~/Qwen3.8-27B --student lalamo:~/Qwen3.8-27B-M

  Compare several backends in one run:
    romakl --reference ~/Qwen3.8-27B \
      --student lcpp:~/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf --student exl:~/Qwen3.8-27B-EXL3

  Run through uvx, which ignores the project's llama.cpp build flags, so pass
  them yourself from a shell with nvcc on PATH:
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=ON" \
      uvx --from git+https://github.com/trymirai/romakl romakl \
      --reference ~/Qwen3.8-27B --student lalamo:~/Qwen3.8-27B-M
```

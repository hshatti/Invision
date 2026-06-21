# A Lightweight FLUX2 or ZImage Inference engine – Written in Pure Pascal for (FPC, Lazarus or Delphi)

TL;DR
{#what-is-it}
> A lightweight, pure‑Pascal inference engine for text‑to‑image and image‑to‑image generation (FLUX2 / ZImage). Nativly parses .SafeTensor and .JSON files. Runs out‑of‑the‑box with no external dependencies. If OpenBLAS is present on Windows/Linux it will automatically bind to it and gain ~20× speed‑up. On macOS the native Accelerate framework is used.

## Table of Contents

- [Why Pure Pascal?](#why-pure-pascal)
- [Supported Models](#supported-models)



### Why Pure Pascal?{#why-pure-pascal}

- **Zero Runtime** – The binary can be shipped to any machine without requiring a Pascal compiler or DLLs.
- **Safety & Readability** – Pascal’s strong typing and modularity reduce bugs in tensor manipulation.
- **Cross‑Platform** – Works on Windows, Linux, macOS with minimal changes.

---

## Supported Models {#supported-models}

*Currently only the following weights are shipped:*

| Model | Path | Size |
|-------|------|------|
| FLUX2‑Klein-4B | SafeTensor | ~14 GB |
| ZImage‑V1.0 | `models/Z-Image-turbo.bin` | ~40 GB |

This is an initial README, More descriptions installation guides to come

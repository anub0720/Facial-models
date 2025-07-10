# Facial-models: ComSys Hackathon 2025

This repository contains the unified inference pipeline (`test.py`) for both **Task A** (gender classification) and **Task B** (face identity matching) from the ComSys Hackathon.

---

## 🔧 Setup Instructions

> ✅ All required Python packages are automatically installed by `test.py` if missing.

However, you must install a few tools **before cloning** this repository (especially if running locally).

### 🛠️ Prerequisites

| Tool                | Required Version    | Notes                                                                 |
|---------------------|---------------------|-----------------------------------------------------------------------|
| **Python**          | `>= 3.10`           | Recommended version: **3.11** for compatibility with latest packages |
| **Git LFS**         | `>= 3.x`            | Required to download large checkpoint files from GitHub              |
| **Torch**           | `== 2.6.0+cu124`    | Avoids CUDA/NumPy compatibility issues                               |
| **NumPy**           | `< 2.0`             | NumPy ≥ 2.0 causes PyTorch runtime errors with some packages         |
| **pytorch-lightning** | `== 2.5.1`        | Must match the version used during model training                    |

### 🧱 Install Git LFS

```bash
sudo apt-get install git-lfs
git lfs install

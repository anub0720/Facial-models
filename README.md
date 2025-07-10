# Facial-models: ComSys Hackathon 2025

This repository contains the unified inference pipeline (`test.py`) for both **Task A** (gender classification) and **Task B** (face identity matching) from the ComSys Hackathon.

---

## 🔧 Setup Instructions

> ✅ All required Python packages are automatically installed by `test.py` if missing.

However, you must install a few tools **before cloning** this repository (especially if running locally).

### 🛠️ Prerequisites

| Tool                | Required Version    | Notes                                                                 |
|---------------------|---------------------|-----------------------------------------------------------------------|
| **Git LFS**         | `>= 3.x`            | Required to download large checkpoint files from GitHub              |

Other dependencies, if unavailable, are installed by the code itself.

### 🧱 Install Git LFS

```bash
sudo apt-get install git-lfs
git lfs install
```
## 📦 Folder Structure & Expected Input Format
```
Facial-models/
│
├── test.py # Main pipeline entrypoint for Task A and Task B
├── .gitattributes
├── Task-A/
│ └── Source_Code.ipynb
│ └── Testing_Code.ipynb
│ └── best_model_taskA.ckpt # Lightning checkpoint file for Task A
├── Task-B/
│ └── Source_Code.ipynb
│ └── Testing_Code.ipynb
│ └── best_embedding_model_TaskB.pth # PyTorch model file for Task B
└── README.md
```

### 👤 Task A (Gender Classification)

Expected test dataset format:
```
task-a/
├── male/
│ ├── img1.jpg
│ └── img2.jpg
└── female/
├── img3.jpg
└── img4.jpg
```

Same as validation dataset format


### 😷 Task B (Face Recognition with Distortions)

Expected test dataset format:

```
task-b/
├── 001_frontal/
│ ├── 001_frontal.jpg # Reference image
│ └── distortion/
│ ├── distorted1.jpg # Distorted query images
│ └── distorted2.jpg
├── 002_frontal/
│ ├── 002_frontal.jpg
│ └── distortion/
│ ├── distorted1.jpg
│ └── distorted2.jpg
...
```

Same as validation dataset format

```bash
python test.py
```
## 📝 Notes for Reproducibility

- ✅ The **same models** and **test pipeline** were used to generate the submitted results.  
- ✅ Ensure that all **paths are correct** and that the **test images** are structured as described in the folder structure section.  
- ✅ Face detection model files are **auto-downloaded** during execution unless manually provided in advance.  
- ⚠️ **Important:** All dependencies are handled within the code, but version-specific issues (e.g., **NumPy < 2.0**) are critical for compatibility. Using the wrong version may lead to runtime errors.  

To get the same results as submitted, pass the same dataset in the code.

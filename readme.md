# 🧠 SV-RAG: Semantic Video RAG

**<div align="center">
A Training-Free, Multimodal Video Question-Answering Pipeline**
</div>


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Website](https://img.shields.io/badge/Website-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/)
[![AI Model](https://img.shields.io/badge/Model-BLIP%20%2B%20FlanT5-yellow)](https://huggingface.co/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-success)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/streamlit-logo.png" width="600" alt="Project Banner">
</p> -->

> **"Engineered a training-free Multimodal Video RAG pipeline utilizing unsupervised CLIP-based semantic clustering to achieve ~85% frame redundancy reduction, enabling latency-optimized spatio-temporal insight retrieval on low-resource hardware."**

</div>

---

## 📖 Abstract
Traditional Video QA systems require massive GPU resources and end-to-end training. **SV-RAG** proposes a modular, resource-efficient architecture that enables **semantic video understanding on consumer hardware (CPU/MPS)**.

By leveraging **Unsupervised K-Means Clustering** on CLIP embeddings, the system mathematically identifies key narrative scenes, removing 85%+ of redundant frames. These keyframes are processed by a Vision-Language Model (BLIP) and indexed into a Vector Database (FAISS) for retrieval-augmented generation (RAG).

## ✨ Key Features

* **🎥 Semantic Keyframe Extraction:** Uses `CLIP (ViT-B/32)` + `K-Means` to extract only distinct scenes, ignoring repetitive frames.
* **👁️ Multimodal Insight Generation:** Generates descriptive captions with confidence scores using `BLIP-Large`.
* **🕸️ Spatio-Temporal Knowledge Graph:** Visualizes how objects and concepts connect over time (Node-Link Diagram).
* **💬 Hallucination-Resistant Chat:** Uses `Flan-T5-Large` with strict prompt engineering to answer questions based *only* on visual evidence.
* **⚡ Hardware Optimized:** Runs entirely on **Apple Metal (MPS)** or Standard **CPU** (No NVIDIA GPU required).

---

## ⚙️ System Architecture

The pipeline follows a 4-stage modular approach:

```mermaid
graph LR
    A[Video Input] --> B[Frame Sampling]
    B --> C{Semantic Clustering}
    C -- CLIP Embeddings --> D[Keyframe Selection]
    D -- Top 10% Frames --> E[BLIP Captioning]
    E --> F["Vector Store (FAISS)"]
    G[User Question] --> H[RAG Retrieval]
    F --> H
    H --> I[Flan-T5 LLM]
    I --> J[Final Answer]
```
---
## 🎥 Video Processing Pipeline Overview

### 🎬 **1. Ingestion**
Video is sampled at **1 FPS**, grabbing clean, lightweight snapshots of the timeline.

### 🧩 **2. Clustering (Our Novel Trick)**
Frames are embedded into a vector space →  
**K-Means picks the most representative frames** (centroids) →  
Instant video summary without losing the story.

### 👁️ **3. Vision Insights**
Each keyframe is captioned to generate rich **visual descriptions**  
(objects, actions, scenes, context).

### 🔎🤖 **4. RAG Intelligence**
Insights are indexed. User queries trigger a semantic search to retrieve relevant timestamps before generating an answer.

---

## 🚀 Installation

### 🔧 Prerequisites
- **Python 3.9+**
- **FFmpeg**  
  - macOS: `brew install ffmpeg`  
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: [Follow this](https://www.geeksforgeeks.org/installation-guide/how-to-install-ffmpeg-on-windows/)

### 📦 Setup

#### **1. Clone the Repository**
```bash
git clone https://github.com/M-SaiCharan/Semantic-Video-RAG
cd Semantic-Video-RAG
```

#### **2. Create Virtual Environment***
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **3. Install Dependencies***
```bash
pip install -r requirements.txt
```
---
## 🏃 Usage
1. Run the Streamlit App
```bash
streamlit run app.py
```
2. **Open Browser:** Go to http://localhost:8501.
3. **Upload Video:** Drag and drop an MP4 file.
4. **Configure:** Adjust K-Clusters slider in the sidebar to change sensitivity.
5. **Chat:** Ask questions like "What color is the car?" or "Describe the sequence of events."
---

## 📊 Performance Metrics

Comparison of **uniform 1 FPS sampling** vs **SV-RAG cluster-based sampling**:

| **Metric**            | **Uniform Sampling (1 FPS)** | **SV-RAG (Cluster-Based)** | **Improvement**        |
|----------------------|------------------------------|----------------------------|-------------------------|
| Frames Processed     | 62 frames                    | 10 frames                  | **83.8% Reduction**     |
| Processing Time      | ~45 seconds                  | ~8 seconds                 | **5× Faster**           |
| Info Retention       | High Redundancy              | High Entropy (Unique Info) | —                       |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom CSS / Glassmorphism)  
- **Orchestration:** LangChain  
- **Vision Model:** Salesforce BLIP  
- **LLM:** Google Flan-T5-Large  
- **Embeddings:** OpenAI CLIP (Vision), MiniLM (Text)  
- **Vector DB:** FAISS  
- **Graph Viz:** Streamlit-Agraph
---
## 📜 License  
Distributed under the **MIT License**.  
See the `LICENSE` file for more information.





























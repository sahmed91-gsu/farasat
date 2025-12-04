---
title: Farasat Clinical AI
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🧬 Farasat: Forensic Clinical AI Auditor

**Farasat** ("Insight" or "Physiognomy") is a Multi-Agent Retrieval-Augmented Generation (RAG) system designed to predict patient retention in substance use treatment programs.

Unlike black-box models that output a simple probability, Farasat acts as a **Forensic Auditor**, retrieving historical precedents, analyzing risk factors against clinical guidelines, and producing a legally defensible "Length-of-Stay" (LOS) forecast.

---

## 🏗️ System Architecture

The system orchestrates four specialized AI Agents in a vertical reasoning pipeline:

1.  **📥 The Parser Agent:**
    *   Converts unstructured natural language clinical notes into strict ICD/SAMHSA codes.
    *   Handles messy inputs (e.g., *"Patient is hooked on crystal meth"*) and maps them to standardized integers (e.g., `SUB1: 10`).

2.  **🔀 The Routing Agent:**
    *   Evaluates the parsed profile against clinical protocols.
    *   Dynamically assigns the case to a **Specialist Persona** (e.g., *Stimulant Expert*, *Compliance Officer*, *Opioid Specialist*).

3.  **🔍 The Retrieval Engine (RAG):**
    *   Encodes the patient profile using `BAAI/bge-large-en-v1.5`.
    *   Queries a **202,000-vector ChromaDB** knowledge base to find the 5 most semantically similar historical cases.

4.  **🧠 The Forensic Auditor (Synthesis):**
    *   Synthesizes three sources of truth:
        1.  **Statistical Baselines:** Real-time probability calculation from the training set.
        2.  **Historical Precedent:** Outcomes of the retrieved similar cases.
        3.  **Long-Term Memory:** Corrections learned from previous user feedback.
    *   Generates a balanced argument ("Case for Short Stay" vs "Case for Long Stay").

5.  **⚖️ The Resource Manager (Decision):**
    *   Makes the final allocation decision (`Short` vs `Long`) based on maximizing facility resource efficiency.

---

## 🧪 Performance Metrics

Evaluated on a held-out test set of **10,000 records**:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Balanced Accuracy** | **63.3%** | Significant lift over random chance (50%) in high-entropy human behavior prediction. |
| **Retrieval Concordance** | **69.5%** | In ~70% of cases, the vector database retrieves neighbors that accurately predict the outcome. |
| **Semantic Relevance** | **0.98** | High cosine similarity confirms the embedding model effectively maps clinical narratives. |

---

## 🚀 Deployment Stack

*   **Frontend:** HTML5, TailwindCSS, Glassmorphism UI.
*   **Backend:** Flask (Python 3.10).
*   **Database:** ChromaDB (Disk-based persistence).
*   **AI Engine:** Fireworks AI (Llama-3-405b / GPT-OSS-120b).
*   **Memory:** JSONL-based active learning loop.

---

### 👨‍💻 Author
**Shehzad Ahmed**  
# AI Resume Analyzer and Career Assistant using NLP and Generative AI

⚡ **NVIDIA AI Internship Capstone Project Submission** ⚡  
An end-to-end AI-powered system that evaluates candidate resumes against target job descriptions contextually and provides intelligent career development and interview preparation insights.

---

## 📖 Project Overview
Traditional screening software filters candidates using primitive, rigid keyword scans (lexical matching). This system overcomes these limitations by integrating **semantic understanding** and **generative AI recommendations** into a streamlined, high-performance pipeline. 

By mapping document texts into high-dimensional vector spaces using **Sentence Transformers** (`all-MiniLM-L6-v2`) and generating actionable growth plans using **FLAN-T5**, the system provides candidates with deep alignment scores, skill gap analyses, resume phrasing optimizations, and customized interview preparation materials. The entire pipeline supports GPU acceleration via **NVIDIA CUDA**.

---

## 📐 System Architecture
The application flow separates presentation, parsing, matching, diagnostics, and text generation logic:

```
[Resume PDF] ──> PDF Reader ──> Raw Text ─┐
                                          ├──> [Sentence Transformer (GPU)] ──> Similarity Score (0-100%)
[Job Desc]   ──────────────────> Raw Text ─┘
                                          │
                                          ├──> [Skills Taxonomy Parser] ─────> Matching & Missing Skills
                                          │
                                          └──> [Generative LLM (FLAN-T5)] ───> Improvement Tips & Interview Prep
```

*See the high-resolution architecture diagram at [docs/architecture.png](docs/architecture.png).*

---

## ✨ Features
1. **Resume Text Extraction**: Automatically extracts text from uploaded PDF resumes using a dual-parser engine (`pdfplumber` + `pypdf` fallback).
2. **Semantic Similarity Scoring**: Computes compatibility percentages using dense vector embeddings on NVIDIA GPUs rather than basic keywords.
3. **Skill Gap Diagnostics**: Maps resume profiles to an industry-standard skill taxonomy, classifying them into matching, missing, and auxiliary skill sets.
4. **Generative Career Mentoring**: Leverages FLAN-T5 to compile:
   * Actionable resume improvement tips.
   * ATS-friendly, metrics-focused professional wording upgrades.
5. **Interactive Interview Kit**: Generates tailored Technical, HR/Behavioral, and Project Architecture practice questions.
6. **Real-time Hardware Telemetry**: Interactive Streamlit sidebar showing GPU usage, device index, and active VRAM metrics.

---

## 🛠️ Technologies Used
* **Programming Language**: Python
* **Web UI Framework**: Streamlit
* **Deep Learning Framework**: PyTorch (with NVIDIA CUDA support)
* **NLP & Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Generative LLM**: Hugging Face Transformers (`google/flan-t5-base`)
* **PDF Processing**: `pdfplumber` & `pypdf`
* **Visualization & Reporting**: `reportlab`, `matplotlib`, `pandas`

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
Ensure you have **Python 3.9+** and a CUDA-capable GPU (optional but recommended for faster model inference).

### 2. Clone and Setup
```bash
# Clone the repository (or navigate to directory)
cd AI-Resume-Analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Data
Run the built-in generator script to create sample resume PDFs and test data:
```bash
python generate_samples.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
This will open the dashboard in your local browser at `http://localhost:8501`.

---

## 📓 Running on Google Colab (GPU Accelerated)
We provide a fully self-contained notebook to run this project in the cloud with free T4 GPU access:
1. Open Google Colab and upload the [AI_Resume_Analyzer.ipynb](AI_Resume_Analyzer.ipynb) file.
2. Select **Runtime > Change runtime type** and choose **T4 GPU** as the accelerator.
3. Click **Runtime > Run all** to run dependencies, check GPU status, compute embedding similarities, run skill analysis, and launch the Streamlit interface using `localtunnel`.
4. Click the generated tunnel link to open the active UI in a new tab.

---

## 🔮 Future Enhancements
* **Model Quantization (TensorRT)**: Quantize model pipelines to 8-bit precision (INT8) and optimize using NVIDIA TensorRT for sub-millisecond execution speeds.
* **OCR Support**: Incorporate OCR pipelines (e.g. Tesseract) to parse scanned/image-based resume uploads.
* **Direct PDF Modification**: Allow candidates to download updated versions of their resumes directly from the application after recommendations are applied.

# YieldBoost: Agentic Precision Agriculture Platform

YieldBoost is a hybrid, two-stage artificial intelligence ecosystem that bridges edge-level deep learning and classic machine learning with a Generative AI multi-agent orchestration framework. The platform ingests real-time environmental data, tabular soil chemistry metrics, and crop leaf tensors to synthesize localized, conflict-free biosecurity prescriptions for optimal crop management.

## 🚀 System Architecture

The platform operates via a tiered inference pipeline designed to minimize LLM token overhead while ensuring semantic contextual awareness:

1. **The Edge/Inference Layer (Local ML/DL):** High-speed classification engines process raw inputs. A Scikit-Learn `RandomForestClassifier` handles multi-variable tabular soil metrics to recommend optimal crops. Simultaneously, a custom PyTorch `ResNet9` Convolutional Neural Network analyzes leaf image tensors for automated plant pathology diagnosis.
2. **The Cognitive Orchestration Layer (GenAI Multi-Agent System):** A decentralized multi-agent network powered by the **Groq API (Llama 3)** asynchronously ingests local model predictions along with geo-localized live weather data via the **OpenWeatherMap REST API**. Specialized domain agents (Soil Chemist, Plant Pathologist, Meteorological Analyst) pass sub-reports to a **Coordinator Jury Agent**, which programmatically resolves domain conflicts (e.g., assessing if a soil-enriching nitrogen fertilizer would accelerate an active fungal outbreak) before streaming a safe, unified action plan.

---

## 📊 Model Training & Hardware Caveat

* **Crop Recommendation Engine:** Trained successfully using Scikit-Learn's Random Forest architecture on multi-variable environmental features ($N, P, K, \text{pH, temperature, humidity, rainfall}$), yielding stable production-ready inference.
* **Plant Pathology Vision Engine:** The vision system utilizes a compressed **ResNet9** architecture built from scratch in PyTorch to output raw logits across 20 distinct disease classes. 

> ⚠️ **Technical Note on Vision Accuracy:** Due to local system hardware limitations and computational overhead during training cycles, the CNN was trained on a highly restricted subset of the image corpus. Consequently, while the structural routing and edge inference pipeline function perfectly, the classification accuracy is currently altered. Performance will scale directly when trained over the full dataset on high-compute clusters.

---

## 🛠️ Tech Stack & Microservice Architecture

* **Core Backend & MLOps:** Python 3.12, Flask, Docker
* **Deep Learning & ML:** PyTorch, Torchvision, Scikit-Learn, Pandas, NumPy
* **Generative AI Framework:** Groq API (Llama 3 Orchestration), Semantic Reasoning
* **Data & Tooling:** REST API Integrations (OpenWeatherMap), Markdown-to-HTML Parsing Engines

---


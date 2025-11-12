# InventoryStandardization

Step 1: Create a Gemini api key from the this website
Step 2: Add the generated key to the .env file GEMINI_API_KEY = ""
Step 3: open terminal go agents path and create a virtual venv and activate it 
Step 4: run the command below
pip install -r requirements.txt
Step 5: run the command below to start the agent
python inputProcessing.py file then you could see the outputs in the data folder


# 🧠 InventoryStandardization

A modular AI-powered pipeline designed to **standardize, clean, and generate structured product data (SKUs, confidence scores, mappings)** from raw multi-category inventory files such as Jewelry, Textiles, Electronics, and more.

---

## 📦 Overview

The `InventoryStandardization` project uses a series of AI and rule-based **agents** to automate inventory data transformation and SKU generation.

The goal is to:

* Process messy raw inventory data files (with 60+ columns)
* Extract and normalize only the relevant fields
* Generate standardized **SKUs**
* Compute **confidence scores**
* Route low-confidence records to **HITL (Human-in-the-Loop)** review

---

## ⚙️ Architecture Overview

The complete pipeline includes the following agents:

| Agent                                   | Purpose                                                                              | Type              |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| 🧹 **InputProcessingAgent**             | Cleans & extracts necessary fields from raw CSVs (e.g. Brand, Category, Description) | AI agent   |
| 🧠 **ConfidenceSKUAgent**               | Generates standardized SKUs & assigns confidence scores (row + column level)         | AI-assisted       |
| 🔍 **MappingEngineAgent**               | Classifies data into known internal schemas & identifies missing/ambiguous fields    | ML Classifier     |
| 👤 **HITL Review Agent**                | Routes low-confidence mappings (< threshold) to a review interface                   | Human-in-the-Loop |
| 📊 **StatisticsAgent** *(optional)* | Aggregates logs, generates audit reports for QA                                      | AI            |

---

## 🪄 Features

✅ Automatic SKU generation from cleaned input
✅ Confidence scoring for both row-level & attribute-level data
✅ Multi-domain support (Jewelry, Textiles, Electronics, etc.)
✅ AI-assisted data enrichment and normalization
✅ Human-in-the-loop fallback for uncertain predictions
✅ Exports final standardized datasets to `.csv` and `.parquet`

---

## 🧰 Tech Stack

* **Python 3.10+**
* **Pandas** for data handling
* **FastAPI (optional)** for service orchestration
* **Google Gemini API** for AI-powered text interpretation
* **dotenv** for key management
* **scikit-learn** (planned) for classification and mapping engine

---

## 🧩 Setup Instructions

### 1️⃣ Generate a Gemini API Key

Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
Copy your key and keep it secure.

---

### 2️⃣ Add API Key to `.env`

Create a `.env` file in the project root and paste:

```bash
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

### 3️⃣ Create and Activate a Virtual Environment

In the project root:

```bash
cd agents
python -m venv venv
# Activate venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Run the Input Processing Agent

```bash
python inputProcessing.py
```

This step:

* Reads raw data from `data/raw/`
* Cleans and extracts essential fields
* Saves the intermediate output to `data/processed/`

---

### 6️⃣ Run the Confidence + SKU Agent

```bash
python confidence_sku_agent.py
```

This step:

* Loads the processed file from `data/processed/`
* Generates standardized **SKUs**
* Calculates **confidence scores**
* Flags low-confidence rows for review
* Saves final results to:

```
data/output/standardized_inventory.csv
data/output/standardized_inventory.parquet
```

---

## 📈 Sample Output

| SKU               | Brand     | Category | Description        | Confidence_Score | Review_Flag |
| ----------------- | --------- | -------- | ------------------ | ---------------- | ----------- |
| JW-BR-GLD-NEC-001 | Bluestone | Jewelry  | Gold Necklace 18K  | 0.94             | No          |
| TX-FB-COT-SHR-022 | FabIndia  | Textile  | Cotton Shirt Men’s | 0.87             | Yes         |

---

## 🧠 Recommended Development Order

1️⃣ `inputProcessing.py` → Data extraction and normalization
2️⃣ `confidence_sku_agent.py` → SKU generation + confidence scoring
3️⃣ `mapping_engine.py` → ML-based attribute classifier
4️⃣ `hitl_agent.py` → HITL flag routing + review dashboard
5️⃣ `audit_reporting.py` *(optional)* → Logging and QA reports

---

## 🧾 Example Folder Structure

InventoryStandardization/
│
├── agents/
│   ├── inputProcessing.py
│   ├── confidence_sku_agent.py
│   ├── mapping_engine.py
│   ├── hitl_agent.py
│   └── audit_reporting.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── .env
├── requirements.txt
└── README.md

import os
import re
import json
import ast
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import google.generativeai as genai
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables (expects GEMINI_API_KEY in .env)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# =============================================================
#              AI-POWERED INPUT PROCESSING AGENT
# =============================================================
class InputProcessingAgent:
    """
    AI-powered Input Processing Agent.
    Features:
      - Load Excel/CSV
      - Gemini-based schema mapping
      - Value normalization (text cleaning)
      - Numeric & text cleanup
      - Attribute extraction (color, material, volume, quantity)
      - Category standardization
      - Metadata enrichment (row_id, timestamp, source)
      - Save output (Excel + Parquet)
    """

    def __init__(self, target_schema: Dict[str, str], verbose: bool = True):
        self.target_schema = target_schema
        self.verbose = verbose
        self.schema_model = genai.GenerativeModel("gemini-2.5-flash")
        self.normalize_model = genai.GenerativeModel("gemini-2.5-flash")
        self.df_cleaned = None
        self.cleaning_stats = {
            "rows_original": 0,
            "rows_after_cleaning": 0,
            "duplicates_removed": 0,
            "invalid_rows_removed": 0
        }

    # ---------------------------------------------------------
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV or Excel into a DataFrame."""
        try:
            if file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            self.cleaning_stats["rows_original"] = len(df)
            if self.verbose:
                print(f"✅ Loaded file: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            raise RuntimeError(f"❌ Error loading file: {str(e)}")

    # ---------------------------------------------------------
    def _safe_parse_json(self, text: str) -> dict:
        """Safely parse JSON or Python-like dict from Gemini."""
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except Exception:
                raise ValueError(f"Failed to parse Gemini response:\n{text}")

    # ---------------------------------------------------------
    def ai_schema_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer schema mapping using Gemini AI."""
        prompt = f"""
        You are a data normalization agent.
        Target schema: {self.target_schema}.
        Raw columns: {list(df.columns)}.
        Map each raw column to the most semantically similar target schema field.
        Respond strictly as JSON.
        """
        response = self.schema_model.generate_content(prompt)
        mapping = self._safe_parse_json(response.text)
        if self.verbose:
            print("🧠 Gemini Schema Mapping:")
            print(json.dumps(mapping, indent=2))
        return mapping

    # ---------------------------------------------------------
    def rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        df = df.rename(columns=mapping)
        df = df.loc[:, df.columns.notna()]
        if self.verbose:
            print("✅ Columns renamed and cleaned")
        return df

    # ---------------------------------------------------------
    def ai_normalize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text values using Gemini AI."""
        text_cols = list(df.select_dtypes(include=["object"]).columns)
        for col in text_cols:
            sample_values = df[col].dropna().unique().tolist()[:20]
            if not sample_values: 
                continue
            prompt = f"""
            Normalize values for column '{col}': {sample_values}
            Standardize casing, spelling, and synonyms.
            Respond only as JSON mapping of original → cleaned.
            """
            try:
                response = self.normalize_model.generate_content(prompt)
                mapping = self._safe_parse_json(response.text)
                df[col] = df[col].replace(mapping)
                if self.verbose:
                    print(f"🧹 Normalized '{col}': {len(mapping)} replacements")
            except Exception:
                if self.verbose:
                    print(f"⚠️ Skipped normalization for '{col}'")
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    # ---------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic text & numeric cleanup."""
        if self.verbose: print("🧼 Cleaning text and numeric data...")
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None})
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = df[col].fillna(0)
        if self.verbose: print("✅ Data cleaned")
        return df

    # ---------------------------------------------------------
    def extract_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume, color, material, quantity from description."""
        if "description" not in df.columns:
            df["description"] = ""
        
        def extract_volume(text):
            text = str(text).upper()
            ml = re.search(r"(\d+(?:\.\d+)?)\s*ML", text)
            l = re.search(r"(\d+(?:\.\d+)?)\s*L\b", text)
            return f"{ml.group(1)}ML" if ml else f"{l.group(1)}L" if l else ""
        
        COLORS = ['clear','white','black','blue','red','green','yellow','grey','gray','silver','gold','pink','purple','brown','orange']
        MATERIALS = ['glass','crystal','ceramic','porcelain','stoneware','steel','silver','gold','plastic','wood','metal','copper','brass','aluminum']
        
        def extract_color(text):
            text = str(text).lower()
            for c in COLORS: 
                if c in text: return c
            return ""
        
        def extract_material(text):
            text = str(text).lower()
            for m in MATERIALS: 
                if m in text: return m
            return ""
        
        def extract_quantity(text):
            text = str(text).upper()
            match = re.search(r"(?:SET\s*OF\s*|X)(\d+)|(\d+)\s*(?:PCS|PIECES)", text)
            return match.group(1) or match.group(2) if match else "1"
        
        df["extracted_volume"] = df["description"].apply(extract_volume)
        df["extracted_color"] = df["description"].apply(extract_color)
        df["extracted_material"] = df["description"].apply(extract_material)
        df["extracted_quantity"] = df["description"].apply(extract_quantity)
        
        return df

    # ---------------------------------------------------------
    def remove_duplicates_and_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and rows with missing critical info."""
        initial = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial - len(df)

        mandatory = ["supplier_name"]
        for col in mandatory:
            if col in df.columns:
                df = df[df[col].notna() & (df[col] != "")]
        invalid_removed = initial - duplicates_removed - len(df)

        self.cleaning_stats["duplicates_removed"] = duplicates_removed
        self.cleaning_stats["invalid_rows_removed"] = invalid_removed
        return df

    # ---------------------------------------------------------
    def add_metadata(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        df["processing_timestamp"] = datetime.now().isoformat()
        df["row_id"] = range(len(df))
        df["data_source"] = os.path.basename(file_path)
        df["processing_version"] = "2.0"
        return df

    # ---------------------------------------------------------
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional: normalize category names."""
        if "category" in df.columns:
            mapping = {
                "jewellery": "Jewellery",
                "jewelry": "Jewellery",
                "electronics": "Electronics",
                "electronic": "Electronics",
                "home decor": "Home Decor",
                "homedecor": "Home Decor",
                "textiles": "Textiles"
            }
            df["category"] = df["category"].map(lambda x: mapping.get(str(x).lower(), x) if pd.notna(x) else "")
        return df

    # ---------------------------------------------------------
    def save_data(self, df: pd.DataFrame, output_dir: str = "data/processed") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = os.path.join(output_dir, f"processed_{timestamp}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)
        excel_path = os.path.join(output_dir, f"processed_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        if self.verbose:
            print(f"💾 Data saved → {parquet_path} & {excel_path}")
        return parquet_path

    # ---------------------------------------------------------
    def process(self, file_path: str) -> Dict[str, Any]:
        """Full pipeline execution."""
        df = self.load_file(file_path)
        mapping = self.ai_schema_mapping(df)
        df = self.rename_columns(df, mapping)
        df = self.ai_normalize_values(df)
        df = self.clean_data(df)
        df = self.extract_attributes(df)
        df = self.remove_duplicates_and_invalid(df)
        df = self.standardize_categories(df)
        df = self.add_metadata(df, file_path)
        self.df_cleaned = df
        output_path = self.save_data(df)
        return {
            "mapped_schema": mapping,
            "cleaned_dataframe": df,
            "output_path": output_path,
            "ai_notes": ["Schema mapping + normalization + attribute extraction completed using Gemini AI."]
        }

# =============================================================
#                    MAIN EXECUTION
# =============================================================
if __name__ == "__main__":
    target_schema = {
        "product_name": "Name of the product",
        "category": "Type or category of the item",
        "material": "Material composition",
        "price": "Price in USD",
        "brand": "Product brand name",
        "color": "Color variant",
        "description": "Detailed product description",
    }

    agent = InputProcessingAgent(target_schema)
    result = agent.process("data/raw/06.10.2025.xlsx")

    print(f"\n✅ File processed successfully → {result['output_path']}")


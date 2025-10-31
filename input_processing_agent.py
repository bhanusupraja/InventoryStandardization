# import pandas as pd
# import numpy as np
# import os
# import re
# import pyarrow.parquet as pq
# import pyarrow as pa
# from datetime import datetime
# from tqdm import tqdm

# # ==============================================================
# # 🧩 Input Processing Agent
# # ==============================================================

# class InputProcessingAgent:
#     def __init__(self, input_path: str, output_dir: str = "data/processed"):
#         self.input_path = input_path
#         self.output_dir = output_dir
#         self.df_raw = None
#         self.df_processed = None
#         os.makedirs(self.output_dir, exist_ok=True)

#     # ----------------------------------------------------------
#     # STEP 1: Load file (CSV/XLSX)
#     # ----------------------------------------------------------
#     def load_file(self):
#         print(f"📂 Loading file: {self.input_path}")
#         if self.input_path.endswith(".csv"):
#             self.df_raw = pd.read_csv(self.input_path)
#         elif self.input_path.endswith(".xlsx"):
#             self.df_raw = pd.read_excel(self.input_path, sheet_name=0)
#         else:
#             raise ValueError("Unsupported file type! Must be .csv or .xlsx")

#         print(f"✅ Loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns.")
#         return self.df_raw

#     # ----------------------------------------------------------
#     # STEP 2: Detect and map schema
#     # ----------------------------------------------------------
#     def map_schema(self):
#         colmap = {
#             'CODE': 'product_id',
#             'DESCRIPTION1': 'description1',
#             'DESCRIPTION2': 'description2',
#             'BRAND': 'brand',
#             'CATEGORY': 'category',
#             'Supplier Name': 'supplier_name',
#             'Supplier Account': 'supplier_account',
#             'BARCODE': 'barcode',
#             'QTY IN STOCK': 'stock_qty',
#             'RETAIL': 'price'
#         }

#         # Normalize column names
#         df = self.df_raw.rename(columns=lambda x: x.strip())

#         # Map known columns
#         mapped_cols = {k: v for k, v in colmap.items() if k in df.columns}
#         df = df.rename(columns=mapped_cols)

#         # Combine description fields
#         if 'description1' in df.columns and 'description2' in df.columns:
#             df['description'] = df['description1'].astype(str) + " " + df['description2'].astype(str)
#         elif 'description1' in df.columns:
#             df['description'] = df['description1']
#         elif 'description2' in df.columns:
#             df['description'] = df['description2']

#         # Drop intermediate columns
#         df.drop(columns=[c for c in ['description1', 'description2'] if c in df.columns], inplace=True)
#         self.df_processed = df
#         return df

#     # ----------------------------------------------------------
#     # STEP 3: Text normalization
#     # ----------------------------------------------------------
#     def normalize_text_fields(self):
#         text_fields = ['description', 'brand', 'category', 'supplier_name']

#         for field in text_fields:
#             if field in self.df_processed.columns:
#                 self.df_processed[field] = (
#                     self.df_processed[field]
#                     .astype(str)
#                     .str.lower()                        # convert to lowercase
#                     .str.strip()                         # trim spaces
#                     .replace(r'\s+', ' ', regex=True)    # normalize whitespace
#                 )
#         return self.df_processed

#     # ----------------------------------------------------------
#     # STEP 4: Save processed data
#     # ----------------------------------------------------------
#     def save_parquet(self):
#         date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_path = os.path.join(self.output_dir, f"processed_{date_str}.parquet")

#         table = pa.Table.from_pandas(self.df_processed)
#         pq.write_table(table, out_path)

#         print(f"💾 Saved processed data → {out_path}")
#         return out_path

#     # ----------------------------------------------------------
#     # STEP 5: Generate summary
#     # ----------------------------------------------------------
#     def generate_summary(self):
#         summary = {
#             "file_name": os.path.basename(self.input_path),
#             "rows": len(self.df_processed),
#             "columns": len(self.df_processed.columns),
#             "missing_values": int(self.df_processed.isnull().sum().sum()),
#             "created_at": datetime.now().isoformat()
#         }
#         print("📊 Summary:", summary)
#         return summary

#     # ----------------------------------------------------------
#     # RUN ALL STEPS
#     # ----------------------------------------------------------
#     def run(self):
#         self.load_file()
#         self.map_schema()
#         self.normalize_text_fields()
#         parquet_path = self.save_parquet()
#         summary = self.generate_summary()
#         return parquet_path, summary


# # ==============================================================
# # 🧠 Run as script
# # ==============================================================
# if __name__ == "__main__":
#     agent = InputProcessingAgent(input_path="data/raw/06.10.2025.xlsx")
#     parquet_path, summary = agent.run()
#     print("\n✅ Processing complete!")


import pandas as pd
import numpy as np
import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from tqdm import tqdm

# ==============================================================
# 🧩 Input Processing / Data Cleaning Agent
# ==============================================================

class DataCleaningAgent:
    def __init__(self, input_path: str, output_dir: str = "data/processed"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df_raw = None
        self.df_cleaned = None
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # STEP 1: Load file (CSV/XLSX)
    # ----------------------------------------------------------
    def load_file(self):
        print(f"📂 Loading file: {self.input_path}")
        if self.input_path.endswith(".csv"):
            self.df_raw = pd.read_csv(self.input_path)
        elif self.input_path.endswith(".xlsx"):
            self.df_raw = pd.read_excel(self.input_path, sheet_name=0)
        else:
            raise ValueError("Unsupported file type! Must be .csv or .xlsx")

        print(f"✅ Loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns.")
        return self.df_raw

    # ----------------------------------------------------------
    # STEP 2: Map schema & standardize column names
    # ----------------------------------------------------------
    def map_schema(self):
        colmap = {
            'CODE': 'product_id',
            'DESCRIPTION1': 'description1',
            'DESCRIPTION2': 'description2',
            'BRAND': 'brand',
            'CATEGORY': 'category',
            'Supplier Name': 'supplier_name',
            'Supplier Account': 'supplier_account',
            'BARCODE': 'barcode',
            'QTY IN STOCK': 'stock_qty',
            'RETAIL': 'price'
        }

        # Normalize column names: strip whitespace and lower-case
        df = self.df_raw.rename(columns=lambda x: x.strip())

        # Map known columns
        mapped_cols = {k: v for k, v in colmap.items() if k in df.columns}
        df = df.rename(columns=mapped_cols)

        # Combine description fields
        if 'description1' in df.columns and 'description2' in df.columns:
            df['description'] = df['description1'].astype(str) + " " + df['description2'].astype(str)
        elif 'description1' in df.columns:
            df['description'] = df['description1']
        elif 'description2' in df.columns:
            df['description'] = df['description2']

        # Drop intermediate columns
        df.drop(columns=[c for c in ['description1', 'description2'] if c in df.columns], inplace=True)

        self.df_cleaned = df
        return df

    # ----------------------------------------------------------
    # STEP 3: Normalize text fields
    # ----------------------------------------------------------
    def normalize_text_fields(self):
        text_fields = ['description', 'brand', 'category', 'supplier_name']

        for field in text_fields:
            if field in self.df_cleaned.columns:
                self.df_cleaned[field] = (
                    self.df_cleaned[field]
                    .astype(str)
                    .str.lower()                        # lowercase
                    .str.strip()                         # trim spaces
                    .replace(r'\s+', ' ', regex=True)    # normalize whitespace
                )
        return self.df_cleaned

    # ----------------------------------------------------------
    # STEP 4: Clean numeric fields
    # ----------------------------------------------------------
    def clean_numeric_fields(self):
        numeric_fields = ['stock_qty', 'price', 'AVERAGECOST', 'LAST COST']
        for field in numeric_fields:
            if field in self.df_cleaned.columns:
                self.df_cleaned[field] = (
                    pd.to_numeric(self.df_cleaned[field], errors='coerce')  # convert to numeric
                )
        return self.df_cleaned

    # ----------------------------------------------------------
    # STEP 5: Remove duplicates and missing critical data
    # ----------------------------------------------------------
    def remove_duplicates_and_invalid(self):
        # Drop exact duplicate rows
        self.df_cleaned.drop_duplicates(inplace=True)

        # Remove rows with missing mandatory fields
        mandatory_fields = ['product_id', 'supplier_name', 'category']
        self.df_cleaned.dropna(subset=mandatory_fields, inplace=True)
        return self.df_cleaned

    # ----------------------------------------------------------
    # STEP 6: Fix common spelling errors / standardization
    # ----------------------------------------------------------
    def standardize_categories(self):
        if 'category' in self.df_cleaned.columns:
            # Example mapping, can expand based on dataset
            cat_map = {
                'home decor': 'home decor',
                'jewellery': 'jewellery',
                'textiles': 'textiles',
                'branded watches': 'branded watch',
                'electronics': 'electronics'
            }
            self.df_cleaned['category'] = self.df_cleaned['category'].map(lambda x: cat_map.get(x, x))
        return self.df_cleaned

    # ----------------------------------------------------------
    # STEP 7: Save cleaned dataframe to parquet
    # ----------------------------------------------------------
    def save_parquet(self):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f"processed_{date_str}.parquet")
        table = pa.Table.from_pandas(self.df_cleaned)
        pq.write_table(table, out_path)
        print(f"💾 Saved cleaned data → {out_path}")
        return out_path

    # ----------------------------------------------------------
    # STEP 8: Generate summary
    # ----------------------------------------------------------
    def generate_summary(self):
        summary = {
            "file_name": os.path.basename(self.input_path),
            "rows": len(self.df_cleaned),
            "columns": len(self.df_cleaned.columns),
            "missing_values": int(self.df_cleaned.isnull().sum().sum()),
            "duplicates_removed": int(self.df_raw.shape[0] - self.df_cleaned.shape[0]),
            "created_at": datetime.now().isoformat()
        }
        print("📊 Summary:", summary)
        return summary

    # ----------------------------------------------------------
    # RUN ALL STEPS
    # ----------------------------------------------------------
    def run(self):
        self.load_file()
        self.map_schema()
        self.normalize_text_fields()
        self.clean_numeric_fields()
        self.remove_duplicates_and_invalid()
        self.standardize_categories()
        parquet_path = self.save_parquet()
        summary = self.generate_summary()
        return parquet_path, summary

# ==============================================================
# 🧠 Run as script
# ==============================================================
if __name__ == "__main__":
    agent = DataCleaningAgent(input_path="../data/raw/06.10.2025.xlsx")
    parquet_path, summary = agent.run()
    print("\n✅ Data cleaning complete!")

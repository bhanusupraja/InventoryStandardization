# import pandas as pd
# import os
# import json

# def load_parquet(file_path: str) -> pd.DataFrame:
#     """Loads cleaned data from parquet file."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"❌ File not found: {file_path}")
#     print(f"📂 Loading cleaned data from: {file_path}")
#     df = pd.read_parquet(file_path)
#     print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
#     return df


# def generate_statistics(df: pd.DataFrame) -> dict:
#     """Generates dataset statistics."""
#     print("📊 Generating statistics...")

#     stats = {
#         "total_rows": len(df),
#         "total_columns": len(df.columns),
#         "unique_brands": df['brand'].nunique() if 'brand' in df.columns else None,
#         "unique_stores": df['store_name'].nunique() if 'store_name' in df.columns else None,
#         "unique_categories": df['category'].nunique() if 'category' in df.columns else None,
#         "missing_values": df.isnull().sum().to_dict(),
#         "top_10_stores_by_products": (
#             df['store_name'].value_counts().head(10).to_dict()
#             if 'store_name' in df.columns else {}
#         ),
#         "top_10_categories": (
#             df['category'].value_counts().head(10).to_dict()
#             if 'category' in df.columns else {}
#         ),
#     }

#     print("✅ Statistics generated successfully.")
#     return stats


# def save_statistics(stats: dict, output_dir: str, file_name: str):
#     """Saves statistics as CSV and JSON."""
#     os.makedirs(output_dir, exist_ok=True)

#     csv_path = os.path.join(output_dir, f"{file_name}.csv")
#     json_path = os.path.join(output_dir, f"{file_name}.json")

#     # Save CSV
#     pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]).to_csv(csv_path, index=False)
#     # Save JSON
#     with open(json_path, "w") as f:
#         json.dump(stats, f, indent=4)

#     print(f"📁 Reports saved:\n- {csv_path}\n- {json_path}")


# if __name__ == "__main__":
#     # 👇 Update the path based on your last parquet output
#     parquet_file = "..\data\processed\processed_20251030_214739.parquet"
#     report_dir = "data/reports"

#     # Pipeline execution
#     df_clean = load_parquet(parquet_file)
#     stats = generate_statistics(df_clean)
#     save_statistics(stats, report_dir, "stats_06.10.2025")


# import os
# import json
# import pandas as pd
# from datetime import datetime

# # -----------------------------------------------------------
# # CONFIGURATION
# # -----------------------------------------------------------

# # Path to your processed parquet file (from SKU agent)
# DATA_PATH = os.path.join("..", "data", "processed", "final_with_sku.parquet")

# # Folder for saving reports
# REPORTS_DIR = os.path.join("..", "data", "reports")
# os.makedirs(REPORTS_DIR, exist_ok=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------

# print(f"📂 Loading cleaned data from: {os.path.abspath(DATA_PATH)}")
# df = pd.read_parquet(DATA_PATH)
# print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")

# # -----------------------------------------------------------
# # STATISTICS GENERATION
# # -----------------------------------------------------------

# print("📊 Generating statistics report...")

# stats = {
#     "total_rows": len(df),
# }

# # Optional metrics — only include if columns exist
# cols = [c.lower() for c in df.columns]

# # Total unique suppliers
# for col in df.columns:
#     if "supplier" in col.lower():
#         stats["unique_suppliers"] = df[col].nunique()
#         break

# # Total unique categories
# for col in df.columns:
#     if "category" in col.lower():
#         stats["unique_categories"] = df[col].nunique()
#         break

# # Total unique SKUs
# for col in df.columns:
#     if "sku" in col.lower():
#         stats["unique_skus"] = df[col].nunique()
#         break

# # Largest / smallest store by product count
# for col in df.columns:
#     if "store" in col.lower() or "location" in col.lower():
#         store_col = col
#         counts = df[store_col].value_counts()
#         stats["largest_store"] = counts.index[0]
#         stats["largest_store_products"] = int(counts.iloc[0])
#         stats["smallest_store"] = counts.index[-1]
#         stats["smallest_store_products"] = int(counts.iloc[-1])
#         stats["avg_products_per_store"] = float(counts.mean())
#         break

# # -----------------------------------------------------------
# # SAVE REPORT
# # -----------------------------------------------------------

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# report_path = os.path.join(REPORTS_DIR, f"statistics_report_{timestamp}.json")

# with open(report_path, "w") as f:
#     json.dump(stats, f, indent=4)

# print(f"✅ Statistics report saved at: {os.path.abspath(report_path)}")

# # (Optional) also save as CSV
# csv_path = report_path.replace(".json", ".csv")
# pd.DataFrame([stats]).to_csv(csv_path, index=False)
# print(f"📈 CSV version saved at: {os.path.abspath(csv_path)}")

# print("🎉 Statistics Generation Agent completed successfully.")



import os
import json
import pandas as pd
from datetime import datetime

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
DATA_PATH = os.path.join("..", "data", "processed", "final_with_sku.parquet")
REPORTS_DIR = os.path.join("..", "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
print(f"📂 Loading cleaned data from: {os.path.abspath(DATA_PATH)}")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")

# -----------------------------------------------------------
# STATISTICS GENERATION
# -----------------------------------------------------------
print("📊 Generating statistics report...")

# Detect key columns
supplier_col = [c for c in df.columns if 'supplier' in c.lower()]
category_col = [c for c in df.columns if 'category' in c.lower()]
sku_col = [c for c in df.columns if 'sku' in c.lower()]

if not supplier_col or not category_col or not sku_col:
    raise ValueError("Required columns not found in data!")

supplier_col = supplier_col[0]
category_col = category_col[0]
sku_col = sku_col[0]

# Compute metrics
stats = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "total_stores": df[supplier_col].nunique(),
    "total_categories": df[category_col].nunique(),
    "total_skus": df[sku_col].nunique(),
    "missing_values": int(df.isnull().sum().sum())
}

# Store-wise product counts
store_counts = df[supplier_col].value_counts()
stats["largest_store"] = store_counts.idxmax()
stats["largest_store_products"] = int(store_counts.max())
stats["smallest_store"] = store_counts.idxmin()
stats["smallest_store_products"] = int(store_counts.min())
stats["avg_products_per_store"] = float(store_counts.mean())

# -----------------------------------------------------------
# SAVE REPORT
# -----------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_json_path = os.path.join(REPORTS_DIR, f"statistics_report_{timestamp}.json")
report_csv_path = report_json_path.replace(".json", ".csv")

with open(report_json_path, "w") as f:
    json.dump(stats, f, indent=4)

pd.DataFrame([stats]).to_csv(report_csv_path, index=False)

print(f"✅ Statistics report saved at: {os.path.abspath(report_json_path)}")
print(f"📈 CSV version saved at: {os.path.abspath(report_csv_path)}")
print("🎉 Statistics Generation Agent completed successfully.")

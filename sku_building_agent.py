# import os
# import pandas as pd

# # -----------------------------
# # 1️⃣ Define file paths (relative-safe)
# # -----------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from /agents/
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# input_parquet = os.path.join(DATA_DIR, "processed_20251030_214739.parquet")
# output_parquet = os.path.join(DATA_DIR, "final_with_sku.parquet")

# # -----------------------------
# # 2️⃣ Load cleaned product data
# # -----------------------------
# print(f"📂 Loading cleaned data from: {input_parquet}")

# try:
#     df = pd.read_parquet(input_parquet)
#     print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")
# except Exception as e:
#     raise FileNotFoundError(f"❌ Failed to load parquet file: {e}")

# # -----------------------------
# # 3️⃣ Define SKU generation rules by category
# # -----------------------------
# def generate_sku(row):
#     """Generate SKU string based on category-specific rules."""
#     vendor = str(row.get("Supplier Name", "UNKNOWN")).replace(" ", "")
#     brand = str(row.get("BRAND", "GENERIC")).replace(" ", "")
#     category = str(row.get("CATEGORY", "MISC")).replace(" ", "")
#     subcat = str(row.get("SUBTYPE", "GEN")).replace(" ", "")
#     barcode = str(row.get("BARCODE", "0000000000000"))

#     # ---- Category-specific templates ----
#     if "Electronics" in category:
#         return f"{vendor}-{brand}-{category}-{subcat}-ModelX-Unisex-Adult-220V-25W-5G-Black-128GB-8GB-Snapdragon8-4500mAh-6.7inch-3200x1440-AMOLED-FingerprintFaceID-1Year-2025-India-{barcode}"

#     elif "Textile" in category or "Cloth" in category:
#         return f"{vendor}-{brand}-{category}-{subcat}-Mens-Adult-Cotton-Checked-Blue-M-L-40inch-120cm-200TC-Twill-OfficeWear-Formal-Summer2025-ExecutiveSeries-2025-India-{barcode}"

#     elif "Jewellery" in category or "Jewelry" in category:
#         return f"{vendor}-{brand}-{category}-{subcat}-Womens-Gold-22K-Diamond-White-2Carat-Classic-Necklace-Wedding-18g-SizeM-Glossy-QueenSeries-2025-India-{barcode}"

#     else:
#         return f"{vendor}-{brand}-{category}-{subcat}-{barcode}"

# # -----------------------------
# # 4️⃣ Apply SKU generation
# # -----------------------------
# print("🧠 Generating SKUs based on category rules...")
# df["SKU"] = df.apply(generate_sku, axis=1)
# print("✅ SKUs generated successfully.")

# # -----------------------------
# # 5️⃣ Save final data
# # -----------------------------
# df.to_parquet(output_parquet, index=False)
# print(f"📦 Final SKU data saved at: {output_parquet}")
# print("🎉 SKU Building Agent completed successfully.")


# sku_building_agent.py

# import pandas as pd
# import os

# # === CONFIG ===
# # === CONFIG ===
# # INPUT_FILE = "../data/processed/processed_cleaned.parquet"  # cleaned file path
# # OUTPUT_FILE = "../data/processed/final_with_sku.parquet"
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from /agents/
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# INPUT_FILE = os.path.join(DATA_DIR, "processed_20251030_214739.parquet")
# OUTPUT_FILE  = os.path.join(DATA_DIR, "final_with_sku.parquet")

# # === LOAD CLEANED DATA ===
# print(f"📂 Loading cleaned data from: {INPUT_FILE}")
# df = pd.read_parquet(INPUT_FILE)
# print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# # === SKU GENERATION FUNCTIONS ===
# def generate_electronics_sku(row):
#     # Example Electronics SKU rule
#     return f"{row['suppref']}-{row['brand']}-{row['category']}-{row.get('subtype', 'GENERIC')}-0000000000000"

# def generate_textiles_sku(row):
#     # Example Textiles SKU rule
#     return f"{row['suppref']}-{row['brand']}-{row['category']}-{row.get('subtype', 'GENERIC')}-0000000000000"

# def generate_jewellery_sku(row):
#     # Example Jewellery SKU rule
#     return f"{row['suppref']}-{row['brand']}-{row['category']}-{row.get('subtype', 'GENERIC')}-0000000000000"

# # === APPLY SKU BASED ON CATEGORY ===
# print("🧠 Generating SKUs based on category rules...")
# def build_sku(row):
#     category = row['category'].lower()
#     if "electronics" in category:
#         return generate_electronics_sku(row)
#     elif "textiles" in category:
#         return generate_textiles_sku(row)
#     elif "jewellery" in category:
#         return generate_jewellery_sku(row)
#     else:
#         return f"{row['suppref']}-{row['brand']}-MISC-GENERIC-0000000000000"

# df['SKU'] = df.apply(build_sku, axis=1)
# print("✅ SKUs generated successfully.")

# # === SAVE FINAL DATA WITH SKU ===
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# df.to_parquet(OUTPUT_FILE, index=False)
# print(f"📦 Final SKU data saved at: {OUTPUT_FILE}")
# print("🎉 SKU Building Agent completed successfully.")


# import pandas as pd
# import os

# # === CONFIG ===
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from /agents/
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# INPUT_FILE = os.path.join(DATA_DIR, "processed_20251030_214739.parquet")
# OUTPUT_FILE = os.path.join(DATA_DIR, "final_with_sku.parquet")

# # === LOAD CLEANED DATA ===
# print(f"📂 Loading cleaned data from: {INPUT_FILE}")
# df = pd.read_parquet(INPUT_FILE)
# print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# # === SKU GENERATION FUNCTIONS ===
# def generate_electronics_sku(row):
#     return f"{row['SUPPREF']}-{row['brand']}-{row['category']}-{row.get('SUBTYPE', 'GENERIC')}-0000000000000"

# def generate_textiles_sku(row):
#     return f"{row['SUPPREF']}-{row['brand']}-{row['category']}-{row.get('SUBTYPE', 'GENERIC')}-0000000000000"

# def generate_jewellery_sku(row):
#     return f"{row['SUPPREF']}-{row['brand']}-{row['category']}-{row.get('SUBTYPE', 'GENERIC')}-0000000000000"

# # === APPLY SKU BASED ON CATEGORY ===
# print("🧠 Generating SKUs based on category rules...")

# def build_sku(row):
#     category = str(row['category']).lower()
#     if "electronics" in category:
#         return generate_electronics_sku(row)
#     elif "textiles" in category or "cloth" in category:
#         return generate_textiles_sku(row)
#     elif "jewellery" in category or "jewelry" in category:
#         return generate_jewellery_sku(row)
#     else:
#         return f"{row['SUPPREF']}-{row['brand']}-MISC-GENERIC-0000000000000"

# df['SKU'] = df.apply(build_sku, axis=1)
# print("✅ SKUs generated successfully.")

# # === SAVE FINAL DATA WITH SKU ===
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# df.to_parquet(OUTPUT_FILE, index=False)
# print(f"📦 Final SKU data saved at: {OUTPUT_FILE}")
# print("🎉 SKU Building Agent completed successfully.")


# import os
# import pandas as pd

# # === CONFIG ===
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from /agents/
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# INPUT_FILE = os.path.join(DATA_DIR, "processed_20251030_214739.parquet")
# OUTPUT_FILE = os.path.join(DATA_DIR, "final_with_sku.parquet")

# # === LOAD CLEANED DATA ===
# print(f"📂 Loading cleaned data from: {INPUT_FILE}")
# df = pd.read_parquet(INPUT_FILE)
# print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")

# # === SKU GENERATION FUNCTIONS ===

# def generate_electronics_sku(row):
#     return (
#         f"{row.get('SUPPREF','UNKNOWN')}-{row.get('brand','GENERIC')}-{row.get('category','MISC')}-"
#         f"{row.get('SUBTYPE','GENERIC')}-{row.get('ANA1','ModelX')}-{row.get('ANA2','Unisex')}-"
#         f"{row.get('ANA3','Adult')}-{row.get('ANA4','220V')}-{row.get('ANA5','25W')}-{row.get('ANA6','5G')}-"
#         f"{row.get('ANA7','Black')}-{row.get('ANA8','128GB')}-{row.get('ANA9','8GB')}-{row.get('ANA10','Snapdragon8')}-"
#         f"{row.get('ANA11','4500mAh')}-{row.get('ANA12','6.7inch')}-3200x1440-AMOLED-FingerprintFaceID-1Year-2025-India-"
#         f"{row.get('barcode','0000000000000')}"
#     )

# def generate_textiles_sku(row):
#     return (
#         f"{row.get('SUPPREF','UNKNOWN')}-{row.get('brand','GENERIC')}-{row.get('category','MISC')}-"
#         f"{row.get('SUBTYPE','GENERIC')}-{row.get('ANA1','Mens')}-{row.get('ANA2','Adult')}-"
#         f"{row.get('ANA3','Cotton')}-{row.get('ANA4','Checked')}-{row.get('ANA5','Blue')}-"
#         f"{row.get('ANA6','M-L')}-{row.get('Length','40inch')}-{row.get('Breadth','120cm')}-"
#         f"{row.get('ANA7','200TC')}-{row.get('ANA8','Twill')}-{row.get('ANA9','OfficeWear')}-"
#         f"{row.get('ANA10','Formal')}-{row.get('ANA11','Summer2025')}-{row.get('ANA12','ExecutiveSeries')}-"
#         f"2025-India-{row.get('barcode','0000000000000')}"
#     )

# def generate_jewellery_sku(row):
#     return (
#         f"{row.get('SUPPREF','UNKNOWN')}-{row.get('brand','GENERIC')}-{row.get('category','MISC')}-"
#         f"{row.get('SUBTYPE','GENERIC')}-{row.get('ANA1','Womens')}-{row.get('ANA2','Gold')}-"
#         f"{row.get('ANA3','22K')}-{row.get('ANA4','Diamond')}-{row.get('ANA5','White')}-"
#         f"{row.get('ANA6','2Carat')}-{row.get('ANA7','Classic')}-{row.get('ANA8','Necklace')}-"
#         f"{row.get('ANA9','Wedding')}-{row.get('ANA10','18g')}-{row.get('ANA11','SizeM')}-"
#         f"{row.get('ANA12','Glossy')}-QueenSeries-2025-India-{row.get('barcode','0000000000000')}"
#     )

# # === APPLY SKU BASED ON CATEGORY ===
# print("🧠 Generating SKUs based on category rules...")

# def build_sku(row):
#     category = str(row.get('category','')).lower()
#     if "electronics" in category:
#         return generate_electronics_sku(row)
#     elif "textile" in category or "cloth" in category:
#         return generate_textiles_sku(row)
#     elif "jewellery" in category or "jewelry" in category:
#         return generate_jewellery_sku(row)
#     else:
#         return f"{row.get('SUPPREF','UNKNOWN')}-{row.get('brand','GENERIC')}-MISC-GENERIC-0000000000000"

# df['SKU'] = df.apply(build_sku, axis=1)
# print("✅ SKUs generated successfully.")

# # === SAVE FINAL DATA WITH SKU ===
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# df.to_parquet(OUTPUT_FILE, index=False)
# print(f"📦 Final SKU data saved at: {OUTPUT_FILE}")
# print("🎉 SKU Building Agent completed successfully.")


# import os
# import pandas as pd
# import json

# # === CONFIG ===
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /agents/ -> parent
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# INPUT_FILE = os.path.join(DATA_DIR, "final_with_sku.parquet")
# OUTPUT_FILE = os.path.join(DATA_DIR, "mall_sku.json")

# # === LOAD DATA ===
# df = pd.read_parquet(INPUT_FILE)
# print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# # === CREATE JSON STRUCTURE ===
# mall_data = {
#     "mall": {
#         "mall_id": "MALL001",
#         "name": "Prime City Mall",
#         "location": "Bengaluru, India",
#         "stores": []
#     }
# }

# # Group by store (supplier_name)
# for store_name, group in df.groupby("supplier_name"):
#     store_skus = []
#     for _, row in group.iterrows():
#         sku_entry = {
#             "vendor_id": row.get("SUPPREF", ""),
#             "brand": row.get("brand", ""),
#             "category": row.get("category", ""),
#             "subcategory": row.get("SUBTYPE", ""),
#             "gender": row.get("ANA1", ""),
#             "material": row.get("ANA2", ""),
#             "purity": row.get("ANA3", ""),
#             "stone_type": row.get("ANA4", ""),
#             "stone_color": row.get("ANA5", ""),
#             "stone_carat": row.get("ANA6", ""),
#             "design": row.get("ANA7", ""),
#             "type": row.get("TYPE", ""),
#             "occasion": row.get("ANA8", ""),
#             "weight": row.get("ANA9", ""),
#             "size": row.get("ANA10", ""),
#             "color_finish": row.get("ANA11", ""),
#             "collection": row.get("ANA12", ""),
#             "year": 2025,
#             "country_of_origin": "India",
#             "barcode_id": row.get("barcode", ""),
#             "sku": row.get("SKU", "")
#         }
#         store_skus.append(sku_entry)
    
#     store_entry = {
#         "store_id": f"STORE{len(mall_data['mall']['stores'])+1:03d}",
#         "name": store_name,
#         "type": row.get("category", ""),
#         "skus": store_skus
#     }
#     mall_data["mall"]["stores"].append(store_entry)

# # === SAVE JSON ===
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(mall_data, f, ensure_ascii=False, indent=2)

# print(f"📦 JSON saved to: {OUTPUT_FILE}")


import os
import pandas as pd
import json

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /agents/ -> parent
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

INPUT_FILE = os.path.join(DATA_DIR, "final_with_sku.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "mall_sku.json")

# === LOAD DATA ===
df = pd.read_parquet(INPUT_FILE)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# === CREATE JSON STRUCTURE ===
mall_data = {
    "mall": {
        "mall_id": "MALL001",
        "name": "Prime City Mall",
        "location": "Bengaluru, India",
        "stores": []
    }
}

# Group by store (supplier_name)
for store_name, group in df.groupby("supplier_name"):
    store_skus = []
    for _, row in group.iterrows():
        sku_entry = {
            "vendor_id": row.get("SUPPREF", ""),
            "brand": row.get("brand", ""),
            "category": row.get("category", ""),
            "subcategory": row.get("SUBTYPE", ""),
            "gender": row.get("ANA1", ""),
            "material": row.get("ANA2", ""),
            "purity": row.get("ANA3", ""),
            "stone_type": row.get("ANA4", ""),
            "stone_color": row.get("ANA5", ""),
            "stone_carat": row.get("ANA6", ""),
            "design": row.get("ANA7", ""),
            "type": row.get("TYPE", ""),
            "occasion": row.get("ANA8", ""),
            "weight": row.get("ANA9", ""),
            "size": row.get("ANA10", ""),
            "color_finish": row.get("ANA11", ""),
            "collection": row.get("ANA12", ""),
            "year": 2025,
            "country_of_origin": "India",
            "barcode_id": row.get("barcode", ""),
            "sku": row.get("SKU", "")
        }
        store_skus.append(sku_entry)
    
    store_entry = {
        "store_id": f"STORE{len(mall_data['mall']['stores'])+1:03d}",
        "name": store_name,
        "type": row.get("category", ""),
        "skus": store_skus
    }
    mall_data["mall"]["stores"].append(store_entry)

# === SAVE JSON ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(mall_data, f, ensure_ascii=False, indent=2)

print(f"📦 JSON saved to: {OUTPUT_FILE}")

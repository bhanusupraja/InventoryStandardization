import pandas as pd
import numpy as np
import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataCleaningAgent:
    def __init__(self, input_path: str, output_dir: str = "data/processed"):
        """
        Initialize the Data Cleaning Agent.
        
        Args:
            input_path: Path to the input file (Excel/CSV)
            output_dir: Directory to save processed files
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.df_raw = None
        self.df_cleaned = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics tracking
        self.cleaning_stats = {
            'rows_original': 0,
            'rows_after_cleaning': 0,
            'duplicates_removed': 0,
            'invalid_rows_removed': 0,
            'columns_standardized': 0
        }
    
    def load_file(self):
        """Load the input file (CSV or Excel)."""
        print(f"📂 Loading file: {self.input_path}")
        
        try:
            if self.input_path.endswith('.csv'):
                self.df_raw = pd.read_csv(self.input_path, low_memory=False)
            elif self.input_path.endswith('.xlsx') or self.input_path.endswith('.xls'):
                self.df_raw = pd.read_excel(self.input_path, sheet_name=0)
            else:
                raise ValueError("Unsupported file type! Must be .csv, .xlsx, or .xls")
            
            self.cleaning_stats['rows_original'] = len(self.df_raw)
            print(f"✅ Loaded {len(self.df_raw):,} rows and {len(self.df_raw.columns)} columns")
            
            return self.df_raw
            
        except Exception as e:
            raise Exception(f"❌ Failed to load file: {str(e)}")
    
    def map_schema(self):
        """Map and standardize column names."""
        print("🔧 Mapping schema and standardizing columns...")
        
        # Column mapping dictionary
        column_mapping = {
            # Product identification
            'CODE': 'product_id',
            'SUPPREF': 'supplier_ref',
            'BARCODE': 'barcode',
            
            # Descriptions
            'DESCRIPTION1': 'description1',
            'DESCRIPTION2': 'description2',
            
            # Categories and classification
            'BRAND': 'brand',
            'BRAND_Code': 'brand_code',
            'CATEGORY': 'category',
            'CATEGORY_Code': 'category_code',
            'TYPE': 'type',
            'TYPE_Code': 'type_code',
            'SUBTYPE': 'subtype',
            'SUBTYPE_Code': 'subtype_code',
            
            # Supplier information
            'Supplier Name': 'supplier_name',
            'Supplier Account': 'supplier_account',
            
            # Stock and inventory
            'QTY IN STOCK': 'stock_qty',
            'QTY_In_Stock': 'stock_qty',
            'Warehouse Code': 'warehouse_code',
            'Location Code': 'location_code',
            
            # Pricing
            'RETAIL': 'price_retail',
            'PRICE_Retail': 'price_retail',
            'PRICE_Consumer': 'price_consumer',
            'PRICE_A': 'price_a',
            'PRICE_B': 'price_b',
            'PRICE_C': 'price_c',
            'Wholesale': 'price_wholesale',
            'STAFF': 'price_staff',
            'AVERAGECOST': 'average_cost',
            'LAST COST': 'last_cost',
            
            # Tax information
            'HSN': 'hsn',
            'GST': 'gst',
            'CESS': 'cess',
            
            # Dimensions
            'Length': 'length',
            'Breadth': 'breadth',
            'Height': 'height',
            'GrsWeight': 'gross_weight',
            'NetWeight': 'net_weight',
            
            # Analysis fields
            'ANA1': 'ana1',
            'ANA2': 'ana2',
            'ANA3': 'ana3',
            'ANA4': 'ana4',
            'ANA5': 'ana5',
            'ANA6': 'ana6',
            'ANA7': 'ana7',
            'ANA8': 'ana8',
            
            # Notes
            'ShortNote1': 'short_note1',
            'ShortNote2': 'short_note2',
            'FreeTextNotes': 'free_text_notes',
            
            # Web listing
            'INCLUDE IN WEBSITE': 'include_website'
        }
        
        # Normalize column names (strip whitespace)
        self.df_raw.columns = self.df_raw.columns.str.strip()
        
        # Map known columns
        df = self.df_raw.rename(columns=column_mapping)
        
        # Create combined description field
        if 'description1' in df.columns and 'description2' in df.columns:
            df['description'] = (
                df['description1'].fillna('').astype(str) + ' ' +
                df['description2'].fillna('').astype(str)
            ).str.strip()
        elif 'description1' in df.columns:
            df['description'] = df['description1'].fillna('').astype(str)
        elif 'description2' in df.columns:
            df['description'] = df['description2'].fillna('').astype(str)
        else:
            df['description'] = ''
        
        self.df_cleaned = df
        self.cleaning_stats['columns_standardized'] = len(column_mapping)
        
        return df
    
    def normalize_text_fields(self):
        """Normalize and clean text fields."""
        print("📝 Normalizing text fields...")
        
        text_fields = [
            'description', 'description1', 'description2',
            'brand', 'category', 'type', 'subtype',
            'supplier_name', 'short_note1', 'short_note2',
            'free_text_notes'
        ]
        
        for field in text_fields:
            if field in self.df_cleaned.columns:
                self.df_cleaned[field] = (
                    self.df_cleaned[field]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .replace(r'\s+', ' ', regex=True)  # Normalize whitespace
                    .replace('nan', '', regex=False)    # Remove 'nan' strings
                )
        
        return self.df_cleaned
    
    def extract_attributes(self):
        """Extract product attributes from descriptions."""
        print("🔍 Extracting product attributes...")
        
        df = self.df_cleaned
        
        # Extract volume/capacity
        def extract_volume(text):
            text = str(text).upper()
            # Look for ML patterns
            ml_match = re.search(r'(\d+(?:\.\d+)?)\s*ML', text)
            if ml_match:
                return f"{ml_match.group(1)}ML"
            # Look for L patterns
            l_match = re.search(r'(\d+(?:\.\d+)?)\s*L\b', text)
            if l_match:
                return f"{l_match.group(1)}L"
            return ''
        
        # Extract colors
        def extract_color(text):
            text = str(text).lower()
            colors = ['clear', 'white', 'black', 'blue', 'red', 'green', 
                     'yellow', 'grey', 'gray', 'silver', 'gold', 'pink',
                     'purple', 'brown', 'orange']
            for color in colors:
                if color in text:
                    return color
            return ''
        
        # Extract material
        def extract_material(text):
            text = str(text).lower()
            materials = ['glass', 'crystal', 'ceramic', 'porcelain', 
                        'stoneware', 'steel', 'silver', 'gold', 'plastic',
                        'wood', 'metal', 'copper', 'brass', 'aluminum']
            for material in materials:
                if material in text:
                    return material
            return ''
        
        # Extract quantity/set information
        def extract_quantity(text):
            text = str(text).upper()
            # Look for "SET OF X" or "X PCS" patterns
            set_match = re.search(r'(?:SET\s*OF\s*|X)(\d+)|(\d+)\s*(?:PCS|PIECES)', text)
            if set_match:
                return set_match.group(1) or set_match.group(2)
            return '1'
        
        # Apply extractions
        df['extracted_volume'] = df['description'].apply(extract_volume)
        df['extracted_color'] = df['description'].apply(extract_color)
        df['extracted_material'] = df['description'].apply(extract_material)
        df['extracted_quantity'] = df['description'].apply(extract_quantity)
        
        print(f"   ✓ Extracted volume for {(df['extracted_volume'] != '').sum():,} products")
        print(f"   ✓ Extracted color for {(df['extracted_color'] != '').sum():,} products")
        print(f"   ✓ Extracted material for {(df['extracted_material'] != '').sum():,} products")
        
        self.df_cleaned = df
        return df
    
    def clean_numeric_fields(self):
        """Clean and standardize numeric fields."""
        print("🔢 Cleaning numeric fields...")
        
        numeric_fields = [
            'stock_qty', 'price_retail', 'price_consumer',
            'price_a', 'price_b', 'price_c', 'price_wholesale',
            'price_staff', 'average_cost', 'last_cost',
            'length', 'breadth', 'height', 'gross_weight', 'net_weight',
            'gst', 'cess'
        ]
        
        for field in numeric_fields:
            if field in self.df_cleaned.columns:
                self.df_cleaned[field] = pd.to_numeric(
                    self.df_cleaned[field], 
                    errors='coerce'
                )
        
        return self.df_cleaned
    
    def remove_duplicates_and_invalid(self):
        """Remove duplicate rows and invalid data."""
        print("🧹 Removing duplicates and invalid rows...")
        
        initial_rows = len(self.df_cleaned)
        
        # Remove exact duplicates
        self.df_cleaned.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(self.df_cleaned)
        
        # Remove rows with critical missing data
        mandatory_fields = ['supplier_name']
        initial_rows = len(self.df_cleaned)
        
        for field in mandatory_fields:
            if field in self.df_cleaned.columns:
                self.df_cleaned = self.df_cleaned[
                    (self.df_cleaned[field].notna()) & 
                    (self.df_cleaned[field] != '') &
                    (self.df_cleaned[field] != 'nan')
                ]
        
        invalid_removed = initial_rows - len(self.df_cleaned)
        
        # Remove rows with empty descriptions
        if 'description' in self.df_cleaned.columns:
            initial_rows = len(self.df_cleaned)
            self.df_cleaned = self.df_cleaned[
                (self.df_cleaned['description'] != '') &
                (self.df_cleaned['description'].str.len() > 3)
            ]
            invalid_removed += initial_rows - len(self.df_cleaned)
        
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        self.cleaning_stats['invalid_rows_removed'] = invalid_removed
        
        print(f"   ✓ Removed {duplicates_removed:,} duplicate rows")
        print(f"   ✓ Removed {invalid_removed:,} invalid rows")
        
        return self.df_cleaned
    
    def standardize_categories(self):
        """Standardize category names and fix common issues."""
        print("🏷️ Standardizing categories...")
        
        if 'category' not in self.df_cleaned.columns:
            return self.df_cleaned
        
        # Category standardization mapping
        category_map = {
            # Jewellery variations
            'jewellery': 'Jewellery',
            'jewelry': 'Jewellery',
            'branded jewellery': 'Jewellery',
            'fashion jewellery': 'Jewellery',
            
            # Watch variations
            'branded watch': 'Branded Watch',
            'watches': 'Branded Watch',
            'timepiece': 'Branded Watch',
            
            # Home decor variations
            'home decor': 'Home Decor',
            'homedecor': 'Home Decor',
            'decoration': 'Home Decor',
            'decorative': 'Home Decor',
            
            # Drinkware variations
            'drinkware': 'Drinkware',
            'glassware': 'Drinkware',
            'drink ware': 'Drinkware',
            
            # Tableware variations
            'tableware': 'Tableware',
            'dinnerware': 'Tableware',
            'serveware': 'Tableware',
            
            # Electronics variations
            'electronics': 'Electronics',
            'electronic': 'Electronics',
            'gadgets': 'Electronics',
            
            # Textiles variations
            'textiles': 'Textiles',
            'fabric': 'Textiles',
            'apparel': 'Textiles',
            'clothing': 'Textiles'
        }
        
        # Apply mapping
        self.df_cleaned['category'] = self.df_cleaned['category'].map(
            lambda x: category_map.get(str(x).lower(), x) if pd.notna(x) else ''
        )
        
        return self.df_cleaned
    
    def add_metadata(self):
        """Add metadata fields for tracking."""
        print("📋 Adding metadata...")
        
        self.df_cleaned['processing_timestamp'] = datetime.now().isoformat()
        self.df_cleaned['processing_version'] = '2.0'
        self.df_cleaned['data_source'] = os.path.basename(self.input_path)
        self.df_cleaned['row_id'] = range(len(self.df_cleaned))
        
        return self.df_cleaned
    
    def validate_data_quality(self):
        """Validate data quality and generate warnings."""
        print("✅ Validating data quality...")
        
        warnings_list = []
        
        # Check for high percentage of missing values
        missing_pct = (self.df_cleaned.isnull().sum() / len(self.df_cleaned)) * 100
        high_missing = missing_pct[missing_pct > 50]
        
        if not high_missing.empty:
            warnings_list.append(f"High missing values in: {', '.join(high_missing.index.tolist())}")
        
        # Check for low category coverage
        if 'category' in self.df_cleaned.columns:
            uncategorized = (self.df_cleaned['category'] == '') | (self.df_cleaned['category'].isna())
            if uncategorized.sum() > len(self.df_cleaned) * 0.3:
                warnings_list.append(f"Low category coverage: {uncategorized.sum():,} products uncategorized")
        
        # Print warnings
        if warnings_list:
            print("\n⚠️ Data Quality Warnings:")
            for warning in warnings_list:
                print(f"   - {warning}")
        else:
            print("   ✓ Data quality checks passed")
        
        return warnings_list
    
    def save_cleaned_data(self):
        """Save the cleaned data to parquet and CSV."""
        print("💾 Saving cleaned data...")
        
        # Generate timestamp for filenames
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as parquet (efficient format)
        parquet_path = os.path.join(self.output_dir, f"processed_{date_str}.parquet")
        table = pa.Table.from_pandas(self.df_cleaned)
        pq.write_table(table, parquet_path)
        print(f"   ✓ Parquet saved → {parquet_path}")
        
        # Save as CSV (human-readable backup)
        csv_path = os.path.join(self.output_dir, f"processed_{date_str}.csv")
        self.df_cleaned.to_csv(csv_path, index=False)
        print(f"   ✓ CSV backup saved → {csv_path}")
        
        return parquet_path
    
    def generate_summary(self):
        """Generate processing summary."""
        self.cleaning_stats['rows_after_cleaning'] = len(self.df_cleaned)
        
        summary = {
            'file_name': os.path.basename(self.input_path),
            'processing_timestamp': datetime.now().isoformat(),
            'statistics': {
                'rows_original': self.cleaning_stats['rows_original'],
                'rows_cleaned': self.cleaning_stats['rows_after_cleaning'],
                'rows_removed': self.cleaning_stats['rows_original'] - self.cleaning_stats['rows_after_cleaning'],
                'duplicates_removed': self.cleaning_stats['duplicates_removed'],
                'invalid_rows_removed': self.cleaning_stats['invalid_rows_removed'],
                'columns_total': len(self.df_cleaned.columns),
                'columns_standardized': self.cleaning_stats['columns_standardized']
            },
            'data_quality': {
                'missing_values_total': int(self.df_cleaned.isnull().sum().sum()),
                'completeness_pct': float(100 - (self.df_cleaned.isnull().sum().sum() / 
                                                 (len(self.df_cleaned) * len(self.df_cleaned.columns)) * 100))
            },
            'extracted_attributes': {
                'volume_extracted': int((self.df_cleaned.get('extracted_volume', pd.Series()) != '').sum()),
                'color_extracted': int((self.df_cleaned.get('extracted_color', pd.Series()) != '').sum()),
                'material_extracted': int((self.df_cleaned.get('extracted_material', pd.Series()) != '').sum())
            }
        }
        
        return summary
    
    def run(self):
        """Execute the complete data cleaning pipeline."""
        print("\n" + "=" * 60)
        print("🚀 DATA CLEANING PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load file
            self.load_file()
            
            # Step 2: Map schema
            self.map_schema()
            
            # Step 3: Normalize text
            self.normalize_text_fields()
            
            # Step 4: Extract attributes
            self.extract_attributes()
            
            # Step 5: Clean numeric fields
            self.clean_numeric_fields()
            
            # Step 6: Remove duplicates and invalid rows
            self.remove_duplicates_and_invalid()
            
            # Step 7: Standardize categories
            self.standardize_categories()
            
            # Step 8: Add metadata
            self.add_metadata()
            
            # Step 9: Validate quality
            self.validate_data_quality()
            
            # Step 10: Save cleaned data
            output_path = self.save_cleaned_data()
            
            # Step 11: Generate summary
            summary = self.generate_summary()
            
            print("\n" + "=" * 60)
            print("✅ DATA CLEANING COMPLETE")
            print("=" * 60)
            print(f"📊 Rows processed: {summary['statistics']['rows_original']:,} → {summary['statistics']['rows_cleaned']:,}")
            print(f"🧹 Rows removed: {summary['statistics']['rows_removed']:,}")
            print(f"📈 Data completeness: {summary['data_quality']['completeness_pct']:.1f}%")
            print("=" * 60)
            
            return output_path, summary
            
        except Exception as e:
            print(f"\n❌ Data cleaning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "06.10.2025.xlsx"
    OUTPUT_DIR = "data/processed"
    
    # Initialize and run agent
    agent = DataCleaningAgent(
        input_path=INPUT_FILE,
        output_dir=OUTPUT_DIR
    )
    
    try:
        output_path, summary = agent.run()
        print(f"\n🎉 Data ready for Mapping Engine!")
        print(f"📁 Output: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
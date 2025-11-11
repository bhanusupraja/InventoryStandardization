import os
import pandas as pd
import numpy as np
import json
import glob
import re
from datetime import datetime
import hashlib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SKUBuilder:
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/processed"):
        """
        Initialize Enhanced SKU Builder with domain-specific rules.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.df_mapped = None
        self.df_with_sku = None
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # SKU statistics
        self.sku_stats = {
            'total_generated': 0,
            'unique_skus': 0,
            'duplicates_resolved': 0,
            'category_breakdown': {}
        }
        
        # Define domain-specific SKU structures
        self.sku_structures = self._define_sku_structures()
    
    def _define_sku_structures(self):
        """Define shortened SKU structures for each domain."""
        return {
            'jewellery': {
                'prefix': 'JWL',
                'components': [
                    'vendor_id', 'brand', 'subcategory', 'gender', 
                    'material', 'purity', 'stone_type', 'stone_color',
                    'stone_carat', 'design', 'type', 'occasion',
                    'weight', 'size', 'color_finish', 'collection',
                    'year', 'country', 'barcode'
                ],
                'shortened': {
                    'Ring': 'RNG', 'Necklace': 'NCK', 'Bracelet': 'BRC',
                    'Earrings': 'ERG', 'Pendant': 'PND', 'Bangle': 'BGL',
                    'Chain': 'CHN', 'Anklet': 'ANK', 'Brooch': 'BRH',
                    'Womens': 'W', 'Mens': 'M', 'Unisex': 'U',
                    'Gold': 'GLD', 'Silver': 'SLV', 'Platinum': 'PLT',
                    'Diamond': 'DMD', 'Emerald': 'EMD', 'Ruby': 'RBY',
                    'Sapphire': 'SPH', 'Pearl': 'PRL', 'Topaz': 'TPZ',
                    'White': 'WH', 'Yellow': 'YL', 'Rose': 'RS',
                    'Black': 'BK', 'Blue': 'BL', 'Green': 'GR',
                    'Red': 'RD', 'Pink': 'PK',
                    'Traditional': 'TRD', 'Modern': 'MDN', 'Contemporary': 'CTM',
                    'Floral': 'FLR', 'Geometric': 'GEO', 'Royal': 'RYL',
                    'Engagement': 'ENG', 'Wedding': 'WED', 'Casual': 'CSL',
                    'Festive': 'FST', 'DailyWear': 'DLY', 'Party': 'PTY',
                    'India': 'IN', 'USA': 'US', 'China': 'CN'
                }
            },
            'textiles': {
                'prefix': 'TXT',
                'components': [
                    'vendor_id', 'brand', 'subcategory', 'gender',
                    'age_group', 'material', 'pattern', 'color',
                    'size', 'length', 'width', 'thread_count',
                    'weave', 'occasion', 'style', 'season',
                    'collection', 'year', 'country', 'barcode'
                ],
                'shortened': {
                    'Shirt': 'SHT', 'Pants': 'PNT', 'TShirt': 'TSH',
                    'Dress': 'DRS', 'Suit': 'SUT', 'Jeans': 'JNS',
                    'Jacket': 'JKT', 'Sweater': 'SWT', 'Skirt': 'SKT',
                    'Mens': 'M', 'Womens': 'W', 'Unisex': 'U', 'Kids': 'K',
                    'Adult': 'A', 'Teen': 'T', 'Child': 'C', 'Infant': 'I',
                    'Cotton': 'CTN', 'Linen': 'LNN', 'Silk': 'SLK',
                    'Wool': 'WOL', 'Polyester': 'PLY', 'Denim': 'DNM',
                    'Checked': 'CHK', 'Solid': 'SLD', 'Striped': 'STP',
                    'Printed': 'PRT', 'Floral': 'FLR', 'Geometric': 'GEO',
                    'Blue': 'BL', 'White': 'WH', 'Black': 'BK',
                    'Red': 'RD', 'Green': 'GR', 'Grey': 'GY',
                    'Beige': 'BG', 'Navy': 'NV', 'Brown': 'BR',
                    'Formal': 'FML', 'Casual': 'CSL', 'Business': 'BUS',
                    'Sporty': 'SPT', 'Ethnic': 'ETH', 'Party': 'PTY',
                    'Summer': 'SUM', 'Winter': 'WIN', 'Spring': 'SPR',
                    'India': 'IN', 'USA': 'US', 'Bangladesh': 'BD'
                }
            },
            'electronics': {
                'prefix': 'ELC',
                'components': [
                    'vendor_id', 'brand', 'subcategory', 'model',
                    'power_voltage', 'wattage', 'connectivity', 'color',
                    'storage', 'ram', 'processor', 'battery',
                    'screen_size', 'resolution', 'features', 'warranty',
                    'year', 'country', 'barcode'
                ],
                'shortened': {
                    'Smartphone': 'PHN', 'Laptop': 'LTP', 'Tablet': 'TBL',
                    'Television': 'TV', 'Camera': 'CAM', 'Headphone': 'HPN',
                    'Speaker': 'SPK', 'Smartwatch': 'SWT', 'Router': 'RTR',
                    'Black': 'BK', 'White': 'WH', 'Silver': 'SV',
                    'Gold': 'GD', 'Blue': 'BL', 'Red': 'RD',
                    '5G': '5G', '4G': '4G', 'WiFi': 'WF',
                    'Bluetooth': 'BT', 'USB': 'USB', 'TypeC': 'TC',
                    'AMOLED': 'AMD', 'LCD': 'LCD', 'LED': 'LED',
                    'OLED': 'OLD', 'Retina': 'RTN',
                    'India': 'IN', 'China': 'CN', 'SouthKorea': 'KR',
                    'Japan': 'JP', 'USA': 'US', 'Taiwan': 'TW'
                }
            },
            'home_decor': {
                'prefix': 'HDC',
                'components': [
                    'vendor_id', 'brand', 'subcategory', 'material',
                    'color', 'size', 'style', 'finish', 'collection',
                    'year', 'country', 'barcode'
                ],
                'shortened': {
                    'Vase': 'VSE', 'Frame': 'FRM', 'Mirror': 'MRR',
                    'Candle': 'CDL', 'Sculpture': 'SCP', 'Clock': 'CLK',
                    'Lamp': 'LMP', 'Cushion': 'CSH', 'Rug': 'RUG',
                    'Glass': 'GLS', 'Crystal': 'CRY', 'Ceramic': 'CRM',
                    'Wood': 'WOD', 'Metal': 'MTL', 'Plastic': 'PLS',
                    'Modern': 'MDN', 'Traditional': 'TRD', 'Contemporary': 'CTM',
                    'Vintage': 'VTG', 'Minimalist': 'MIN', 'Rustic': 'RST',
                    'Small': 'S', 'Medium': 'M', 'Large': 'L',
                    'India': 'IN', 'Italy': 'IT', 'China': 'CN'
                }
            },
            'branded_watch': {
                'prefix': 'WCH',
                'components': [
                    'vendor_id', 'brand', 'model', 'gender',
                    'movement', 'strap_material', 'dial_color', 'case_size',
                    'water_resistance', 'features', 'collection',
                    'year', 'country', 'barcode'
                ],
                'shortened': {
                    'Mens': 'M', 'Womens': 'W', 'Unisex': 'U',
                    'Automatic': 'AUT', 'Quartz': 'QTZ', 'Digital': 'DIG',
                    'Smart': 'SMT', 'Mechanical': 'MEC', 'Solar': 'SOL',
                    'Leather': 'LTH', 'Steel': 'STL', 'Rubber': 'RBR',
                    'Titanium': 'TTN', 'Gold': 'GLD', 'Ceramic': 'CRM',
                    'Black': 'BK', 'White': 'WH', 'Blue': 'BL',
                    'Silver': 'SV', 'Gold': 'GD', 'Brown': 'BR',
                    'Switzerland': 'CH', 'Japan': 'JP', 'USA': 'US'
                }
            }
        }
    
    def find_latest_mapped_data(self):
        """Find the latest mapped data file."""
        patterns = [
            os.path.join(self.input_dir, "mapped_data_*.parquet"),
            os.path.join(self.input_dir, "*mapped*.parquet"),
            os.path.join(self.input_dir, "*.parquet")
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                print(f"📁 Found: {os.path.basename(latest_file)}")
                return latest_file
        
        raise FileNotFoundError(f"❌ No mapped data files found in {self.input_dir}")
    
    def load_mapped_data(self):
        """Load the mapped data."""
        input_file = self.find_latest_mapped_data()
        print(f"📂 Loading mapped data from: {input_file}")
        
        self.df_mapped = pd.read_parquet(input_file)
        print(f"✅ Loaded {len(self.df_mapped):,} products")
        
        return self.df_mapped
    
    def _get_shortened_code(self, value, category, field):
        """Get shortened code for a value based on category and field."""
        if not value or pd.isna(value):
            return 'NA'
        
        value = str(value).strip()
        
        # Get category-specific shortened codes
        if category.lower() in self.sku_structures:
            shortened = self.sku_structures[category.lower()].get('shortened', {})
            
            # Try exact match first
            if value in shortened:
                return shortened[value]
            
            # Try case-insensitive match
            for key, code in shortened.items():
                if key.lower() == value.lower():
                    return code
        
        # Default shortening strategies
        if field in ['vendor_id', 'barcode']:
            return value[:8] if len(value) > 8 else value
        elif field == 'brand':
            # Take first 3-4 letters of brand
            clean = re.sub(r'[^A-Za-z0-9]', '', value)
            return clean[:4].upper() if clean else 'GEN'
        elif field == 'year':
            return str(value)[-2:]  # Last 2 digits of year
        elif field in ['weight', 'size', 'carat']:
            # Extract numeric part
            numeric = re.search(r'(\d+(?:\.\d+)?)', str(value))
            if numeric:
                return numeric.group(1)
            return '0'
        else:
            # Default: first 3 characters
            clean = re.sub(r'[^A-Za-z0-9]', '', value)
            return clean[:3].upper() if clean else 'NA'
    
    def generate_jewellery_sku(self, row):
        """Generate shortened SKU for jewellery with full JSON breakdown."""
        structure = self.sku_structures['jewellery']
        sku_components = []
        json_breakdown = {}
        
        # Add prefix
        sku_components.append(structure['prefix'])
        
        # Extract and shorten each component
        # Vendor ID
        vendor = self._get_shortened_code(
            row.get('supplier_ref', row.get('supplier_name', '')), 
            'jewellery', 'vendor_id'
        )
        sku_components.append(vendor)
        json_breakdown['vendor_id'] = vendor
        
        # Brand
        brand = self._get_shortened_code(row.get('brand', ''), 'jewellery', 'brand')
        sku_components.append(brand)
        json_breakdown['brand'] = row.get('brand', 'Generic')
        
        # Subcategory
        desc = str(row.get('description', '')).lower()
        subcat_full = self._detect_jewellery_type(desc)
        subcat = self._get_shortened_code(subcat_full, 'jewellery', 'subcategory')
        sku_components.append(subcat)
        json_breakdown['subcategory'] = subcat_full
        
        # Gender
        gender_full = self._detect_gender(row)
        gender = self._get_shortened_code(gender_full, 'jewellery', 'gender')
        sku_components.append(gender)
        json_breakdown['gender'] = gender_full
        
        # Material
        material_full = self._detect_jewellery_material(row)
        material = self._get_shortened_code(material_full, 'jewellery', 'material')
        sku_components.append(material)
        json_breakdown['material'] = material_full
        
        # Purity
        purity_full = self._detect_purity(row)
        purity = self._get_shortened_code(purity_full, 'jewellery', 'purity')
        sku_components.append(purity)
        json_breakdown['purity'] = purity_full
        
        # Stone Type
        stone_full = self._detect_stone_type(row)
        stone = self._get_shortened_code(stone_full, 'jewellery', 'stone_type')
        sku_components.append(stone)
        json_breakdown['stone_type'] = stone_full
        
        # Stone Color
        stone_color_full = self._detect_stone_color(row)
        stone_color = self._get_shortened_code(stone_color_full, 'jewellery', 'stone_color')
        sku_components.append(stone_color)
        json_breakdown['stone_color'] = stone_color_full
        
        # Stone Carat
        carat = self._extract_carat(row)
        sku_components.append(carat)
        json_breakdown['stone_carat'] = f"{carat}Ct"
        
        # Design
        design_full = self._detect_design(row)
        design = self._get_shortened_code(design_full, 'jewellery', 'design')
        sku_components.append(design)
        json_breakdown['design'] = design_full
        
        # Type
        type_full = self._detect_jewellery_type_category(row)
        type_code = self._get_shortened_code(type_full, 'jewellery', 'type')
        sku_components.append(type_code)
        json_breakdown['type'] = type_full
        
        # Occasion
        occasion_full = self._detect_occasion(row)
        occasion = self._get_shortened_code(occasion_full, 'jewellery', 'occasion')
        sku_components.append(occasion)
        json_breakdown['occasion'] = occasion_full
        
        # Weight
        weight = self._extract_weight(row)
        sku_components.append(weight)
        json_breakdown['weight'] = f"{weight}g"
        
        # Size
        size = self._extract_size(row)
        sku_components.append(size)
        json_breakdown['size'] = size
        
        # Color Finish
        finish_full = self._detect_color_finish(row)
        finish = self._get_shortened_code(finish_full, 'jewellery', 'color_finish')
        sku_components.append(finish)
        json_breakdown['color_finish'] = finish_full
        
        # Collection
        collection = self._extract_collection(row)
        sku_components.append(collection[:6])
        json_breakdown['collection'] = collection
        
        # Year
        year = datetime.now().year
        sku_components.append(str(year)[-2:])
        json_breakdown['year'] = year
        
        # Country
        country = 'IN'
        sku_components.append(country)
        json_breakdown['country_of_origin'] = 'India'
        
        # Barcode
        barcode = self._get_barcode(row)
        sku_components.append(barcode[-6:])
        json_breakdown['barcode_id'] = barcode
        
        # Generate SKU
        sku = '-'.join(sku_components)
        json_breakdown['sku'] = sku
        
        return sku, json_breakdown
    
    def generate_textiles_sku(self, row):
        """Generate shortened SKU for textiles with full JSON breakdown."""
        structure = self.sku_structures['textiles']
        sku_components = []
        json_breakdown = {}
        
        # Add prefix
        sku_components.append(structure['prefix'])
        
        # Similar extraction logic for textiles
        # Vendor ID
        vendor = self._get_shortened_code(
            row.get('supplier_ref', row.get('supplier_name', '')), 
            'textiles', 'vendor_id'
        )
        sku_components.append(vendor)
        json_breakdown['vendor_id'] = vendor
        
        # Brand
        brand = self._get_shortened_code(row.get('brand', ''), 'textiles', 'brand')
        sku_components.append(brand)
        json_breakdown['brand'] = row.get('brand', 'Generic')
        
        # Subcategory
        desc = str(row.get('description', '')).lower()
        subcat_full = self._detect_textile_type(desc)
        subcat = self._get_shortened_code(subcat_full, 'textiles', 'subcategory')
        sku_components.append(subcat)
        json_breakdown['subcategory'] = subcat_full
        
        # Gender
        gender_full = self._detect_gender(row)
        gender = self._get_shortened_code(gender_full, 'textiles', 'gender')
        sku_components.append(gender)
        json_breakdown['gender'] = gender_full
        
        # Age Group
        age_group = 'A'  # Adult default
        sku_components.append(age_group)
        json_breakdown['age_group'] = 'Adult'
        
        # Material
        material_full = self._detect_textile_material(row)
        material = self._get_shortened_code(material_full, 'textiles', 'material')
        sku_components.append(material)
        json_breakdown['material'] = material_full
        
        # Pattern
        pattern_full = self._detect_pattern(row)
        pattern = self._get_shortened_code(pattern_full, 'textiles', 'pattern')
        sku_components.append(pattern)
        json_breakdown['pattern'] = pattern_full
        
        # Color
        color_full = self._extract_color(row)
        color = self._get_shortened_code(color_full, 'textiles', 'color')
        sku_components.append(color)
        json_breakdown['color'] = color_full
        
        # Size
        size = self._extract_textile_size(row)
        sku_components.append(size)
        json_breakdown['size'] = size
        
        # Length
        length = self._extract_dimension(row, 'length')
        sku_components.append(length[:4])
        json_breakdown['length'] = length
        
        # Width
        width = self._extract_dimension(row, 'width')
        sku_components.append(width[:4])
        json_breakdown['width'] = width
        
        # Thread Count
        thread_count = '200'  # Default
        sku_components.append(thread_count)
        json_breakdown['thread_count'] = f"{thread_count}TC"
        
        # Weave
        weave = 'PLN'  # Plain default
        sku_components.append(weave)
        json_breakdown['weave'] = 'Plain'
        
        # Occasion
        occasion_full = 'Casual'
        occasion = self._get_shortened_code(occasion_full, 'textiles', 'occasion')
        sku_components.append(occasion)
        json_breakdown['occasion'] = occasion_full
        
        # Style
        style_full = 'Casual'
        style = self._get_shortened_code(style_full, 'textiles', 'style')
        sku_components.append(style)
        json_breakdown['style'] = style_full
        
        # Season
        season = 'S25'  # Summer 2025
        sku_components.append(season)
        json_breakdown['season'] = 'Summer2025'
        
        # Collection
        collection = 'GEN'
        sku_components.append(collection)
        json_breakdown['collection'] = 'General'
        
        # Year
        year = datetime.now().year
        sku_components.append(str(year)[-2:])
        json_breakdown['year'] = year
        
        # Country
        country = 'IN'
        sku_components.append(country)
        json_breakdown['country_of_origin'] = 'India'
        
        # Barcode
        barcode = self._get_barcode(row)
        sku_components.append(barcode[-6:])
        json_breakdown['barcode_id'] = barcode
        
        # Generate SKU
        sku = '-'.join(sku_components)
        json_breakdown['sku'] = sku
        
        return sku, json_breakdown
    
    def generate_electronics_sku(self, row):
        """Generate shortened SKU for electronics with full JSON breakdown."""
        structure = self.sku_structures['electronics']
        sku_components = []
        json_breakdown = {}
        
        # Add prefix
        sku_components.append(structure['prefix'])
        
        # Vendor ID
        vendor = self._get_shortened_code(
            row.get('supplier_ref', row.get('supplier_name', '')), 
            'electronics', 'vendor_id'
        )
        sku_components.append(vendor)
        json_breakdown['vendor_id'] = vendor
        
        # Brand
        brand = self._get_shortened_code(row.get('brand', ''), 'electronics', 'brand')
        sku_components.append(brand)
        json_breakdown['brand'] = row.get('brand', 'Generic')
        
        # Subcategory
        desc = str(row.get('description', '')).lower()
        subcat_full = self._detect_electronics_type(desc)
        subcat = self._get_shortened_code(subcat_full, 'electronics', 'subcategory')
        sku_components.append(subcat)
        json_breakdown['subcategory'] = subcat_full
        
        # Model
        model = self._extract_model(row)
        sku_components.append(model[:6])
        json_breakdown['model'] = model
        
        # Power Voltage
        voltage = '220V'
        sku_components.append('220')
        json_breakdown['power_voltage'] = voltage
        
        # Wattage
        wattage = '25W'
        sku_components.append('25')
        json_breakdown['wattage'] = wattage
        
        # Connectivity
        connectivity = self._detect_connectivity(row)
        conn_code = self._get_shortened_code(connectivity, 'electronics', 'connectivity')
        sku_components.append(conn_code)
        json_breakdown['connectivity'] = connectivity
        
        # Color
        color_full = self._extract_color(row)
        color = self._get_shortened_code(color_full, 'electronics', 'color')
        sku_components.append(color)
        json_breakdown['color'] = color_full
        
        # Storage
        storage = self._extract_storage(row)
        sku_components.append(storage[:4])
        json_breakdown['memory_storage'] = storage
        
        # RAM
        ram = self._extract_ram(row)
        sku_components.append(ram[:3])
        json_breakdown['ram'] = ram
        
        # Processor
        processor = 'GEN'
        sku_components.append(processor)
        json_breakdown['processor'] = 'Generic'
        
        # Battery
        battery = '4000'
        sku_components.append(battery)
        json_breakdown['battery_capacity'] = f"{battery}mAh"
        
        # Screen Size
        screen = '6.5'
        sku_components.append(screen)
        json_breakdown['screen_size'] = f"{screen}inch"
        
        # Resolution
        resolution = 'FHD'
        sku_components.append(resolution)
        json_breakdown['resolution'] = '1920x1080'
        
        # Features
        features = 'STD'
        sku_components.append(features)
        json_breakdown['features'] = 'Standard'
        
        # Warranty
        warranty = '1Y'
        sku_components.append(warranty)
        json_breakdown['warranty'] = '1Year'
        
        # Year
        year = datetime.now().year
        sku_components.append(str(year)[-2:])
        json_breakdown['release_year'] = year
        
        # Country
        country = 'CN'
        sku_components.append(country)
        json_breakdown['country_of_origin'] = 'China'
        
        # Barcode
        barcode = self._get_barcode(row)
        sku_components.append(barcode[-6:])
        json_breakdown['barcode_id'] = barcode
        
        # Generate SKU
        sku = '-'.join(sku_components)
        json_breakdown['sku'] = sku
        
        return sku, json_breakdown
    
    def generate_generic_sku(self, row):
        """Generate generic shortened SKU."""
        sku_components = []
        json_breakdown = {}
        
        # Prefix
        sku_components.append('GEN')
        
        # Vendor
        vendor = self._get_shortened_code(
            row.get('supplier_ref', row.get('supplier_name', '')), 
            'generic', 'vendor_id'
        )
        sku_components.append(vendor)
        json_breakdown['vendor_id'] = vendor
        
        # Brand
        brand = self._get_shortened_code(row.get('brand', ''), 'generic', 'brand')
        sku_components.append(brand)
        json_breakdown['brand'] = row.get('brand', 'Generic')
        
        # Category
        category = self._get_shortened_code(row.get('mapped_category', ''), 'generic', 'category')
        sku_components.append(category)
        json_breakdown['category'] = row.get('mapped_category', 'Miscellaneous')
        
        # Type
        type_code = self._get_shortened_code(row.get('subtype', ''), 'generic', 'type')
        sku_components.append(type_code)
        json_breakdown['type'] = row.get('subtype', 'General')
        
        # Year
        year = datetime.now().year
        sku_components.append(str(year)[-2:])
        json_breakdown['year'] = year
        
        # Country
        country = 'IN'
        sku_components.append(country)
        json_breakdown['country_of_origin'] = 'India'
        
        # Unique ID
        unique_id = self._generate_unique_id(row, length=8)
        sku_components.append(unique_id)
        json_breakdown['unique_id'] = unique_id
        
        # Barcode
        barcode = self._get_barcode(row)
        sku_components.append(barcode[-6:])
        json_breakdown['barcode_id'] = barcode
        
        # Generate SKU
        sku = '-'.join(sku_components)
        json_breakdown['sku'] = sku
        
        return sku, json_breakdown
    
    # ==============================================================
    # Helper Methods for Extraction
    # ==============================================================
    
    def _detect_jewellery_type(self, desc):
        """Detect jewellery type from description."""
        types = {
            'ring': 'Ring', 'necklace': 'Necklace', 'bracelet': 'Bracelet',
            'earring': 'Earrings', 'pendant': 'Pendant', 'bangle': 'Bangle',
            'chain': 'Chain', 'anklet': 'Anklet', 'brooch': 'Brooch'
        }
        
        for key, value in types.items():
            if key in desc:
                return value
        return 'Jewellery'
    
    def _detect_textile_type(self, desc):
        """Detect textile type from description."""
        types = {
            'shirt': 'Shirt', 'pant': 'Pants', 't-shirt': 'TShirt',
            'dress': 'Dress', 'suit': 'Suit', 'jeans': 'Jeans',
            'jacket': 'Jacket', 'sweater': 'Sweater', 'skirt': 'Skirt'
        }
        
        for key, value in types.items():
            if key in desc:
                return value
        return 'Apparel'
    
    def _detect_electronics_type(self, desc):
        """Detect electronics type from description."""
        types = {
            'phone': 'Smartphone', 'laptop': 'Laptop', 'tablet': 'Tablet',
            'television': 'Television', 'tv': 'Television', 'camera': 'Camera',
            'headphone': 'Headphone', 'speaker': 'Speaker', 'watch': 'Smartwatch'
        }
        
        for key, value in types.items():
            if key in desc:
                return value
        return 'Electronics'
    
    def _detect_gender(self, row):
        """Detect gender from various fields."""
        desc = str(row.get('description', '')).lower()
        ana1 = str(row.get('ana1', '')).lower()
        
        if any(word in desc + ana1 for word in ['women', 'ladies', 'female']):
            return 'Womens'
        elif any(word in desc + ana1 for word in ['men', 'gents', 'male']):
            return 'Mens'
        elif 'unisex' in desc + ana1:
            return 'Unisex'
        elif 'kid' in desc + ana1 or 'child' in desc + ana1:
            return 'Kids'
        
        return 'Unisex'
    
    def _detect_jewellery_material(self, row):
        """Detect jewellery material."""
        desc = str(row.get('description', '')).lower()
        ana2 = str(row.get('ana2', '')).lower()
        text = desc + ' ' + ana2
        
        if 'gold' in text:
            return 'Gold'
        elif 'silver' in text:
            return 'Silver'
        elif 'platinum' in text:
            return 'Platinum'
        elif 'brass' in text:
            return 'Brass'
        elif 'copper' in text:
            return 'Copper'
        
        return 'Metal'
    
    def _detect_textile_material(self, row):
        """Detect textile material."""
        desc = str(row.get('description', '')).lower()
        material = str(row.get('extracted_material', '')).lower()
        text = desc + ' ' + material
        
        materials = ['cotton', 'linen', 'silk', 'wool', 'polyester', 'denim', 'velvet', 'satin']
        
        for mat in materials:
            if mat in text:
                return mat.capitalize()
        
        return 'Cotton'
    
    def _detect_purity(self, row):
        """Detect purity for jewellery."""
        desc = str(row.get('description', '')).lower()
        ana3 = str(row.get('ana3', '')).lower()
        text = desc + ' ' + ana3
        
        # Look for karat
        karat_match = re.search(r'(\d+)k', text)
        if karat_match:
            return f"{karat_match.group(1)}K"
        
        # Look for silver purity
        if '925' in text:
            return '925'
        elif '999' in text:
            return '999'
        elif '916' in text:
            return '916'
        
        return '18K'
    
    def _detect_stone_type(self, row):
        """Detect stone type for jewellery."""
        desc = str(row.get('description', '')).lower()
        ana4 = str(row.get('ana4', '')).lower()
        text = desc + ' ' + ana4
        
        stones = {
            'diamond': 'Diamond', 'emerald': 'Emerald', 'ruby': 'Ruby',
            'sapphire': 'Sapphire', 'pearl': 'Pearl', 'topaz': 'Topaz',
            'amethyst': 'Amethyst', 'garnet': 'Garnet'
        }
        
        for stone, name in stones.items():
            if stone in text:
                return name
        
        return 'None'
    
    def _detect_stone_color(self, row):
        """Detect stone color."""
        desc = str(row.get('description', '')).lower()
        
        colors = ['white', 'yellow', 'blue', 'green', 'red', 'pink', 'black', 'brown']
        
        for color in colors:
            if color in desc:
                return color.capitalize()
        
        return 'Clear'
    
    def _extract_carat(self, row):
        """Extract carat weight."""
        desc = str(row.get('description', '')).lower()
        
        carat_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ct|carat)', desc)
        if carat_match:
            return carat_match.group(1)
        
        return '0.5'
    
    def _detect_design(self, row):
        """Detect design type."""
        desc = str(row.get('description', '')).lower()
        
        designs = ['traditional', 'modern', 'contemporary', 'floral', 'geometric', 'royal', 'vintage', 'classic']
        
        for design in designs:
            if design in desc:
                return design.capitalize()
        
        return 'Modern'
    
    def _detect_jewellery_type_category(self, row):
        """Detect jewellery type category."""
        desc = str(row.get('description', '')).lower()
        
        if any(word in desc for word in ['engagement', 'solitaire']):
            return 'Engagement'
        elif any(word in desc for word in ['wedding', 'bridal']):
            return 'Wedding'
        elif any(word in desc for word in ['casual', 'daily']):
            return 'Casual'
        elif any(word in desc for word in ['festive', 'festival', 'diwali']):
            return 'Festive'
        elif any(word in desc for word in ['party', 'cocktail']):
            return 'Party'
        
        return 'Casual'
    
    def _detect_occasion(self, row):
        """Detect occasion."""
        return self._detect_jewellery_type_category(row)
    
    def _extract_weight(self, row):
        """Extract weight."""
        weight = row.get('net_weight', row.get('gross_weight', ''))
        
        if weight and not pd.isna(weight):
            try:
                return str(int(float(weight)))
            except:
                pass
        
        desc = str(row.get('description', '')).lower()
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*g(?:ram)?', desc)
        if weight_match:
            return weight_match.group(1)
        
        return '5'
    
    def _extract_size(self, row):
        """Extract size."""
        size = row.get('size', '')
        if size and not pd.isna(size):
            return str(size)
        
        desc = str(row.get('description', '')).lower()
        size_match = re.search(r'size\s*(\d+)', desc)
        if size_match:
            return size_match.group(1)
        
        return 'NA'
    
    def _detect_color_finish(self, row):
        """Detect color finish for jewellery."""
        desc = str(row.get('description', '')).lower()
        
        if 'rose gold' in desc:
            return 'Rose'
        elif 'yellow gold' in desc:
            return 'Yellow'
        elif 'white gold' in desc:
            return 'White'
        
        return 'Yellow'
    
    def _extract_collection(self, row):
        """Extract collection name."""
        collection = row.get('collection', '')
        if collection and not pd.isna(collection):
            return str(collection)
        
        return 'GEN'
    
    def _extract_color(self, row):
        """Extract color."""
        color = row.get('extracted_color', '')
        if color:
            return color.capitalize()
        
        desc = str(row.get('description', '')).lower()
        colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'grey', 'brown', 'pink', 'purple']
        
        for color in colors:
            if color in desc:
                return color.capitalize()
        
        return 'Black'
    
    def _detect_pattern(self, row):
        """Detect pattern for textiles."""
        desc = str(row.get('description', '')).lower()
        
        patterns = ['checked', 'solid', 'striped', 'printed', 'floral', 'geometric', 'plain']
        
        for pattern in patterns:
            if pattern in desc:
                return pattern.capitalize()
        
        return 'Solid'
    
    def _extract_textile_size(self, row):
        """Extract textile size."""
        size = row.get('size', '')
        if size and not pd.isna(size):
            return str(size).upper()
        
        desc = str(row.get('description', '')).lower()
        sizes = ['xs', 's', 'm', 'l', 'xl', 'xxl', '28', '30', '32', '34', '36', '38', '40', '42']
        
        for s in sizes:
            if f' {s} ' in f' {desc} ' or f'size {s}' in desc:
                return s.upper()
        
        return 'M'
    
    def _extract_dimension(self, row, dim_type):
        """Extract dimension."""
        if dim_type == 'length':
            value = row.get('length', '')
        else:
            value = row.get('width', row.get('breadth', ''))
        
        if value and not pd.isna(value):
            try:
                return str(int(float(value)))
            except:
                pass
        
        return 'NA'
    
    def _extract_model(self, row):
        """Extract model name."""
        model = row.get('model', '')
        if model and not pd.isna(model):
            return str(model)[:10]
        
        return 'GEN'
    
    def _detect_connectivity(self, row):
        """Detect connectivity type."""
        desc = str(row.get('description', '')).lower()
        
        if '5g' in desc:
            return '5G'
        elif '4g' in desc:
            return '4G'
        elif 'wifi' in desc or 'wi-fi' in desc:
            return 'WiFi'
        elif 'bluetooth' in desc:
            return 'Bluetooth'
        
        return 'WiFi'
    
    def _extract_storage(self, row):
        """Extract storage capacity."""
        desc = str(row.get('description', '')).lower()
        
        storage_match = re.search(r'(\d+)\s*gb', desc)
        if storage_match:
            return f"{storage_match.group(1)}GB"
        
        return '64GB'
    
    def _extract_ram(self, row):
        """Extract RAM."""
        desc = str(row.get('description', '')).lower()
        
        ram_match = re.search(r'(\d+)\s*gb\s*ram', desc)
        if ram_match:
            return f"{ram_match.group(1)}GB"
        
        return '4GB'
    
    def _get_barcode(self, row):
        """Get or generate barcode."""
        barcode = row.get('barcode', '')
        if barcode and not pd.isna(barcode) and str(barcode) != '' and str(barcode) != 'nan':
            return str(barcode)
        
        # Generate barcode
        return '890' + ''.join([str(hash(str(row.get(col, ''))) % 10) for col in ['product_id', 'description']])[:10]
    
    def _generate_unique_id(self, row, length=6):
        """Generate unique identifier."""
        unique_string = str(row.get('product_id', '')) + \
                       str(row.get('barcode', '')) + \
                       str(row.get('description', '')) + \
                       str(row.get('supplier_ref', ''))
        
        hash_obj = hashlib.md5(unique_string.encode())
        hash_hex = hash_obj.hexdigest()
        
        return hash_hex[:length].upper()
    
    # ==============================================================
    # Main SKU Generation
    # ==============================================================
    
    def generate_sku_for_product(self, row):
        """Route to appropriate SKU generator based on category."""
        category = str(row.get('mapped_category', '')).lower()
        
        if 'jewel' in category:
            return self.generate_jewellery_sku(row)
        elif 'textile' in category or 'apparel' in category or 'clothing' in category:
            return self.generate_textiles_sku(row)
        elif 'electronic' in category:
            return self.generate_electronics_sku(row)
        elif 'watch' in category:
            # Use jewellery SKU structure for watches (can be customized)
            return self.generate_jewellery_sku(row)
        else:
            return self.generate_generic_sku(row)
    
    def generate_all_skus(self):
        """Generate SKUs for all products with JSON breakdowns."""
        print("\n🔧 Generating Enhanced SKUs...")
        
        skus = []
        json_breakdowns = []
        category_counts = {}
        
        for idx, row in tqdm(self.df_mapped.iterrows(), total=len(self.df_mapped), desc="Generating SKUs"):
            sku, breakdown = self.generate_sku_for_product(row)
            skus.append(sku)
            json_breakdowns.append(breakdown)
            
            # Track statistics
            category = row.get('mapped_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Add to dataframe
        self.df_with_sku = self.df_mapped.copy()
        self.df_with_sku['generated_sku'] = skus
        self.df_with_sku['sku_breakdown'] = json_breakdowns
        
        # Add individual breakdown columns
        for idx, breakdown in enumerate(json_breakdowns):
            for key, value in breakdown.items():
                if key != 'sku':
                    col_name = f'sku_{key}'
                    if col_name not in self.df_with_sku.columns:
                        self.df_with_sku[col_name] = ''
                    self.df_with_sku.at[idx, col_name] = value
        
        # Statistics
        self.sku_stats['total_generated'] = len(skus)
        self.sku_stats['unique_skus'] = pd.Series(skus).nunique()
        self.sku_stats['category_breakdown'] = category_counts
        
        print(f"✅ Generated {self.sku_stats['total_generated']:,} SKUs")
        print(f"   Unique SKUs: {self.sku_stats['unique_skus']:,}")
        
        return self.df_with_sku
    
    def save_enhanced_data(self):
        """Save SKU data with JSON breakdowns."""
        print("💾 Saving SKU data...")
        # Fix sku_year dtype issues
        if 'sku_year' in self.df_with_sku.columns:
            self.df_with_sku['sku_year'] = (
                pd.to_numeric(self.df_with_sku['sku_year'], errors='coerce')
                .fillna(datetime.now().year)
                .astype('int64')
            )
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete data as parquet
        parquet_path = os.path.join(self.output_dir, f"sku_data_{date_str}.parquet")
        self.df_with_sku.to_parquet(parquet_path)
        print(f"   ✓ Parquet saved → {parquet_path}")
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"sku_data_{date_str}.csv")
        self.df_with_sku.to_csv(csv_path, index=False)
        print(f"   ✓ CSV saved → {csv_path}")
        
        # Save JSON breakdowns separately
        json_path = os.path.join(self.output_dir, f"sku_breakdowns_{date_str}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.df_with_sku['sku_breakdown'].tolist(), f, ensure_ascii=False, indent=2)
        print(f"   ✓ JSON breakdowns saved → {json_path}")
        
        return parquet_path
    
    def generate_mall_hierarchy_json(self):
        """Generate mall hierarchy with domain-specific SKUs."""
        print("🏢 Generating mall hierarchy with SKUs...")
        
        mall_data = {
            "mall": {
                "mall_id": "MALL001",
                "name": "Prime City Mall",
                "location": "Bengaluru, India",
                "total_products": len(self.df_with_sku),
                "total_suppliers": self.df_with_sku['supplier_name'].nunique(),
                "total_categories": self.df_with_sku['mapped_category'].nunique(),
                "unique_skus": self.sku_stats['unique_skus'],
                "generation_timestamp": datetime.now().isoformat(),
                "stores": []
            }
        }
        
        # Group by supplier/store
        supplier_groups = self.df_with_sku.groupby('supplier_name')
        
        for store_id, (supplier_name, group) in enumerate(supplier_groups, 1):
            # Get store type from most common category
            store_type = group['mapped_category'].mode()[0] if len(group) > 0 else "Mixed"
            
            # Sample SKUs with full breakdown
            sample_skus = []
            for _, row in group.head(10).iterrows():  # Limit to 10 for JSON size
                if 'sku_breakdown' in row and isinstance(row['sku_breakdown'], dict):
                    sample_skus.append(row['sku_breakdown'])
            
            store_entry = {
                "store_id": f"STORE{store_id:03d}",
                "name": supplier_name,
                "type": store_type,
                "total_products": len(group),
                "unique_skus": group['generated_sku'].nunique(),
                "average_confidence": float(group['final_confidence'].mean()),
                "skus": sample_skus
            }
            
            mall_data["mall"]["stores"].append(store_entry)
        
        # Save JSON
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"mall_hierarchy_enhanced_{date_str}.json")
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mall_data, f, ensure_ascii=False, indent=2)
        
        print(f"   ✓ Mall hierarchy saved → {json_path}")
        print(f"   Total stores: {len(mall_data['mall']['stores'])}")
        
        return json_path
    
    def run(self):
        """Execute enhanced SKU building pipeline."""
        print("\n" + "=" * 60)
        print("🏗️ ENHANCED SKU BUILDING PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            self.load_mapped_data()
            
            # Generate enhanced SKUs
            self.generate_all_skus()
            
            # Save data
            parquet_path = self.save_enhanced_data()
            
            # Generate mall hierarchy
            json_path = self.generate_mall_hierarchy_json()
            
            print("\n" + "=" * 60)
            print("✅ ENHANCED SKU BUILDING COMPLETE")
            print("=" * 60)
            print(f"📊 Total SKUs: {self.sku_stats['total_generated']:,}")
            print(f"🔑 Unique SKUs: {self.sku_stats['unique_skus']:,}")
            print("=" * 60)
            
            return {
                'data_file': parquet_path,
                'mall_json': json_path,
                'statistics': self.sku_stats
            }
            
        except Exception as e:
            print(f"\n❌ Enhanced SKU Building failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "data/processed"
    OUTPUT_DIR = "data/processed"
    
    # Initialize and run
    builder = SKUBuilder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
    
    try:
        results = builder.run()
        print(f"\n🎉 Enhanced SKU Building successful!")
        print(f"📁 Data: {results['data_file']}")
        print(f"📁 Mall JSON: {results['mall_json']}")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
import os
import json
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StatisticsGenerationAgent:
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/reports"):
        """
        Initialize the Statistics Generation Agent.
        
        Args:
            input_dir: Directory containing final SKU data
            output_dir: Directory to save reports and visualizations
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.df_final = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize statistics container
        self.statistics = {}
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def find_latest_sku_data(self):
        """Find the latest SKU data file."""
        print("🔍 Looking for latest SKU data file...")
        
        patterns = [
            os.path.join(self.input_dir, "final_with_sku_*.parquet"),
            os.path.join(self.input_dir, "*sku*.parquet"),
            os.path.join(self.input_dir, "*.parquet")
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                print(f"📁 Found: {os.path.basename(latest_file)}")
                return latest_file
        
        raise FileNotFoundError(f"❌ No SKU data files found in {self.input_dir}")
    
    def load_final_data(self):
        """Load the final SKU data."""
        input_file = self.find_latest_sku_data()
        
        print(f"📂 Loading final data from: {input_file}")
        
        try:
            self.df_final = pd.read_parquet(input_file)
            print(f"✅ Loaded {len(self.df_final):,} products")
            print(f"📊 Columns available: {len(self.df_final.columns)}")
            
            return self.df_final
            
        except Exception as e:
            raise Exception(f"❌ Failed to load data: {str(e)}")
    
    def _find_column(self, possible_names):
        """Find the first matching column name."""
        if self.df_final is None:
            return None
        
        for name in possible_names:
            if name in self.df_final.columns:
                return name
        return None
    
    # ==============================================================
    # Statistical Analysis Functions
    # ==============================================================
    
    def generate_basic_statistics(self):
        """Generate basic dataset statistics."""
        print("\n📊 Generating Basic Statistics...")
        
        basic_stats = {
            'total_products': int(len(self.df_final)),
            'total_columns': int(len(self.df_final.columns)),
            'memory_usage_mb': float(self.df_final.memory_usage(deep=True).sum() / 1024 / 1024),
            'missing_values_total': int(self.df_final.isnull().sum().sum()),
            'missing_values_percentage': float(
                self.df_final.isnull().sum().sum() / (len(self.df_final) * len(self.df_final.columns)) * 100
            )
        }
        
        # Data types distribution
        dtype_counts = self.df_final.dtypes.value_counts()
        basic_stats['data_types'] = {str(k): int(v) for k, v in dtype_counts.items()}
        
        self.statistics['basic'] = basic_stats
        
        print(f"   Total Products: {basic_stats['total_products']:,}")
        print(f"   Missing Values: {basic_stats['missing_values_percentage']:.1f}%")
        
        return basic_stats
    
    def generate_supplier_analysis(self):
        """Analyze supplier/store distribution."""
        print("\n🏪 Analyzing Suppliers...")
        
        supplier_col = self._find_column(['supplier_name', 'supplier'])
        if not supplier_col:
            return {}
        
        supplier_stats = {}
        
        # Basic counts
        supplier_counts = self.df_final[supplier_col].value_counts()
        supplier_stats['total_suppliers'] = int(len(supplier_counts))
        supplier_stats['average_products_per_supplier'] = float(supplier_counts.mean())
        supplier_stats['median_products_per_supplier'] = float(supplier_counts.median())
        supplier_stats['std_products_per_supplier'] = float(supplier_counts.std())
        
        # Top suppliers
        supplier_stats['top_10_suppliers'] = {
            str(k): int(v) for k, v in supplier_counts.head(10).items()
        }
        
        # Supplier size distribution
        supplier_stats['size_distribution'] = {
            'large_suppliers_100plus': int((supplier_counts >= 100).sum()),
            'medium_suppliers_50_99': int(((supplier_counts >= 50) & (supplier_counts < 100)).sum()),
            'small_suppliers_10_49': int(((supplier_counts >= 10) & (supplier_counts < 50)).sum()),
            'tiny_suppliers_below10': int((supplier_counts < 10).sum())
        }
        
        # Concentration metrics
        top_3_share = supplier_counts.head(3).sum() / len(self.df_final) * 100
        top_10_share = supplier_counts.head(10).sum() / len(self.df_final) * 100
        supplier_stats['concentration'] = {
            'top3_market_share': float(top_3_share),
            'top10_market_share': float(top_10_share),
            'herfindahl_index': float((supplier_counts / len(self.df_final) ** 2).sum())
        }
        
        self.statistics['suppliers'] = supplier_stats
        
        print(f"   Total Suppliers: {supplier_stats['total_suppliers']}")
        print(f"   Top 10 Market Share: {supplier_stats['concentration']['top10_market_share']:.1f}%")
        
        return supplier_stats
    
    def generate_category_analysis(self):
        """Analyze product category distribution."""
        print("\n🏷️ Analyzing Categories...")
        
        category_col = self._find_column(['mapped_category', 'category'])
        if not category_col:
            return {}
        
        category_stats = {}
        
        # Basic counts
        category_counts = self.df_final[category_col].value_counts()
        category_stats['total_categories'] = int(len(category_counts))
        category_stats['category_distribution'] = {
            str(k): int(v) for k, v in category_counts.items()
        }
        
        # Category percentages
        category_stats['category_percentages'] = {
            str(k): float(v / len(self.df_final) * 100) 
            for k, v in category_counts.items()
        }
        
        # Uncategorized products
        uncategorized_count = (
            (self.df_final[category_col] == 'Uncategorized') | 
            (self.df_final[category_col].isna())
        ).sum()
        category_stats['uncategorized_products'] = int(uncategorized_count)
        category_stats['uncategorized_percentage'] = float(uncategorized_count / len(self.df_final) * 100)
        
        # Category diversity metrics
        category_stats['diversity_metrics'] = {
            'shannon_entropy': float(-sum([
                (c/len(self.df_final)) * np.log(c/len(self.df_final)) 
                for c in category_counts.values if c > 0
            ])),
            'simpson_index': float(sum([(c/len(self.df_final))**2 for c in category_counts.values]))
        }
        
        self.statistics['categories'] = category_stats
        
        print(f"   Total Categories: {category_stats['total_categories']}")
        print(f"   Uncategorized: {category_stats['uncategorized_percentage']:.1f}%")
        
        return category_stats
    
    def generate_brand_analysis(self):
        """Analyze brand distribution."""
        print("\n🏢 Analyzing Brands...")
        
        brand_col = self._find_column(['brand'])
        if not brand_col:
            return {}
        
        brand_stats = {}
        
        # Clean brand data
        brand_series = self.df_final[brand_col].fillna('Generic').str.strip()
        brand_counts = brand_series.value_counts()
        
        brand_stats['total_brands'] = int(len(brand_counts))
        brand_stats['top_20_brands'] = {
            str(k): int(v) for k, v in brand_counts.head(20).items()
        }
        
        # Brand concentration
        brand_stats['brand_metrics'] = {
            'single_product_brands': int((brand_counts == 1).sum()),
            'brands_with_10plus_products': int((brand_counts >= 10).sum()),
            'generic_products': int((brand_series == 'Generic').sum()),
            'average_products_per_brand': float(brand_counts.mean())
        }
        
        self.statistics['brands'] = brand_stats
        
        print(f"   Total Brands: {brand_stats['total_brands']}")
        print(f"   Single-Product Brands: {brand_stats['brand_metrics']['single_product_brands']}")
        
        return brand_stats
    
    def generate_sku_analysis(self):
        """Analyze SKU generation results."""
        print("\n🔑 Analyzing SKUs...")
        
        sku_col = self._find_column(['generated_sku', 'sku'])
        if not sku_col:
            return {}
        
        sku_stats = {}
        
        sku_series = self.df_final[sku_col].dropna()
        sku_stats['total_skus_generated'] = int(len(sku_series))
        sku_stats['unique_skus'] = int(sku_series.nunique())
        sku_stats['duplicate_skus'] = int(len(sku_series) - sku_series.nunique())
        sku_stats['sku_generation_rate'] = float(len(sku_series) / len(self.df_final) * 100)
        sku_stats['uniqueness_ratio'] = float(
            sku_series.nunique() / len(sku_series) * 100
        ) if len(sku_series) > 0 else 0
        
        # SKU format analysis
        if 'sku_category' in self.df_final.columns:
            sku_categories = self.df_final['sku_category'].value_counts()
            sku_stats['sku_categories'] = {
                str(k): int(v) for k, v in sku_categories.head(10).items()
            }
        
        # SKU length distribution
        sku_lengths = sku_series.str.len()
        sku_stats['sku_length_stats'] = {
            'min_length': int(sku_lengths.min()) if not sku_lengths.empty else 0,
            'max_length': int(sku_lengths.max()) if not sku_lengths.empty else 0,
            'avg_length': float(sku_lengths.mean()) if not sku_lengths.empty else 0,
            'std_length': float(sku_lengths.std()) if not sku_lengths.empty else 0
        }
        
        self.statistics['skus'] = sku_stats
        
        print(f"   Unique SKUs: {sku_stats['unique_skus']:,}")
        print(f"   SKU Generation Rate: {sku_stats['sku_generation_rate']:.1f}%")
        
        return sku_stats
    
    def generate_mapping_quality_analysis(self):
        """Analyze mapping confidence and quality."""
        print("\n🎯 Analyzing Mapping Quality...")
        
        confidence_col = self._find_column(['final_confidence', 'confidence'])
        if not confidence_col:
            return {}
        
        quality_stats = {}
        
        confidence_series = self.df_final[confidence_col].dropna()
        
        # Confidence statistics
        quality_stats['confidence_statistics'] = {
            'mean': float(confidence_series.mean()),
            'median': float(confidence_series.median()),
            'std': float(confidence_series.std()),
            'min': float(confidence_series.min()),
            'max': float(confidence_series.max()),
            'q25': float(confidence_series.quantile(0.25)),
            'q75': float(confidence_series.quantile(0.75))
        }
        
        # Confidence distribution
        quality_stats['confidence_distribution'] = {
            'auto_approved_90plus': int((confidence_series >= 90).sum()),
            'review_recommended_70_89': int(
                ((confidence_series >= 70) & (confidence_series < 90)).sum()
            ),
            'manual_review_50_69': int(
                ((confidence_series >= 50) & (confidence_series < 70)).sum()
            ),
            'low_confidence_below50': int((confidence_series < 50).sum())
        }
        
        # Calculate percentages
        total_with_confidence = len(confidence_series)
        quality_stats['confidence_percentages'] = {
            'auto_approval_rate': float(
                quality_stats['confidence_distribution']['auto_approved_90plus'] / 
                total_with_confidence * 100
            ) if total_with_confidence > 0 else 0,
            'review_rate': float(
                quality_stats['confidence_distribution']['review_recommended_70_89'] / 
                total_with_confidence * 100
            ) if total_with_confidence > 0 else 0,
            'manual_review_rate': float(
                (quality_stats['confidence_distribution']['manual_review_50_69'] +
                 quality_stats['confidence_distribution']['low_confidence_below50']) / 
                total_with_confidence * 100
            ) if total_with_confidence > 0 else 0
        }
        
        # Method performance
        if 'confidence_deterministic' in self.df_final.columns:
            quality_stats['method_performance'] = {
                'deterministic_avg': float(self.df_final['confidence_deterministic'].mean()),
                'fuzzy_avg': float(self.df_final['confidence_fuzzy'].mean()) 
                    if 'confidence_fuzzy' in self.df_final.columns else 0,
                'ml_avg': float(self.df_final['confidence_ml'].mean())
                    if 'confidence_ml' in self.df_final.columns else 0
            }
        
        self.statistics['mapping_quality'] = quality_stats
        
        print(f"   Average Confidence: {quality_stats['confidence_statistics']['mean']:.1f}%")
        print(f"   Auto-Approval Rate: {quality_stats['confidence_percentages']['auto_approval_rate']:.1f}%")
        
        return quality_stats
    
    def generate_price_analysis(self):
        """Analyze pricing data."""
        print("\n💰 Analyzing Pricing...")
        
        price_col = self._find_column(['price_retail', 'price', 'retail'])
        if not price_col:
            return {}
        
        price_stats = {}
        
        # Clean price data
        price_series = pd.to_numeric(self.df_final[price_col], errors='coerce').dropna()
        price_series = price_series[price_series > 0]  # Remove zero/negative prices
        
        if not price_series.empty:
            price_stats['price_statistics'] = {
                'count': int(len(price_series)),
                'mean': float(price_series.mean()),
                'median': float(price_series.median()),
                'std': float(price_series.std()),
                'min': float(price_series.min()),
                'max': float(price_series.max()),
                'q25': float(price_series.quantile(0.25)),
                'q75': float(price_series.quantile(0.75))
            }
            
            # Price ranges
            price_stats['price_ranges'] = {
                'below_100': int((price_series < 100).sum()),
                '100_500': int(((price_series >= 100) & (price_series < 500)).sum()),
                '500_1000': int(((price_series >= 500) & (price_series < 1000)).sum()),
                '1000_5000': int(((price_series >= 1000) & (price_series < 5000)).sum()),
                '5000_10000': int(((price_series >= 5000) & (price_series < 10000)).sum()),
                'above_10000': int((price_series >= 10000).sum())
            }
            
            # Price by category if available
            category_col = self._find_column(['mapped_category', 'category'])
            if category_col:
                category_prices = []
                for category in self.df_final[category_col].unique():
                    if pd.notna(category):
                        cat_prices = price_series[self.df_final[category_col] == category]
                        if len(cat_prices) > 0:
                            category_prices.append({
                                'category': str(category),
                                'avg_price': float(cat_prices.mean()),
                                'median_price': float(cat_prices.median()),
                                'count': int(len(cat_prices))
                            })
                
                # Sort by average price and take top 10
                category_prices.sort(key=lambda x: x['avg_price'], reverse=True)
                price_stats['price_by_category'] = category_prices[:10]
        
        self.statistics['pricing'] = price_stats
        
        if price_stats:
            print(f"   Products with Price: {price_stats['price_statistics']['count']:,}")
            print(f"   Average Price: ₹{price_stats['price_statistics']['mean']:.2f}")
        
        return price_stats
    
    def generate_inventory_analysis(self):
        """Analyze inventory/stock data."""
        print("\n📦 Analyzing Inventory...")
        
        stock_col = self._find_column(['stock_qty', 'qty_in_stock', 'stock'])
        if not stock_col:
            return {}
        
        inventory_stats = {}
        
        # Clean stock data
        stock_series = pd.to_numeric(self.df_final[stock_col], errors='coerce').dropna()
        stock_series = stock_series[stock_series >= 0]  # Remove negative stock
        
        if not stock_series.empty:
            inventory_stats['stock_statistics'] = {
                'total_stock': int(stock_series.sum()),
                'products_in_stock': int((stock_series > 0).sum()),
                'products_out_of_stock': int((stock_series == 0).sum()),
                'avg_stock_per_product': float(stock_series.mean()),
                'median_stock': float(stock_series.median()),
                'max_stock': int(stock_series.max())
            }
            
            # Stock levels
            inventory_stats['stock_levels'] = {
                'out_of_stock': int((stock_series == 0).sum()),
                'low_stock_1_10': int(((stock_series > 0) & (stock_series <= 10)).sum()),
                'medium_stock_11_50': int(((stock_series > 10) & (stock_series <= 50)).sum()),
                'high_stock_51_100': int(((stock_series > 50) & (stock_series <= 100)).sum()),
                'very_high_stock_above100': int((stock_series > 100).sum())
            }
            
            # Stock value if price available
            price_col = self._find_column(['price_retail', 'price', 'retail'])
            if price_col:
                df_with_both = self.df_final[[stock_col, price_col]].copy()
                df_with_both['stock_value'] = pd.to_numeric(df_with_both[stock_col], errors='coerce') * \
                                              pd.to_numeric(df_with_both[price_col], errors='coerce')
                total_value = df_with_both['stock_value'].sum()
                inventory_stats['stock_value'] = {
                    'total_inventory_value': float(total_value) if not pd.isna(total_value) else 0
                }
        
        self.statistics['inventory'] = inventory_stats
        
        if inventory_stats:
            print(f"   Total Stock: {inventory_stats['stock_statistics']['total_stock']:,}")
            print(f"   Products in Stock: {inventory_stats['stock_statistics']['products_in_stock']:,}")
        
        return inventory_stats
    
    def generate_data_quality_report(self):
        """Generate data quality metrics."""
        print("\n🔍 Analyzing Data Quality...")
        
        quality_report = {}
        
        # Completeness by column
        completeness = (1 - self.df_final.isnull().mean()) * 100
        quality_report['column_completeness'] = {
            col: float(comp) for col, comp in completeness.items()
        }
        
        # Critical fields completeness
        critical_fields = [
            'supplier_name', 'mapped_category', 'generated_sku',
            'brand', 'description', 'mapped_hsn'
        ]
        
        critical_completeness = {}
        for field in critical_fields:
            if field in self.df_final.columns:
                critical_completeness[field] = float(
                    (self.df_final[field].notna() & (self.df_final[field] != '')).mean() * 100
                )
        
        quality_report['critical_fields_completeness'] = critical_completeness
        
        # Overall quality score
        if critical_completeness:
            quality_report['overall_quality_score'] = float(np.mean(list(critical_completeness.values())))
        else:
            quality_report['overall_quality_score'] = 0.0
        
        self.statistics['data_quality'] = quality_report
        
        print(f"   Overall Quality Score: {quality_report['overall_quality_score']:.1f}%")
        
        return quality_report
    
    def generate_business_insights(self):
        """Generate actionable business insights."""
        print("\n💡 Generating Business Insights...")
        
        insights = []
        recommendations = []
        
        # Supplier concentration insights
        if 'suppliers' in self.statistics:
            top10_share = self.statistics['suppliers']['concentration']['top10_market_share']
            if top10_share > 60:
                insights.append(f"High supplier concentration: Top 10 suppliers control {top10_share:.1f}% of inventory")
                recommendations.append("Consider diversifying supplier base to reduce dependency risk")
        
        # Category insights
        if 'categories' in self.statistics:
            uncategorized_pct = self.statistics['categories']['uncategorized_percentage']
            if uncategorized_pct > 10:
                insights.append(f"Product categorization needs improvement: {uncategorized_pct:.1f}% uncategorized")
                recommendations.append("Implement better categorization rules or manual review process")
        
        # Mapping quality insights
        if 'mapping_quality' in self.statistics:
            auto_approval = self.statistics['mapping_quality']['confidence_percentages']['auto_approval_rate']
            if auto_approval < 80:
                insights.append(f"Mapping confidence below target: Only {auto_approval:.1f}% auto-approved")
                recommendations.append("Consider training ML models with more data for better accuracy")
            elif auto_approval > 95:
                insights.append(f"Excellent mapping performance: {auto_approval:.1f}% auto-approval rate")
        
        # Inventory insights
        if 'inventory' in self.statistics:
            out_of_stock = self.statistics['inventory']['stock_statistics']['products_out_of_stock']
            total_products = self.statistics['basic']['total_products']
            oos_percentage = (out_of_stock / total_products) * 100
            if oos_percentage > 20:
                insights.append(f"High out-of-stock rate: {oos_percentage:.1f}% products have zero stock")
                recommendations.append("Review inventory management and reorder points")
        
        # SKU insights
        if 'skus' in self.statistics:
            uniqueness = self.statistics['skus']['uniqueness_ratio']
            if uniqueness < 99:
                insights.append(f"SKU duplication detected: {100-uniqueness:.2f}% duplicate SKUs")
                recommendations.append("Review SKU generation logic to ensure uniqueness")
        
        # Price insights
        if 'pricing' in self.statistics:
            price_count = self.statistics['pricing']['price_statistics']['count']
            total_products = self.statistics['basic']['total_products']
            price_coverage = (price_count / total_products) * 100
            if price_coverage < 80:
                insights.append(f"Incomplete pricing data: Only {price_coverage:.1f}% products have prices")
                recommendations.append("Update pricing information for complete catalog")
        
        business_insights = {
            'insights': insights,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        self.statistics['business_insights'] = business_insights
        
        print(f"   Generated {len(insights)} insights")
        print(f"   Generated {len(recommendations)} recommendations")
        
        return business_insights
    
    # ==============================================================
    # Report Generation Functions
    # ==============================================================
    
    def save_statistics_reports(self):
        """Save all statistics to various formats."""
        print("\n💾 Saving Reports...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON report
        json_path = os.path.join(self.output_dir, f"statistics_report_{timestamp}.json")
        json_data = self._convert_to_json_serializable(self.statistics)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"   ✓ JSON report saved → {json_path}")
        
        # Save summary CSV
        summary_data = self._create_summary_dataframe()
        csv_path = os.path.join(self.output_dir, f"statistics_summary_{timestamp}.csv")
        summary_data.to_csv(csv_path, index=False)
        print(f"   ✓ Summary CSV saved → {csv_path}")
        
        # Save detailed Excel report with multiple sheets
        excel_path = os.path.join(self.output_dir, f"detailed_report_{timestamp}.xlsx")
        self._save_excel_report(excel_path)
        print(f"   ✓ Excel report saved → {excel_path}")
        
        # Save insights text file
        insights_path = os.path.join(self.output_dir, f"business_insights_{timestamp}.txt")
        self._save_insights_text(insights_path)
        print(f"   ✓ Insights saved → {insights_path}")
        
        return {
            'json': json_path,
            'csv': csv_path,
            'excel': excel_path,
            'insights': insights_path
        }
    
    def _create_summary_dataframe(self):
        """Create summary dataframe for CSV export."""
        summary_rows = []
        
        # Basic metrics
        if 'basic' in self.statistics:
            for key, value in self.statistics['basic'].items():
                if not isinstance(value, dict):
                    summary_rows.append({
                        'Category': 'Basic',
                        'Metric': key,
                        'Value': value
                    })
        
        # Supplier metrics
        if 'suppliers' in self.statistics:
            summary_rows.append({
                'Category': 'Suppliers',
                'Metric': 'Total Suppliers',
                'Value': self.statistics['suppliers']['total_suppliers']
            })
            summary_rows.append({
                'Category': 'Suppliers',
                'Metric': 'Average Products per Supplier',
                'Value': self.statistics['suppliers']['average_products_per_supplier']
            })
        
        # Category metrics
        if 'categories' in self.statistics:
            summary_rows.append({
                'Category': 'Categories',
                'Metric': 'Total Categories',
                'Value': self.statistics['categories']['total_categories']
            })
            summary_rows.append({
                'Category': 'Categories',
                'Metric': 'Uncategorized Percentage',
                'Value': self.statistics['categories']['uncategorized_percentage']
            })
        
        # SKU metrics
        if 'skus' in self.statistics:
            summary_rows.append({
                'Category': 'SKUs',
                'Metric': 'Unique SKUs',
                'Value': self.statistics['skus']['unique_skus']
            })
            summary_rows.append({
                'Category': 'SKUs',
                'Metric': 'SKU Generation Rate',
                'Value': self.statistics['skus']['sku_generation_rate']
            })
        
        # Quality metrics
        if 'mapping_quality' in self.statistics:
            summary_rows.append({
                'Category': 'Mapping Quality',
                'Metric': 'Average Confidence',
                'Value': self.statistics['mapping_quality']['confidence_statistics']['mean']
            })
            summary_rows.append({
                'Category': 'Mapping Quality',
                'Metric': 'Auto-Approval Rate',
                'Value': self.statistics['mapping_quality']['confidence_percentages']['auto_approval_rate']
            })
        
        return pd.DataFrame(summary_rows)
    
    def _save_excel_report(self, filepath):
        """Save detailed Excel report with multiple sheets."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self._create_summary_dataframe()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Top suppliers
            if 'suppliers' in self.statistics:
                suppliers_df = pd.DataFrame([
                    {'Supplier': k, 'Products': v}
                    for k, v in self.statistics['suppliers']['top_10_suppliers'].items()
                ])
                suppliers_df.to_excel(writer, sheet_name='Top Suppliers', index=False)
            
            # Category distribution
            if 'categories' in self.statistics:
                categories_df = pd.DataFrame([
                    {'Category': k, 'Products': v, 'Percentage': self.statistics['categories']['category_percentages'][k]}
                    for k, v in self.statistics['categories']['category_distribution'].items()
                ])
                categories_df.to_excel(writer, sheet_name='Categories', index=False)
            
            # Top brands
            if 'brands' in self.statistics:
                brands_df = pd.DataFrame([
                    {'Brand': k, 'Products': v}
                    for k, v in list(self.statistics['brands']['top_20_brands'].items())[:20]
                ])
                brands_df.to_excel(writer, sheet_name='Top Brands', index=False)
    
    def _save_insights_text(self, filepath):
        """Save business insights as readable text."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BUSINESS INSIGHTS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            if 'business_insights' in self.statistics:
                f.write("KEY INSIGHTS:\n")
                f.write("-" * 40 + "\n")
                for i, insight in enumerate(self.statistics['business_insights']['insights'], 1):
                    f.write(f"{i}. {insight}\n")
                
                f.write("\n" + "RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(self.statistics['business_insights']['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "=" * 60 + "\n")
    
    def display_key_insights(self):
        """Display key insights in console."""
        print("\n" + "=" * 60)
        print("📈 KEY BUSINESS METRICS")
        print("=" * 60)
        
        # Basic metrics
        if 'basic' in self.statistics:
            print(f"📦 Total Products: {self.statistics['basic']['total_products']:,}")
        
        # Supplier metrics
        if 'suppliers' in self.statistics:
            print(f"🏪 Total Suppliers: {self.statistics['suppliers']['total_suppliers']}")
            print(f"   Top 10 Market Share: {self.statistics['suppliers']['concentration']['top10_market_share']:.1f}%")
        
        # Category metrics
        if 'categories' in self.statistics:
            print(f"🏷️ Total Categories: {self.statistics['categories']['total_categories']}")
            print(f"   Uncategorized: {self.statistics['categories']['uncategorized_percentage']:.1f}%")
        
        # SKU metrics
        if 'skus' in self.statistics:
            print(f"🔑 Unique SKUs: {self.statistics['skus']['unique_skus']:,}")
            print(f"   Generation Rate: {self.statistics['skus']['sku_generation_rate']:.1f}%")
        
        # Quality metrics
        if 'mapping_quality' in self.statistics:
            print(f"🎯 Mapping Confidence: {self.statistics['mapping_quality']['confidence_statistics']['mean']:.1f}%")
            print(f"   Auto-Approval: {self.statistics['mapping_quality']['confidence_percentages']['auto_approval_rate']:.1f}%")
        
        # Business insights
        if 'business_insights' in self.statistics:
            insights = self.statistics['business_insights']['insights']
            if insights:
                print("\n💡 TOP INSIGHTS:")
                for insight in insights[:3]:
                    print(f"   • {insight}")
        
        print("=" * 60)
    
    def run(self):
        """Execute complete statistics generation pipeline."""
        print("\n" + "=" * 60)
        print("📊 STATISTICS GENERATION PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            self.load_final_data()
            
            # Generate all statistics
            self.generate_basic_statistics()
            self.generate_supplier_analysis()
            self.generate_category_analysis()
            self.generate_brand_analysis()
            self.generate_sku_analysis()
            self.generate_mapping_quality_analysis()
            self.generate_price_analysis()
            self.generate_inventory_analysis()
            self.generate_data_quality_report()
            self.generate_business_insights()
            
            # Save reports
            report_files = self.save_statistics_reports()
            
            # Display insights
            self.display_key_insights()
            
            print("\n" + "=" * 60)
            print("✅ STATISTICS GENERATION COMPLETE")
            print("=" * 60)
            
            return {
                'statistics': self.statistics,
                'report_files': report_files
            }
            
        except Exception as e:
            print(f"\n❌ Statistics generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "data/processed"
    OUTPUT_DIR = "data/reports"
    
    # Initialize and run
    agent = StatisticsGenerationAgent(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
    
    try:
        results = agent.run()
        print(f"\n🎉 Statistics Generation successful!")
        print(f"📊 Reports saved to: {OUTPUT_DIR}")
        print(f"✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Statistics Generation failed: {e}")
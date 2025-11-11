import pandas as pd
import numpy as np
import os
import json
import re
import glob
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from rapidfuzz import fuzz, process
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================
# Mapping Engine Agent with ML Integration
# ==============================================================

class MappingEngine:
    def __init__(self, input_path: str = None, input_dir: str = "data/processed", 
                 output_dir: str = "data/processed", model_dir: str = "models"):
        self.input_path = input_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.df_raw = None
        self.df_mapped = None
        
        # Initialize enhanced modules
        self.deterministic_mapper = DeterministicMapper()
        self.fuzzy_matcher = FuzzyMatcher()
        self.ml_classifier = AdvancedMLClassifier(model_dir=model_dir)
        self.confidence_combiner = ConfidenceCombiner()
        self.review_tagger = ReviewTagger()
        
        # Create directories
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    def find_latest_processed_data(self):
        """Find the latest processed data file."""
        print("🔍 Looking for latest processed data file...")
        
        patterns = [
            os.path.join(self.input_dir, "processed_*.parquet"),
            os.path.join(self.input_dir, "*process*.parquet"),
            os.path.join(self.input_dir, "*.parquet")
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                print(f"📁 Found: {os.path.basename(latest_file)}")
                return latest_file
        
        raise FileNotFoundError(f"❌ No processed data files found in {self.input_dir}")
    
    def load_processed_data(self):
        """Load processed data with auto-discovery support."""
        if self.input_path and os.path.exists(self.input_path):
            input_file = self.input_path
        else:
            input_file = self.find_latest_processed_data()
            self.input_path = input_file
        
        print(f"📂 Loading data from: {input_file}")
        
        if input_file.endswith('.parquet'):
            self.df_raw = pd.read_parquet(input_file)
        elif input_file.endswith('.csv'):
            self.df_raw = pd.read_csv(input_file, low_memory=False)
        else:
            raise ValueError("❌ Unsupported file format")
        
        print(f"✅ Loaded {len(self.df_raw):,} rows and {len(self.df_raw.columns)} columns")
        return self.df_raw
    
    def run_mapping_pipeline(self):
        """Execute the enhanced 5-module mapping pipeline."""
        print("\n🚀 Starting Enhanced 5-Module Mapping Pipeline...")
        print("=" * 60)
        
        # Prepare data
        descriptions = self.df_raw['description'].fillna('').tolist()
        existing_categories = self.df_raw['category'].fillna('').tolist()
        brands = self.df_raw['brand'].fillna('').tolist()
        
        # Module 1: Deterministic Mapping
        print("\n1️⃣ DETERMINISTIC MAPPING")
        print("-" * 40)
        det_results = self.deterministic_mapper.map_categories(descriptions, existing_categories)
        det_confidence = np.mean([r['confidence'] for r in det_results])
        print(f"   Average Confidence: {det_confidence:.1f}%")
        
        # Module 2: Fuzzy Matching
        print("\n2️⃣ FUZZY MATCHING")
        print("-" * 40)
        fuzzy_results = self.fuzzy_matcher.match_categories(descriptions, existing_categories)
        fuzzy_confidence = np.mean([r['confidence'] for r in fuzzy_results])
        print(f"   Average Confidence: {fuzzy_confidence:.1f}%")
        
        # Module 3: ML Classification (Enhanced)
        print("\n3️⃣ ML CLASSIFICATION (with BERT embeddings)")
        print("-" * 40)
        ml_results = self.ml_classifier.predict_categories(self.df_raw)
        ml_confidence = np.mean([r['confidence'] for r in ml_results])
        print(f"   Average Confidence: {ml_confidence:.1f}%")
        
        # Module 4: Confidence Combination
        print("\n4️⃣ CONFIDENCE COMBINATION")
        print("-" * 40)
        final_results = self.confidence_combiner.combine_predictions(det_results, fuzzy_results, ml_results)
        final_confidence = np.mean([r['final_confidence'] for r in final_results])
        print(f"   Final Average Confidence: {final_confidence:.1f}%")
        
        # Module 5: Review Tagging
        print("\n5️⃣ REVIEW TAGGING")
        print("-" * 40)
        tagged_results = self.review_tagger.tag_for_review(final_results)
        
        # Calculate review statistics
        auto_approved = sum(1 for r in tagged_results if 'auto_approved' in r['tags'])
        needs_review = sum(1 for r in tagged_results if 'review_recommended' in r['tags'])
        manual_review = sum(1 for r in tagged_results if 'manual_review_required' in r['tags'])
        
        print(f"   ✅ Auto-Approved: {auto_approved:,} ({auto_approved/len(tagged_results)*100:.1f}%)")
        print(f"   ⚠️ Review Recommended: {needs_review:,} ({needs_review/len(tagged_results)*100:.1f}%)")
        print(f"   ❌ Manual Review: {manual_review:,} ({manual_review/len(tagged_results)*100:.1f}%)")
        
        # Apply results
        self.apply_results_to_dataframe(tagged_results)
        
        print("\n" + "=" * 60)
        print("✅ Enhanced Mapping Pipeline Complete!")
        print(f"🎯 Final Confidence Score: {final_confidence:.1f}%")
        print("=" * 60)
        
        return self.df_mapped
    
    def apply_results_to_dataframe(self, results):
        """Apply mapping results to the dataframe."""
        self.df_mapped = self.df_raw.copy()
        
        self.df_mapped['mapped_category'] = [r['final_category'] for r in results]
        self.df_mapped['mapped_subcategory'] = [r.get('subcategory', '') for r in results]
        self.df_mapped['confidence_deterministic'] = [r['det_confidence'] for r in results]
        self.df_mapped['confidence_fuzzy'] = [r['fuzzy_confidence'] for r in results]
        self.df_mapped['confidence_ml'] = [r['ml_confidence'] for r in results]
        self.df_mapped['final_confidence'] = [r['final_confidence'] for r in results]
        self.df_mapped['mapped_hsn'] = [r['hsn_code'] for r in results]
        self.df_mapped['review_tags'] = [json.dumps(r['tags']) for r in results]
        self.df_mapped['review_priority'] = [r['review_priority'] for r in results]
        self.df_mapped['mapping_timestamp'] = datetime.now().isoformat()
        
        return self.df_mapped
    
    def save_mapped_data(self):
        """Save mapped data for next pipeline stage."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"mapped_data_{date_str}.parquet")
        
        table = pa.Table.from_pandas(self.df_mapped)
        pq.write_table(table, output_path)
        print(f"\n💾 Mapped data saved → {output_path}")
        
        # CSV backup
        csv_path = output_path.replace('.parquet', '.csv')
        self.df_mapped.to_csv(csv_path, index=False)
        print(f"📊 CSV backup saved → {csv_path}")
        
        return output_path
    
    def generate_mapping_summary(self):
        """Generate comprehensive mapping statistics."""
        if self.df_mapped is None:
            return {}
        
        summary = {
            'total_products': len(self.df_mapped),
            'average_confidence': float(self.df_mapped['final_confidence'].mean()),
            'confidence_std': float(self.df_mapped['final_confidence'].std()),
            'auto_approved': int((self.df_mapped['final_confidence'] >= 90).sum()),
            'needs_review': int(((self.df_mapped['final_confidence'] >= 70) & 
                                (self.df_mapped['final_confidence'] < 90)).sum()),
            'manual_review': int((self.df_mapped['final_confidence'] < 70).sum()),
            'category_distribution': self.df_mapped['mapped_category'].value_counts().to_dict(),
            'hsn_distribution': self.df_mapped['mapped_hsn'].value_counts().to_dict()
        }
        
        return summary
    
    def run(self):
        """Execute complete enhanced mapping pipeline."""
        try:
            # Load data
            self.load_processed_data()
            
            # Run mapping
            self.run_mapping_pipeline()
            
            # Save results
            output_path = self.save_mapped_data()
            
            # Generate summary
            summary = self.generate_mapping_summary()
            
            return output_path, summary
            
        except Exception as e:
            print(f"\n❌ Mapping failed: {e}")
            raise


# ==============================================================
# MODULE 1: Deterministic Mapper
# ==============================================================

class DeterministicMapper:
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self):
        """Load comprehensive deterministic mapping rules."""
        return {
            'jewellery': {
                'keywords': ['ring', 'necklace', 'bracelet', 'earring', 'pendant', 
                           'chain', 'bangle', 'anklet', 'brooch', 'jewel'],
                'patterns': [r'(\d+k\s*gold)', r'(diamond)', r'(silver\s*925)'],
                'confidence': 95
            },
            'watch': {
                'keywords': ['watch', 'timepiece', 'chronograph', 'wristwatch'],
                'patterns': [r'(automatic)', r'(quartz)', r'(analog)', r'(digital)'],
                'confidence': 95
            },
            'home_decor': {
                'keywords': ['vase', 'decor', 'ornament', 'figurine', 'sculpture',
                           'candle', 'holder', 'frame', 'mirror'],
                'patterns': [r'(decorative)', r'(ornamental)'],
                'confidence': 90
            },
            'drinkware': {
                'keywords': ['glass', 'mug', 'cup', 'tumbler', 'flute', 'decanter',
                           'carafe', 'bottle', 'pitcher'],
                'patterns': [r'(\d+\s*ml)', r'(\d+\s*l\b)', r'(set\s*of\s*\d+)'],
                'confidence': 90
            },
            'tableware': {
                'keywords': ['plate', 'bowl', 'dish', 'platter', 'tray', 
                           'serving', 'dinnerware'],
                'patterns': [r'(dinner\s*set)', r'(serving\s*set)'],
                'confidence': 90
            },
            'cutlery': {
                'keywords': ['knife', 'fork', 'spoon', 'cutlery', 'flatware'],
                'patterns': [r'(stainless\s*steel)', r'(silver\s*plated)'],
                'confidence': 90
            },
            'textiles': {
                'keywords': ['fabric', 'cloth', 'textile', 'cotton', 'silk', 
                           'wool', 'linen', 'polyester'],
                'patterns': [r'(\d+\s*thread\s*count)', r'(\d+\s*gsm)'],
                'confidence': 85
            }
        }
    
    def map_categories(self, descriptions, existing_categories):
        """Map categories using deterministic rules."""
        results = []
        
        for i, desc in enumerate(descriptions):
            desc_lower = desc.lower()
            matched = False
            
            for category, rule in self.rules.items():
                # Check keywords
                if any(keyword in desc_lower for keyword in rule['keywords']):
                    results.append({
                        'index': i,
                        'mapped_category': category.replace('_', ' ').title(),
                        'confidence': rule['confidence'],
                        'method': 'deterministic_keyword'
                    })
                    matched = True
                    break
                
                # Check patterns
                if any(re.search(pattern, desc_lower) for pattern in rule['patterns']):
                    results.append({
                        'index': i,
                        'mapped_category': category.replace('_', ' ').title(),
                        'confidence': rule['confidence'] - 5,
                        'method': 'deterministic_pattern'
                    })
                    matched = True
                    break
            
            if not matched:
                # Use existing category if available
                if existing_categories[i]:
                    results.append({
                        'index': i,
                        'mapped_category': existing_categories[i],
                        'confidence': 50,
                        'method': 'existing'
                    })
                else:
                    results.append({
                        'index': i,
                        'mapped_category': 'Uncategorized',
                        'confidence': 0,
                        'method': 'none'
                    })
        
        return results


# ==============================================================
# MODULE 2: Fuzzy Matcher
# ==============================================================

class FuzzyMatcher:
    def __init__(self):
        self.known_categories = [
            'Jewellery', 'Branded Watch', 'Home Decor', 'Drinkware',
            'Tableware', 'Cutlery', 'Textiles', 'Electronics',
            'Accessories', 'Serveware', 'Kitchen', 'Glass', 'Ceramic'
        ]
    
    def match_categories(self, descriptions, existing_categories):
        """Match categories using fuzzy string matching."""
        results = []
        
        for i, desc in enumerate(descriptions):
            # Try to match against known categories
            if existing_categories[i]:
                match = process.extractOne(
                    existing_categories[i],
                    self.known_categories,
                    scorer=fuzz.WRatio
                )
                
                if match:
                    category, score, _ = match
                    results.append({
                        'index': i,
                        'mapped_category': category,
                        'confidence': int(score * 0.95),  # Scale to 0-95
                        'method': 'fuzzy_match'
                    })
                else:
                    results.append({
                        'index': i,
                        'mapped_category': 'Uncategorized',
                        'confidence': 20,
                        'method': 'fuzzy_no_match'
                    })
            else:
                # Try to extract category from description
                desc_words = desc.lower().split()
                best_match = None
                best_score = 0
                
                for category in self.known_categories:
                    for word in desc_words:
                        score = fuzz.ratio(word, category.lower())
                        if score > best_score:
                            best_score = score
                            best_match = category
                
                if best_match and best_score > 60:
                    results.append({
                        'index': i,
                        'mapped_category': best_match,
                        'confidence': int(best_score * 0.8),
                        'method': 'fuzzy_description'
                    })
                else:
                    results.append({
                        'index': i,
                        'mapped_category': 'Uncategorized',
                        'confidence': 10,
                        'method': 'fuzzy_no_match'
                    })
        
        return results


# ==============================================================
# MODULE 3: ML Classifier with BERT
# ==============================================================

class AdvancedMLClassifier:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models_loaded = False
        self.sentence_model = None
        self.category_model = None
        self.subcategory_model = None
        self.hsn_model = None
        self.category_encoder = None
        self.subcategory_encoder = None
        self.hsn_encoder = None
        
        # Try to load models if they exist
        self.load_models()
    
    def load_models(self):
        """Load trained models if available."""
        try:
            print("   Loading ML models...")
            
            # Load sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load LightGBM models
            category_model_path = os.path.join(self.model_dir, 'category_model.txt')
            if os.path.exists(category_model_path):
                self.category_model = lgb.Booster(model_file=category_model_path)
                self.category_encoder = joblib.load(os.path.join(self.model_dir, 'category_encoder.pkl'))
                print("   ✅ Category model loaded")
            
            subcategory_model_path = os.path.join(self.model_dir, 'subcategory_model.txt')
            if os.path.exists(subcategory_model_path):
                self.subcategory_model = lgb.Booster(model_file=subcategory_model_path)
                self.subcategory_encoder = joblib.load(os.path.join(self.model_dir, 'subcategory_encoder.pkl'))
                print("   ✅ Subcategory model loaded")
            
            hsn_model_path = os.path.join(self.model_dir, 'hsn_model.txt')
            if os.path.exists(hsn_model_path):
                self.hsn_model = lgb.Booster(model_file=hsn_model_path)
                self.hsn_encoder = joblib.load(os.path.join(self.model_dir, 'hsn_encoder.pkl'))
                print("   ✅ HSN model loaded")
            
            self.models_loaded = True if self.category_model else False
            
        except Exception as e:
            print(f"   ⚠️ Could not load ML models: {e}")
            print("   Using baseline classifier")
            self.models_loaded = False
    
    def _create_features(self, df):
        """Create features for ML prediction."""
        # Generate embeddings
        descriptions = df['description'].fillna('').tolist()
        
        if self.sentence_model:
            print("   Generating BERT embeddings...")
            embeddings = self.sentence_model.encode(
                descriptions,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            # Fallback to simple features
            embeddings = np.zeros((len(descriptions), 384))
        
        # Create feature dataframe
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        df_features = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
        
        # Add text features
        df_features['desc_length'] = df['description'].str.len()
        df_features['word_count'] = df['description'].str.split().str.len()
        df_features['has_ml'] = df['description'].str.contains(r'\d+\s*ml', case=False, na=False).astype(int)
        df_features['has_litre'] = df['description'].str.contains(r'\d+\s*l\b', case=False, na=False).astype(int)
        df_features['has_gold'] = df['description'].str.contains('gold|18k|22k', case=False, na=False).astype(int)
        df_features['has_silver'] = df['description'].str.contains('silver|925', case=False, na=False).astype(int)
        df_features['has_diamond'] = df['description'].str.contains('diamond', case=False, na=False).astype(int)
        df_features['has_glass'] = df['description'].str.contains('glass|crystal', case=False, na=False).astype(int)
        df_features['has_ceramic'] = df['description'].str.contains('ceramic|porcelain', case=False, na=False).astype(int)
        
        # Brand frequency
        brand_freq = df['brand'].fillna('Generic').value_counts().to_dict()
        df_features['brand_frequency'] = df['brand'].fillna('Generic').map(brand_freq)
        
        return df_features
    
    def predict_categories(self, df):
        """Predict categories using ML models."""
        results = []
        
        if self.models_loaded and self.category_model:
            # Use trained models
            features = self._create_features(df)
            X = features.values
            
            # Predict categories
            cat_probs = self.category_model.predict(X, num_iteration=self.category_model.best_iteration)
            cat_preds = np.argmax(cat_probs, axis=1)
            cat_confidences = np.max(cat_probs, axis=1) * 100
            
            categories = self.category_encoder.inverse_transform(cat_preds)
            
            # Predict subcategories if model available
            subcategories = [''] * len(df)
            if self.subcategory_model:
                subcat_probs = self.subcategory_model.predict(X, num_iteration=self.subcategory_model.best_iteration)
                subcat_preds = np.argmax(subcat_probs, axis=1)
                subcategories = self.subcategory_encoder.inverse_transform(subcat_preds)
            
            # Predict HSN codes if model available
            hsn_codes = ['9999'] * len(df)
            if self.hsn_model:
                hsn_probs = self.hsn_model.predict(X, num_iteration=self.hsn_model.best_iteration)
                hsn_preds = np.argmax(hsn_probs, axis=1)
                hsn_codes = self.hsn_encoder.inverse_transform(hsn_preds)
            
            for i in range(len(df)):
                results.append({
                    'index': i,
                    'mapped_category': categories[i],
                    'subcategory': subcategories[i],
                    'hsn_code': hsn_codes[i],
                    'confidence': int(cat_confidences[i]),
                    'method': 'ml_advanced'
                })
        else:
            # Fallback to baseline
            print("   Using baseline classifier (models not loaded)")
            results = self._baseline_classifier(df)
        
        return results
    
    def _baseline_classifier(self, df):
        """Baseline classifier when models are not available."""
        print("USING BASELINE ML CLASSIFIER")

        results = []
        
        for i, row in df.iterrows():
            desc = str(row.get('description', '')).lower()
            confidence = 60
            
            # Simple keyword-based classification
            if any(word in desc for word in ['ring', 'necklace', 'jewel', 'gold']):
                category = 'Jewellery'
                hsn = '7113'
                confidence = 75
            elif any(word in desc for word in ['watch', 'timepiece']):
                category = 'Branded Watch'
                hsn = '9102'
                confidence = 75
            elif any(word in desc for word in ['glass', 'mug', 'cup', 'flute']):
                category = 'Drinkware'
                hsn = '7013'
                confidence = 70
            elif any(word in desc for word in ['plate', 'bowl', 'dish']):
                category = 'Tableware'
                hsn = '6911'
                confidence = 70
            elif any(word in desc for word in ['decor', 'vase', 'ornament']):
                category = 'Home Decor'
                hsn = '7020'
                confidence = 65
            else:
                category = 'Uncategorized'
                hsn = '9999'
                confidence = 30
            
            results.append({
                'index': i,
                'mapped_category': category,
                'subcategory': '',
                'hsn_code': hsn,
                'confidence': confidence,
                'method': 'ml_baseline'
            })
        
        return results


# ==============================================================
# MODULE 4: Confidence Combiner
# ==============================================================

class ConfidenceCombiner:
    def __init__(self):
        self.weights = {
            'deterministic': 0.4,
            'fuzzy': 0.25,
            'ml': 0.35
        }
        
        self.hsn_mapping = {
            'Jewellery': '7113',
            'Branded Watch': '9102',
            'Home Decor': '7020',
            'Drinkware': '7013',
            'Tableware': '6911',
            'Cutlery': '8215',
            'Textiles': '5209',
            'Electronics': '8517',
            'Accessories': '6217',
            'Uncategorized': '9999'
        }
    def normalize_category(self, cat):
        if not cat:
            return "Uncategorized"
    
        cat = cat.strip().lower()

        mapping = {
            "jewellery": "Jewellery",
            "gold jewellery": "Jewellery",
            "branded jewellery": "Jewellery",
            "watch": "Branded Watch",
            "branded watch": "Branded Watch",
            "home decor": "Home Decor",
            "decor": "Home Decor",
            "branded accessories": "Accessories",
            "accessories": "Accessories",
            "branded accessories": "Accessories",
        }

        return mapping.get(cat, cat.title())

    
    def combine_predictions(self, det_results, fuzzy_results, ml_results):
        """Combine predictions using weighted voting."""
        combined_results = []
        
        for i in range(len(det_results)):
            det = det_results[i]
            fuzzy = fuzzy_results[i]
            ml = ml_results[i]
            
            # Weighted scoring
            category_scores = {}
            
            # Deterministic score
            det_cat = self.normalize_category(det['mapped_category'])
            #print("Deterministic   " + det_cat)
            category_scores[det_cat] = category_scores.get(det_cat, 0) + \
                                      det['confidence'] * self.weights['deterministic']
            
            # Fuzzy score
            fuzzy_cat = self.normalize_category(fuzzy['mapped_category'])
            #print("Fuzzy   " +fuzzy_cat)
            category_scores[fuzzy_cat] = category_scores.get(fuzzy_cat, 0) + \
                                        fuzzy['confidence'] * self.weights['fuzzy']
            
            # ML score
            ml_cat = self.normalize_category(ml['mapped_category'])
            #print("ML Classifier  "+ml_cat)
            category_scores[ml_cat] = category_scores.get(ml_cat, 0) + \
                                     ml['confidence'] * self.weights['ml']
            
            # Get best category
            if category_scores:
                final_category = max(category_scores, key=category_scores.get)
                final_confidence = min(int(category_scores[final_category]), 100)
                
            else:
                final_category = 'Uncategorized'
                final_confidence = 0
            
            # Use ML HSN if available, otherwise map from category
            hsn_code = ml.get('hsn_code', self.hsn_mapping.get(final_category, '9999'))
            
            # Boost confidence if all methods agree
            if det_cat == fuzzy_cat == ml_cat:
                final_confidence = min(final_confidence + 10, 100)
            
            combined_results.append({
                'index': i,
                'final_category': final_category,
                'subcategory': ml.get('subcategory', ''),
                'final_confidence': final_confidence,
                'det_confidence': det['confidence'],
                'fuzzy_confidence': fuzzy['confidence'],
                'ml_confidence': ml['confidence'],
                'hsn_code': hsn_code,
                'category_scores': category_scores
            })
            #print(category_scores)
        return combined_results


# ==============================================================
# MODULE 5: Review Tagger
# ==============================================================

class ReviewTagger:
    def __init__(self):
        self.thresholds = {
            'auto_approve': 90,
            'review_recommended': 70,
            'manual_review': 0
        }
    
    def tag_for_review(self, combined_results):
        """Tag products for review based on confidence and other factors."""
        tagged_results = []
        
        for result in combined_results:
            confidence = result['final_confidence']
            category = result['final_category']
            
            tags = []
            
            # Confidence-based tagging
            if confidence >= self.thresholds['auto_approve']:
                tags.extend(['auto_approved', 'high_confidence'])
                review_priority = 'none'
            elif confidence >= self.thresholds['review_recommended']:
                tags.extend(['review_recommended', 'medium_confidence'])
                review_priority = 'medium'
            else:
                tags.extend(['manual_review_required', 'low_confidence'])
                review_priority = 'high'
            
            # Special case tagging
            if category == 'Uncategorized':
                tags.append('uncategorized')
                review_priority = 'high'
            
            if confidence < 50:
                tags.append('very_low_confidence')
                review_priority = 'urgent'
            
            # Check for conflicts
            det_conf = result['det_confidence']
            fuzzy_conf = result['fuzzy_confidence']
            ml_conf = result['ml_confidence']
            
            conf_std = np.std([det_conf, fuzzy_conf, ml_conf])
            if conf_std > 30:
                tags.append('conflicting_predictions')
                if review_priority == 'none':
                    review_priority = 'medium'
            
            # High-value items
            if category in ['Jewellery', 'Branded Watch']:
                tags.append('high_value_item')
                if confidence < 95 and review_priority == 'none':
                    review_priority = 'low'
            
            result['tags'] = tags
            result['review_priority'] = review_priority
            
            tagged_results.append(result)
        
        return tagged_results


# ==============================================================
# Main Execution
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ENHANCED MAPPING ENGINE")
    print("=" * 60)
    
    # Configuration
    INPUT_DIR = "data/processed"
    OUTPUT_DIR = "data/processed"
    MODEL_DIR = "models"
    
    # Initialize enhanced engine
    engine = MappingEngine(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR
    )
    
    try:
        output_path, summary = engine.run()
        
        print(f"\n🎉 Enhanced Mapping Complete!")
        print(f"📁 Output: {output_path}")
        print(f"🎯 Average Confidence: {summary['average_confidence']:.1f}%")
        print(f"✅ Auto-Approved: {summary['auto_approved']:,} products")
        print(f"📊 Ready for SKU Building Agent")
        
    except Exception as e:
        print(f"\n❌ Mapping failed: {e}")
        import traceback
        traceback.print_exc()
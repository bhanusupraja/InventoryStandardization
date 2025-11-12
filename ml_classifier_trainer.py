import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MLClassifierTrainer:
    def __init__(self, model_dir: str = "models", data_path: str = None):
        """
        Initialize the ML Classifier Trainer with advanced embeddings.
        
        Args:
            model_dir: Directory to save trained models
            data_path: Path to training data
        """
        self.model_dir = model_dir
        self.data_path = data_path
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize Sentence-BERT model for embeddings
        print("🚀 Loading Sentence-BERT model for embeddings...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize encoders and models
        self.category_encoder = LabelEncoder()
        self.subcategory_encoder = LabelEncoder()
        self.hsn_encoder = LabelEncoder()
        
        self.category_model = None
        self.subcategory_model = None
        self.hsn_model = None
        
        # Training history
        self.training_history = {
            'category': {},
            'subcategory': {},
            'hsn': {}
        }
    
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare training data with proper category mappings."""
        if data_path:
            self.data_path = data_path
        
        print("📂 Loading training data...")
        if self.data_path.endswith('.xlsx'):
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_csv(self.data_path)
        
        print(f"✅ Loaded {len(df):,} products")
        
        # Clean and prepare data
        df = self._clean_data(df)
        
        # Create training features
        df = self._create_features(df)
        
        return df
    
    def _clean_data(self, df):
        """Clean and standardize the data."""
        # Combine description fields
        df['description'] = df['DESCRIPTION1'].fillna('') + ' ' + df['DESCRIPTION2'].fillna('')
        df['description'] = df['description'].str.lower().str.strip()
        
        # Clean category fields
        df['category'] = df['CATEGORY'].fillna('Uncategorized').str.strip()
        df['subcategory'] = df['SUBTYPE'].fillna('Generic').str.strip()
        df['brand'] = df['BRAND'].fillna('Generic').str.strip()
        
        # Map HSN codes based on categories
        df['hsn'] = self._map_hsn_codes(df)
        
        # Remove rows with empty descriptions
        df = df[df['description'].str.len() > 0]
        
        return df
    
    def _map_hsn_codes(self, df):
        """Map HSN codes based on product categories and descriptions."""
        hsn_mapping = {
            # Jewellery
            'jewellery': '7113', 'ring': '7113', 'necklace': '7113', 
            'bracelet': '7113', 'earring': '7113', 'pendant': '7113',
            
            # Watches
            'watch': '9102', 'timepiece': '9102', 'chronograph': '9102',
            
            # Home Decor / Glassware
            'glass': '7013', 'crystal': '7013', 'decanter': '7013',
            'flute': '7013', 'tumbler': '7013', 'vase': '7020',
            
            # Ceramics / Porcelain
            'ceramic': '6912', 'porcelain': '6912', 'stoneware': '6912',
            'mug': '6912', 'bowl': '6911', 'plate': '6911',
            
            # Cutlery
            'cutlery': '8215', 'knife': '8215', 'fork': '8215', 'spoon': '8215',
            
            # Textiles
            'textile': '5209', 'fabric': '5209', 'cloth': '5209',
            'shirt': '6205', 'dress': '6204', 'apparel': '6203'
        }
        
        def get_hsn(row):
            desc = str(row['description']).lower()
            cat = str(row.get('category', '')).lower()
            
            # Check description and category for HSN mapping
            for keyword, hsn_code in hsn_mapping.items():
                if keyword in desc or keyword in cat:
                    return hsn_code
            
            return '9999'  # Default HSN for uncategorized
        
        return df.apply(get_hsn, axis=1)
    
    def _create_features(self, df):
        """Create advanced features using embeddings and text analysis."""
        print("🔧 Creating advanced features...")
        
        # Generate BERT embeddings for descriptions
        print("  - Generating BERT embeddings...")
        descriptions = df['description'].tolist()
        embeddings = self._generate_embeddings_batch(descriptions)
        
        # Add embeddings as features
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        df_embeddings = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
        df = pd.concat([df, df_embeddings], axis=1)
        
        # Add text-based features
        df['desc_length'] = df['description'].str.len()
        df['word_count'] = df['description'].str.split().str.len()
        df['has_ml'] = df['description'].str.contains(r'\d+\s*ml', case=False, na=False).astype(int)
        df['has_litre'] = df['description'].str.contains(r'\d+\s*l\b', case=False, na=False).astype(int)
        df['has_gold'] = df['description'].str.contains('gold|18k|22k|24k', case=False, na=False).astype(int)
        df['has_silver'] = df['description'].str.contains('silver|925|sterling', case=False, na=False).astype(int)
        df['has_diamond'] = df['description'].str.contains('diamond|solitaire', case=False, na=False).astype(int)
        df['has_glass'] = df['description'].str.contains('glass|crystal', case=False, na=False).astype(int)
        df['has_ceramic'] = df['description'].str.contains('ceramic|porcelain|stoneware', case=False, na=False).astype(int)
        
        # Brand encoding
        brand_freq = df['brand'].value_counts().to_dict()
        df['brand_frequency'] = df['brand'].map(brand_freq)
        
        print(f"✅ Created {len(embedding_cols) + 11} features")
        
        return df
    
    def _generate_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings in batches for efficiency."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.sentence_model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def train_category_classifier(self, df):
        """Train the main category classifier."""
        print("\n🎯 Training Category Classifier...")
        
        # Remove categories with only 1 sample
        category_counts = df['category'].value_counts()
        df = df[df['category'].isin(category_counts[category_counts > 1].index)]
        
        # Prepare features and labels
        feature_cols = [col for col in df.columns if 'embedding_' in col or col in [
            'desc_length', 'word_count', 'has_ml', 'has_litre', 'has_gold',
            'has_silver', 'has_diamond', 'has_glass', 'has_ceramic', 'brand_frequency'
        ]]
        
        X = df[feature_cols].values
        y = self.category_encoder.fit_transform(df['category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.category_model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred = self.category_model.predict(X_test, num_iteration=self.category_model.best_iteration)
        y_pred = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Category Classifier Accuracy: {accuracy:.2%}")
        
        # Save training history
        self.training_history['category'] = {
            'accuracy': accuracy,
            'num_classes': len(np.unique(y)),
            'features': len(feature_cols),
            'training_samples': len(X_train)
        }
        
        return accuracy
    
    def train_subcategory_classifier(self, df):
        """Train the subcategory classifier."""
        print("\n🎯 Training Subcategory Classifier...")
        
        # Similar to category classifier but for subcategories
        feature_cols = [col for col in df.columns if 'embedding_' in col or col in [
            'desc_length', 'word_count', 'has_ml', 'has_litre', 'has_gold',
            'has_silver', 'has_diamond', 'has_glass', 'has_ceramic', 'brand_frequency'
        ]]
        
        X = df[feature_cols].values
        y = self.subcategory_encoder.fit_transform(df['subcategory'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.subcategory_model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = self.subcategory_model.predict(X_test, num_iteration=self.subcategory_model.best_iteration)
        y_pred = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Subcategory Classifier Accuracy: {accuracy:.2%}")
        
        self.training_history['subcategory'] = {
            'accuracy': accuracy,
            'num_classes': len(np.unique(y)),
            'features': len(feature_cols),
            'training_samples': len(X_train)
        }
        
        return accuracy
    
    def train_hsn_classifier(self, df):
        """Train the HSN code classifier."""
        print("\n🎯 Training HSN Code Classifier...")
        
        feature_cols = [col for col in df.columns if 'embedding_' in col or col in [
            'desc_length', 'word_count', 'has_ml', 'has_litre', 'has_gold',
            'has_silver', 'has_diamond', 'has_glass', 'has_ceramic', 'brand_frequency'
        ]]
        
        X = df[feature_cols].values
        y = self.hsn_encoder.fit_transform(df['hsn'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.hsn_model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = self.hsn_model.predict(X_test, num_iteration=self.hsn_model.best_iteration)
        y_pred = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ HSN Classifier Accuracy: {accuracy:.2%}")
        
        self.training_history['hsn'] = {
            'accuracy': accuracy,
            'num_classes': len(np.unique(y)),
            'features': len(feature_cols),
            'training_samples': len(X_train)
        }
        
        return accuracy
    
    def save_models(self):
        """Save all trained models and encoders."""
        print("\n💾 Saving models...")
        
        # Save LightGBM models
        if self.category_model:
            self.category_model.save_model(os.path.join(self.model_dir, 'category_model.txt'))
        if self.subcategory_model:
            self.subcategory_model.save_model(os.path.join(self.model_dir, 'subcategory_model.txt'))
        if self.hsn_model:
            self.hsn_model.save_model(os.path.join(self.model_dir, 'hsn_model.txt'))
        
        # Save encoders
        joblib.dump(self.category_encoder, os.path.join(self.model_dir, 'category_encoder.pkl'))
        joblib.dump(self.subcategory_encoder, os.path.join(self.model_dir, 'subcategory_encoder.pkl'))
        joblib.dump(self.hsn_encoder, os.path.join(self.model_dir, 'hsn_encoder.pkl'))
        
        # Save sentence transformer info (the model itself will be loaded by name)
        model_info = {
            'sentence_model': 'all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'training_history': self.training_history,
            'training_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.model_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Models saved to {self.model_dir}/")
        
        # Print summary
        print("\n📊 Training Summary:")
        print(f"  Category Accuracy: {self.training_history['category'].get('accuracy', 0):.2%}")
        print(f"  Subcategory Accuracy: {self.training_history['subcategory'].get('accuracy', 0):.2%}")
        print(f"  HSN Code Accuracy: {self.training_history['hsn'].get('accuracy', 0):.2%}")
    
    def train_all_models(self, data_path=None):
        """Train all classifiers in sequence."""
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        
        # Train models
        cat_acc = self.train_category_classifier(df)
        subcat_acc = self.train_subcategory_classifier(df)
        hsn_acc = self.train_hsn_classifier(df)
        
        # Calculate overall accuracy
        overall_accuracy = (cat_acc + subcat_acc + hsn_acc) / 3
        print(f"\n🎯 Overall Model Accuracy: {overall_accuracy:.2%}")
        
        # Save models
        self.save_models()
        
        return {
            'category_accuracy': cat_acc,
            'subcategory_accuracy': subcat_acc,
            'hsn_accuracy': hsn_acc,
            'overall_accuracy': overall_accuracy
        }


if __name__ == "__main__":
    # Train the ML classifiers
    print("=" * 60)
    print("🚀 ML CLASSIFIER TRAINING MODULE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MLClassifierTrainer(
        model_dir="models",
        data_path="06.10.2025.xlsx"
    )
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"🎯 Final Overall Accuracy: {results['overall_accuracy']:.2%}")
    print("=" * 60)
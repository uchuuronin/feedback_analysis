"""
Using pre-trained transformer models from HuggingFace for advanced sentiment analysis
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import confusion_matrix, classification_report


class LLMSentimentAnalyzer:
    MODELS = {
        'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english',
        'roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'bert': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'finbert': 'ProsusAI/finbert'  
    }
    
    def __init__(
        self,
        model_name: str = 'distilbert',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
        
        if model_name in self.MODELS:
            model_path = self.MODELS[model_name]
        else:
            model_path = model_name
        
        print(f"Loading model: {model_path}")
        print(f"Device: {'GPU' if self.device == 0 else 'CPU'}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                truncation=True,
                max_length=512
            )
            
            print(f"Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = True
    ) -> Union[Dict, List[Dict]]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Predicting"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
        
        if single_input:
            return results[0]
        
        return results
    
    def analyze_reviews(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        include_probabilities: bool = True
    ) -> pd.DataFrame:
        print(f"Analyzing {len(df)} reviews with {self.model_name}...")
        
        # Get predictions
        texts = df[text_column].fillna('').tolist()
        predictions = self.predict(texts, return_probabilities=include_probabilities)
        
        df = df.copy()
        df['llm_sentiment'] = [pred['label'] for pred in predictions]
        df['llm_confidence'] = [pred['score'] for pred in predictions]
        
        # Standardize labels to positive/negative/neutral
        df['llm_sentiment_normalized'] = df['llm_sentiment'].apply(self._normalize_label)
        
        print(f"Analysis complete")
        
        return df
    
    def _normalize_label(self, label: str) -> str:
        """Normalize different label formats to positive/negative/neutral"""
        label = label.lower()
        
        # Handle different labeling schemes
        if 'positive' in label or label in ['pos', '5 stars', '4 stars', 'label_2']:
            return 'positive'
        elif 'negative' in label or label in ['neg', '1 star', '2 stars', 'label_0']:
            return 'negative'
        else:
            return 'neutral'
    
    def batch_analyze(
        self,
        texts: List[str],
        aggregate: bool = True
    ) -> Dict:
        predictions = self.predict(texts)
        
        results = {
            'predictions': predictions,
            'total_texts': len(texts)
        }
        
        if aggregate:
            # Count sentiments
            sentiment_counts = {}
            for pred in predictions:
                label = self._normalize_label(pred['label'])
                sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            
            # Calculate percentages
            sentiment_percentages = {
                label: (count / len(texts)) * 100
                for label, count in sentiment_counts.items()
            }
            
            # Average confidence
            avg_confidence = np.mean([pred['score'] for pred in predictions])
            
            results['sentiment_counts'] = sentiment_counts
            results['sentiment_percentages'] = sentiment_percentages
            results['average_confidence'] = avg_confidence
        
        return results
    
    def compare_with_ratings(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        rating_column: str = 'rating'
    ) -> Dict:
        # Analyze reviews
        df = self.analyze_reviews(df, text_column)
        
        # Create sentiment from ratings
        df['rating_sentiment'] = df[rating_column].apply(
            lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
        )
        
        # Calculate agreement
        agreement = (df['llm_sentiment_normalized'] == df['rating_sentiment']).mean()
        
        # Confusion matrix
        
        
        cm = confusion_matrix(
            df['rating_sentiment'],
            df['llm_sentiment_normalized'],
            labels=['negative', 'neutral', 'positive']
        )
        
        report = classification_report(
            df['rating_sentiment'],
            df['llm_sentiment_normalized'],
            labels=['negative', 'neutral', 'positive']
        )
        
        results = {
            'agreement': agreement,
            'confusion_matrix': cm,
            'classification_report': report,
            'sample_size': len(df)
        }
        
        print(f"\nAgreement with ratings: {agreement:.2%}")
        print("\nClassification Report:")
        print(report)
        
        return results


class ZeroShotSentimentAnalyzer:    
    def __init__(self, model_name: str = 'facebook/bart-large-mnli'):
        self.model_name = model_name
        
        device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading zero-shot model: {model_name}")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        print("Model loaded")
    
    def classify(
        self,
        texts: Union[str, List[str]],
        candidate_labels: List[str],
        multi_label: bool = False
    ) -> Union[Dict, List[Dict]]:
        return self.classifier(
            texts,
            candidate_labels=candidate_labels,
            multi_label=multi_label
        )


def compare_llm_models(
    texts: List[str],
    true_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    models_to_test = ['distilbert', 'roberta']
    results = []
    
    for model_name in models_to_test:
        print(f"Testing {model_name.upper()}")
        try:
            analyzer = LLMSentimentAnalyzer(model_name=model_name)
            predictions = analyzer.predict(texts)
            
            avg_confidence = np.mean([pred['score'] for pred in predictions])
            
            result = {
                'Model': model_name,
                'Avg Confidence': avg_confidence,
                'Predictions': predictions
            }
            
            if true_labels:
                predicted_labels = [
                    analyzer._normalize_label(pred['label']) 
                    for pred in predictions
                ]
                accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(true_labels)
                result['Accuracy'] = accuracy
            
            results.append(result)
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Testing
    sample_reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Broke after one day. Complete waste of money.",
        "It's okay, nothing special but does what it claims.",
        "Fast shipping and great customer service. Product works well.",
        "Disappointed with this purchase. Not worth the price."
    ]
    
    # Test single model
    analyzer = LLMSentimentAnalyzer(model_name='distilbert')
    results = analyzer.predict(sample_reviews)
    
    print("\nSentiment Analysis Results:")
    for text, result in zip(sample_reviews, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")
    
    # Test batch analysis with aggregation
    batch_results = analyzer.batch_analyze(sample_reviews, aggregate=True)
    print("\n\nBatch Analysis Summary:")
    print(f"Total texts: {batch_results['total_texts']}")
    print(f"\nSentiment Distribution:")
    for sentiment, percentage in batch_results['sentiment_percentages'].items():
        print(f"  {sentiment}: {percentage:.1f}%")
    print(f"\nAverage Confidence: {batch_results['average_confidence']:.3f}")

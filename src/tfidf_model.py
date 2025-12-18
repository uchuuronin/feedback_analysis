import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_amazon_reviews
from src.preprocessor import prepare_review_data

class TFIDFSentimentAnalyzer:
    def __init__(
        self,
        classifier_type: str = 'logistic',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.classifier = self._get_classifier(classifier_type)
        
        self.metrics = {}
        self.is_trained = False
    
    def _get_classifier(self, classifier_type: str):
        classifiers = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm': LinearSVC(random_state=42, max_iter=2000),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_type}")
        
        return classifiers[classifier_type]
    
    def train(
        self,
        texts: list,
        labels: list,
        test_size: float = 0.2,
        validation_split: bool = True
    ) -> Dict:
        print(f"Training TF-IDF {self.classifier_type} classifier...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        print("Vectorizing text with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
        
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        y_pred_train = self.classifier.predict(X_train_tfidf)
        y_pred_test = self.classifier.predict(X_test_tfidf)
        
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'test_predictions': y_pred_test,
            'test_labels': y_test
        }
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred_test, average='weighted'
        )
        
        self.metrics['precision'] = precision
        self.metrics['recall'] = recall
        self.metrics['f1_score'] = f1
        
        self.is_trained = True
        
        print(f"Train Accuracy:\t{self.metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:\t{self.metrics['test_accuracy']:.4f}")
        print(f"Precision:\t{precision:.4f}")
        print(f"Recall:\t{recall:.4f}")
        print(f"F1 Score:\t{f1:.4f}")
        
        return self.metrics
    
    def predict(self, texts: list) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        x_tfidf = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(x_tfidf)
        
        return predictions
    
    def predict_proba(self, texts: list) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        x_tfidf = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(x_tfidf)
        
        return probabilities
    
    def get_top_features(self, n_features: int = 20) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        if hasattr(self.classifier, 'coef_'):
            # For logistic regression and SVM
            coef = self.classifier.coef_
        elif hasattr(self.classifier, 'feature_log_prob_'):
            # For Naive Bayes
            coef = self.classifier.feature_log_prob_
        else:
            return {}
        
        top_features = {}
        
        for idx, class_label in enumerate(self.classifier.classes_):
            top_indices = np.argsort(coef[idx])[-n_features:][::-1]
            top_features[class_label] = feature_names[top_indices].tolist()
        
        return top_features
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        if not self.is_trained or 'confusion_matrix' not in self.metrics:
            raise ValueError("Model must be trained first")
        
        cm = self.metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classifier.classes_,
            yticklabels=self.classifier.classes_
        )
        plt.title(f'Confusion Matrix - {self.classifier_type.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved confusion matrix to {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.metrics = model_data['metrics']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def compare_classifiers(
    texts: list,
    labels: list,
    test_size: float = 0.2
) -> pd.DataFrame:
    classifier_types = ['logistic', 'svm', 'naive_bayes', 'random_forest']
    results = []
    
    for clf_type in classifier_types:
        print(f"Training {clf_type.upper()} classifier")
        analyzer = TFIDFSentimentAnalyzer(classifier_type=clf_type)
        metrics = analyzer.train(texts, labels, test_size=test_size)
        
        results.append({
            'Classifier': clf_type,
            'Train Accuracy': metrics['train_accuracy'],
            'Test Accuracy': metrics['test_accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    df = load_amazon_reviews('Electronics', n_samples=5000)
    df = prepare_review_data(df)
    
    analyzer = TFIDFSentimentAnalyzer(classifier_type='logistic')
    metrics = analyzer.train(
        texts=df['processed_text'].tolist(),
        labels=df['sentiment_label'].tolist()
    )
    top_features = analyzer.get_top_features(n_features=10)
    for sentiment, features in top_features.items():
        print(f"\n{sentiment.upper()}: {', '.join(features)}")
    
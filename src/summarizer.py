"""
Generates summaries and insights from customer reviews using LLMs
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
from transformers import pipeline
import torch
from sklearn.feature_extraction.text import CountVectorizer

class CustomerInsightSummarizer:
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        self.model_name = model_name
        
        device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading summarization model: {model_name}")
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=device
        )
        print("Summarizer loaded")
    
    def summarize_reviews(
        self,
        reviews: List[str],
        max_length: int = 150,
        min_length: int = 50,
        do_sample: bool = False
    ) -> str:
        # Combine reviews (limit to avoid token limits)
        combined_text = " ".join(reviews[:20]) 
        
        # Truncate if too long
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000]
        
        try:
            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary."
    
    def generate_product_insights(
        self,
        df: pd.DataFrame,
        product_id: Optional[str] = None,
        text_column: str = 'text',
        rating_column: str = 'rating',
        sentiment_column: str = 'sentiment_label'
    ) -> Dict:
        insights = {
            'product_id': product_id,
            'total_reviews': len(df),
            'average_rating': df[rating_column].mean() if rating_column in df.columns else None,
            'rating_distribution': df[rating_column].value_counts().to_dict() if rating_column in df.columns else None
        }
        
        # Sentiment distribution
        if sentiment_column in df.columns:
            sentiment_counts = df[sentiment_column].value_counts().to_dict()
            insights['sentiment_distribution'] = sentiment_counts
            insights['sentiment_percentages'] = {
                k: (v / len(df)) * 100 for k, v in sentiment_counts.items()
            }
        
        # Separate positive and negative reviews
        if rating_column in df.columns:
            positive_reviews = df[df[rating_column] >= 4][text_column].tolist()
            negative_reviews = df[df[rating_column] <= 2][text_column].tolist()
            
            # Generate summaries
            if positive_reviews:
                insights['positive_summary'] = self.summarize_reviews(positive_reviews)
            
            if negative_reviews:
                insights['negative_summary'] = self.summarize_reviews(negative_reviews)
        
        # Extract common keywords
        insights['common_keywords'] = self._extract_keywords(df[text_column].tolist())
        
        return insights
    
    def _extract_keywords(
        self,
        texts: List[str],
        top_n: int = 20
    ) -> List[tuple]:
        try:
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum counts across all documents
            word_counts = X.sum(axis=0).A1
            
            # Get top words
            top_indices = word_counts.argsort()[-top_n:][::-1]
            top_words = [(feature_names[i], int(word_counts[i])) for i in top_indices]
            
            return top_words
        except:
            return []
    
    def generate_comparison_insights(
        self,
        tfidf_results: Dict,
        llm_results: Dict
    ) -> Dict:
        insights = {
            'tfidf_accuracy': tfidf_results.get('test_accuracy', 0),
            'llm_accuracy': llm_results.get('accuracy', 0),
            'accuracy_difference': abs(
                tfidf_results.get('test_accuracy', 0) - llm_results.get('accuracy', 0)
            ),
            'better_model': 'LLM' if llm_results.get('accuracy', 0) > tfidf_results.get('test_accuracy', 0) else 'TF-IDF'
        }
        
        return insights
    
    def create_txt_report(
        self,
        insights: Dict,
        output_file: Optional[str] = None
    ) -> str:
        report_lines = [
            "CUSTOMER FEEDBACK ANALYSIS REPORT",
            ""
        ]
        
        # Product information
        if 'product_id' in insights and insights['product_id']:
            report_lines.append(f"Product ID: {insights['product_id']}")
        
        report_lines.append(f"Total Reviews Analyzed: {insights.get('total_reviews', 0)}")
        
        # Rating information
        if 'average_rating' in insights and insights['average_rating']:
            report_lines.extend([
                "",
                f"Average Rating: {insights['average_rating']:.2f}/5.0",
                ""
            ])
        
        # Sentiment distribution
        if 'sentiment_distribution' in insights:
            report_lines.extend([
                "Sentiment Distribution:",
            ])
            for sentiment, count in insights['sentiment_distribution'].items():
                percentage = insights['sentiment_percentages'][sentiment]
                report_lines.append(f"  {sentiment.upper()}: {count} reviews ({percentage:.1f}%)")
            report_lines.append("")
        
        # Positive insights
        if 'positive_summary' in insights:
            report_lines.extend([
                "Positive Feedback Summary:",
                insights['positive_summary'],
                ""
            ])
        
        # Negative insights
        if 'negative_summary' in insights:
            report_lines.extend([
                "Negative Feedback Summary:",
                insights['negative_summary'],
                ""
            ])
        
        # Common keywords
        if 'common_keywords' in insights and insights['common_keywords']:
            report_lines.extend([
                "Most Frequent Topics:",
            ])
            for keyword, count in insights['common_keywords'][:10]:
                report_lines.append(f"  • {keyword} ({count} mentions)")
            report_lines.append("")
        
        # Model comparison
        if 'tfidf_accuracy' in insights:
            report_lines.extend([
                "Model Performance Comparison:",
                f"\tTF-IDF Model Accuracy:\t{insights['tfidf_accuracy']:.2%}",
                f"\tLLM Model Accuracy:\t{insights['llm_accuracy']:.2%}",
                f"\tBest Performing Model:\t{insights['better_model']}",
                ""
            ])
        
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text


def generate_insights_from_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    rating_column: str = 'rating',
    sentiment_column: str = 'sentiment_label'
) -> Dict:
    summarizer = CustomerInsightSummarizer()
    insights = summarizer.generate_product_insights(
        df,
        text_column=text_column,
        rating_column=rating_column,
        sentiment_column=sentiment_column
    )
    
    return insights


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'text': [
            "Great product! Works perfectly and arrived quickly.",
            "Excellent quality, highly recommend to everyone.",
            "Amazing! Best purchase I've made this year.",
            "Poor quality, broke after a week of use.",
            "Disappointed with this product, not as described.",
            "Terrible experience, waste of money."
        ],
        'rating': [5, 5, 5, 1, 2, 1],
        'sentiment_label': ['positive', 'positive', 'positive', 'negative', 'negative', 'negative']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Generate insights
    summarizer = CustomerInsightSummarizer()
    insights = summarizer.generate_product_insights(df)
    
    # Create report
    report = summarizer.create_txt_report(insights)
    print(report)

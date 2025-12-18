"""
LLM-Based Sentiment Analysis and Model Comparison
Advanced pipeline using transformer models and comparative analysis
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import AmazonReviewLoader
from src.preprocessor import prepare_review_data
from src.llm_model import LLMSentimentAnalyzer, compare_llm_models
from src.tfidf_model import TFIDFSentimentAnalyzer
from src.summarizer import CustomerInsightSummarizer
from src.visualizer import SentimentVisualizer


def run_llm_results_pipeline(
    category: str = 'Electronics',
    n_samples: int = 2000,  # Smaller sample for LLM due to speed
    output_dir: str = 'results',
    use_tfidf_results_data: bool = True,
    llm_model: str = 'distilbert'
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/reports', exist_ok=True)
    
    if use_tfidf_results_data and os.path.exists(f'{output_dir}/tfidf_results/preprocessed_data.csv'):
        print("Loading data from tfidf_results...")
        df = pd.read_csv(f'{output_dir}/tfidf_results/preprocessed_data.csv')
        # Sample if larger than n_samples
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
    else:
        print("Loading fresh data...")
        loader = AmazonReviewLoader()
        df = loader.load_reviews(category, n_samples=n_samples)
        df = prepare_review_data(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Processing {len(df):,} reviews with LLM...")
    
    # Run LLM Analysis
    llm_analyzer = LLMSentimentAnalyzer(model_name=llm_model)
    df = llm_analyzer.analyze_reviews(df, text_column='full_text')
    
    # Load TF-IDF Model for Comparison
    tfidf_model_path = f'{output_dir}/tfidf_results/models/tfidf_sentiment_model.pkl'
    
    if os.path.exists(tfidf_model_path):
        print("Loading TF-IDF model...")
        tfidf_analyzer = TFIDFSentimentAnalyzer()
        tfidf_analyzer.load_model(tfidf_model_path)
        
        # Get TF-IDF predictions
        tfidf_predictions = tfidf_analyzer.predict(df['processed_text'].tolist())
        df['tfidf_prediction'] = tfidf_predictions
    else:
        print("TF-IDF model not found. Training new model...")
        tfidf_analyzer = TFIDFSentimentAnalyzer(classifier_type='logistic')
        tfidf_metrics = tfidf_analyzer.train(
            df['processed_text'].tolist(),
            df['sentiment_label'].tolist(),
            test_size=0.2
        )
        df['tfidf_prediction'] = tfidf_analyzer.predict(df['processed_text'].tolist())
    
    # Compare Models
    comparison_results = compare_models(df)
    
    print("\nModel Comparison Results:")
    for model, metrics in comparison_results.items():
        print(f"\n{model.upper()}:")
        print(f"\tAccuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"\tPrecision: {metrics['precision']:.4f}")
        print(f"\tRecall: {metrics['recall']:.4f}")
        print(f"\tF1 Score: {metrics['f1_score']:.4f}")
        
    summarizer = CustomerInsightSummarizer()
    insights = summarizer.generate_product_insights(
        df,
        product_id=category,
        text_column='full_text',
        rating_column='rating',
        sentiment_column='llm_sentiment_normalized'
    )
    
    print(f"Positive reviews: {len(df[df['rating'] >= 4])} reviews")
    print(f"Negative reviews: {len(df[df['rating'] <= 2])} reviews")
    
    visualizer = SentimentVisualizer()
    
    # Model comparison plot
    visualizer.plot_model_comparison(
        comparison_results,
        save_path=f'{output_dir}/figures/model_comparison.png'
    )
    
    # Confidence distribution
    if 'llm_confidence' in df.columns:
        visualizer.plot_confidence_distribution(
            df,
            save_path=f'{output_dir}/figures/llm_confidence_dist.png'
        )
    
    # Create dashboard with LLM predictions
    visualizer.create_dashboard(
        df,
        sentiment_column='llm_sentiment_normalized',
        save_path=f'{output_dir}/figures/llm_dashboard.png'
    )
    
    llm_results_report = generate_llm_txt_report(df, comparison_results, insights, category, n_samples)
    llm_results_report_path = f'{output_dir}/reports/llm_results_report.txt'
    
    with open(llm_results_report_path, 'w') as f:
        f.write(llm_results_report)
    
    print(llm_results_report)
    print(f"\nSaved report to {llm_results_report_path}")
    
    # Customer insights report
    insights_report = summarizer.create_txt_report(insights)
    insights_report_path = f'{output_dir}/reports/customer_insights.txt'
    
    with open(insights_report_path, 'w') as f:
        f.write(insights_report)
    
    print(f"Saved customer insights to {insights_report_path}")
    
    results_path = f'{output_dir}/llm_results_results.csv'
    df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv(f'{output_dir}/reports/detailed_comparison.csv')
    
    return df, llm_analyzer, comparison_results, insights


def compare_models(df: pd.DataFrame) -> dict:
    results = {}
    
    # TF-IDF metrics
    if 'tfidf_prediction' in df.columns:
        tfidf_acc = accuracy_score(df['sentiment_label'], df['tfidf_prediction'])
        tfidf_p, tfidf_r, tfidf_f1, _ = precision_recall_fscore_support(
            df['sentiment_label'], 
            df['tfidf_prediction'], 
            average='weighted'
        )
        
        results['tfidf'] = {
            'accuracy': tfidf_acc,
            'precision': tfidf_p,
            'recall': tfidf_r,
            'f1_score': tfidf_f1
        }
    
    # LLM metrics
    if 'llm_sentiment_normalized' in df.columns:
        llm_acc = accuracy_score(df['sentiment_label'], df['llm_sentiment_normalized'])
        llm_p, llm_r, llm_f1, _ = precision_recall_fscore_support(
            df['sentiment_label'],
            df['llm_sentiment_normalized'],
            average='weighted'
        )
        
        results['llm'] = {
            'accuracy': llm_acc,
            'precision': llm_p,
            'recall': llm_r,
            'f1_score': llm_f1
        }
    
    return results


def generate_llm_txt_report(
    df: pd.DataFrame,
    comparison_results: dict,
    insights: dict,
    category: str,
    n_samples: int
) -> str:
    report_lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: Amazon Reviews 2023 - {category}",
        f"Sample Size: {n_samples:,} reviews",
        "1. MODEL PERFORMANCE COMPARISON",
        ""
    ]
    
    for model, metrics in comparison_results.items():
        report_lines.extend([
            f"{model.upper()} MODEL:",
            f"\tAccuracy:\t{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
            f"\tPrecision:\t{metrics['precision']:.4f}",
            f"\tRecall:\t{metrics['recall']:.4f}",
            f"\tF1 Score:\t{metrics['f1_score']:.4f}",
            ""
        ])
    
    # Determine winner
    if 'tfidf' in comparison_results and 'llm' in comparison_results:
        tfidf_acc = comparison_results['tfidf']['accuracy']
        llm_acc = comparison_results['llm']['accuracy']
        
        if llm_acc > tfidf_acc:
            winner = "LLM"
            diff = (llm_acc - tfidf_acc) * 100
        else:
            winner = "TF-IDF"
            diff = (tfidf_acc - llm_acc) * 100
        
        report_lines.extend([
            f"BEST MODEL: {winner}",
            f"Performance improvement: {diff:.2f} percentage points",
            ""
        ])
    
    report_lines.extend([
        "2. CUSTOMER INSIGHTS SUMMARY",
        "",
        f"Total Reviews: {insights.get('total_reviews', 0):,}",
        f"Average Rating: {insights.get('average_rating', 0):.2f}/5.0",
        ""
    ])
    
    if 'sentiment_distribution' in insights:
        report_lines.extend([
            "Sentiment Breakdown:",
        ])
        for sentiment, count in insights['sentiment_distribution'].items():
            percentage = insights['sentiment_percentages'][sentiment]
            report_lines.append(f"  {sentiment.upper()}: {count:,} ({percentage:.1f}%)")
        report_lines.append("")
    
    if 'positive_summary' in insights:
        report_lines.extend([
            "Key Positive Themes:",
            insights['positive_summary'],
            ""
        ])
    
    if 'negative_summary' in insights:
        report_lines.extend([
            "Key Negative Themes:",
            insights['negative_summary'],
            ""
        ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2: LLM Sentiment Analysis')
    parser.add_argument('--category', type=str, default='Electronics',
                       help='Amazon product category')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--use_tfidf_results_data', action='store_true', default=True,
                       help='Use data from Phase 1')
    parser.add_argument('--llm_model', type=str, default='distilbert',
                       help='LLM model to use (distilbert, roberta, bert)')
    
    args = parser.parse_args()
    
    run_llm_results_pipeline(
        category=args.category,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        use_tfidf_results_data=args.use_tfidf_results_data,
        llm_model=args.llm_model
    )

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from src.data_loader import AmazonReviewLoader
from src.preprocessor import prepare_review_data
from src.tfidf_model import TFIDFSentimentAnalyzer, compare_classifiers
from src.visualizer import SentimentVisualizer, gen_plots

def tf_idf_implementation(
    category: str = 'Electronics',
    n_samples: int = 10000,
    output_dir: str = 'results',
    compare_models: bool = True
):
    subdirs = ['figures', 'models', 'reports']
    for subdir in subdirs:
        os.makedirs(f'{output_dir}/{subdir}', exist_ok=True)
    
    loader = AmazonReviewLoader()
    df = loader.load_reviews(category, n_samples=n_samples)
    df = prepare_review_data(df)
    
    preprocessed_path = f'{output_dir}/preprocessed_data.csv'
    df.to_csv(preprocessed_path, index=False)
    
    visualizer = SentimentVisualizer()
    visualizer.plot_sentiment_distribution(
        df, 'sentiment_label',
        save_path=f'{output_dir}/figures/initial_sentiment_dist.png'
    )
    
    visualizer.plot_rating_distribution(
        df, 'rating',
        save_path=f'{output_dir}/figures/initial_rating_dist.png'
    )
    
    # Train TF-IDF Model
    texts = df['processed_text'].tolist()
    labels = df['sentiment_label'].tolist()
    
    if compare_models:
        comparison_df = compare_classifiers(texts, labels, test_size=0.2)
        comparison_df.to_csv(f'{output_dir}/reports/model_comparison.csv', index=False)
        best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Classifier']
        analyzer = TFIDFSentimentAnalyzer(classifier_type=best_model)
    else:
        analyzer = TFIDFSentimentAnalyzer(classifier_type='logistic')
    
    metrics = analyzer.train(texts, labels, test_size=0.2)
    
    top_features = analyzer.get_top_features(n_features=15)
    print("\nTop Features by Sentiment:")
    for sentiment, features in top_features.items():
        print(f"\n{sentiment.upper()}:")
        print(f"  {', '.join(features)}")
    
    analyzer.plot_confusion_matrix(
        save_path=f'{output_dir}/figures/confusion_matrix_tfidf.png'
    )
    
    # Predictions
    predictions = analyzer.predict(texts)
    df['tfidf_prediction'] = predictions
    
    agreement = (df['tfidf_prediction'] == df['sentiment_label']).mean()
    print(f"Prediction agreement with labels: {agreement:.2%}")
    
    # Visualizations
    gen_plots(
        df,
        output_dir=f'{output_dir}/figures',
        rating_column='rating',
        sentiment_column='tfidf_prediction'
    )

    # Save model
    model_path = f'{output_dir}/models/tfidf_sentiment_model.pkl'
    analyzer.save_model(model_path)
    
    # Save results
    results_path = f'{output_dir}/tfidf_results.csv'
    df.to_csv(results_path, index=False)
    
    report = log_result(df, metrics, top_features, category, n_samples)
    report_path = f'{output_dir}/reports/tfidf_results_report.txt'
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    return df, analyzer, metrics


def log_result(
    df: pd.DataFrame,
    metrics: dict,
    top_features: dict,
    category: str,
    n_samples: int
) -> str:
    
    report_lines = [
        f"Dataset: Amazon Reviews 2023 - {category}",
        f"Sample Size: {n_samples:,} reviews",
        f"Total Reviews Analyzed: {len(df):,}",
        f"Average Rating: {df['rating'].mean():.2f}/5.0",
        "",
        "Rating Distribution:",
    ]
    
    for rating in sorted(df['rating'].unique()):
        count = (df['rating'] == rating).sum()
        percentage = (count / len(df)) * 100
        report_lines.append(f"  {rating} stars: {count:,} reviews ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "Sentiment Distribution:",
    ])
    
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in df['sentiment_label'].values:
            count = (df['sentiment_label'] == sentiment).sum()
            percentage = (count / len(df)) * 100
            report_lines.append(f"  {sentiment.upper()}: {count:,} reviews ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        f"\tClassifier Type:\t{metrics.get('classifier_type', 'Logistic Regression')}",
        f"\tTraining Accuracy:\t{metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)",
        f"\tTesting Accuracy:\t{metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)",
        f"\tPrecision:\t{metrics['precision']:.4f}",
        f"\tRecall:\t{metrics['recall']:.4f}",
        f"\tF1 Score:\t{metrics['f1_score']:.4f}",
        ""
    ])
    
    for sentiment, features in top_features.items():
        report_lines.append(f"{sentiment.upper()} Indicators:")
        report_lines.append("-" * 40)
        report_lines.append(f"  {', '.join(features[:10])}")
        report_lines.append("")
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 1: TF-IDF Sentiment Analysis')
    parser.add_argument('--category', type=str, default='Electronics',
                       help='Amazon product category')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='results/tfidf_results',
                       help='Output directory')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare multiple classifiers')
    
    args = parser.parse_args()
    
    tf_idf_implementation(
        category=args.category,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        compare_models=args.compare_models
    )

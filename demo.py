"""
Run complete sentiment analysis pipeline with one command
"""

import os
import sys
import argparse
from datetime import datetime
import traceback
import nltk

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_amazon_reviews
from src.preprocessor import prepare_review_data
from src.tfidf_model import TFIDFSentimentAnalyzer
from src.visualizer import SentimentVisualizer 
from llm import llm_implementation
from tf_idf import tf_idf_implementation
        

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════╗
    ║     Customer Feedback Sentiment Analyzer     ║
    ║     TF-IDF + LLM-Based Analysis Pipeline     ║
    ╚══════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'nltk', 
        'transformers', 'torch', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies installed")
    return True


def download_nltk_data():
    print("\nDownloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("NLTK data downloaded")
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False


def run_quick_demo():
    print("DEMO (500 reviews)")
    
    print("\nLoading data...")
    df = load_amazon_reviews('Electronics', n_samples=500)
    
    print("\nPreprocessing...")
    df = prepare_review_data(df)
    
    print("\nTraining TF-IDF model...")
    analyzer = TFIDFSentimentAnalyzer()
    metrics = analyzer.train(
        df['processed_text'].tolist(),
        df['sentiment_label'].tolist(),
        test_size=0.2
    )

    print("\nCreating visualizations...")
    viz = SentimentVisualizer()
    os.makedirs('demo_results', exist_ok=True)
    viz.plot_sentiment_distribution(df, save_path='demo_results/sentiment_dist.png')
    
    print("DEMO COMPLETE!")
    print(f"Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Results saved to: demo_results/")


def run_full_pipeline(args):
    print("RUNNING FULL PIPELINE")
    
    # Part 1
    if not args.skip_tfidf:
        print("\nTF-IDF Analysis")
        
        tf_idf_implementation(
            category=args.category,
            n_samples=args.n_samples,
            output_dir=f'results/tfidf_results',
            compare_models=args.compare_models
        )
    
    # Part 2
    if not args.skip_llm:
        print("\nLLM Analysis")
       
        llm_implementation(
            category=args.category,
            n_samples=min(args.n_samples, 2000),  
            output_dir='results',
            use_tfidf_results_data=True,
            llm_model=args.llm_model
        )
    
def main():
    parser = argparse.ArgumentParser(
        description='Customer Feedback Sentiment Analyzer - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run quick demo
            python demo.py 

            # Run full pipeline with defaults
            python demo.py --full-demo

            # Customize parameters
            python demo.py --category Books --n_samples 5000

            # Run only Phase 1
            python demo.py --skip_llm

            # Compare multiple classifiers
            python demo.py --compare_models
        """
    )
    
    parser.add_argument('--full-demo', action='store_true',
                       help='Run full pipeline instead of quick demo')
    parser.add_argument('--category', type=str, default='Electronics',
                       help='Product category (default: Electronics)')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples (default: 10000)')
    parser.add_argument('--skip_tfidf', action='store_true',
                       help='Skip Part 1 (TF-IDF)')
    parser.add_argument('--skip_llm', action='store_true',
                       help='Skip Part 2 (LLM)')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare multiple classifiers in Part 1')
    parser.add_argument('--llm_model', type=str, default='distilbert',
                       choices=['distilbert', 'roberta', 'bert'],
                       help='LLM model to use (default: distilbert)')
    parser.add_argument('--skip_checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    print_banner()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not args.skip_checks:
        if not check_dependencies():
            return
        download_nltk_data()
    
    if args.full_demo:
        run_full_pipeline(args)
    else:
        run_quick_demo()
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nError: {e}")
        traceback.print_exc()

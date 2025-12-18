"""
Visualization Module
Creates charts and plots for sentiment analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from wordcloud import WordCloud
import os


class SentimentVisualizer:
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (10, 6)):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        self.default_figsize = figsize
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        sentiment_column: str = 'sentiment_label',
        title: str = 'Sentiment Distribution',
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        sentiment_counts = df[sentiment_column].value_counts()
        
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        bar_colors = [colors.get(sent, '#3498db') for sent in sentiment_counts.index]
        
        sentiment_counts.plot(kind='bar', ax=ax, color=bar_colors)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_rating_distribution(
        self,
        df: pd.DataFrame,
        rating_column: str = 'rating',
        title: str = 'Rating Distribution',
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        rating_counts = df[rating_column].value_counts().sort_index()
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
        
        rating_counts.plot(kind='bar', ax=ax, color=colors)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Rating (Stars)', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Add value labels
        for i, v in enumerate(rating_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_sentiment_by_rating(
        self,
        df: pd.DataFrame,
        rating_column: str = 'rating',
        sentiment_column: str = 'sentiment_label',
        title: str = 'Sentiment Distribution by Rating',
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create cross-tabulation
        ct = pd.crosstab(df[rating_column], df[sentiment_column], normalize='index') * 100
        
        ct.plot(kind='bar', stacked=True, ax=ax, 
                color={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Rating (Stars)', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        title: str = 'Model Performance Comparison',
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(metrics_dict.keys())
        metrics = ['Accuracy', 'Precision', 'F1 Score']
        
        for idx, metric in enumerate(metrics):
            values = [metrics_dict[model].get(metric.lower().replace(' ', '_'), 0) 
                     for model in models]
            
            axes[idx].bar(models, values, color=['#3498db', '#e74c3c'])
            axes[idx].set_title(metric, fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_ylim([0, 1])
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def create_wordcloud(
        self,
        texts: List[str],
        sentiment: str = 'all',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(combined_text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        if title is None:
            title = f'Word Cloud - {sentiment.title()} Reviews'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved word cloud to {save_path}")
        
        return fig
    
    def plot_confidence_distribution(
        self,
        df: pd.DataFrame,
        confidence_column: str = 'llm_confidence',
        sentiment_column: str = 'llm_sentiment_normalized',
        title: str = 'Model Confidence Distribution',
        save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        for sentiment in df[sentiment_column].unique():
            subset = df[df[sentiment_column] == sentiment][confidence_column]
            ax.hist(subset, alpha=0.6, label=sentiment, bins=20)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        df: pd.DataFrame,
        rating_column: str = 'rating',
        sentiment_column: str = 'sentiment_label',
        save_path: Optional[str] = None
    ):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Sentiment distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts = df[sentiment_column].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        bar_colors = [colors.get(sent, '#3498db') for sent in sentiment_counts.index]
        sentiment_counts.plot(kind='bar', ax=ax1, color=bar_colors)
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Rating distribution
        ax2 = fig.add_subplot(gs[0, 1])
        rating_counts = df[rating_column].value_counts().sort_index()
        rating_colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
        rating_counts.plot(kind='bar', ax=ax2, color=rating_colors)
        ax2.set_title('Rating Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rating (Stars)')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=0)
        
        # Sentiment by rating
        ax3 = fig.add_subplot(gs[1, :])
        ct = pd.crosstab(df[rating_column], df[sentiment_column], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=ax3,
                color={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
        ax3.set_title('Sentiment Distribution by Rating', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Rating (Stars)')
        ax3.set_ylabel('Percentage (%)')
        ax3.tick_params(axis='x', rotation=0)
        ax3.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle('Sentiment Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dashboard to {save_path}")
        
        return fig


def gen_plots(
    df: pd.DataFrame,
    output_dir: str = 'results/figures',
    rating_column: str = 'rating',
    sentiment_column: str = 'sentiment_label'
):
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = SentimentVisualizer()
    
    # Create all plots
    visualizer.plot_sentiment_distribution(
        df, sentiment_column, 
        save_path=f'{output_dir}/sentiment_distribution.png'
    )
    
    visualizer.plot_rating_distribution(
        df, rating_column,
        save_path=f'{output_dir}/rating_distribution.png'
    )
    
    visualizer.plot_sentiment_by_rating(
        df, rating_column, sentiment_column,
        save_path=f'{output_dir}/sentiment_by_rating.png'
    )
    
    visualizer.create_dashboard(
        df, rating_column, sentiment_column,
        save_path=f'{output_dir}/dashboard.png'
    )
    
    # Word clouds for each sentiment
    for sentiment in df[sentiment_column].unique():
        texts = df[df[sentiment_column] == sentiment]['text'].tolist()
        visualizer.create_wordcloud(
            texts, sentiment,
            save_path=f'{output_dir}/wordcloud_{sentiment}.png'
        )
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'rating': [5, 5, 4, 4, 3, 2, 1, 1, 5, 4],
        'sentiment_label': ['positive', 'positive', 'positive', 'positive', 'neutral',
                           'negative', 'negative', 'negative', 'positive', 'positive'],
        'text': ['Great product!', 'Love it!', 'Good quality', 'Works well', 'It\'s okay',
                'Not great', 'Poor quality', 'Waste of money', 'Excellent!', 'Very satisfied']
    }
    
    df = pd.DataFrame(sample_data)
    
    visualizer = SentimentVisualizer()
    
    # Create dashboard
    visualizer.create_dashboard(df)
    plt.show()

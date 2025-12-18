"""
Handles cleaning and preprocessing of review text for NLP analysis
"""

import re
import string
import nltk
import pandas as pd
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        remove_numbers: bool = False,
        lemmatize: bool = True,
        stem: bool = False
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        if self.stem:
            self.stemmer = PorterStemmer()
    
    def _download_nltk_data(self):
        packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for package in packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    pass
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize or stem
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        elif self.stem:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Join tokens back
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_column: str = 'processed_text'
    ) -> pd.DataFrame:
        print(f"Preprocessing {len(df)} reviews...")
        df[output_column] = df[text_column].apply(self.preprocess)
        
        # Remove empty processed texts
        initial_count = len(df)
        df = df[df[output_column].str.len() > 0]
        removed = initial_count - len(df)
        
        if removed > 0:
            print(f"Removed {removed} empty reviews after preprocessing")
        
        return df
    
    def extract_features(self, text: str) -> dict:
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'sentence_count': len(re.split(r'[.!?]+', text))
        }
        
        return features


def prepare_review_data(df: pd.DataFrame) -> pd.DataFrame:
    # Select relevant columns
    columns_to_keep = ['rating', 'title', 'text', 'verified_purchase', 'timestamp']
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns].copy()
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['full_text'] = df['text'].fillna('')
    else:
        raise ValueError("No text column found in DataFrame")
    
    # Create sentiment labels from ratings
    if 'rating' in df.columns:
        df['sentiment_label'] = df['rating'].apply(lambda x: 
            'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
        )
        df['sentiment_numeric'] = df['rating'].apply(lambda x:
            1 if x >= 4 else (-1 if x <= 2 else 0)
        )
    
    # Preprocess text
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,  # Keep some punctuation for sentiment
        remove_stopwords=False,    # Keep stopwords for better context
        lemmatize=True
    )
    
    df = preprocessor.preprocess_dataframe(df, 'full_text', 'processed_text')
    
    # Extract features
    print("Extracting text features...")
    feature_dicts = df['full_text'].apply(preprocessor.extract_features)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    df = pd.concat([df, feature_df], axis=1)
    
    print(f"Prepared {len(df)} reviews for analysis")
    
    return df


if __name__ == "__main__":
    # Tesing TextPreprocessor
    sample_reviews = [
        "This product is absolutely amazing! Best purchase ever!!!",
        "Terrible quality. Broke after one day. Don't waste your money.",
        "It's okay, nothing special but does the job.",
        "LOVE IT! Works perfectly and shipping was fast."
    ]
    
    preprocessor = TextPreprocessor()
    
    print("Original vs Preprocessed Text:\n")
    for review in sample_reviews:
        processed = preprocessor.preprocess(review)
        print(f"Original:    {review}")
        print(f"Processed:   {processed}")
        print(f"Features:    {preprocessor.extract_features(review)}\n")

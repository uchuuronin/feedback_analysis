"""
    Data Loader for Amazon Reviews 2023 Dataset
    Dataset info: https://amazon-reviews-2023.github.io/
"""

import os
import json
import gzip
import requests
import pandas as pd
from typing import Optional, List, Dict
from tqdm import tqdm


class AmazonReviewLoader:
    BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
    CATEGORIES = [
        'All_Beauty',
        'Amazon_Fashion', 
        'Appliances',
        'Arts_Crafts_and_Sewing',
        'Automotive',
        'Books',
        'CDs_and_Vinyl',
        'Cell_Phones_and_Accessories',
        'Clothing_Shoes_and_Jewelry',
        'Digital_Music',
        'Electronics',
        'Gift_Cards',
        'Grocery_and_Gourmet_Food',
        'Handmade_Products',
        'Health_and_Household',
        'Health_and_Personal_Care',
        'Home_and_Kitchen',
        'Industrial_and_Scientific',
        'Kindle_Store',
        'Magazine_Subscriptions',
        'Movies_and_TV',
        'Musical_Instruments',
        'Office_Products',
        'Patio_Lawn_and_Garden',
        'Pet_Supplies',
        'Prime_Pantry',
        'Software',
        'Sports_and_Outdoors',
        'Tools_and_Home_Improvement',
        'Toys_and_Games',
        'Video_Games'
    ]
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_category(self, category: str, force: bool = False) -> str:
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")
        
        filename = f"{category}.jsonl.gz"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath) and not force:
            print(f"{filename} already exists")
            return filepath
        url = f"{self.BASE_URL}{filename}"
        print(f"Downloading {category} reviews...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=category
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            return filepath
            
        except Exception as e:
            print(f"ERROR downloading {category}: {str(e)}")
            raise
    
    def load_reviews(
        self, 
        category: str,
        n_samples: Optional[int] = None,
        rating_filter: Optional[List[int]] = None
    ) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, f"{category}.jsonl.gz")
        
        # Download if category does not exist
        if not os.path.exists(filepath):
            self.download_category(category)
        
        reviews = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reading reviews")):
                if n_samples and i >= n_samples:
                    break
                
                try:
                    review = json.loads(line)
                    if rating_filter and review.get('rating') not in rating_filter:
                        continue
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(reviews)
        print(f"Loaded {len(df)} reviews")
        return df
    
    def create_sample_dataset(
        self,
        categories: List[str],
        samples_per_category: int = 1000,
        output_file: str = 'data/processed/mixed_review_set.csv'
    ) -> pd.DataFrame:
        all_reviews = []
        
        for category in categories:
            try:
                df = self.load_reviews(category, n_samples=samples_per_category)
                df['category'] = category
                all_reviews.append(df)
            except Exception as e:
                print(f"Skipping {category}: {str(e)}")
                continue
        
        combined_df = pd.concat(all_reviews, ignore_index=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} reviews to {output_file}")
        return combined_df


def load_amazon_reviews(
    category: str = 'Electronics',
    n_samples: int = 10000,
    data_dir: str = 'data/raw'
) -> pd.DataFrame:
    loader = AmazonReviewLoader(data_dir)
    return loader.load_reviews(category, n_samples=n_samples)


if __name__ == "__main__":
    loader = AmazonReviewLoader()
    df = loader.load_reviews('Electronics', n_samples=5000)
    sample_df = loader.create_sample_dataset(
        categories=['Electronics', 'Cell_Phones_and_Accessories', 'Home_and_Kitchen','Appliances'],
        samples_per_category=1000
    )

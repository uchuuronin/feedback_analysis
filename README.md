# Customer Feedback Sentiment Analyzer

A NLP-exploratory project that analyzes Amazon product reviews using both traditional (TF-IDF) and recent (LLM-based) approaches to extract sentiment insights and generate customer feedback summaries.

## Dataset

Using the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset, which contains:
- 571M+ reviews
- 48M+ products
- Multiple product categories

## Execution Help

#### Installations

```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

#### Running main files

```bash
# Part 1: TF-IDF Analysis
python src/tf_idf.py --data data/raw/reviews.json --output results/

python src/tf_idf.py --category Electronics --n_samples 5000 --compare_models

# Part 2: LLM Analysis and Comparison
python src/llm.py --data data/raw/reviews.json --output results/
python src/llm.py --n_samples 2000 --llm_model distilbert

# Or run demo 
python demo.py
```

## Tech Stack

- **Python 3.10+**
- **NLP**: scikit-learn, NLTK, spaCy
- **LLM**: HuggingFace Transformers, OpenAI API
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

## Context History
Exploratory project done to understand NLP concepts practically on 2024 
Version: 1.2 (last updated Dec 2025)
Created: 2024

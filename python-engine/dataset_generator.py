"""
Dataset Generator for creating synthetic sentiment analysis test data
"""

import pandas as pd
import random

class DatasetGenerator:
    """Generate synthetic datasets for sentiment analysis testing"""
    
    def __init__(self):
        # Sample texts for different sentiments
        self.positive_texts = [
            "This product is absolutely amazing! Best purchase ever.",
            "I love this so much! Highly recommend to everyone.",
            "Fantastic quality and excellent service!",
            "This exceeded all my expectations. So happy!",
            "Great experience from start to finish!",
            "Wonderful product, will definitely buy again!",
            "Outstanding quality and fast delivery!",
            "I'm very pleased with this purchase!",
            "Excellent value for money!",
            "This is exactly what I needed!",
            "Superb quality and great customer service!",
            "Absolutely delighted with this product!",
            "Five stars all the way!",
            "This is a game changer!",
            "Could not be happier with my purchase!"
        ]
        
        self.negative_texts = [
            "This is the worst product I've ever bought.",
            "Terrible quality and poor customer service.",
            "Very disappointed with this purchase.",
            "Do not buy this! Complete waste of money.",
            "Awful experience, would not recommend.",
            "Poor quality and overpriced.",
            "This broke after one use. Horrible!",
            "Terrible experience from start to finish.",
            "Not worth the money at all.",
            "Very unhappy with this product.",
            "Worst purchase I've made in years.",
            "Completely useless and disappointing.",
            "Save your money and avoid this.",
            "Terrible quality control issues.",
            "This is garbage, don't waste your time."
        ]
        
        self.neutral_texts = [
            "The product arrived on time.",
            "It's okay, nothing special.",
            "Average quality for the price.",
            "It does what it's supposed to do.",
            "Neither good nor bad, just average.",
            "It's fine, meets basic expectations.",
            "Standard product, nothing remarkable.",
            "As described, no surprises.",
            "It's acceptable for the price point.",
            "Meets minimum requirements.",
            "Functional but not impressive.",
            "It works as intended.",
            "Pretty standard, no complaints.",
            "Delivered as expected.",
            "It's a product that exists."
        ]
    
    def generate(self, count: int = 100, distribution: dict = None):
        """
        Generate synthetic dataset
        
        Args:
            count: Number of samples to generate
            distribution: Dictionary with sentiment distribution
                         {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
        
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        if distribution is None:
            distribution = {
                "positive": 0.4,
                "negative": 0.3,
                "neutral": 0.3
            }
        
        # Calculate counts for each sentiment
        pos_count = int(count * distribution.get("positive", 0.33))
        neg_count = int(count * distribution.get("negative", 0.33))
        neu_count = count - pos_count - neg_count
        
        texts = []
        sentiments = []
        
        # Generate positive samples
        for _ in range(pos_count):
            texts.append(random.choice(self.positive_texts))
            sentiments.append("positive")
        
        # Generate negative samples
        for _ in range(neg_count):
            texts.append(random.choice(self.negative_texts))
            sentiments.append("negative")
        
        # Generate neutral samples
        for _ in range(neu_count):
            texts.append(random.choice(self.neutral_texts))
            sentiments.append("neutral")
        
        # Create DataFrame and shuffle
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

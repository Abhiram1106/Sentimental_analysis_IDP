"""
Premium Transformer-based Sentiment Analysis
Uses DistilBERT for high-accuracy sentiment analysis
"""

import time
from typing import List, Dict
from multiprocessing import Pool, cpu_count
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class TransformerSentimentAnalyzer:
    """
    Premium sentiment analyzer using DistilBERT
    More accurate than VADER but slower
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize transformer model
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading transformer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def analyze_single(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text using transformer
        
        Returns:
            Dict with sentiment label and confidence scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        probs = predictions[0].cpu().numpy()
        
        # Model outputs: [negative, positive]
        negative_score = float(probs[0])
        positive_score = float(probs[1])
        
        # Determine sentiment
        if positive_score > 0.6:
            sentiment = 'positive'
        elif negative_score > 0.6:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate compound score (-1 to 1)
        compound = positive_score - negative_score
        
        return {
            'sentiment': sentiment,
            'compound': float(compound),
            'positive': float(positive_score),
            'negative': float(negative_score),
            'neutral': float(1.0 - abs(compound)),
            'confidence': float(max(positive_score, negative_score))
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment of multiple texts in batches (more efficient)
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
        
        Returns:
            List of sentiment results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process each result in batch
            for j, probs in enumerate(predictions.cpu().numpy()):
                negative_score = float(probs[0])
                positive_score = float(probs[1])
                
                if positive_score > 0.6:
                    sentiment = 'positive'
                elif negative_score > 0.6:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                compound = positive_score - negative_score
                
                results.append({
                    'sentiment': sentiment,
                    'compound': float(compound),
                    'positive': float(positive_score),
                    'negative': float(negative_score),
                    'neutral': float(1.0 - abs(compound)),
                    'confidence': float(max(positive_score, negative_score))
                })
        
        return results
    
    def analyze_sequential(self, texts: List[str]) -> Dict:
        """
        Sequential analysis with batching for efficiency
        
        Returns:
            Dict with results and timing
        """
        start_time = time.time()
        
        detailed_results = self.analyze_batch(texts, batch_size=32)
        
        # Aggregate results
        results = {
            'positive': sum(1 for r in detailed_results if r['sentiment'] == 'positive'),
            'negative': sum(1 for r in detailed_results if r['sentiment'] == 'negative'),
            'neutral': sum(1 for r in detailed_results if r['sentiment'] == 'neutral')
        }
        
        processing_time = time.time() - start_time
        
        return {
            'summary': results,
            'detailed_results': detailed_results,
            'processing_time': processing_time,
            'method': 'transformer_sequential',
            'total_processed': len(texts),
            'avg_confidence': float(np.mean([r['confidence'] for r in detailed_results]))
        }
    
    def analyze_parallel(self, texts: List[str], num_workers: int = None) -> Dict:
        """
        Parallel analysis (splits into chunks, processes sequentially within each)
        
        Note: Transformers are already optimized with batching,
        so parallel processing may not provide significant speedup
        """
        return self.analyze_sequential(texts)  # Batching is more efficient than multiprocessing


if __name__ == "__main__":
    # Test the transformer analyzer
    analyzer = TransformerSentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this! It's amazing!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
        "Best product ever! Highly recommend!",
        "Waste of money, very disappointed."
    ]
    
    print("\nTesting transformer analyzer:")
    for text in test_texts:
        result = analyzer.analyze_single(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"Scores - Positive: {result['positive']:.2f}, Negative: {result['negative']:.2f}")

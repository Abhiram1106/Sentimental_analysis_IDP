"""
Trend Analysis for Sentiment Over Time
Analyzes sentiment patterns and trends
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression


class TrendAnalyzer:
    """Analyze sentiment trends over time"""
    
    def __init__(self):
        """Initialize trend analyzer"""
        pass
    
    def create_time_series(
        self,
        sentiments: List[Dict],
        timestamps: Optional[List[datetime]] = None,
        interval: str = 'hour'
    ) -> Dict:
        """
        Create time series aggregation of sentiments
        
        Args:
            sentiments: List of sentiment results
            timestamps: Optional timestamps for each sentiment
            interval: Aggregation interval ('minute', 'hour', 'day', 'week')
        
        Returns:
            Time series data with aggregated sentiments
        """
        # If no timestamps provided, create sequential ones
        if timestamps is None:
            base_time = datetime.now() - timedelta(hours=len(sentiments))
            timestamps = [base_time + timedelta(hours=i) for i in range(len(sentiments))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sentiment': [s['sentiment'] for s in sentiments],
            'compound': [s.get('compound', 0) for s in sentiments],
            'confidence': [s.get('confidence', 0) for s in sentiments]
        })
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Determine resampling rule
        resample_rule = {
            'minute': '1T',
            'hour': '1H',
            'day': '1D',
            'week': '1W'
        }.get(interval, '1H')
        
        # Aggregate by time interval
        time_series = []
        
        # Resample and calculate metrics
        resampled = df.resample(resample_rule)
        
        for timestamp, group in resampled:
            if len(group) == 0:
                continue
            
            positive_count = (group['sentiment'] == 'positive').sum()
            negative_count = (group['sentiment'] == 'negative').sum()
            neutral_count = (group['sentiment'] == 'neutral').sum()
            total = len(group)
            
            time_series.append({
                'timestamp': timestamp.isoformat(),
                'positive': int(positive_count),
                'negative': int(negative_count),
                'neutral': int(neutral_count),
                'total': int(total),
                'avg_compound': float(group['compound'].mean()),
                'avg_confidence': float(group['confidence'].mean()) if 'confidence' in group else 0,
                'positive_ratio': float(positive_count / total) if total > 0 else 0,
                'negative_ratio': float(negative_count / total) if total > 0 else 0
            })
        
        return {
            'time_series': time_series,
            'interval': interval,
            'total_intervals': len(time_series),
            'total_samples': len(sentiments)
        }
    
    def analyze_trend(self, time_series_data: List[Dict]) -> Dict:
        """
        Analyze trend direction and strength
        
        Args:
            time_series_data: List of time series points
        
        Returns:
            Trend analysis results
        """
        if len(time_series_data) < 2:
            return {'error': 'Need at least 2 data points for trend analysis'}
        
        # Extract compound scores over time
        compounds = [point['avg_compound'] for point in time_series_data]
        positive_ratios = [point['positive_ratio'] for point in time_series_data]
        negative_ratios = [point['negative_ratio'] for point in time_series_data]
        
        # Time points (0, 1, 2, ...)
        time_points = np.arange(len(compounds)).reshape(-1, 1)
        
        # Linear regression for compound scores
        model = LinearRegression()
        model.fit(time_points, compounds)
        slope = model.coef_[0]
        r_squared = model.score(time_points, compounds)
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'improving'
        else:
            trend_direction = 'declining'
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(time_points.flatten(), compounds)
        
        # Calculate moving average (if enough points)
        moving_avg = []
        window = min(3, len(compounds))
        for i in range(len(compounds)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(float(np.mean(compounds[start_idx:i+1])))
        
        # Volatility (standard deviation)
        volatility = float(np.std(compounds))
        
        # Peak positive and negative periods
        max_positive_idx = np.argmax(positive_ratios)
        max_negative_idx = np.argmax(negative_ratios)
        
        return {
            'trend_direction': trend_direction,
            'slope': float(slope),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'p_value': float(p_value),
            'volatility': volatility,
            'moving_average': moving_avg,
            'peak_positive': {
                'timestamp': time_series_data[max_positive_idx]['timestamp'],
                'ratio': float(positive_ratios[max_positive_idx])
            },
            'peak_negative': {
                'timestamp': time_series_data[max_negative_idx]['timestamp'],
                'ratio': float(negative_ratios[max_negative_idx])
            },
            'overall_avg_compound': float(np.mean(compounds)),
            'trend_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
        }
    
    def predict_next(self, time_series_data: List[Dict], periods: int = 3) -> Dict:
        """
        Predict next sentiment values using linear regression
        
        Args:
            time_series_data: Historical time series data
            periods: Number of periods to predict
        
        Returns:
            Predictions for next periods
        """
        if len(time_series_data) < 3:
            return {'error': 'Need at least 3 data points for prediction'}
        
        # Extract data
        compounds = [point['avg_compound'] for point in time_series_data]
        time_points = np.arange(len(compounds)).reshape(-1, 1)
        
        # Train model
        model = LinearRegression()
        model.fit(time_points, compounds)
        
        # Predict next periods
        future_points = np.arange(len(compounds), len(compounds) + periods).reshape(-1, 1)
        predictions = model.predict(future_points)
        
        # Determine predicted sentiments
        predicted_sentiments = []
        for pred in predictions:
            if pred >= 0.05:
                sentiment = 'positive'
            elif pred <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            predicted_sentiments.append({
                'compound': float(pred),
                'sentiment': sentiment
            })
        
        return {
            'predictions': predicted_sentiments,
            'periods_ahead': periods,
            'confidence': float(model.score(time_points, compounds))
        }
    
    def generate_summary(self, sentiments: List[Dict], timestamps: Optional[List[datetime]] = None) -> Dict:
        """
        Generate comprehensive trend summary
        
        Args:
            sentiments: List of sentiment results
            timestamps: Optional timestamps
        
        Returns:
            Complete trend analysis summary
        """
        # Create time series
        ts_result = self.create_time_series(sentiments, timestamps, interval='hour')
        
        # Analyze trend
        trend_result = self.analyze_trend(ts_result['time_series'])
        
        # Predict future
        prediction_result = self.predict_next(ts_result['time_series'], periods=3)
        
        return {
            'time_series': ts_result,
            'trend_analysis': trend_result,
            'predictions': prediction_result,
            'generated_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    from sentiment_analyzer import SentimentAnalyzer
    from dataset_generator import DatasetGenerator
    
    # Generate sample data
    analyzer = SentimentAnalyzer()
    generator = DatasetGenerator()
    df = generator.generate(count=100)
    
    # Analyze sentiments
    texts = df['text'].tolist()
    result = analyzer.analyze_parallel(texts)
    sentiments = result['detailed_results']
    
    # Analyze trends
    trend_analyzer = TrendAnalyzer()
    summary = trend_analyzer.generate_summary(sentiments)
    
    print("\nTrend Analysis Summary:")
    print(f"Direction: {summary['trend_analysis']['trend_direction']}")
    print(f"Strength: {summary['trend_analysis']['trend_strength']}")
    print(f"Correlation: {summary['trend_analysis']['correlation']:.3f}")
    print(f"\nPredicted next 3 periods:")
    for i, pred in enumerate(summary['predictions']['predictions'], 1):
        print(f"  {i}. {pred['sentiment']} (compound: {pred['compound']:.3f})")

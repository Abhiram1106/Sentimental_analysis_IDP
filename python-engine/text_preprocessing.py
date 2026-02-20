"""
Advanced Text Preprocessing for Higher Accuracy
Cleans and normalizes text before sentiment analysis
"""

import re
import html
from typing import List, Dict
import emoji


class TextPreprocessor:
    """
    Advanced text preprocessing to improve sentiment analysis accuracy
    Handles emojis, URLs, mentions, hashtags, and more
    """
    
    def __init__(self, 
                 lowercase: bool = False,
                 remove_urls: bool = False,
                 remove_mentions: bool = False,
                 remove_hashtags: bool = False,
                 expand_contractions: bool = True,
                 handle_emojis: str = 'convert'):  # 'convert', 'remove', or 'keep'
        """
        Initialize preprocessor with configuration
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            expand_contractions: Expand contractions (don't -> do not)
            handle_emojis: How to handle emojis ('convert', 'remove', 'keep')
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.expand_contractions = expand_contractions
        self.handle_emojis = handle_emojis
        
        # Contraction mappings
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have",
            "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would",
            "he's": "he is", "how'd": "how did",
            "how's": "how is", "i'd": "i would",
            "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it's": "it is",
            "let's": "let us", "might've": "might have",
            "must've": "must have", "needn't": "need not",
            "shan't": "shall not", "she'd": "she would",
            "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "that'd": "that would",
            "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not",
            "we'd": "we would", "we'll": "we will",
            "we're": "we are", "we've": "we have",
            "weren't": "were not", "what're": "what are",
            "what's": "what is", "when's": "when is",
            "where'd": "where did", "where's": "where is",
            "who'd": "who would", "who's": "who is",
            "won't": "will not", "wouldn't": "would not",
            "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        
        # Emoji sentiment mappings (basic)
        self.emoji_sentiments = {
            '😊': 'happy', '😃': 'happy', '😄': 'happy', '😁': 'happy',
            '😍': 'love', '🥰': 'love', '❤️': 'love', '💕': 'love',
            '😢': 'sad', '😭': 'sad', '😞': 'sad', '😔': 'sad',
            '😠': 'angry', '😡': 'angry', '👿': 'angry',
            '👍': 'good', '👎': 'bad', '✅': 'good', '❌': 'bad'
        }
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text
        
        Args:
            text: Raw text input
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # HTML decode
        text = html.unescape(text)
        
        # Handle emojis
        if self.handle_emojis == 'convert':
            text = self._convert_emojis(text)
        elif self.handle_emojis == 'remove':
            text = self._remove_emojis(text)
        # else keep emojis as-is
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (but keep the text)
        if self.remove_hashtags:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Trim
        text = text.strip()
        
        return text
    
    def _convert_emojis(self, text: str) -> str:
        """Convert emojis to text descriptions"""
        try:
            # Try using emoji library if available
            text = emoji.demojize(text, delimiters=(" ", " "))
            # Clean up underscores
            text = text.replace('_', ' ')
        except:
            # Fallback: use basic mappings
            for em, sentiment in self.emoji_sentiments.items():
                text = text.replace(em, f' {sentiment} ')
        
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove all emojis from text"""
        try:
            return emoji.replace_emoji(text, replace='')
        except:
            # Fallback: regex pattern for emojis
            emoji_pattern = re.compile(
                "["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            return emoji_pattern.sub(r'', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions (don't -> do not)"""
        words = text.split()
        expanded = []
        
        for word in words:
            # Check lowercase version against contractions
            lower_word = word.lower()
            if lower_word in self.contractions:
                expanded.append(self.contractions[lower_word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts
        
        Args:
            texts: List of raw texts
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def analyze_text_quality(self, text: str) -> Dict:
        """
        Analyze text characteristics that may affect sentiment analysis
        
        Returns:
            Dict with text quality metrics
        """
        has_urls = bool(re.search(r'http\S+|www\.\S+', text))
        has_mentions = bool(re.search(r'@\w+', text))
        has_hashtags = bool(re.search(r'#\w+', text))
        
        try:
            emoji_count = emoji.emoji_count(text)
        except:
            emoji_count = sum(text.count(em) for em in self.emoji_sentiments.keys())
        
        word_count = len(text.split())
        char_count = len(text)
        
        # Check for excessive punctuation (may indicate emotion)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'has_urls': has_urls,
            'has_mentions': has_mentions,
            'has_hashtags': has_hashtags,
            'emoji_count': emoji_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': round(caps_ratio, 3),
            'quality_score': self._calculate_quality_score(
                word_count, emoji_count, caps_ratio, exclamation_count
            )
        }
    
    def _calculate_quality_score(self, word_count, emoji_count, 
                                 caps_ratio, exclamation_count) -> float:
        """Calculate text quality score (0-1)"""
        score = 1.0
        
        # Penalize very short texts
        if word_count < 3:
            score -= 0.3
        
        # Penalize excessive emojis
        if emoji_count > word_count / 2:
            score -= 0.2
        
        # Penalize excessive caps
        if caps_ratio > 0.5:
            score -= 0.2
        
        # Penalize excessive punctuation
        if exclamation_count > 3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))


# Preset configurations
PRESETS = {
    'social_media': TextPreprocessor(
        lowercase=False,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        expand_contractions=True,
        handle_emojis='convert'
    ),
    'reviews': TextPreprocessor(
        lowercase=False,
        remove_urls=True,
        remove_mentions=False,
        remove_hashtags=False,
        expand_contractions=True,
        handle_emojis='convert'
    ),
    'formal': TextPreprocessor(
        lowercase=False,
        remove_urls=False,
        remove_mentions=False,
        remove_hashtags=False,
        expand_contractions=True,
        handle_emojis='remove'
    ),
    'minimal': TextPreprocessor(
        lowercase=False,
        remove_urls=False,
        remove_mentions=False,
        remove_hashtags=False,
        expand_contractions=False,
        handle_emojis='keep'
    )
}


def get_preprocessor(preset: str = 'reviews') -> TextPreprocessor:
    """
    Get preprocessor with preset configuration
    
    Args:
        preset: 'social_media', 'reviews', 'formal', or 'minimal'
    
    Returns:
        TextPreprocessor instance
    """
    if preset in PRESETS:
        return PRESETS[preset]
    else:
        return PRESETS['reviews']


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = get_preprocessor('social_media')
    
    test_texts = [
        "I can't believe how amazing this is! 😍 #love",
        "This is terrible don't buy it 😠",
        "Check out this link: https://example.com @user",
        "IT'S THE BEST PRODUCT EVER!!!!"
    ]
    
    print("Text Preprocessing Examples:\n" + "="*50)
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        quality = preprocessor.analyze_text_quality(text)
        
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")
        print(f"Quality Score: {quality['quality_score']:.2f}")

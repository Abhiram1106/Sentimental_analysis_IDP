"""
Ultra Advanced Sentiment Dataset Generator
Creates realistic internet-style comments

Includes:
✔ long paragraph reviews
✔ mixed sentiment paragraphs
✔ social media comments
✔ Hinglish style comments
✔ sarcasm & comparisons
✔ Twitter-length comments
"""

import pandas as pd
import random


class DatasetGenerator:
    """Advanced dataset generator compatible with existing API"""

    def __init__(self):

        # Emojis
        self.emojis_positive = ["😍", "🔥", "✨", "🙌", "❤️", "😊"]
        self.emojis_negative = ["😡", "🤦", "😤", "👎", "💔", "😑"]
        self.emojis_neutral = ["🙂", "🤔", "😐"]

        # Hashtags
        self.hashtags = [
            "#happy", "#worthit", "#fail", "#notbad",
            "#disappointed", "#loveit", "#average",
            "#bestpurchase", "#wasteofmoney"
        ]

        # Hinglish slang
        self.hinglish_positive = [
            "yeh product sach mein solid hai",
            "paisa vasool item hai",
            "full satisfaction mila",
            "bilkul mast kaam kar raha hai"
        ]

        self.hinglish_negative = [
            "bilkul bekaar nikla",
            "paisa barbaad ho gaya",
            "itna bakwaas expect nahi kiya",
            "total time waste hai"
        ]

        self.hinglish_neutral = [
            "theek thaak hai",
            "normal hi laga use karne mein",
            "zyada special nahi hai",
            "kaam chalau hai"
        ]

        # Experience phrases
        self.positive_experience = [
            "performance is outstanding",
            "works flawlessly every day",
            "quality feels premium",
            "super smooth and reliable",
            "exceeded my expectations",
            "makes my daily work easier"
        ]

        self.negative_experience = [
            "performance is terrible",
            "stopped working suddenly",
            "quality feels cheap",
            "very frustrating to use",
            "completely disappointing",
            "not worth the price"
        ]

        self.neutral_experience = [
            "works as expected",
            "quality is acceptable",
            "nothing stands out",
            "performance is average",
            "does the job",
            "meets basic needs"
        ]

    # -------- SHORT & MEDIUM STYLES -------- #

    def review_style(self, sentiment):
        if sentiment == "positive":
            return f"I have been using this for weeks and the {random.choice(self.positive_experience)}. Delivery was quick and overall I am very satisfied {random.choice(self.emojis_positive)}"
        if sentiment == "negative":
            return f"After a few uses, the {random.choice(self.negative_experience)}. Support was unhelpful and I regret buying this {random.choice(self.emojis_negative)}"
        return f"I used it for some time and it {random.choice(self.neutral_experience)}. Nothing remarkable but works fine {random.choice(self.emojis_neutral)}"

    def social_media_style(self, sentiment):
        phrase = random.choice(
            self.positive_experience if sentiment == "positive"
            else self.negative_experience if sentiment == "negative"
            else self.neutral_experience
        )

        emoji = random.choice(
            self.emojis_positive if sentiment == "positive"
            else self.emojis_negative if sentiment == "negative"
            else self.emojis_neutral
        )

        return f"Just tried this today — {phrase}! {emoji} {random.choice(self.hashtags)}"

    def hinglish_style(self, sentiment):
        if sentiment == "positive":
            return f"{random.choice(self.hinglish_positive)}, seriously impressed!"
        if sentiment == "negative":
            return f"{random.choice(self.hinglish_negative)}, totally disappointed."
        return f"{random.choice(self.hinglish_neutral)}, overall okay experience."

    def sarcasm_style(self, sentiment):
        if sentiment == "positive":
            return "Wow, something that actually works as promised. What a rare surprise."
        if sentiment == "negative":
            return "Amazing… broke on day one. Exactly what I needed."
        return "Well, it works… I guess that's something."

    def story_style(self, sentiment):
        if sentiment == "positive":
            return "I bought this for daily use and after a month it has made my routine easier and faster."
        if sentiment == "negative":
            return "I bought this hoping it would help, but it ended up creating more problems."
        return "I started using it last week and so far the experience has been normal."

    def comparison_style(self, sentiment):
        if sentiment == "positive":
            return "Compared to similar products, this performs better and feels more reliable."
        if sentiment == "negative":
            return "Compared to others I've used, this performs worse and feels poorly made."
        return "Compared to similar options, this performs about the same."

    def question_style(self, sentiment):
        if sentiment == "positive":
            return "Is it just me or is this surprisingly good for the price?"
        if sentiment == "negative":
            return "Why does this stop working every time I need it most?"
        return "Has anyone else noticed it works fine but nothing special?"

    def twitter_style(self, sentiment):
        if sentiment == "positive":
            tweets = [
                "Just bought this and wow 😍 performance is smooth and fast! #loveit",
                "Didn't expect this quality 🔥 totally worth it!",
                "Using this daily and it never disappoints 🙌",
                "Finally something that works perfectly ❤️",
                "Super impressed ✨"
            ]
        elif sentiment == "negative":
            tweets = [
                "Stopped working in 2 days 😡 waste of money #fail",
                "Looks good but performs terribly 👎",
                "App keeps crashing… so frustrating 😤",
                "Totally not worth the price 💔",
                "Support not responding at all 🤦"
            ]
        else:
            tweets = [
                "It works fine 🙂 nothing extraordinary.",
                "Average performance 🤔",
                "Not bad, not great.",
                "Seems decent so far.",
                "Fair for the price 😐"
            ]
        return random.choice(tweets)

    # -------- LONG PARAGRAPH COMMENTS -------- #

    def long_review_style(self, sentiment):

        if sentiment == "positive":
            return (
                "I have been using this product daily for the past few weeks and the experience has been excellent. "
                "The build quality feels premium and the performance remains smooth even during extended use. "
                "Setup was quick and everything worked right out of the box. "
                "It has improved my daily routine and made tasks more efficient. "
                "Considering the price and features, this is one of the best purchases I've made recently."
            )

        elif sentiment == "negative":
            return (
                "After using this product for several days, I am very disappointed with its performance. "
                "The build quality feels fragile and problems started appearing sooner than expected. "
                "Basic features do not function reliably and the overall experience has been frustrating. "
                "Customer support was slow and unhelpful when I reported the issue. "
                "Given the price and promises made, this product fails to deliver the reliability I expected."
            )

        else:
            return (
                "I have been using this product for a short time and my experience has been fairly average. "
                "Setup was simple and the product performs its basic functions without major issues. "
                "The design and build quality are acceptable, though nothing stands out as exceptional. "
                "It works as intended and meets basic needs, but there is room for improvement. "
                "At this price point, it offers a standard overall experience."
            )

    def mixed_sentiment_style(self):
        templates = [
            "The design looks amazing and performance is smooth, but the battery drains faster than expected and heats up during long use.",
            "Setup was simple and the interface is user-friendly, however the software occasionally freezes which can be frustrating.",
            "The product feels premium and everyday tasks run smoothly, but performance drops significantly under heavy usage.",
            "Delivery was fast and initial performance was satisfying, but small bugs started appearing after a week of use.",
            "It works great for basic tasks and looks stylish, although the price feels a bit high for the performance offered."
        ]
        return random.choice(templates)

    # -------- GENERATOR LOGIC -------- #

    def generate_comment(self, sentiment):

        # 20% mixed sentiment paragraphs
        if random.random() < 0.2:
            return self.mixed_sentiment_style()

        # 30% long detailed reviews
        if random.random() < 0.3:
            return self.long_review_style(sentiment)

        styles = [
            self.review_style,
            self.social_media_style,
            self.hinglish_style,
            self.sarcasm_style,
            self.story_style,
            self.comparison_style,
            self.question_style,
            self.twitter_style
        ]

        return random.choice(styles)(sentiment)

    def generate(self, count=100, distribution=None):
        """
        Generate dataset compatible with existing API
        
        Args:
            count: Total number of texts to generate
            distribution: Dictionary with sentiment distribution
                         {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
        
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
            texts.append(self.generate_comment("positive"))
            sentiments.append("positive")
        
        # Generate negative samples
        for _ in range(neg_count):
            texts.append(self.generate_comment("negative"))
            sentiments.append("negative")
        
        # Generate neutral samples
        for _ in range(neu_count):
            texts.append(self.generate_comment("neutral"))
            sentiments.append("neutral")
        
        # Create DataFrame and shuffle
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

    def generate_dataset(self, per_category=100, save_path=None):
        """
        Alternative method for standalone usage
        
        Args:
            per_category: Number of samples per sentiment category
            save_path: Optional path to save CSV file
        
        Returns:
            DataFrame with generated data
        """
        texts = []
        sentiments = []

        for _ in range(per_category):
            for sentiment in ["positive", "negative", "neutral"]:
                texts.append(self.generate_comment(sentiment))
                sentiments.append(sentiment)

        df = pd.DataFrame({"text": texts, "sentiment": sentiments})
        df = df.sample(frac=1).reset_index(drop=True)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✅ Dataset saved to {save_path}")

        return df


# Backward compatibility
AdvancedDatasetGenerator = DatasetGenerator


# RUN
if __name__ == "__main__":
    generator = DatasetGenerator()

    # Test the generate method (API compatible)
    print("Testing API-compatible generate() method:")
    df1 = generator.generate(count=10)
    print(df1.head())
    print()

    # Test the standalone method
    print("Testing standalone generate_dataset() method:")
    dataset = generator.generate_dataset(
        per_category=5,
        save_path="advanced_sentiment_dataset.csv"
    )
    print(dataset.head())

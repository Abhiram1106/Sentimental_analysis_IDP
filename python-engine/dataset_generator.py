"""
Advanced Dynamic Dataset Generator
Creates 100+ unique long comments per sentiment category
Perfect for ML training & Parallel Processing experiments
"""

import pandas as pd
import random

class DatasetGenerator:

    def __init__(self):

        # Sentence pools for realistic generation

        self.positive_parts = {
            "openings": [
                "I recently bought this product and I must say",
                "After using this for a few weeks",
                "From the moment I started using it",
                "Honestly, I was surprised to see",
                "At first I wasn’t sure, but now I feel",
                "I ordered this without high expectations but",
                "Right out of the box",
                "I’ve tried many similar products, but",
                "This has been one of the best purchases because",
                "After continuous daily use"
            ],
            "experience": [
                "the performance has been outstanding",
                "everything works smoothly and efficiently",
                "the build quality feels premium and durable",
                "the user experience is simple and intuitive",
                "it performs exactly as promised",
                "the speed and responsiveness are impressive",
                "the design is modern and visually appealing",
                "the features are extremely useful in daily life",
                "it integrates perfectly into my routine",
                "the functionality exceeded my expectations"
            ],
            "support": [
                "customer support was quick and helpful",
                "delivery was fast and packaging was secure",
                "the setup process was simple and quick",
                "installation took only a few minutes",
                "the instructions were clear and easy to follow",
                "updates and maintenance are seamless",
                "the company clearly values customer satisfaction",
                "their service team responded professionally",
                "everything arrived in perfect condition",
                "the overall service experience was excellent"
            ],
            "closing": [
                "I highly recommend this to anyone looking for quality.",
                "this is definitely worth the money.",
                "I would happily purchase this again.",
                "it’s rare to find something this reliable.",
                "this exceeded all my expectations.",
                "I’m extremely satisfied with my decision.",
                "this product truly stands out in its category.",
                "five stars without hesitation.",
                "I’ll be recommending this to friends and family.",
                "overall, a fantastic experience."
            ]
        }

        self.negative_parts = {
            "openings": [
                "I regret buying this product because",
                "After only a few days of use",
                "From the first time I used it",
                "I had high hopes but unfortunately",
                "This turned out to be disappointing since",
                "I wish I had read the reviews earlier because",
                "My experience has been frustrating as",
                "It looked promising online but",
                "I feel like I wasted my money because",
                "Right after unboxing"
            ],
            "experience": [
                "the performance has been extremely poor",
                "it stopped working unexpectedly",
                "the build quality feels cheap and fragile",
                "the interface is confusing and outdated",
                "it fails to perform basic functions",
                "the speed is painfully slow",
                "the product feels poorly designed",
                "it crashes or freezes frequently",
                "the features don’t work as advertised",
                "it creates more problems than it solves"
            ],
            "support": [
                "customer support has been unresponsive",
                "return and refund process is complicated",
                "delivery was delayed without updates",
                "the packaging arrived damaged",
                "the instructions were unclear",
                "service response time is unacceptable",
                "they keep sending automated replies",
                "no proper assistance was provided",
                "the company seems careless about customers",
                "support representatives were not helpful"
            ],
            "closing": [
                "I would not recommend this to anyone.",
                "this was a complete waste of money.",
                "I strongly advise avoiding this product.",
                "it is not worth the price at all.",
                "I will never buy from this brand again.",
                "this has been a terrible experience.",
                "save your money and choose something better.",
                "I’m extremely dissatisfied.",
                "this product needs serious improvement.",
                "overall, very disappointing."
            ]
        }

        self.neutral_parts = {
            "openings": [
                "I have been using this product for a short time and",
                "After trying it out for a few days",
                "My initial impression is that",
                "Based on my experience so far",
                "I purchased this recently and",
                "After setting it up",
                "I decided to try this product and",
                "So far, my experience shows",
                "Upon first use",
                "From a practical standpoint"
            ],
            "experience": [
                "it performs as expected",
                "the quality is acceptable for the price",
                "it delivers standard functionality",
                "there is nothing particularly impressive",
                "it works similarly to other products",
                "the performance is average",
                "the design is simple and functional",
                "it meets basic requirements",
                "the features are fairly standard",
                "it does what it is intended to do"
            ],
            "support": [
                "delivery was on time",
                "packaging was adequate",
                "setup was straightforward",
                "instructions were clear enough",
                "customer service response was average",
                "installation took some time but was manageable",
                "no major issues were encountered",
                "everything arrived as described",
                "the process was routine",
                "service quality was acceptable"
            ],
            "closing": [
                "overall, it meets expectations.",
                "it works fine for basic use.",
                "nothing exceptional but acceptable.",
                "it is an average product.",
                "suitable for everyday needs.",
                "it does the job adequately.",
                "performance is neither good nor bad.",
                "it is fine for the price.",
                "no strong opinions so far.",
                "a standard experience overall."
            ]
        }

    def build_comment(self, parts):
        """Build a long realistic comment"""
        return (
            random.choice(parts["openings"]) + ", " +
            random.choice(parts["experience"]) + ". " +
            random.choice(parts["support"]) + ", and " +
            random.choice(parts["closing"])
        )

    def generate(self, per_category=100, save_path=None):

        texts = []
        sentiments = []

        for _ in range(per_category):
            texts.append(self.build_comment(self.positive_parts))
            sentiments.append("positive")

            texts.append(self.build_comment(self.negative_parts))
            sentiments.append("negative")

            texts.append(self.build_comment(self.neutral_parts))
            sentiments.append("neutral")

        df = pd.DataFrame({"text": texts, "sentiment": sentiments})
        df = df.sample(frac=1).reset_index(drop=True)

        if save_path:
            df.to_csv(save_path, index=False)
            print("Dataset saved to", save_path)

        return df


# Example usage
if __name__ == "__main__":
    generator = DatasetGenerator()

    # 100 comments per category (total 300)
    dataset = generator.generate(per_category=100, save_path="sentiment_dataset.csv")

    print(dataset.head())


import re
import csv
import os
import time
from collections import Counter


class StoryEvaluator:

    def tokenize_words(self, text):
        return re.findall(r"\b[\w']+\b", text.lower())

    def distinct_n(self, text, n):
        words = self.tokenize_words(text)

        if len(words) < n:
            return 0.0

        ngrams = [
            tuple(words[i:i + n])
            for i in range(len(words) - n + 1)
        ]

        return len(set(ngrams)) / len(ngrams)

    def repetition_rate(self, text, n=3):
        words = self.tokenize_words(text)

        if len(words) < n:
            return 0.0

        ngrams = [
            tuple(words[i:i + n])
            for i in range(len(words) - n + 1)
        ]

        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values())

        return repeated / len(ngrams)

    def lexical_diversity(self, text):
        words = self.tokenize_words(text)

        if not words:
            return 0.0

        return len(set(words)) / len(words)

    def evaluate_story(self, text):
        words = self.tokenize_words(text)

        return {
            "word_count": len(words),
            "distinct_1": self.distinct_n(text, 1),
            "distinct_2": self.distinct_n(text, 2),
            "repetition_rate": self.repetition_rate(text),
            "lexical_diversity": self.lexical_diversity(text)
        }

    def measure_generation_time(self, generate_function, *args, **kwargs):
        start_time = time.perf_counter()

        story = generate_function(*args, **kwargs)

        end_time = time.perf_counter()

        return story, end_time - start_time

    def save_to_csv(self, metrics, model_name, filename="results/evaluation_results.csv"):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file_exists = os.path.isfile(filename)

        row = {
            "model_name": model_name,
            **metrics
        }

        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())

            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()

            writer.writerow(row)


if __name__ == "__main__":

    evaluator = StoryEvaluator()

    sample_story = """
    Once upon a time, there was a young boy who lived near a forest.
    Every day he explored the forest and discovered something new.
    One day he found a mysterious glowing stone.
    """

    metrics = evaluator.evaluate_story(sample_story)

    print("\n========== STORY EVALUATION ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    evaluator.save_to_csv(
        metrics,
        model_name="sample_model"
    )

    print("\nResults saved to results/evaluation_results.csv")
    print("======================================")

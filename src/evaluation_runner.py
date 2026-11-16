
import csv
import json
import time
from pathlib import Path
from datetime import datetime

from src.evaluator import StoryEvaluator


class EvaluationRunner:

    def __init__(self):

        self.evaluator = StoryEvaluator()

        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.csv_file = self.results_dir / "combined_evaluation.csv"
        self.json_file = self.results_dir / "combined_evaluation.json"

    # =====================================
    # EVALUATE GENERATED STORY
    # =====================================

    def evaluate_story(
        self,
        model_id,
        model_name,
        architecture,
        tokenizer,
        objective,
        prompt_id,
        prompt,
        story,
        generation_time=None,
        perplexity=None
    ):

        metrics = self.evaluator.evaluate_story(story)

        word_count = metrics["word_count"]

        seconds_per_word = None
        words_per_second = None

        if generation_time is not None and generation_time > 0:

            seconds_per_word = generation_time / max(word_count, 1)

            words_per_second = word_count / generation_time

        record = {

            "timestamp": datetime.now().isoformat(),

            "model_id": model_id,
            "model_name": model_name,
            "architecture": architecture,
            "tokenizer": tokenizer,
            "objective": objective,

            "prompt_id": prompt_id,
            "prompt": prompt,
            "story": story,

            "word_count": word_count,
            "distinct_1": metrics["distinct_1"],
            "distinct_2": metrics["distinct_2"],
            "repetition_rate": metrics["repetition_rate"],
            "lexical_diversity": metrics["lexical_diversity"],

            "generation_time_seconds": generation_time,
            "seconds_per_word": seconds_per_word,
            "words_per_second": words_per_second,

            "perplexity": perplexity
        }

        return record

    # =====================================
    # MEASURE GENERATION TIME
    # =====================================

    def generate_and_evaluate(
        self,
        generate_function,
        model_id,
        model_name,
        architecture,
        tokenizer,
        objective,
        prompt_id,
        prompt
    ):

        start_time = time.perf_counter()

        story = generate_function(prompt)

        end_time = time.perf_counter()

        generation_time = end_time - start_time

        result = self.evaluate_story(

            model_id=model_id,
            model_name=model_name,
            architecture=architecture,
            tokenizer=tokenizer,
            objective=objective,
            prompt_id=prompt_id,
            prompt=prompt,
            story=story,
            generation_time=generation_time

        )

        return result

    # =====================================
    # EXPORT RESULTS TO CSV
    # =====================================

    def export_csv(self, record):

        fieldnames = [

            "timestamp",
            "model_id",
            "model_name",
            "architecture",
            "tokenizer",
            "objective",
            "prompt_id",
            "prompt",
            "generated_story",
            "generation_time_seconds",
            "seconds_per_word",
            "words_per_second",
            "perplexity",
            "word_count",
            "distinct_1",
            "distinct_2",
            "repetition_rate",
            "lexical_diversity"

        ]

        csv_record = {

            "timestamp": record.get("timestamp"),
            "model_id": record.get("model_id"),
            "model_name": record.get("model_name"),
            "architecture": record.get("architecture"),
            "tokenizer": record.get("tokenizer"),
            "objective": record.get("objective"),
            "prompt_id": record.get("prompt_id"),
            "prompt": record.get("prompt"),

            "generated_story": record.get("story"),

            "generation_time_seconds": record.get("generation_time_seconds"),
            "seconds_per_word": record.get("seconds_per_word"),
            "words_per_second": record.get("words_per_second"),

            "perplexity": record.get("perplexity"),

            "word_count": record.get("word_count"),
            "distinct_1": record.get("distinct_1"),
            "distinct_2": record.get("distinct_2"),
            "repetition_rate": record.get("repetition_rate"),
            "lexical_diversity": record.get("lexical_diversity")

        }

        file_exists = (
            self.csv_file.exists()
            and self.csv_file.stat().st_size > 0
        )

        with open(
            self.csv_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore"
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(csv_record)

        print("CSV export successful.")

    # =====================================
    # EXPORT RESULTS TO JSON
    # =====================================

    def export_json(self, record):

        if self.json_file.exists():

            with open(
                self.json_file,
                "r",
                encoding="utf-8"
            ) as file:

                try:
                    data = json.load(file)

                    if not isinstance(data, list):
                        data = []

                except json.JSONDecodeError:
                    data = []

        else:
            data = []

        data.append(record)

        with open(
            self.json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("JSON export successful.")

    # =====================================
    # RUN COMPLETE EVALUATION
    # =====================================

    def run_evaluation(self, **story_data):

        record = self.evaluate_story(**story_data)

        self.export_csv(record)
        self.export_json(record)

        self.display_result(record)

        return record

    # =====================================
    # DISPLAY RESULTS
    # =====================================

    def display_result(self, record):

        print("\n========== EVALUATION RESULT ==========")

        print("Model:", record["model_name"])
        print("Architecture:", record["architecture"])
        print("Tokenizer:", record["tokenizer"])
        print("Objective:", record["objective"])

        print("\nSTORY METRICS")

        print("Word Count:", record["word_count"])
        print("Distinct-1:", record["distinct_1"])
        print("Distinct-2:", record["distinct_2"])
        print("Repetition Rate:", record["repetition_rate"])
        print("Lexical Diversity:", record["lexical_diversity"])

        print("\nGENERATION PERFORMANCE")

        print("Generation Time:", record["generation_time_seconds"])
        print("Seconds Per Word:", record["seconds_per_word"])
        print("Words Per Second:", record["words_per_second"])

        print("Perplexity:", record["perplexity"])

        print("\nResults saved successfully.")

        print("=======================================\n")


# =====================================
# MAIN PROGRAM
# =====================================

if __name__ == "__main__":

    runner = EvaluationRunner()

    print("========== GENERATION TIME TEST ==========")

    prompt = "A boy discovers a mysterious forest."

    # Dummy generator for testing only
    def dummy_generator(prompt):

        time.sleep(2)

        return (
            "A young boy was walking through a forest. "
            "He discovered a mysterious glowing stone. "
            "He picked it up and suddenly heard a voice. "
            "The stone showed him a magical path."
        )

    result = runner.generate_and_evaluate(

        generate_function=dummy_generator,

        model_id="TEST01",
        model_name="Dummy Model",
        architecture="GRU",
        tokenizer="BPE",
        objective="MLE",

        prompt_id="P01",
        prompt=prompt

    )

    runner.export_csv(result)

    runner.export_json(result)

    runner.display_result(result)

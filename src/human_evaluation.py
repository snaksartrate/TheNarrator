
import csv
from pathlib import Path


class HumanEvaluator:

    CRITERIA = [
        "grammar",
        "coherence",
        "creativity",
        "continuity",
        "relevance",
        "completeness"
    ]

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.output_file = self.results_dir / "human_evaluation.csv"

    def collect_evaluation(
        self,
        evaluator_id,
        model_name,
        story_id,
        prompt,
        ratings,
        comments=""
    ):

        # Validate that all six criteria have ratings
        for criterion in self.CRITERIA:
            if criterion not in ratings:
                raise ValueError(
                    f"Missing rating for {criterion}"
                )

            if not isinstance(ratings[criterion], (int, float)):
                raise ValueError("Ratings must be numeric.")

            if not 1 <= ratings[criterion] <= 5:
                raise ValueError(
                    "Every rating must be between 1 and 5."
                )

        average_score = sum(
            ratings[criterion] for criterion in self.CRITERIA
        ) / len(self.CRITERIA)

        record = {
            "evaluator_id": evaluator_id,
            "model_name": model_name,
            "story_id": story_id,
            "prompt": prompt
        }

        for criterion in self.CRITERIA:
            record[criterion] = ratings[criterion]

        record["average_score"] = round(average_score, 2)
        record["comments"] = comments

        file_exists = self.output_file.exists()

        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=record.keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

        print("\nHuman evaluation saved successfully!")
        print("Average score:", round(average_score, 2), "/ 5")

        return record


if __name__ == "__main__":

    evaluator = HumanEvaluator()

    print("========== HUMAN STORY EVALUATION ==========")

    evaluator_id = input("Evaluator ID: ")
    model_name = input("Model name: ")
    story_id = input("Story ID: ")
    prompt = input("Original prompt: ")

    print("\nEnter ratings from 1 to 5.")

    ratings = {}

    for criterion in HumanEvaluator.CRITERIA:

        while True:
            try:
                score = int(
                    input(f"{criterion.capitalize()} (1-5): ")
                )

                if 1 <= score <= 5:
                    ratings[criterion] = score
                    break

                print("Please enter a score between 1 and 5.")

            except ValueError:
                print("Please enter a valid number.")

    comments = input("\nAdditional comments: ")

    evaluator.collect_evaluation(
        evaluator_id=evaluator_id,
        model_name=model_name,
        story_id=story_id,
        prompt=prompt,
        ratings=ratings,
        comments=comments
    )

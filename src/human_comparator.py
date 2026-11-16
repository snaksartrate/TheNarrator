
import csv
import os
from collections import defaultdict


class HumanComparator:

    CRITERIA = [
        "grammar",
        "coherence",
        "creativity",
        "continuity",
        "relevance",
        "completeness"
    ]

    def __init__(
        self,
        input_file="results/human_evaluation.csv",
        output_file="results/human_comparison.csv"
    ):
        self.input_file = input_file
        self.output_file = output_file

    def load_results(self):

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"Human evaluation file not found: {self.input_file}"
            )

        with open(self.input_file, "r", encoding="utf-8") as file:
            results = list(csv.DictReader(file))

        if not results:
            raise ValueError("No human evaluation records found.")

        return results

    def compare_models(self):

        results = self.load_results()

        model_data = defaultdict(list)

        for row in results:
            model_data[row["model_name"]].append(row)

        comparison = []

        for model_name, ratings in model_data.items():

            summary = {
                "model_name": model_name,
                "number_of_evaluations": len(ratings),
                "number_of_stories": len(
                    set(row["story_id"] for row in ratings)
                ),
                "number_of_evaluators": len(
                    set(row["evaluator_id"] for row in ratings)
                )
            }

            for criterion in self.CRITERIA:

                scores = [
                    int(row[criterion])
                    for row in ratings
                    if row.get(criterion, "").strip()
                ]

                average = (
                    sum(scores) / len(scores)
                    if scores else 0
                )

                summary[f"avg_{criterion}"] = round(average, 2)

            summary["overall_average"] = round(
                sum(summary[f"avg_{c}"] for c in self.CRITERIA)
                / len(self.CRITERIA),
                2
            )

            comparison.append(summary)

        return comparison

    def save_results(self, comparison):

        if not comparison:
            raise ValueError("No comparison results available.")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=comparison[0].keys()
            )

            writer.writeheader()
            writer.writerows(comparison)

        print(f"\nSaved human comparison to: {self.output_file}")

    def display_results(self, comparison):

        print("\n========== HUMAN MODEL COMPARISON ==========")

        for model in comparison:

            print(f"\nModel: {model['model_name']}")
            print(f"Evaluations: {model['number_of_evaluations']}")
            print(f"Stories: {model['number_of_stories']}")
            print(f"Evaluators: {model['number_of_evaluators']}")

            for criterion in self.CRITERIA:
                print(
                    f"Average {criterion}: "
                    f"{model[f'avg_{criterion}']}/5"
                )

            print(
                f"Overall average: "
                f"{model['overall_average']}/5"
            )

        print("\n============================================")


if __name__ == "__main__":

    comparator = HumanComparator()

    try:
        comparison = comparator.compare_models()
        comparator.display_results(comparison)
        comparator.save_results(comparison)

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")

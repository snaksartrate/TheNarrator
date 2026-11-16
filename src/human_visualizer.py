
import csv
import os
import matplotlib.pyplot as plt


class HumanVisualizer:

    def __init__(
        self,
        input_file="results/human_comparison.csv",
        output_dir="results/visualizations"
    ):
        self.input_file = input_file
        self.output_dir = output_dir

        self.criteria = [
            "grammar",
            "coherence",
            "creativity",
            "continuity",
            "relevance",
            "completeness"
        ]

    def load_results(self):

        if not os.path.exists(self.input_file):
            raise FileNotFoundError("Human comparison CSV not found.")

        with open(self.input_file, "r", encoding="utf-8") as file:
            results = list(csv.DictReader(file))

        if not results:
            raise ValueError("No human comparison results found.")

        return results

    def plot_criteria(self, results):

        models = [row["model_name"] for row in results]

        x = list(range(len(self.criteria)))
        width = 0.8 / len(models)

        plt.figure(figsize=(11, 6))

        for i, row in enumerate(results):

            values = [
                float(row[f"avg_{criterion}"])
                for criterion in self.criteria
            ]

            positions = [
                position - 0.4 + width / 2 + i * width
                for position in x
            ]

            plt.bar(
                positions,
                values,
                width,
                label=row["model_name"]
            )

        plt.xticks(x, self.criteria, rotation=20)
        plt.ylim(0, 5)
        plt.xlabel("Evaluation Criteria")
        plt.ylabel("Average Rating (1-5)")
        plt.title("Human Evaluation: Model Comparison")
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(
            self.output_dir,
            "human_criteria_comparison.png"
        )

        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved: {filename}")

    def plot_overall_scores(self, results):

        models = [row["model_name"] for row in results]

        scores = [
            float(row["overall_average"])
            for row in results
        ]

        plt.figure(figsize=(9, 5))
        plt.bar(models, scores)

        plt.ylim(0, 5)
        plt.xlabel("Models")
        plt.ylabel("Overall Human Rating (1-5)")
        plt.title("Overall Human Evaluation Scores")
        plt.xticks(rotation=25, ha="right")

        plt.tight_layout()

        filename = os.path.join(
            self.output_dir,
            "human_overall_comparison.png"
        )

        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved: {filename}")

    def generate_graphs(self):

        results = self.load_results()

        os.makedirs(self.output_dir, exist_ok=True)

        self.plot_criteria(results)
        self.plot_overall_scores(results)

        print("\nHuman evaluation graphs generated successfully!")


if __name__ == "__main__":

    visualizer = HumanVisualizer()

    try:
        visualizer.generate_graphs()

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")

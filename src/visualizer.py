
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class EvaluationVisualizer:

    def __init__(self):
        self.results_file = Path("results/comparison_summary.csv")
        self.output_dir = Path("results/visualizations")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self):

        if not self.results_file.exists():
            raise FileNotFoundError(
                "comparison_summary.csv not found. "
                "Run comparator.py first."
            )

        df = pd.read_csv(self.results_file)

        if df.empty:
            raise ValueError("No evaluation results available.")

        return df

    def create_graph(self, df, metric, title, filename, ylabel):

        if metric not in df.columns:
            print(f"Skipping {metric}: column not found.")
            return

        values = pd.to_numeric(df[metric], errors="coerce")

        plt.figure(figsize=(10, 6))

        plt.bar(df["model_name"].astype(str), values)

        plt.title(title)
        plt.xlabel("Models")
        plt.ylabel(ylabel)

        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()

        output_path = self.output_dir / filename

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved: {output_path}")

    def generate_all_graphs(self):

        df = self.load_results()

        graphs = [
            (
                "avg_distinct_1",
                "Distinct-1 Comparison",
                "distinct1_comparison.png",
                "Average Distinct-1"
            ),
            (
                "avg_distinct_2",
                "Distinct-2 Comparison",
                "distinct2_comparison.png",
                "Average Distinct-2"
            ),
            (
                "avg_repetition_rate",
                "Repetition Rate Comparison",
                "repetition_comparison.png",
                "Average Repetition Rate"
            ),
            (
                "avg_lexical_diversity",
                "Lexical Diversity Comparison",
                "lexical_diversity_comparison.png",
                "Average Lexical Diversity"
            ),
            (
                "avg_word_count",
                "Average Story Length Comparison",
                "word_count_comparison.png",
                "Average Word Count"
            )
        ]

        for metric, title, filename, ylabel in graphs:

            self.create_graph(
                df,
                metric,
                title,
                filename,
                ylabel
            )

        print("\nAll comparison graphs generated successfully!")

        print("\nGraphs saved in:")
        print(self.output_dir)


if __name__ == "__main__":

    visualizer = EvaluationVisualizer()
    visualizer.generate_all_graphs()

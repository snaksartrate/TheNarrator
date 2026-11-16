
import csv
import pandas as pd
from pathlib import Path


class ModelComparator:

    def __init__(self):
        self.results_dir = Path("results")

        self.automatic_file = self.results_dir / "combined_evaluation.csv"
        self.human_file = self.results_dir / "human_evaluation.csv"

        self.output_file = self.results_dir / "comparison_summary.csv"
        self.detailed_file = self.results_dir / "final_model_comparison.csv"

    def load_automatic_results(self):

        if not self.automatic_file.exists():
            raise FileNotFoundError("combined_evaluation.csv not found.")

        valid_rows = []
        malformed_count = 0

        with open(self.automatic_file, "r", encoding="utf-8-sig", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                if not row or not row.get("model_name"):
                    malformed_count += 1
                    continue

                # Remove unwanted test records
                if row["model_name"] in ["Dummy Model", "sample_model"]:
                    continue

                valid_rows.append(row)

        if malformed_count:
            print(
                f"Warning: Skipped {malformed_count} malformed or empty rows."
            )

        if not valid_rows:
            raise ValueError(
                "No valid actual model records found. "
                "You need actual model evaluation results."
            )

        df = pd.DataFrame(valid_rows)

        return df

    def calculate_automatic_averages(self, df):

        metrics = [
            "word_count",
            "distinct_1",
            "distinct_2",
            "repetition_rate",
            "lexical_diversity",
            "generation_time_seconds",
            "seconds_per_word",
            "words_per_second",
            "perplexity"
        ]

        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")

        available_metrics = [
            metric for metric in metrics
            if metric in df.columns
        ]

        summary = df.groupby("model_name")[available_metrics].mean(
            numeric_only=True
        ).reset_index()

        summary = summary.rename(
            columns={
                metric: "avg_" + metric
                for metric in available_metrics
            }
        )

        counts = df.groupby("model_name").size().rename(
            "number_of_stories"
        )

        summary = summary.merge(
            counts,
            on="model_name"
        )

        return summary

    def load_human_results(self):

        if not self.human_file.exists():
            print("Human evaluation file not found.")
            return None

        df = pd.read_csv(self.human_file)

        if df.empty:
            return None

        criteria = [
            "grammar",
            "coherence",
            "creativity",
            "continuity",
            "relevance",
            "completeness",
            "average_score"
        ]

        available = [
            column for column in criteria
            if column in df.columns
        ]

        for column in available:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        summary = df.groupby("model_name")[available].mean(
            numeric_only=True
        ).reset_index()

        summary = summary.rename(
            columns={
                column: "human_avg_" + column
                for column in available
            }
        )

        return summary

    def generate_comparison(self):

        automatic = self.load_automatic_results()

        final_table = self.calculate_automatic_averages(automatic)

        human = self.load_human_results()

        if human is not None:
            final_table = pd.merge(
                final_table,
                human,
                on="model_name",
                how="left"
            )

        final_table = final_table.sort_values("model_name")

        final_table.to_csv(self.output_file, index=False)
        final_table.to_csv(self.detailed_file, index=False)

        print("\n========== MODEL COMPARISON ==========")
        print(final_table.to_string(index=False))

        print("\nComparison files saved:")
        print(self.output_file)
        print(self.detailed_file)

        return final_table


if __name__ == "__main__":

    comparator = ModelComparator()
    comparator.generate_comparison()

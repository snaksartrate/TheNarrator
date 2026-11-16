
import pytest
from src.evaluator import StoryEvaluator


@pytest.fixture
def evaluator():
    return StoryEvaluator()


def test_empty_story(evaluator):
    metrics = evaluator.evaluate_story("")

    assert metrics["word_count"] == 0
    assert metrics["distinct_1"] == 0
    assert metrics["distinct_2"] == 0
    assert metrics["repetition_rate"] == 0
    assert metrics["lexical_diversity"] == 0


def test_word_count(evaluator):
    metrics = evaluator.evaluate_story("The boy likes apples")

    assert metrics["word_count"] == 4


def test_distinct_1(evaluator):
    metrics = evaluator.evaluate_story("cat dog cat")

    assert metrics["distinct_1"] == pytest.approx(2 / 3)


def test_distinct_2(evaluator):
    metrics = evaluator.evaluate_story("cat dog cat")

    assert metrics["distinct_2"] == pytest.approx(1.0)


def test_repetition_rate(evaluator):
    metrics = evaluator.evaluate_story(
        "a b c a b c"
    )

    assert metrics["repetition_rate"] == pytest.approx(0.25)


def test_lexical_diversity(evaluator):
    metrics = evaluator.evaluate_story("cat dog cat")

    assert metrics["lexical_diversity"] == pytest.approx(2 / 3)


def test_csv_export(evaluator, tmp_path):

    metrics = evaluator.evaluate_story("The boy likes apples")

    output_file = tmp_path / "evaluation.csv"

    evaluator.save_to_csv(
        metrics,
        model_name="test_model",
        filename=str(output_file)
    )

    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")

    assert "test_model" in content
    assert "word_count" in content

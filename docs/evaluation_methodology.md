
# TheNarrator - Model Evaluation Methodology

## 1. Introduction

TheNarrator is a language model project focused on generating stories.
The evaluation module measures the quality and performance of generated
stories using automatic evaluation metrics and human evaluation.

The final comparison is intended to cover six model configurations.

## 2. Evaluation Objectives

The objectives are:

- Evaluate the quality of generated stories.
- Measure lexical diversity.
- Identify repetitive text generation.
- Measure story length.
- Measure generation time and speed.
- Calculate perplexity when model interfaces are available.
- Compare six model configurations using common evaluation criteria.
- Include human judgment in the final evaluation.

## 3. Automatic Evaluation Metrics

### 3.1 Perplexity

Perplexity measures how well a language model predicts a sequence of tokens.

Lower perplexity generally indicates better next-token prediction.

Perplexity implementation and model-specific calculation are pending
integration with the trained models.

### 3.2 Distinct-1

Distinct-1 measures the proportion of unique unigrams
(single words) in generated text.

Distinct-1 = Unique unigrams / Total unigrams

Higher values indicate greater word-level diversity.

### 3.3 Distinct-2

Distinct-2 measures the proportion of unique bigrams
(two-word sequences) in generated text.

Distinct-2 = Unique bigrams / Total bigrams

Higher values indicate greater phrase-level diversity.

### 3.4 Repetition Rate

Repetition rate measures repeated n-grams in generated text.

Repetition Rate =
Repeated n-gram occurrences / Total n-grams

Lower values indicate less repetition.

### 3.5 Lexical Diversity

Lexical diversity measures the proportion of unique words.

Lexical Diversity =
Unique words / Total words

Higher values indicate greater vocabulary variety.
This metric can be affected by story length.

### 3.6 Story Length

Story length is measured using the total number of words
in the generated story.

### 3.7 Generation Time

Generation time is measured using Python's time.perf_counter().

The evaluation system also calculates:

Seconds Per Word = Generation Time / Word Count

Words Per Second = Word Count / Generation Time

## 4. Human Evaluation

Human evaluation measures story quality using six criteria.

Each criterion is rated on a scale of 1 to 5.

| Criterion | Description |
|---|---|
| Grammar | Correctness of language and sentence structure |
| Coherence | Logical flow of sentences |
| Creativity | Originality and interesting ideas |
| Continuity | Consistency between story events |
| Relevance | Alignment with the input prompt |
| Completeness | Meaningful story development and ending |

### Rating Scale

1 - Very Poor
2 - Poor
3 - Average
4 - Good
5 - Excellent

The overall human score is calculated as the arithmetic mean
of the six criterion scores.

## 5. Evaluation Pipeline

The evaluation workflow is:

1. Receive a prompt.
2. Generate a story using a language model.
3. Measure generation time.
4. Calculate automatic evaluation metrics.
5. Store evaluation results.
6. Export results to CSV and JSON.
7. Collect human ratings.
8. Calculate model-wise averages.
9. Generate comparison tables.
10. Generate comparison graphs.

## 6. Result Storage

The evaluation system stores results in the following files:

- combined_evaluation.csv
- combined_evaluation.json
- human_evaluation.csv
- comparison_summary.csv
- final_model_comparison.csv

Graphs are stored in:

results/visualizations/

## 7. Model Comparison

Six model configurations will be evaluated using
the same evaluation methodology.

The comparison will consider:

- Average story length
- Distinct-1
- Distinct-2
- Repetition rate
- Lexical diversity
- Generation time
- Words per second
- Perplexity, when available
- Human evaluation scores

All models should be evaluated using consistent prompts,
comparable generation settings and the same metric definitions.

## 8. Limitations

- Automatic diversity metrics do not fully measure story quality.
- Lexical diversity depends on generated story length.
- Perplexity requires access to the trained model and its tokenizer.
- Human evaluation can involve subjective judgments.
- Generation time depends on hardware and generation settings.
- Dummy generator results are used only for testing and are not
  included in final model comparisons.

## 9. Conclusion

The evaluation framework combines automatic metrics,
generation performance measurements and human judgment.

This provides a structured approach to comparing the six
language model configurations developed in TheNarrator.

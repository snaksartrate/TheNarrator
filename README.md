# TheNarrator – A Comparative Study of Story Generation Using Small Language Models

## 1. About the Project

The goal of this project is to build a basic **Small Language Model (SLM)** capable of generating short stories and study how different training strategies and tokenization techniques affect its output.

Instead of building just one model, we will train multiple models using different combinations of training strategies and tokenization patterns, then compare their generated stories.

The project is primarily focused on experimentation and comparative analysis.

## 2. What Are We Trying to Compare?

We have two main experimental dimensions.

### A. Training Strategies (m)

We will experiment with two different training strategies:

1. AdamW optimizer with a fixed learning rate.
2. SGD optimizer with momentum.

The objective is to understand how different training strategies affect the quality of generated stories and the model's learning performance.

### B. Tokenization Patterns (n)

We will experiment with three tokenization approaches:

1. Character-level tokenization.
2. Word-level tokenization.
3. Subword-level tokenization using Byte Pair Encoding (BPE).

The objective is to understand how breaking text into different token units affects:

- Story generation quality.
- Vocabulary size.
- Training efficiency.
- Context handling.
- Coherence and readability.

### C. Total Models

With two training strategies and three tokenization patterns:

**Total models = m × n = 2 × 3 = 6 models**

Each model will use the same underlying model architecture, dataset splits and comparable generation settings wherever possible.

This ensures that the comparison focuses on the effects of training strategies and tokenization rather than differences in model architecture.

## 3. Project Workflow

Our planned workflow:

1. **Dataset Collection:** Gather a suitable collection of short stories.
2. **Data Preprocessing:** Clean and prepare the text for training.
3. **Tokenization:** Prepare datasets using the three different tokenization patterns.
4. **Model Development:** Build a basic Transformer-based language model.
5. **Training:** Train all six combinations of training strategies and tokenization patterns.
6. **Story Generation:** Generate stories using each trained model.
7. **Evaluation:** Evaluate and compare generated stories using quantitative and qualitative metrics.
8. **Analysis:** Identify which combinations perform better and discuss why.
9. **Documentation:** Prepare a formal ML project report containing the methodology, experiments, results and conclusions.

## 4. Evaluation

We intend to compare the models using the following metrics:

- **Perplexity:** How well the model predicts the next token on held-out data.
- **Repetition:** How frequently the model repeats words or phrases.
- **Distinct-1:** Measures unigram diversity.
- **Distinct-2:** Measures bigram diversity.
- **Lexical Diversity:** Measures the variety of words used.
- **Coherence:** Whether the story maintains a consistent flow.
- **Grammar and Readability:** How understandable the generated text is.
- **Generation Time:** How long the model takes to generate a story.

We will also manually inspect generated stories to understand differences that numerical metrics might miss.

Human evaluation will be used for subjective qualities such as coherence, grammar and storytelling quality.

Note: Perplexity values across different tokenization methods must be interpreted carefully because their token units and vocabulary distributions differ.

## 5. Technology Stack

Proposed technologies:

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Tokenization:** Custom tokenizers / Python libraries
- **Visualization:** Matplotlib
- **Development:** VS Code / Jupyter Notebook
- **Version Control:** Git and GitHub

## 6. Project Structure

The planned project directory structure is:

```text
TheNarrator/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── checkpoints/
│
├── tokenizers/
│   ├── base_tokenizer.py
│   ├── char_tokenizer.py
│   ├── word_tokenizer.py
│   └── bpe_tokenizer.py
│
├── src/
│   ├── model.py
│   ├── trainer.py
│   ├── generator.py
│   └── data/
│       ├── dataset.py
│       └── preprocess.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── diversity.py
│   ├── repetition.py
│   ├── readability.py
│   └── timing.py
│
├── experiments/
│   ├── configs/
│   ├── outputs/
│   └── analysis/
│
├── reports/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

This is the target structure. Some directories and files may still need to be created.

## 7. Team Responsibilities

We have four team members. Each member has an independent module to implement to minimize conflicts while working on the same GitHub repository.

### A. Project Lead — Model Architecture, Training and Integration

**Responsibility:** Build the actual language model and integrate all modules.

Tasks:

- Design and implement the Transformer-based language model.
- Implement token embeddings and positional embeddings.
- Implement self-attention and feed-forward layers.
- Implement causal masking.
- Implement next-token prediction and loss calculation.
- Implement the training loop using PyTorch.
- Support AdamW and SGD with momentum.
- Implement model checkpoint saving and loading.
- Implement autoregressive story generation.
- Integrate the dataset and tokenizer interfaces.
- Implement experiment configuration handling.
- Create the main execution pipeline.
- Run all six experimental combinations.
- Debug, benchmark and validate the complete pipeline.

Files owned:

```text
models/
src/model.py
src/trainer.py
src/generator.py
config.py
main.py
```

### B. Pratik — Evaluation, Metrics and Visualization

**Responsibility:** Build the evaluation system that compares the generated stories and model performance.

Tasks:

1. Research evaluation metrics for language models.
2. Implement perplexity calculation.
3. Implement Distinct-1 and Distinct-2.
4. Implement repetition measurement.
5. Implement lexical diversity measurement.
6. Measure generated story length.
7. Measure generation time.
8. Build a combined evaluation runner.
9. Save results in CSV and JSON formats.
10. Generate comparison graphs.
11. Prepare qualitative evaluation rubrics.
12. Prepare model comparison tables.
13. Document evaluation methodology.

Files owned:

```text
evaluation/
experiments/analysis/
```

Expected files:

```text
evaluation/
├── __init__.py
├── perplexity.py
├── diversity.py
├── repetition.py
├── readability.py
├── timing.py
├── evaluate.py
└── README.md

experiments/
└── analysis/
    ├── compare_models.py
    ├── generate_graphs.py
    └── results_summary.py
```

Acceptance criteria:

- Evaluation scripts must accept generated stories and model outputs in a consistent format.
- Results must be reproducible.
- Metrics must be clearly documented.
- Graphs and tables must allow comparison across all six models.

### C. Nihira — Tokenization

**Responsibility:** Implement and compare different tokenization techniques.

Tasks:

1. Research character-level, word-level and subword-level tokenization.
2. Implement character-level tokenization.
3. Implement word-level tokenization.
4. Implement BPE tokenization.
5. Implement vocabulary construction.
6. Implement text-to-token conversion using `encode()`.
7. Implement token-to-text conversion using `decode()`.
8. Implement unknown-token handling where applicable.
9. Implement special tokens such as PAD, BOS and EOS.
10. Implement tokenizer vocabulary saving and loading.
11. Calculate vocabulary sizes and tokenization statistics.
12. Write unit tests for tokenizer correctness.

Files owned:

```text
tokenizers/
```

Expected files:

```text
tokenizers/
├── __init__.py
├── base_tokenizer.py
├── char_tokenizer.py
├── word_tokenizer.py
├── bpe_tokenizer.py
├── train_tokenizers.py
└── README.md
```

Integration requirement:

All tokenizers must expose a common interface:

```python
tokenizer.train(text)

token_ids = tokenizer.encode(text)

decoded_text = tokenizer.decode(token_ids)

vocab_size = tokenizer.vocab_size
```

The implementation can differ, but the interface must remain consistent.

### D. Rudrakshi — Dataset Collection and Preprocessing

**Responsibility:** Collect, clean, organize and prepare the story dataset for training.

Tasks:

1. Research suitable public-domain story datasets.
2. Select one dataset suitable for small-scale language model training.
3. Download and organize the original data.
4. Remove empty and malformed samples.
5. Remove duplicate stories.
6. Normalize whitespace and unwanted characters.
7. Preserve story boundaries and separators.
8. Split the dataset into training, validation and test sets.
9. Generate dataset statistics.
10. Implement a reusable dataset loader.
11. Document dataset sources and licensing.
12. Ensure reproducibility using fixed random seeds.

Files owned:

```text
dataset/
src/data/
```

Expected files:

```text
dataset/
├── raw/
├── processed/
│   ├── train.txt
│   ├── validation.txt
│   ├── test.txt
│   └── dataset_stats.json
└── README.md

src/
└── data/
    ├── __init__.py
    ├── preprocess.py
    └── dataset.py
```

Acceptance criteria:

- Preprocessing must produce reproducible dataset splits.
- Duplicate stories should be removed.
- Training, validation and test datasets must not overlap.
- The model developer must be able to load the processed dataset without knowing how it was collected or cleaned.

## 8. Integration Between Team Members

The modules must work independently but follow common interfaces.

| From | To | Deliverable |
|---|---|---|
| Rudrakshi | Project Lead | Clean dataset and dataset loader |
| Nihira | Project Lead | Three tokenizers with common interface |
| Project Lead | Pratik | Generated stories, checkpoints and training logs |
| Pratik | Project Lead | Evaluation results and comparison reports |

The project lead is responsible for integrating all modules.

Each member should avoid modifying another member's assigned files without prior discussion.

## 9. GitHub Collaboration Rules

Repository:

https://github.com/snaksartrate/TheNarrator

Each member will work on a separate branch.

| Member | Branch |
|---|---|
| Project Lead | `feature/model-training` |
| Pratik | `feature/evaluation` |
| Nihira | `feature/tokenizers` |
| Rudrakshi | `feature/dataset` |

Workflow:

1. Pull the latest changes from `main`.
2. Work only on the assigned module.
3. Commit changes regularly.
4. Push changes to the assigned branch.
5. Create a Pull Request.
6. Get project lead approval before merging into `main`.

Example:

```bash
git checkout main
git pull origin main

git checkout -b feature/evaluation

# Work on assigned files

git add .
git commit -m "Implement evaluation metrics"
git push -u origin feature/evaluation
```

Important rules:

- Nobody should directly push experimental code to `main`.
- Do not modify another member's module without discussing it.
- Avoid committing generated datasets, large checkpoints or unnecessary output files to Git.
- Keep code modular and documented.
- Communicate interface changes before implementing them.

## 10. Implementation Milestones

### Milestone 1 — Independent Modules

- Project Lead: Basic Transformer architecture and training loop.
- Pratik: Evaluation functions and test cases.
- Nihira: Three tokenizer implementations.
- Rudrakshi: Dataset collection and preprocessing.

**Completion condition:** All four modules work independently.

### Milestone 2 — Integration

- Connect the dataset loader.
- Connect all three tokenizers.
- Train one baseline model.
- Generate a sample story.
- Verify that evaluation scripts can process its output.

**Completion condition:** One complete end-to-end experiment works.

### Milestone 3 — Experimental Training

- Finalize two training strategies.
- Finalize three tokenization patterns.
- Train all six combinations.
- Save checkpoints and training logs.

**Completion condition:** Six trained model variants.

### Milestone 4 — Comparative Analysis

- Generate stories using all six models.
- Evaluate all outputs.
- Generate comparison graphs.
- Conduct qualitative evaluation.

**Completion condition:** Complete experimental results.

### Milestone 5 — Final Report

Prepare the formal project report, covering:

- Introduction and problem statement.
- Literature review.
- Dataset description.
- Model architecture.
- Tokenization methods.
- Training strategies.
- Experimental methodology.
- Results and visualizations.
- Comparative analysis.
- Discussion and limitations.
- Conclusion and future scope.

**Completion condition:** Final project report and reproducible code.

## 11. Final Expected Outcome

By the end of this project, we aim to have:

1. A basic working story-generating language model.
2. Three tokenization implementations.
3. Two training strategies.
4. Six trained model variants.
5. Generated stories from every experimental configuration.
6. Quantitative and qualitative comparisons.
7. Graphs and tables illustrating experimental results.
8. A formal comparative study explaining the effect of training strategies and tokenization patterns on story generation.

## 12. Main Research Question

**How do different training strategies and tokenization patterns influence the quality, diversity, coherence and efficiency of stories generated by a small language model?**

This is the central question that our entire project and final report will attempt to answer.

---

*This README describes the agreed project direction and current implementation plan. Experimental configurations, implementation details and results will be updated as development progresses.*
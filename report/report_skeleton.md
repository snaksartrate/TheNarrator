# TheNarrator: Effect of Optimizer and Tokenization on Small-LM Story Generation
Title page: names, roll numbers, course, date.

1. Abstract  [write last, from real results]
2. Introduction / Problem statement / Objectives and research question (copy README Sec. 2)
3. Background  [cite: Transformer; AdamW; SGD momentum; BPE; TinyStories; perplexity; Distinct-n]
4. Dataset: TinyStories, first 30,000 train stories, cdla-sharing-1.0, cleaning, seed-42 80/10/10 split (dataset/README.md)
5. Tokenizers: char (84), word (min_freq 2, 10,920), BPE (4,000). Fit on train only. Word unk rate: test 0.21%.
6. Model: decoder-only, block 256, d=128, 4 heads, 4 layers, ff x4, dropout 0.1, pre-norm
7. Training: AdamW lr 3e-4 vs SGD lr 0.1 mom 0.9, batch 8, 2000 steps, clip 1.0, no scheduler, CPU, seed 42
8. Reproducibility: seeds, run_meta.json, history.csv, commands
9. Metrics: definitions from evaluation/README.md
10. Quantitative results  [Table 1: six-row comparison; Fig: loss curves, perplexity per tokenizer, story metrics]
11. Human evaluation: protocol (blind, shuffled, rubric.md), n evaluators = [fill], results [Table 2, Fig]
12. Discussion: answer analysis_tables.md questions 1-13 from real numbers only
13. Limitations (mandatory):
    - Fixed step budget, not epochs; models undertrained (best step = last step)
    - Same 256-token context covers different raw text per tokenizer
    - Perplexity is token-space, never ranked across tokenizers
    - Non-overlapping test windows lose context for later windows of long stories
    - 5 stories/model, no significance claims; Distinct-1 equals lexical diversity
    - Distinct/repetition depend on story length (300-token cap gives different word counts)
    - Timing is CPU-load dependent
14. Conclusion  15. Future scope  16. References [verify each before submitting]

## Reference candidates (verify authors, venue, year, DOI/URL yourself)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
- Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units", ACL 2016
- Eldan & Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", arXiv:2305.07759, 2023
- Li et al., "A Diversity-Promoting Objective Function for Neural Conversation Models", NAACL 2016
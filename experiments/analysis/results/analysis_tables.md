# Analysis tables (generated, numbers only)

> Loss/perplexity are token-space: compare only within a tokenizer. Diversity and repetition depend on story length (see word count). 5 stories/model: descriptive only.

## Within char: AdamW vs SGD
```
optimizer                      adamw     sgd  sgd_minus_adamw
best_val_loss                 1.7650  2.0329           0.2679
test_perplexity               5.8302  7.5951           1.7649
mean_word_count              63.4000 38.8000         -24.6000
mean_distinct_1               0.7355  0.7018          -0.0337
mean_distinct_2               0.9968  0.8000          -0.1968
mean_repetition_rate          0.0000  0.0000           0.0000
mean_generation_time_seconds  6.0739  1.7788          -4.2951
mean_words_per_second        10.7268 32.9415          22.2147
human_average_score_mean         NaN     NaN              NaN
```

## Within word: AdamW vs SGD
```
optimizer                       adamw      sgd  sgd_minus_adamw
best_val_loss                  3.3584   3.4645           0.1061
test_perplexity               28.4481  31.3871           2.9390
mean_word_count              187.0000 157.2000         -29.8000
mean_distinct_1                0.4886   0.4797          -0.0089
mean_distinct_2                0.8606   0.8152          -0.0454
mean_repetition_rate           0.0282   0.0647           0.0365
mean_generation_time_seconds   7.5674   5.8698          -1.6976
mean_words_per_second         30.0939  27.1961          -2.8978
human_average_score_mean          NaN      NaN              NaN
```

## Within bpe: AdamW vs SGD
```
optimizer                      adamw      sgd  sgd_minus_adamw
best_val_loss                 3.3898   3.4759           0.0861
test_perplexity              30.5291  33.0351           2.5060
mean_word_count              89.4000 176.0000          86.6000
mean_distinct_1               0.6478   0.4936          -0.1541
mean_distinct_2               0.9330   0.8542          -0.0789
mean_repetition_rate          0.0198   0.0423           0.0225
mean_generation_time_seconds  1.7181   2.5197           0.8016
mean_words_per_second        51.9781  73.3911          21.4130
human_average_score_mean         NaN      NaN              NaN
```

## Within adamw: across tokenizers
```
tokenizer                       char     word     bpe
best_val_loss                 1.7650   3.3584  3.3898
test_perplexity               5.8302  28.4481 30.5291
mean_word_count              63.4000 187.0000 89.4000
mean_distinct_1               0.7355   0.4886  0.6478
mean_distinct_2               0.9968   0.8606  0.9330
mean_repetition_rate          0.0000   0.0282  0.0198
mean_generation_time_seconds  6.0739   7.5674  1.7181
mean_words_per_second        10.7268  30.0939 51.9781
human_average_score_mean         NaN      NaN     NaN
```

## Within sgd: across tokenizers
```
tokenizer                       char     word      bpe
best_val_loss                 2.0329   3.4645   3.4759
test_perplexity               7.5951  31.3871  33.0351
mean_word_count              38.8000 157.2000 176.0000
mean_distinct_1               0.7018   0.4797   0.4936
mean_distinct_2               0.8000   0.8152   0.8542
mean_repetition_rate          0.0000   0.0647   0.0423
mean_generation_time_seconds  1.7788   5.8698   2.5197
mean_words_per_second        32.9415  27.1961  73.3911
human_average_score_mean         NaN      NaN      NaN
```

## Best model per metric
```
mean_distinct_1: best=adamw_char (highest), n_models=6
mean_distinct_2: best=adamw_char (highest), n_models=6
mean_repetition_rate: best=adamw_char (lowest), n_models=6
```

## Spearman(word_count, metric), pooled stories (tokenizer-confounded)
```
                    rho
distinct_1      -0.7357
distinct_2      -0.5953
repetition_rate  0.6019
```

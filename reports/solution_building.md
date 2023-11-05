# Solution Building

## Choosing metrics

I inspected the repository from the dataset link and have found that they used multiple metrics:
BLEU score, inverse toxicity score predicted by Roberta-like model and similarity between inputs
and predictions. I decided to use these metrics but instead Wieting similarity I chose BERT score
because there
exists [nice function](https://torchmetrics.readthedocs.io/en/stable/text/bert_score.html#torchmetrics.functional.text.bert.bert_score)
in `torchmetrics` to evaluate it.

## Baseline #0: Identity

As a first baseline I decided to implement a model that just returns input as prediction, so I would
get the lowest BLEU and non-toxicity scores for sanity checks in more complex models. However, this
model achieved the best similarity :-)

| Metric            | Value on test set |
|-------------------|-------------------|
| Mean BLEU         | 0.173             |
| Mean Similarity   | 1.0               |
| Mean Non-toxicity | 0.129             |

## Baseline #1: Profanity filter

My second idea was to try another simple approach: just removing profanities from sentences. I used
word list from [here](https://github.com/surge-ai/profanity/blob/main/profanity_en.csv), but took
only single-word examples. Also, I am using lemmatizer from `nltk` to match different forms of word.
As a result, non-toxicity score significantly increased, while similarity dropped dramatically. BLEU
score is noticeable smaller than in previous baseline as well.

| Metric            | Value on test set |
|-------------------|-------------------|
| Mean BLEU         | 0.102             |
| Mean Similarity   | 0.456             |
| Mean Non-toxicity | 0.507             |

## Fine-tune T5

## References

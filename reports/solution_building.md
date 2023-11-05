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

## References

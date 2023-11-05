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
get the lowest scores for sanity checks in more complex models. However, this
model achieved the best similarity :-)

| Metric            | Value on test set (5% of samples) |
|-------------------|-----------------------------------|
| Mean BLEU         | 0.173                             |
| Mean Similarity   | 1.0                               |
| Mean Non-toxicity | 0.129                             |

## Baseline #1: Profanity filter

My second idea was to try another simple approach: just removing profanities from sentences. I used
word list from [here](https://github.com/surge-ai/profanity/blob/main/profanity_en.csv), but took
only single-word examples. Also, I am using lemmatizer from `nltk` to match different forms of a
word.
As a result, non-toxicity score significantly increased, while similarity dropped dramatically. BLEU
score is noticeable smaller than in previous baseline as well.

| Metric            | Value on test set (5% of samples) |
|-------------------|-----------------------------------|
| Mean BLEU         | 0.102                             |
| Mean Similarity   | 0.456                             |
| Mean Non-toxicity | 0.507                             |

## Use T5 model fine-tuned on ParaNMT

Fortunately, I was able to find
a [T5 model already trained on ParaNMT dataset](https://huggingface.co/s-nlp/t5-paranmt-detox), so I
do not need to fine-tune it by myself.
This model shows better translation quality indicated by higher BLEU and non-toxicity, but performs
slightly worse than the second baseline in terms of text similarity.

| Metric            | Value on test set (0.5% of samples) |
|-------------------|-------------------------------------|
| Mean BLEU         | 0.204                               |
| Mean Similarity   | 0.421                               |
| Mean Non-toxicity | 0.675                               |

## Zero-shot learning with OpenAI LLM

Why do we even need to train or fine-tune models for this task? Let us just ask ChatGPT (I actually
used `text-davinci-003`) to do all the work! I used the following prompt templates:

1. For one sentence:
    ```
   Paraphrase the following sentence to make it less toxic:
   ```{input_sentence}```
   Output only the result sentence.
   ```
2. For multiple sentences:
    ```
   Paraphrase the following sentences to make them less toxic:
   ```{all_sentences_separated_by_line_breaks}```
   Output only the result sentences separated by line breaks.
   ```

This model demonstrated the best non-toxicity and similarity scores among other solutions so far.

| Metric            | Value on test set (96 samples) |
|-------------------|--------------------------------|
| Mean BLEU         | 0.122                          |
| Mean Similarity   | 0.506                          |
| Mean Non-toxicity | 0.750                          |

Unfortunately, this solution fails sometimes because LLM breaks the output format.

## Few-shot learning

What if we provide some examples for LLM? Would it perform better?
I used the following prompt template:

```
Input sentences:
{reference_examples}
Paraphrased and less toxic:
{translated_examples}
Input sentences:
{input_sentences}
```

| Metric            | Value on test set (192 samples) |
|-------------------|---------------------------------|
| Mean BLEU         | 0.148                           |
| Mean Similarity   | 0.532                           |
| Mean Non-toxicity | 0.625                           |

Few-shot technique improved BLEU and similarity scores, but non-toxicity score decreased. Also, I
would
test the last two models with more data.

I have added `rpm_limiter` decorator to satisfy the API request rate limit and was able to test both
OpenAI-based solutions with more data.
Zero-shot one tends to fail due to the problems with output format, while the second one works fine almost every time.
Using training examples with the greater toxicity decrease did not help with non-toxicity score. 
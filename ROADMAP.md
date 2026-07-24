## Planning Phase

Generally what we are trying to do is to:

```
Create bigger matrix

↓

Copy old weights

↓

Initialize new weights

↓

Replace module

↓

Save
```

And

```
Modify tokenizer vocabulary

↓

Update tokenizer.json

↓

Reload tokenizer

```

But before doing that it is a good idea to skim through the [paper](https://arxiv.org/pdf/2402.01613) of the model to find anything that could help us in this task, and disregard anything that doesn't

## Paper Reading

This is what's inside the paper:
1. Each document from BooksCorpus and Wikipedia is tokenized using the bert-base-uncased tokenizer from Devlin et al. (We know that this was trained using masked lanugage modeling (MLM))
2. To train a long sequence length and efficient BERT, we adapt the BERT architecture. We make the following
architecture changes to BERT base (Devlin et al., 2019):
    - Substituting absolute positional embeddings for rotary positional embeddings (Su et al., 2023b)
    - 2.2 Using SwiGLU activation instead of GeLU (Shazeer, 2020)
    - Using Flash Attention (Dao et al., 2022)
    - Setting Dropout to 0 (Geiping & Goldstein, 2022)
    - Vocab size as a multiple of 64 (Portes et al., 2023; Shoeybi et al., 2020)


## Paper Analysis
From 1. we know that it's using the bert-base-uncased tokenizer, this allows us to know that it is using (WordPiece), and for training it used MLM

And from 2. We notice the 64-padding vocabulary rule.

## Action Plan
Since we want to add 10-15 words we will not be able to add these tokens straight into the padded space, since remember that:
In standard bert `vocab.txt` file we have: 30,522 tokens
However in `model.config.vocab_size` we have the size: 30,528
note: $30,528/64 = 477$ so we know it is because of the padding.
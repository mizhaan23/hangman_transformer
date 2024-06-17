# Using an Encoder-only Transformer Network to play the game of Hangman
Implementing a Transformer network to play the game of Hangman. Code courtesy : https://github.com/SamLynnEvans/Transformer

## Inspiration
It is a well known fact that transformer models are the *in* thing in the fields of Deep Learning shortly after the paper "Attention is all you need" by. The BERT (Bi-directional Encoder Representations from Transformers) model by Google has shown state-of-the-art results in NLP tasks. The model is training by performing masking on a series of words (i.e. sentences) with the target being the true unmasked sentence. For e.g. the masked sentence,

    I am going to ____.

would be predicted with the words `bed`, `sleep`, `cry` etc. with different levels of probabilities. Hence, we take inspiration from this BERT model for our Hangman problem.

## Implementation
However, the issues that arise are due to the fact that BERT uses word embeddings and we wish to input letters from our fixed dataset. We now implement a learn-able embedding layer that would convert each letter to some vector in the embedding space which then be fed into our encoder blocks inside our transformer.

We use the encoder-only Transformer model. The inputs to this layer are the embeddings from the embedding layer with spit out a $`d_{model} = 128`$ dimensional vector, along with positional encoding from the above-mentioned paper, given as follows:

$`$ PE_{(pos, 2i)} = \sin \left( \frac{pos}{1000^{2i / d_{model}}} \right), $`$
$`$ PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{1000^{2i / d_{model}}} \right), $`$

where $`pos`$ is the position of the character in our word and $`i`$ is the dimension. The label encoding for each character in the English alphabet was done as follows

$`$ \{\texttt{[PAD]} : 0, \texttt{a} : 1, \ldots, \texttt{z} : 26, \texttt{\_} : 27 \} , $`$

here the $`\texttt{[PAD]}`$ label represents a padding so that shorter words can be padded on their right. We constrain a max length of 32 for a word, for e.g. the label encoding for the input $`\texttt{[a \_ \_ l e]}`$ would be the following vector of length 32

$`$ [1, 27, 27, 12, 5, 0, \ldots, 0] .$`$

## Inference
The model outputs a tensor of dimension $`[batch, length, vocab\_size]`$, where the parameters represent the batch size, length of the word (32 in our case), and the vocab size (28 in our case). We train the model by predicting the original word, e.g. $`\texttt{[a p p l e]}`$ from continuing the example from before. The raw output is then converted into probabilities by using the softmax function. This gives us a matrix of dimensions $`32\times28`$, where each row corresponds to the position in our word while the columns represents the probability distribution over the alphabets. We expect this probability distribution to correspond to the actual word.

We then implement a cross-entropy loss as is best for the case of probability distributions and multi-class classification tasks. The model is then trained for about 25 epochs where the loss stabilizes and doesn't decrease further. A longer training time resulted in good results in validation and test. One can train for longer epochs as well.

## Results
The model now picks the letter with the highest probability from the distribution it has learnt and is able to achieve a win rate of $>60%$ on the validation set. One can tune the hyperparameters and train for longer for better results.

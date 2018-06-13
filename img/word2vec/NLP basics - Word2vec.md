+++
title = "NLP basics - Word2vec: Skip-gram, CBOW, GloVe" 
date = 2018-06-01T21:00:00
math = true
highlight = true

# List format.
#   0 = Simple
#   1 = Detailed
list_format = 1


tags = ["NLP", "word2vec", "deep learning", "tensorflow"]
summary = "The post talks about a series of word2vec model, analyzes their pros and cons, also implementing details (TensorFlow code)"

+++

## Table of Content

# Overview

First of all, we need to figure out how do we represent	the	meaning	of a word. A common solution is using WordNet: a	resource containing lists	of synonym	sets and hypernyms ("is	a" relationships). 

<img src="/img/word2vec/WordNet.png" width=700>

But this method is not so good. It is great	as a resource but	missing	nuance. Also it misses the new meaning of a word. 

Another way is representing	words	as discrete symbols. For example, like "hotel" and "motel". We can represent them as $hotel = [0, 0, 0, 1]$ while $motel = [0, 0, 1, 0]$. In this way, word vector dimension is equal to the number of words in	vocabulary. But apparently, this method is not perfect. The word "hotel" and "motel" is quite similar in meaning, but the word vectors of them are orthogonal, which is hard to quantify their similarity. 

So, here comes the word2vec models. The core idea is: A	word’s meaning is	given	by the words that frequently appear close-by. This is one of the most successful ideas in modern NLP. We will build a dense word vector for	each word (dimension far less than the number of words in vocabulary)	so	that	the words	that	appear	in	similar	contexts have similar word vectors.  

[Word2vec (Mikolov et	al.	2013)](https://arxiv.org/pdf/1310.4546.pdf)	is	a	framework	for	learning
word vectors. The idea is as follows: 

1. We	have	a	large	corpus	of	text
2. Every	word	in	a	fixed	vocabulary	is	represented	by	a	vector
3. Go	through	each	position	t in	the	text,	which	has	a	center	word c and	context	(“outside”)	words	o
4. Use	the	similarity	of	the	word	vectors	for	c	and	o to	calculate	the	probability	of	o given	c	(or	vice	versa)
5. Keep	adjusting	the	word	vectors	to	maximize	this	probability

Here is an example of calculating the	probability	of c given o: 

<img src="/img/word2vec/wordwindow.png" width=800>

So the total objective function is: 

$$L(\theta) = \prod\_{t=1}^{T} \prod\_{j  \in [-m, 0) \bigcup (0, m]} P(w\_{t+j}|w_t;\theta)  $$

Where the $T$ is corpus size, and m is the half window size. Our goal would be minimizing the negative log-likelihood: 

$$J(\theta) = -\frac{1}{T} \sum\_{t=1}^{T} \sum\_{j  \in [-m, 0) \bigcup (0, m] }  P(w\_{t+j}|w_t;\theta)  $$

So, the question is: how to calculate $P(w\_{t+j}|w_t;\theta)$, the probability	of	o given	c. Here we will use the most simple one: vector inner product and softmax function to present the probability: 

$$P(o|c) = \frac{exp(u_o^Tv_c)}{\sum\_{w \in V}exp(u_w^Tv_c)}$$

Hmm...Softmax again. Neural network then comes to play. 

# Basic Algorithm 

## Skip-Gram

### Intuition

The [skip-gram algorithm](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) is surprisingly simple and elegant. We’re going to train a simple neural network with a single hidden layer to minimizing the negative log-likelihood listed above. The input word is the **center word** of windows on the text corpus, while the output word is the **surrounding words**. And every word have two word vectors, $v$ for center and $u$ for context. But we are not actually using the neural work afterwards, instead just using the weight we trained in the  hidden layer. This weight matrix is actually the word vectors we want. 

First, say we have a vacabulary of 10,000 words. We’re going to represent an input word like "hotel" as a one-hot vector. So the one hot vector will be 10000 dimensional. The output of the network is a single vector (also 10,000 dimensional), predicting the probability that certain word is appearing nearby the input word. 

The architecture of skip-gram network is simple: 

<img src="/img/word2vec/skip-gram.png" width=800>

There is no activation function on the hidden layer neurons, but the output neurons use softmax to predict the probability of each word's appearance in each nearby location.

Since the input is a one-hot vector, it just select the matrix row corresponding to the "1". So, this is just a simple lookup table on the weight $W$ in the above figure. **The corresponding row in weight W is the word vector of input word**. 

The word vector will be sent into the output layer to produce the probability of each word's appearance in each location. And number of prediction head is equal to the window size - 1.

The brilliance of skip-gram is that it makes sure the words frequently appearing in similar context would have similar word vectors, because the network only uses context words as ground truth. So, similar context words will certainly produce similar results. 

### Implementation details

The authors proposed some [improvements](https://arxiv.org/pdf/1310.4546.pdf) to the original skip-gram model. There are three main modifications: 

1. Use Word Pairs and "phrases"
2. Subsample frequent words
3. Negative sampling

**Use Word Pairs and “phrases”**: The author pointed out that some common word phrases, like "New York City", should be represented as a single word with its own word vector. This is quite natural to understand. We often use word phrase as a word and their meanings are different when splitting them apart. 

The phrase detection mechanism is quite simple, too. We just need to count the number of times each combination of two words appears in the training text, and then these counts are used in an equation to determine which word combinations to turn into phrases. This equation should be related to words co-occurence and individual occurence. 

**Subsample frequent words**: 

There are two “problems” with common words like “the”:

1. When looking at word pairs, ("fox"", "the") doesn’t tell us much about the meaning of "fox". "the" appears in the context of pretty much every word.
2. We will have many more samples of ("the", …) than we need to learn a good vector for “the”.

So, we can subsample frequent words using the follow equation: 

$$P(w_i) = (\sqrt{\frac{z(w_i)}{0.001}} + 1) \times \frac{0.001}{z(w_i)}$$

Where $P(w_i)$ is probability of keeping the word and $z(w_i)$ is the frequency of the word. For example, if "we" appears 100 time in a corpus of 10000 words, $z("we")$ should be 0.01. 

**Negative sampling**: Although the network only has three layers, but the weight size is huge because of a large vacabulary. And the nature of softmax (gradient on all input) function will result in huge amount of computation in single training step. 

Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. We are instead going to randomly select just a small number of “not ground truth” words (let’s say 5) to update their weights. In the paper, the authors suggested 5-20 words for smaller datasets, and you can get away with only 2-5 words for large datasets.

Here a totally random selection is not good. some rare words may have little probability to be selected. So, the sampling prob comes from an equation: 

$$P(w_i) = \frac{f(w_i)^{\frac{3}{4}}}{\sum\_{j=0}^{n} {f(w_i)^{\frac{3}{4}}}}$$

where $f(w_i)$ is the word appearance count. And the power $\frac{3}{4}$ is an empirical value, which works very well. Recent research result also proved the effectiveness of this value. 

[Here](https://www.tensorflow.org/tutorials/word2vec) is an official tutorial of word2vec on TensorFlow. 

## CBOW

In the other hand, Continuous	Bag	of	Words	(CBOW) does the opposite thing compared to skip-gram. It uses the context words to predict the center word. (image [source](https://zhuanlan.zhihu.com/p/35074402))

<img src="/img/word2vec/CBOW.jpg" width=800>

Note the weight matrix for every input is **shared**. All hidden layer weights are **added** and send into next layer. 

**Skip-Gram and CBOW difference**: CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). For the most part, this turns out to be a useful thing for smaller datasets. However, skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasets. Also, this makes the skip-gram model performs better on infrequent words.

## Co-occurrence Models

## GloVe



## Evaluation Metrics


# Limitation


# Reference Materials

1. [Stanford cs224n lectures](http://web.stanford.edu/class/cs224n/syllabus.html)
2. [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
3. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)
4. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)





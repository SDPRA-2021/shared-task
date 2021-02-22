This is the repository of the [shared task](https://sdpra-2021.github.io/website/) at PAKDD2021 on scholar text (abstract) classification for the solution from team **parklize**. 



There are two main ```.ipynb``` notebooks for the solution including:

- ```pakdd2021_fasttext_entityembeddings.ipynb``` and 
- [Google Colab notebook](https://colab.research.google.com/drive/1x9MUQxXa2BnSVYjUMrgfy3oZa_p0YFXu?usp=sharing)



# Details

```pakdd2021_fasttext_entityembeddings.ipynb``` does two things:

- training a [fasttext](https://fasttext.cc/) classifier
- get sentence embeddings with extracted entities using [TagMe](https://tagme.d4science.org/tagme/) and [wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/)



Regarding training a fasttext classifier, there are several steps (cells):

- read challenge data
- split validation set further into *internal* validation & test sets 
- change data to fasttext format
- train a fasttext classifier using [fasttext](https://fasttext.cc/)
- predict on test set(s)



Regarding getting sentence embeddings with extracted entities

- extract Wikipedia entities/articles using TagMe
- get abstract embeddings by aggregating entity embeddings for those entities mentioned in each abstract
  - the entities further filtered by applying k-means clustering (with two clusters) by choosing the large cluster with the assumption that the smaller one consists of noisy entities



The Google Colab notebook does several things such as:

- training Sentence-BERT classifiers (7) with [sentence transformers](https://www.sbert.net/), and testing with those classifiers
- training a classifier with ```universal-sentence-encoder``` from [Tensorflow Hub](https://www.tensorflow.org/hub) for encoding abstract texts, and testing with this classifier
- loading the fasttext classifier's prediction result
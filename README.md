# Symptom Classification With Natural Language Processing
When creating defect reports it is common for a Quality Technician to select an incorrect symptom, and making correction to those reports in the midst of production chaos is very difficult. To effectively and accurately analyze manufacturing quality defects, I wanted to create a Natural Language Processing (NLP) model that could classify each defect description text into accurate symptom type. Output from this project has great potential to benefit the production team with analyzing defects data without the need to manually fix incorrect symptom documentation.

### Library Imports
```ruby
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from string import punctuation
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Loading text data of quality defect comments and symptoms
```ruby
df = pd.read_csv('data_type.csv')
df = df.astype('string')
```
This is the overall dataset available for training and testing the model. The dataset is not large enough to train the model to classify all symptom types, as many classes have few samples which if used for training will render model more biased towards the first three classes, especially biased towards the first class - **"5 - Quality Issue - Assembly".**

In cases where sufficient data size for training is not available, Data Augmentation can be used to augment data for training NLP models. With some techniques, a dataset with small size can be utlized to create more data for training. Some of the techniques are
1. Synonymn Replacement
2. Back Translation
3. Bigram Flipping
4. Replacing Entities
5. Adding Noise to Data

For this particular project, first three classes with the largest samples are considered for this training to keep things simple.

```ruby
df['Symptom Type'].value_counts()
```
**OUTPUT:**

_5 - Quality Issue - Assembly      5011_</br>
_2 - Quality Issue - Appearance    1594_</br>
_4 - Quality Issue - Functional    1016_</br>
_Name: Symptom Type, dtype: Int64_


```ruby
df.dropna(inplace=True)
df = df[df['Symptom Type'].isin(['5 - Quality Issue - Assembly','2 - Quality Issue - Appearance','4 - Quality Issue - Functional'])]
x = df['Reporter Comment']
y = df['Symptom Type']
```
### Text Pre-Processing
To prepare the text for training and testing model, the following needs to be performed:

1. Tokenization. Splitting the setences or text into words.
2. Lowercasing the words.
3. Removing stopwords
4. Removing punctuations and digits

```ruby
def text_processor(texts):
    mystopwords = set(stopwords.words("english"))
    def remove_stpwrd_punc_dig(tokens):
        return [token.lower() for token in tokens if token.lower() not in mystopwords and not token.isdigit() and token not in punctuation]
    return [remove_stpwrd_punc_dig(word_tokenize(text)) for text in texts]

processed_x = text_processor(x)
```

Here is the breakdown of training and testing data
```ruby
print('Size of processed text: ',len(processed_x), '\nSize of classes: ', len(y))
print('\nClasses and their sizes:\n', y.value_counts())
```

**OUTPUT:**

Size of processed text:  7621 
Size of classes:  7621

_Classes and their sizes:_</br>
_5 - Quality Issue - Assembly      5011_</br>
_2 - Quality Issue - Appearance    1594_</br>
_4 - Quality Issue - Functional    1016_</br>
_Name: Symptom Type, dtype: Int64_</br>

### Feature Engineering/Text Representation with Word Embeddings

The pre-processed text is represented in a numerical form so that it can be fed to a machine learning (ML) algorithm, and this step is referred to as text representation in NLP. More often in NLP, an optimal text representation yields far greater results even when used with an ordinary ML algorithm.

There are different techniques available for performing text representations in NLP. Some of them that I am faimiliar with are
1. Basic Vectorization of Text
2. Word Embeddings

One way to perform text representation is through basic vector representation of pre-processed text data, such using One-Hote encoding, Bag of Words, Bag of N-Grams, N-Grams, and Term Frequency-Inverse Document Frequency (TF-IDF). However, using basic vectorization of text come with drawbacks:
- words are treated as atomic units, so relationship among them cannot be established
- features vector size and sparisty incraeses which makes NLP model computationally expensive and can cause overfitting
- the NLP model cannot handle out of vocabulary words

Another way is to use Word Embeddings, which utlize the concept of Distributional Representations. Here the goal is to come up with a text representation that enables the model to derive meaning of the words from their contexts. A Data Scientist can train her/his own word embeddings for text representation or load existing, pre-trained word embeddings. Some popular pre-trained word embeddings are Word2Vec by Google, fasttext embeddings by Facebook, and GloVe word embeddings by Stanford University.

For this particular task in hand, I will use pre-trained Word2Vec model. When creating this model, researchers came up with two architecures that could be used to train this model - Continous Bag of Words (CBOW) and SkipGram. The architectures are similar. With both architectures, the basic idea is to generate small size (between 25 to 600) numerical feature vectors for each word in the corpus, and use those feature vectors as means of comparing with other words in the corpus. Cosine similarity is generally used to compare between the words (feature vectors). Although there are some differences between CBOW and SkipGram, but the main conceptual difference is that in SkipGram a center word is used to predict its surrounding words; whereas, in CBOW, the surrounding (context) words are used to predict the center word.

Word Embeddings, whether pre-trained or trained, also come with drawbacks. Two common drawbacks are:
1. Word embeddings model size is large which makes the NLP model difficult to deploy. As a student of Computer Science, it will be a challenge for me to deploy an NLP model that used word2vec embeddings into Heroku's free tier server. 

2. Out of vocabulary issue as mentioned above

As shown below, word2vec model is being loaded.

```ruby
w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
```
Below is an example cosine similarities of the most similar words to 'tesla' are displayed using the word2vec model. As can be seen, the context words for tesla are related to physics or electricity. This points out that the corpus used for modelling the Word2Vec embeddings did not contain or had very much less information related to the car manufacturing company, Tesla.

**Note-** All words shown here, including 'universe', are present in the model's corpus, which means these words were present in the corpus that used for creating the Word2Vec embeddings model.

```ruby
w2v_model.most_similar('tesla')
```

**_OUTPUT:_**
_[('gauss', 0.6623970866203308),_</br>
_('FT_ICR', 0.5639052391052246),_</br>
_('MeV', 0.5619181990623474),_</br>
_('keV', 0.5605965256690979),_</br>
_('superconducting_magnet', 0.5567352175712585),_</br>
_('electron_volt', 0.5503560900688171),_</br>
_('SQUIDs', 0.5393732786178589),_</br>
_('nT', 0.5386143326759338),_</br>
_('electronvolts', 0.5377055406570435),_</br>
_('kelvin', 0.5367920994758606)]_</br>

### If a particular word is not present in Word2Vec model corpus then an 'Key not present' error will display!
```ruby
w2v_model.most_similar('AWordNotPresentInTheModelCorpus')
```
![image](https://user-images.githubusercontent.com/67841104/166082909-8f7a9367-c489-4958-a949-d64b31442dc1.png)
</br>

Feature vector of a word present in model's corpus can be acquired. The code below shows an example of getting feature vector of the word 'tesla'. Also, the feature vector size was set to 300 during Word2Vec development, which results in all feature vectors of words in corpus to have same size feature vector of 300.

```ruby
print(len(w2v_model['tesla']))
w2v_model['tesla']
```
**_OUTPUT:_**
_300_

![image](https://user-images.githubusercontent.com/67841104/166083298-2fb7e275-307d-4a79-be4f-dd60774918bd.png)

### Function to acquire feature vectors of words in the pro-processed training corpus

The **embedding_features** function is to acquire feature vectors of words in the pro-processed training corpus. The function goes through each word in training corpus and verifies whether the word is present in embeddings model. If a word exists in the w2v_model then its feature vector is generated. For a single sentence, feature vectors of all words are averaged to represent an average feature vector of each sentence in the training corpus.

As mentioned above, one drawback of Word2Vec model is the out of vocabulary issue. In this function below, any word in the training corpus that is not present in the w2v_model is discarded. Therefore, when using a pre-trained embeddings model it is important to keep in mind the domain under which the embeddings model was trained. Word2Vec was trained using large corpus from Google News; however, my training dataset is a corpus from manufacturing, which not ideal. A pre-trained word embeddings model trained using manufacturing text corpus would yeild a much higher end result.

```ruby
def embedding_features(list_of_lists):
    DIMENSION = 300
    zero_vector = np.zeros(DIMENSION)
    features = []
    for tokens in list_of_lists:
        count_for_this = 0 + 1e-5
        feat_for_this = np.zeros(DIMENSION)
        for token in tokens:
            if token in w2v_model:
                feat_for_this += w2v_model[token]
                count_for_this += 1
        if (count_for_this != 0):
            features.append(feat_for_this/count_for_this)
        else:
            features.append(zero_vector)
    return features

features = embedding_features(processed_x)
```
Although the size of processed_x (mentioned few cells above) and features are same, 7621, within each list (pre-processed sentence) of processed_x, some words have been eliminated because of domain discrepancy between manufacturing and news.
```ruby
features = np.array(features)
features.shape
```
**_OUTPUT:_**
_(7621, 300)_

### Random Forest Classifier for Prediction
```ruby
#Take any classifier (LogisticRegression here, and train/test it like before.
classifier = RandomForestClassifier()
train_data, test_data, train_cats, test_cats = train_test_split(features, y)
classifier.fit(train_data, train_cats)
preds = classifier.predict(test_data)
```
### Model Evaluation
Model's f1-score with respect to **5 - Quality Issue - Assembly** is the highest compared to the other two categories. As assumed before due to imbalanced sample sizes of the categories, the classifier's prediction for this 'assembly' symptom was more precise (on average, out of 100 samples predicted as 'assembly', 96 of them were accurate/true positive) compared to its prediction for other two symptom's.

```ruby
print(classification_report(preds, test_cats))
```
![image](https://user-images.githubusercontent.com/67841104/166083113-54a3d53e-6d4f-47c6-bc75-90247c7967f2.png)

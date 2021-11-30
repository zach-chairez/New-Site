---
title: IMDB Sentiment Analysis
date: 2021-11-30T07:13:01.608Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
## <u> CSE 5335 Assignment #3:  Naive-Bayes Classifier </u>

### <u> References</u>

* I slightly altered the function $\textbf{word_count}$ from [here](https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-12.php) to create a dictionary of the words from the datasets.
* The image used for explaining 5-fold cross validation was from [here](https://aiaspirant.com/cross-validation/)
* I referred to to [this link](https://docs.python.org/2/library/collections.html) for the function $\textbf{Counter}$.  
* All other code and algorithms are of my original work.  

#### <u>Goal:</u>

The goal of this assignment is to build a simple text classifier for a set of movie reviews provided by the [IMBD Sentiment Analysis Sub-Dataset](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set) using a Naive-Bayes classification scheme.  We'll begin by introducing Bayes Theorem and the concept of the Naive-Bayes assumption.   

## <u>Bayes Theorem</u>

Given two events $A$ and $B$, the conditional probability of A happening given that B has already happened is given by Bayes formula
<br>

$$
\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{P(B)}, \ \ \ \ \ \  P(B) \neq 0,
\end{equation}
$$
where $P(B|A)$ is the conditional probability of $B$ happening given that $A$ has already happened, $P(A)$ is the probability of $A$ happening, and $P(B)$ is the probability of B happening.  The probability $P(B)$ can be written in terms of a sum over all of its potential marginal distributions.  When written in this way, Bayes formula is alternatively recognized as

<br>

$$
\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{\sum_{A'} P(B|A')P(A')}
\end{equation}
$$

The denominator $\sum_{A'} P(B|A')P(A')$ is adding up all possible events $A'$, where $A'$ belongs to some event space.  Note that the event $A$ is also an event that needs to be summed over.  The value or function $P(A)$ is most notably called the $\textbf{prior probability}$.  The prior probability is exactly what it sounds like.  It's the prior information we have available to us going into a few decision.  Similarly, the value or function $P(A|B)$ is notably called the $\textbf{posterior probability}$.  The posterior probability is an informed probability, utilizing the information gathered in the prior to make a more educated and accurate prediction.  

## <u>Naive-Bayes</u>

The backbone of Naive-Bayes is the assumption of independence of conditional probabilities.  Let $A_1,A_2,\dots,A_n$ be a sequence of events.  Then the assumption of Naive-Bayes states

<br>
$$
\begin{equation}
P(A_1 \cap A_2 \cap \dots \cap A_{n-1}|A_n) = \prod_{j=1}^{n-1}P(A_j|A_n).
\end{equation}
$$
    
<br>
In general, the events $A_1,A_2,\dots,A_n$ are dependent on each other.  Assuming they're indepedent or conditionally independent is a $\textit{naive}$ assumption, hence the name Naive-Bayes.  However, implementing this assumption within a naive-bayes classifier, specifically for text classification, works fairly well.  In this blog post, we'll implement the assumption of naive-bayes to create a classifier that will predict the sentiment (good or bad) of a movie review.  

### <u> IMBD Sentiment Analysis </u>

The original IMBD dataset contains 50,000 movie reviews with a positive (1) or negative (0) review as a label.  The data set is split between 25,000 positive and 25,000 negative reviews.  In the dataset we'll be working in this blog post, we'll deal with a subset of the original dataset.  The original data set can be found [here](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) and the sub-datasetset can be found [here](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set) with descriptions.  The sub-data set has

* 500 movie reviews, with
* 250 positive/negative reviews.     

#### <u> Uploading the Sub-Dataset </u>

Let's begin by uploading the sub-dataset and split the data into training/testing. 

```python
# Importing the Pandas library and uploading the dataset to the variable "data"
import pandas as pd
from collections import Counter
import numpy as np

data = pd.read_csv("C:\\Users\zachc\OneDrive\Desktop\Yea\School\Fall 2021\Data Mining CSE\A1\Assignment #1 Data Mining\imdb_sentiment_analysis.csv")
data = pd.DataFrame(data)
data = data.rename(columns={'Column1': 'Text_Entries', 'Column2': 'Sentiment'})

# Displaying the first 10 text entries of the dataset and their associated sentiment score
# 1 = positive, 0 = negative
print(data.head(10))

# Splitting the dataset between the text and sentiment columns
text = data.Text_Entries; sentiment = data.Sentiment

# Splitting the text and sentiment columns by their positive and negative reviews
pos = sentiment[sentiment == 1]; neg = sentiment[sentiment == 0]
ptext = text[pos.index]; ntext = text[neg.index]

# Gluing the positive and negative texts with their associated sentiments.
pos_data = pd.concat([pos,ptext], axis = 1); neg_data = pd.concat([neg,ntext], axis = 1)
```

```
                                        Text_Entries  Sentiment
0  A very very very slow moving aimless movie abo...          0
1  Not sure who was more lost the flat characters...          0
2  Attempting artiness with black and white and c...          0
3        Very little music or anything to speak of.           0
4  The best scene in the movie was when Gerardo i...          1
5  The rest of the movie lacks art charm meaning ...          0
6                                 Wasted two hours.           0
7  Saw the movie today and thought it was a good ...          1
8                                A bit predictable.           0
9  Loved the casting of Jimmy Buffet as the scien...          1
```

 As we see above, the data set contains a column of $\textbf{Text Entries}$ and $\textbf{Sentiments}$.  We'll now randomly shuffle the data so we can unbiasedly split the data into training/development/testing data sets.  

```python
# Randomly shuffling data 
pos_data = pos_data.sample(frac=1); neg_data = neg_data.sample(frac=1)
```

As part of an exercise, we'll start by finding the probability of finding the word $\textbf{"the"}$ in a single text entry for the entire data set, i.e.,

<br>
$$\begin{equation}
P(\text{"the"}) = \frac{\text{# of text entries containing the word "the"}}{\text{total number of text entries}}
\end{equation}$$

We'll run through each text entry and identify whether the word "the" appears at least once, then move onto the next text entry until we've gone through every one.  

```python
agg1 = 0; 
for i in range(0,len(text)):
    if 'the' in text.iloc[i].lower():
        agg1 += 1
print('The probability of randomly selecting a text entry and it contains the word "the" is', (agg1/len(text)*100), '%')
```

```
The probability of randomly selecting a text entry and it contains the word "the" is 58.9 %
```

As stated above, the probability of finding the word "the" in a single text entry is $58.9 %$.  Roughly 3/5 text entries contain the word "the".  Now, we'll find the probability of finding the word "the" in a positive text entry, i.e., 

<br>
$$\begin{equation}
 P(\text{"the"}| \text{positive text entry}) = \frac{\text{# of text entries containing the word "the" in a positive text entry}}{\text{total number of positive text entries}}
\end{equation}$$

```python
agg2 = 0; 
for i in range(0,len(ptext)):
    if 'the' in ptext.iloc[i].lower():
        agg2 += 1
print('The probability of randomly selecting a text entry and it contains the word "the" given that it\'s a positive entry is', (agg2/len(ptext)*100), '%')
```

```
The probability of randomly selecting a text entry and it contains the word "the" given that it's a positive entry is 58.4 %
```

Similar to the probability of finding the word "the" in any text entry, the probablity of finding the word "the" given a positive text entry is approximately 3/5 or, more specifically, $58.4%$.  

In this next section, let's recall the process of 5-fold cross validation.  

#### <u> 5-fold cross validation </u>

<img src = "https://i2.wp.com/thatdatatho.com/wp-content/uploads/2018/10/5-fold-cross-validation-split.png?resize=807%2C491&is-pending-load=1#038;ssl=1" width = "600"/>
    
The image above displays the basic concept of the structure of 5-fold cross validation.  We'll split our development into five subsets.  During the first trial, the subsets 2-4 will be joined together and used as a single training set, while the $1^{st}$ will be used as a testing set.  In the second trial, subsets 1 and 3-5 will be used as training sets while the $2^{nd}$ data set will be used as the testing set.  And so on.  In the end, we can we establish a prediction from the training set and check the accuracy against the testing set.  We can then use the prediction which closely predicts the testing set, or we have the option of using the average of the predictions as our final prediction.  This process is known as 5-fold cross validation since we're running five different training experiments and validating the accuracy of our predictions on the training sets.  

Up next, we'll split our data set into three sections:  $\textbf{Training, Development, and Testing}$.   

```python
# Splitting data into train, dev, and test by 80/10/10.
train_pos = pos_data[0:400]; train_neg = neg_data[0:400]
dev_pos = pos_data[400:450]; dev_neg = neg_data[400:450]
test_pos = pos_data[450:]; test_neg = neg_data[450:];
```

```python
# Gluing the train and testing data back together, then shuffling one more time.
train = pd.concat([train_pos,train_neg]); train = train.sample(frac=1)  
dev = pd.concat([dev_pos,dev_neg]); dev = dev.sample(frac=1) 
test = pd.concat([test_pos,test_neg]); test = test.sample(frac=1)
```

Using $\textbf{5-fold cross validation}$ to check the accuracy of our probabilities.

```python
d1 = dev[0:20]; d2 = dev[20:40]; d3 = dev[40:60]; d4 = dev[60:80]; d5 = dev[80:]
train1 = pd.concat([d1,d2,d3,d4]); test1 = d5;
train2 = pd.concat([d1,d2,d3,d5]); test2 = d4;
train3 = pd.concat([d1,d2,d4,d5]); test3 = d3;
train4 = pd.concat([d1,d3,d4,d5]); test4 = d2;
train5 = pd.concat([d2,d3,d4,d5]); test5 = d1;

# Checking the accuracy of P('the') using 5-fold cross validation
agg_train1 = 0; agg_test1 = 0
for i in range(0,len(train1)):
    if 'the' in train1.Text_Entries.iloc[i].lower():
        agg_train1 += 1
for i in range(0,len(test1)):
    if 'the' in test1.Text_Entries.iloc[i].lower():
        agg_test1 += 1
print('We predict (in Test set 1) the probability of finding the word "the" is', ("{:.2f}".format(agg_train1/len(train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_test1/len(test1)*100)), '%')

agg_train2 = 0; agg_test2 = 0 
for i in range(0,len(train1)):
    if 'the' in train2.Text_Entries.iloc[i].lower():
        agg_train2 += 1
for i in range(0,len(test1)):
    if 'the' in test2.Text_Entries.iloc[i].lower():
        agg_test2 += 1
print('We predict (in Test set 2) the probability of finding the word "the" is', ("{:.2f}".format(agg_train2/len(train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_test2/len(test1)*100)), '%')

agg_train3 = 0; agg_test3 = 0 
for i in range(0,len(train1)):
    if 'the' in train3.Text_Entries.iloc[i].lower():
        agg_train3 += 1
for i in range(0,len(test1)):
    if 'the' in test3.Text_Entries.iloc[i].lower():
        agg_test3 += 1
print('We predict (in Test set 3) the probability of finding the word "the" is', ("{:.2f}".format(agg_train3/len(train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_test3/len(test1)*100)), '%')

agg_train4 = 0; agg_test4 = 0 
for i in range(0,len(train1)):
    if 'the' in train4.Text_Entries.iloc[i].lower():
        agg_train4 += 1
for i in range(0,len(test1)):
    if 'the' in test4.Text_Entries.iloc[i].lower():
        agg_test4 += 1
print('We predict (in Test set 4) the probability of finding the word "the" is', ("{:.2f}".format(agg_train4/len(train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_test4/len(test1)*100)), '%')

agg_train5 = 0; agg_test5 = 0 
for i in range(0,len(train1)):
    if 'the' in train5.Text_Entries.iloc[i].lower():
        agg_train5 += 1
for i in range(0,len(test1)):
    if 'the' in test5.Text_Entries.iloc[i].lower():
        agg_test5 += 1
print('We predict (in Test set 5) the probability of finding the word "the" is', ("{:.2f}".format(agg_train5/len(train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_test5/len(test1)*100)), '%')
```

```
We predict (in Test set 1) the probability of finding the word "the" is 63.75 % while the actual probability is 50.00 %
We predict (in Test set 2) the probability of finding the word "the" is 63.75 % while the actual probability is 50.00 %
We predict (in Test set 3) the probability of finding the word "the" is 55.00 % while the actual probability is 85.00 %
We predict (in Test set 4) the probability of finding the word "the" is 57.50 % while the actual probability is 75.00 %
We predict (in Test set 5) the probability of finding the word "the" is 65.00 % while the actual probability is 45.00 %
```

As we see above, the predictions are roughly $60 %$.  Some of the prediction accuracies are perfect, while some are far off.  For example, in the first experiment, we predicted a  $65 %$ probability of the word "the" appearing in a text entry while the actual probability was  $40 %$.  Overall, we can safely say the true probability probably falls close to  $60 %$.  Up next, we'll perform the same task, however we'll check the probabilities of find the word "the" given a positive and negative text entry.      

Here, we're splitting the positive and negative text entries from the previously made training sets with their appropriate sentiments.  Then, we'll use 5-fold cross validation again to check our prediction accuracies.  

```python
# Positive and Negative text separation for each training set.
pos_train1 = train1.Text_Entries[train1.Sentiment == 1] 
neg_train1 = train1.Text_Entries[train1.Sentiment == 0]
pos_train2 = train2.Text_Entries[train2.Sentiment == 1]
neg_train2 = train2.Text_Entries[train2.Sentiment == 0]
pos_train3 = train3.Text_Entries[train3.Sentiment == 1]
neg_train3 = train3.Text_Entries[train3.Sentiment == 0]
pos_train4 = train4.Text_Entries[train4.Sentiment == 1]
neg_train4 = train4.Text_Entries[train4.Sentiment == 0]
pos_train5 = train5.Text_Entries[train5.Sentiment == 1]
neg_train5 = train5.Text_Entries[train5.Sentiment == 0]

# Positive and Negative index separation for each testing set.
pos_test1 = test1.Text_Entries[test1.Sentiment == 1] 
neg_test1 = test1.Text_Entries[test1.Sentiment == 0]
pos_test2 = test2.Text_Entries[test2.Sentiment == 1]
neg_test2 = test2.Text_Entries[test2.Sentiment == 0]
pos_test3 = test3.Text_Entries[test3.Sentiment == 1]
neg_test3 = test3.Text_Entries[test3.Sentiment == 0]
pos_test4 = test4.Text_Entries[test4.Sentiment == 1]
neg_test4 = test4.Text_Entries[test4.Sentiment == 0]
pos_test5 = test5.Text_Entries[test5.Sentiment == 1]
neg_test5 = test5.Text_Entries[test5.Sentiment == 0]
```

```python
# Checking P('the' | NEGATIVE TEXTS) via 5-fold cross validation
agg_ntrain1 = 0; agg_ntest1 = 0
for i in range(0,len(neg_train1)):
    if 'the' in neg_train1.iloc[i].lower():
        agg_ntrain1 += 1
for i in range(0,len(neg_test1)):
    if 'the' in neg_test1.iloc[i].lower():
        agg_ntest1 += 1
print('We predict (in Test set 1) the probability of finding the word "the" given a negative text entry is', ("{:.2f}".format(agg_ntrain1/len(neg_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ntest1/len(neg_test1)*100)), '%')

agg_ntrain2 = 0; agg_ntest2 = 0 
for i in range(0,len(neg_train2)):
    if 'the' in neg_train2.iloc[i].lower():
        agg_ntrain2 += 1
for i in range(0,len(neg_test2)):
    if 'the' in neg_test2.iloc[i].lower():
        agg_ntest2 += 1
print('We predict (in Test set 2) the probability of finding the word "the" given a negative text entry is', ("{:.2f}".format(agg_ntrain2/len(neg_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ntest2/len(neg_test1)*100)), '%')

agg_ntrain3 = 0; agg_ntest3 = 0 
for i in range(0,len(neg_train3)):
    if 'the' in neg_train3.iloc[i].lower():
        agg_ntrain3 += 1
for i in range(0,len(neg_test3)):
    if 'the' in neg_train3.iloc[i].lower():
        agg_ntest3 += 1
print('We predict (in Test set 3) the probability of finding the word "the" given a negative text entry is', ("{:.2f}".format(agg_ntrain3/len(neg_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ntest3/len(neg_test1)*100)), '%')

agg_ntrain4 = 0; agg_ntest4 = 0 
for i in range(0,len(neg_train4)):
    if 'the' in neg_train4.iloc[i].lower():
        agg_ntrain4 += 1
for i in range(0,len(neg_test4)):
    if 'the' in neg_test4.iloc[i].lower():
        agg_ntest4 += 1
print('We predict (in Test set 4) the probability of finding the word "the" given a negative text entry is', ("{:.2f}".format(agg_ntrain4/len(neg_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ntest4/len(neg_test1)*100)), '%')

agg_ntrain5 = 0; agg_ntest5 = 0 
for i in range(0,len(neg_train5)):
    if 'the' in neg_train5.iloc[i].lower():
        agg_ntrain5 += 1
for i in range(0,len(neg_test5)):
    if 'the' in neg_train5.iloc[i].lower():
        agg_ntest5 += 1
print('We predict (in Test set 5) the probability of finding the word "the" given a negative text entry is', ("{:.2f}".format(agg_ntrain5/len(neg_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ntest5/len(neg_test1)*100)), '%')
```

```
We predict (in Test set 1) the probability of finding the word "the" given a negative text entry is 63.41 % while the actual probability is 44.44 %
We predict (in Test set 2) the probability of finding the word "the" given a negative text entry is 56.10 % while the actual probability is 77.78 %
We predict (in Test set 3) the probability of finding the word "the" given a negative text entry is 58.54 % while the actual probability is 44.44 %
We predict (in Test set 4) the probability of finding the word "the" given a negative text entry is 56.10 % while the actual probability is 77.78 %
We predict (in Test set 5) the probability of finding the word "the" given a negative text entry is 58.54 % while the actual probability is 66.67 %
```

As shown above, the predicted probablities of finding the word "the" given a negative text entry, on average, are close to the actual accuracies of $ \sim60%$, however the predictions themselves are not entirely accurate.  We'll need to keep this in mind when we're establishing prior probablities later.  

```python
# Checking P('the' | POSITIVE TEXTS)
# Checking P('the' | NEGATIVE TEXTS) via 5-fold cross validation
agg_ptrain1 = 0; agg_ptest1 = 0
for i in range(0,len(pos_train1)):
    if 'the' in pos_train1.iloc[i].lower():
        agg_ptrain1 += 1
for i in range(0,len(pos_test1)):
    if 'the' in pos_test1.iloc[i].lower():
        agg_ptest1 += 1
print('We predict (in Test set 1) the probability of finding the word "the" given a positive text entry is', ("{:.2f}".format(agg_ptrain1/len(pos_train1)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ptest1/len(pos_test1)*100)), '%')

agg_ptrain2 = 0; agg_ptest2 = 0 
for i in range(0,len(pos_train2)):
    if 'the' in pos_train2.iloc[i].lower():
        agg_ptrain2 += 1
for i in range(0,len(pos_test2)):
    if 'the' in pos_test2.iloc[i].lower():
        agg_ptest2 += 1
print('We predict (in Test set 2) the probability of finding the word "the" given a positive text entry is', ("{:.2f}".format(agg_ptrain2/len(pos_train2)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ptest2/len(pos_test2)*100)), '%')

agg_ptrain3 = 0; agg_ptest3 = 0 
for i in range(0,len(pos_train3)):
    if 'the' in pos_train3.iloc[i].lower():
        agg_ptrain3 += 1
for i in range(0,len(pos_test3)):
    if 'the' in pos_train3.iloc[i].lower():
        agg_ptest3 += 1
print('We predict (in Test set 3) the probability of finding the word "the" given a positive text entry is', ("{:.2f}".format(agg_ptrain3/len(pos_train3)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ptest3/len(pos_test3)*100)), '%')

agg_ptrain4 = 0; agg_ptest4 = 0 
for i in range(0,len(pos_train4)):
    if 'the' in pos_train4.iloc[i].lower():
        agg_ptrain4 += 1
for i in range(0,len(pos_test4)):
    if 'the' in pos_test4.iloc[i].lower():
        agg_ptest4 += 1
print('We predict (in Test set 4) the probability of finding the word "the" given a positive text entry is', ("{:.2f}".format(agg_ptrain4/len(pos_train4)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ptest4/len(pos_test4)*100)), '%')

agg_ptrain5 = 0; agg_ptest5 = 0 
for i in range(0,len(pos_train5)):
    if 'the' in pos_train5.iloc[i].lower():
        agg_ptrain5 += 1
for i in range(0,len(pos_test5)):
    if 'the' in pos_train5.iloc[i].lower():
        agg_ptest5 += 1
print('We predict (in Test set 5) the probability of finding the word "the" given a positive text entry is', ("{:.2f}".format(agg_ptrain5/len(pos_train5)*100)), '%'
      ' while the actual probability is', ("{:.2f}".format(agg_ptest5/len(pos_test5)*100)), '%')
```

```
We predict (in Test set 1) the probability of finding the word "the" given a positive text entry is 66.67 % while the actual probability is 36.36 %
We predict (in Test set 2) the probability of finding the word "the" given a positive text entry is 64.10 % while the actual probability is 45.45 %
We predict (in Test set 3) the probability of finding the word "the" given a positive text entry is 55.26 % while the actual probability is 75.00 %
We predict (in Test set 4) the probability of finding the word "the" given a positive text entry is 55.81 % while the actual probability is 85.71 %
We predict (in Test set 5) the probability of finding the word "the" given a positive text entry is 58.54 % while the actual probability is 77.78 %
```

As shown above, the predicted probablities of finding the word "the" given a positive text entry, on average, are close to the actual accuracies of $ \sim60%$, however the predictions themselves are not entirely accurate.  This follows in parallel to the results we saw above with the negative text entries.   

In this next section, we'll create dictionaries for the words that appear in all, positive, and negative text entries.  We'll then locate the most common, and in turn, the most useful words we believe will help us to identify a future potential text entry as a positive or negative movie review.  

```python
# Number of positive/negative in the training set of length 800. 
def word_count(str):
    counts = dict()
    words = str.split()
    
    for word in words:
            counts[word.lower()] = 1
    return counts

# Making a count of all the words that appear in all/positive/negative text entries.  
wordbag_all, wordbag_pos, wordbag_neg = '', '', '' 
agg_all, agg_pos, agg_neg = Counter(), Counter(), Counter()
for i in range(0,len(train)):
    word_all = word_count(train.Text_Entries.iloc[i])
    ctr_all = Counter(word_all)
    agg_all += ctr_all
for i in range(0,len(train_pos)):
    word_pos = word_count(train_pos.Text_Entries.iloc[i])
    ctr_pos = Counter(word_pos)
    agg_pos += ctr_pos
for i in range(0,len(train_neg)):
    word_neg = word_count(train_neg.Text_Entries.iloc[i])
    ctr_neg = Counter(word_neg)
    agg_neg += ctr_neg


# Creating dictionaries for all of the words that appear in all/positive/negative texts in the training set.
wordbag_all = dict(agg_all)
wordbag_pos = dict(agg_pos)
wordbag_neg = dict(agg_neg)

# Deleting words that show up less than 5 times
for key,val in dict(wordbag_all).items():
    if val < 5:
        del[wordbag_all[key]]
for key,val in dict(wordbag_pos).items():
    if val < 5:
        del[wordbag_pos[key]]
for key,val in dict(wordbag_neg).items():
    if val < 5:
        del[wordbag_neg[key]]

# Creating sub-dictionaries with the 50 most common words in the positive/negative dictionaries.
wordbag_pcommon = dict(Counter(wordbag_pos).most_common(50))
wordbag_ncommon = dict(Counter(wordbag_neg).most_common(50))
```

```python
# Displaying the most common positive words
print(wordbag_pcommon)
```

```
{'the': 205, 'and': 170, 'a': 159, 'of': 136, 'is': 124, 'this': 110, 'i': 95, 'to': 93, 'in': 77, 'film': 66, 'it': 65, 'was': 58, 'movie': 53, 'that': 49, 'for': 38, 'but': 38, 'with': 36, 'as': 35, 'are': 34, 'one': 31, 'great': 29, 'good': 28, "it's": 28, 'by': 25, 'an': 24, 'you': 23, 'it.': 23, 'all': 22, 'on': 21, 'so': 20, 'from': 19, 'at': 19, 'about': 18, 'just': 18, 'out': 18, 'if': 18, 'who': 17, 'his': 17, 'wonderful': 17, 'my': 17, 'very': 17, 'really': 16, 'see': 16, 'like': 16, 'not': 16, 'most': 16, 'be': 16, 'think': 16, 'best': 16, 'more': 15}
```

```python
# Displaying the most common negative words
print(wordbag_ncommon)
```

```
{'the': 203, 'a': 116, 'is': 115, 'of': 108, 'and': 107, 'this': 104, 'to': 79, 'it': 76, 'was': 76, 'i': 75, 'movie': 61, 'in': 61, 'that': 60, 'but': 42, 'film': 37, 'not': 36, 'bad': 34, 'just': 34, 'with': 34, 'for': 33, 'as': 32, 'on': 31, 'even': 27, 'you': 25, 'acting': 25, 'so': 24, 'be': 24, 'one': 24, 'very': 23, 'at': 23, 'all': 23, 'there': 23, 'or': 21, 'only': 20, "it's": 20, 'no': 20, 'are': 20, 'have': 20, 'like': 19, 'how': 18, 'an': 17, 'by': 17, 'if': 16, "didn't": 16, 'plot': 16, 'really': 16, 'about': 16, 'were': 16, 'story': 15, 'from': 15}
```

We've displayed the 50 most common words that appear in the positive and negative text entries, respectively.  By observation, many of the words are common and overly used words like "the", "and", "a", etc.  These words will have no weight on how they classify a text entry.  In the next section, we've carefully created sub-dictionaries from our full training set for the 10 most useful words (we believe) that will help us identify a positive or negative text entry.  Then, we'll create additional dictionaries which contain the training set's probablities of containing those words given a positive or negative text entry.  These probabilities will be the basis for the classifier we'll use on the testing set.     

```python
# Calculating number of words that appear in the positive/negative texts. 
num_words_pos = len(wordbag_pos)
num_words_neg = len(wordbag_neg)

# Creating a sub-dictionary for top 10 positive/negative words
top10_pos = {'good', 'great', 'film', 'a', 'best', 'wonderful','love','excellent', 'film', 'see'}
top10_neg = {'not','bad','even', 'no', 'just', 'waste', 'don\'t', 'only', 'worst', 'didn\'t'}

wordbag_pimpt = {i:wordbag_pos[i] for i in top10_pos if i in wordbag_pos}
wordbag_nimpt = {i:wordbag_neg[i] for i in top10_neg if i in wordbag_neg}

# Conditional Probabilities of top 10 common (most useful) words in positive/negative texts
# Recall that len(train) = 800
# P(+) = P(-) = 400/800 = 1/2 = 0.5
n = 400
pdict_common = dict(); ndict_common = dict(); 
pdict = dict(); ndict = dict()

for wordp in wordbag_pimpt:
    pdict_common[wordp] = wordbag_pimpt.get(wordp.lower())/n
for wordn in wordbag_nimpt:
    ndict_common[wordn] = wordbag_nimpt.get(wordn.lower())/n
    
# Conditional Probabilities of all words in positive/negative texts
for wordp in wordbag_pos:
    pdict[wordp] = wordbag_pos.get(wordp)/n
for wordn in wordbag_neg:
    ndict[wordn] = wordbag_neg.get(wordn)/n
```

Before we test our classifier, let's make a quick note on how we'll classify a text entry.  Recall that 

$$
\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{P(B)}, \ \ \ \ \ \  P(B) \neq 0.\
\end{equation}
$$

Let $A = \textbf{positive (or negative) text entry}$ and $B = B*j = \textbf{j}^{th} \ \textbf{text entry} = \cap*{i=1}^{k*j} B*{ji}$, where $B_{ji} = \text{i}^{th} \ \textbf{word in the j}^{th} \ \textbf{text entry}$ and $k_j = \textbf{number of words in text entry j}$.  Along with our naive bayes assumption, the above equation (in terms of a positive text entry) can be written as 

$$
\begin{equation}
P(+|B*j) = \frac{P(B_j|+)P(+)}{P(B_j)} = \frac{P(\cap*{i=1}^{k*j} B*{ji}|+)P(+)}{P(\cap*{i=1}^{k_j} B*{ji})} = \frac{P(+) \prod*{j=1}^{k_j} P(B*{ji}|+)}{P(B_j)}.  \ \ \ \ \ \\
\end{equation}
$$

Following in this manner, the conditional probability for a negative text entry can be written as 

$$
\begin{equation}
P(-|B*j) = \frac{P(-) \prod*{j=1}^{k*j} P(B*{ji}|-)}{P(B_j)}.  \ \ \ \ \ \\
\end{equation}
$$

The two above equations, which represent their appropriate conditional probabilities, have the same denominator.  For computional purposes, we'll ignore the denominators and compare their numertors.  Our classifier will work, in a general way, as follows:

* If $ \left( P(-) \prod*{j=1}^{k_j} P(B*{ji}|-) \right) < \left( P(+) \prod*{j=1}^{k_j} P(B*{ji}|+) \right )$

  * Classify text entry as Positive (1)
* Otherwise

  * Classifiy text entry as Negative (0)

Lastly, the values $P(+) = P(-) = 1/2$.  Therefore, we'll simply compare values $\prod*{j=1}^{k_j} P(B*{ji}|-)$ and $\prod*{j=1}^{k_j} P(B*{ji}|+)$

Now it's time to take our probabilities and check the testing set

```python
# NEED TO FIX SOMEHOW
test_classifier = np.zeros(len(test),dtype = 'f')
p_neg = 1; p_pos = 1; p_smooth = 1/(len(test)/2+1)

for i in range(0,len(test)):
    for word in test.Text_Entries.iloc[i].split():
        if word.lower() in wordbag_pimpt:
            p_pos = pdict_common[word.lower()]*p_pos
        if word.lower() in wordbag_nimpt:
            p_neg = ndict_common[word.lower()]*p_neg
        if word.lower() not in wordbag_pimpt and word.lower() not in wordbag_nimpt:
            p_pos = p_smooth*p_pos
            p_neg = p_smooth*p_neg
    if (p_neg < p_pos):
        test_classifier[i] = 1
        p_neg = 1; p_pos = 1
    else:
        test_classifier[i] = 0
        
# Checking the accuracy
testSent = np.array(test.Sentiment)
acc = (1-sum(abs(testSent - test_classifier)/len(test)))*100

print("The prediction accuracy on the test set was", "{:.2f}".format(acc), '%')
```

```
The prediction accuracy on the test set was 50.00 %
```

As we see by our classifier, we made a $50%$ prediction accuracy on our test set.  The naive-bayes classifier worked half of the time, which is not ideal.  

### <u> Discussion and Conclusion</u>

For future experiments, we would like to run some potential Monte Carlo simulations, where we randomize any 10 words for our positive or negative sub dictionaries, and predict on the test set.  The words we chose, initially, for our sub-dictionaries are suboptimal.  There must exist a collection of words for each sub-dictionary which optimizes the naive-bayes classifier.  In conclusion, this blog post was an introduction into the Naive-Bayes classification scheme for a text entry sentiment analysis problem.  There's much to explore in this post.  We've only begun to scratch the surface of optimal performance.
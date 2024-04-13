# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Our goal with this experiment is to create a sophisticated model that utilizes Bidirectional Recurrent Neural Networks within an LSTM framework. We're focusing on training this model to accurately identify named entities within text data. Our dataset comprises numerous sentences, each containing multiple words with associated tags. Through the use of Embedding techniques, we transform these words into vectors, facilitating the training process. Bidirectional Recurrent Neural Networks enable us to connect two hidden layers in opposite directions to the same output. This architecture allows the output layer to gather information from both past and future states concurrently, enhancing the model's predictive capabilities.

![image](https://github.com/S-Priyadharshan/named-entity-recognition/assets/145854138/ad6f0a46-f327-4633-9c3d-ee6187d46f77)

## DESIGN STEPS

### STEP 1:
Import the necessary packages.

### STEP 2:
Read the dataset, and fill the null values using forward fill.

### STEP 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### STEP 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.

### STEP 5:
We do this by padding the sequences,This is done to acheive the same length of input data

### STEP 6:
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

### STEP 7:
We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis,A custom prediction is done to test the model manually.

## PROGRAM

```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data=pd.read_csv("ner_dataset.csv",encoding="latin1")

data.head(50)

data=data.fillna(method="ffill")

words=list(data["Word"].unique())
words.append("ENDPAD")
tags=list(data["Tag"].unique())

num_words=len(words)
num_tag=len(tags)

class SentenceGetter(object):
  def __init__(self,data):
    self.n_sent=1
    self.data=data
    self.empty=False
    agg_func=lambda s:[(w,p,t) for w, p, t in zip(s["Word"].values.tolist(),
                                                  s["POS"].values.tolist(),
                                                  s["Tag"].values.tolist())]
    self.grouped=self.data.groupby("Sentence #").apply(agg_func)
    self.sentences=[s for s in self.grouped]
  def get_next(self):
    try:
      s=self.grouped["Senatence: {}".format(self.n_sent)]
      self.n_sent+=1
      return s
    except:
      return none

getter=SentenceGetter(data)
sentences=getter.sentences

word2Idx={w: i+1 for i,w in enumerate(words)}
tag2Idx={t: i for i,t in enumerate(tags)}

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

x1=[[word2Idx[w[0]] for w in s] for s in sentences]
max_len=50

X=sequence.pad_sequences(maxlen=max_len,
                         sequences=x1,padding="post",
                         value=num_words-1)

y1=[[tag2Idx[w[2]] for w in s]for s in sentences]

y=sequence.pad_sequences(maxlen=max_len,
                          sequences=y1,
                          padding="post",
                          value=tag2Idx["O"])
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

input_word=layers.Input(shape=(max_len,))

embedding_layer=layers.Embedding(input_dim=num_words,
                                 output_dim=50,
                                 input_length=max_len
)(input_word)

dropout_layer=layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm=layers.Bidirectional(
    layers.LSTM(units=100,
                return_sequences=True,
                recurrent_dropout=0.1)
)(dropout_layer)

output=layers.TimeDistributed(
    layers.Dense(num_tag,activation="softmax")
)(bidirectional_lstm)

model=Model(input_word,output)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history=model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test,y_test),
    batch_size=32,
    epochs=3,
)

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

i = 20
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
print("Priyadharshan S")
print("212223240127")
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/S-Priyadharshan/named-entity-recognition/assets/145854138/e39a035c-1c7b-45a6-b7e9-005897db222a)

![image](https://github.com/S-Priyadharshan/named-entity-recognition/assets/145854138/68e42ed8-9ead-496f-bc8c-e333fedb1f12)


### Sample Text Prediction

![image](https://github.com/S-Priyadharshan/named-entity-recognition/assets/145854138/2ec8a8eb-95c9-491f-acf5-f4384a1fa716)


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text is developed.



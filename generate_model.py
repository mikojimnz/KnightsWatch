import datetime
import json
import nltk
import numpy
import random
import tensorflow
import tflearn
import pickle

from nltk.stem.lancaster import LancasterStemmer

modelName = f'Iteration-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = LancasterStemmer()

with open('training/intents.json') as file:
    data = json.load(file)

with open("settings.json") as jsonFile1:
    cfg = json.load(jsonFile1)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

with open("model/data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir=f'./logs/{modelName}') if (cfg['model']['enable_log']) else tflearn.DNN(net, tensorboard_verbose=3)

model.fit(training, output, n_epoch=cfg['model']['n_epoch'], batch_size=cfg['model']['batch_size'], show_metric=True)
model.save("model/model.tflearn")

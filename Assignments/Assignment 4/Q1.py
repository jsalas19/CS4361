'''
Name: Joshua Salas
Student ID: 80644497
Student Email: jsalas19@miners.utep.edu
'''

from collections import Counter
import numpy as np

'''
1.  Our 'X' is the given input sentences and 'Y' is sentiment labels associated with them
'''

print("\n1.   Our 'X' is the given input sentences and 'Y' is sentiment labels associated with them. \n")

'''
2.  Below we define our vocabulary/dictionary
'''
sentences = [
    ["This restaurant is trash.","negative"],
    ["Taco Bell serves better Mexican food than this.","negative"],
    ["The food is fabulous.","positive"],
    ["Good food, looong wait for the food.","neutral"]
]

clean_sentences = [''.join(char for char in sentence[0] if char.isalnum() or char.isspace()).lower() for sentence in sentences]

stopwords = set(['the', 'is', 'for', 'this', 'and'])
words = [word for sentence in clean_sentences for word in sentence.split() if word not in stopwords]

word_counts = Counter(words)
vocabulary = sorted(word_counts, key=word_counts.get, reverse=True)

special_tokens = ["<PAD>", "<START>", "<END>"]
vocabulary = special_tokens + vocabulary

print("2.   Dictionary:")
print(vocabulary)
print()

'''
3.  Adding numerical values via index in dictionary
'''
word_to_index = {word: index for index, word in enumerate(vocabulary)}
print("3.   Num_values:")
print(word_to_index)
print()


numerical_sentences = []
for sentence in clean_sentences:
    numerical_sentence = [word_to_index[word] for word in sentence.split() if word in word_to_index]
    numerical_sentences.append(numerical_sentence)


'''
4.  Setting length of longest sentence to max sentence length and making all sentences the same length.
'''
max_length = max(len(sentence) for sentence in numerical_sentences)
print("4.   Max sentence length: ", max_length)




padded_sentences = []
for sentence in numerical_sentences:
    sentence = [word_to_index["<START>"]] + sentence + [word_to_index["<END>"]]
    padded_sentence = sentence + [word_to_index["<PAD>"]] * (max_length - len(sentence) + 2)
    padded_sentences.append(padded_sentence)


print("Resulting Dataset: " , (padded_sentences))
print()



#### Defining functions we'll use later
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_z = np.exp(x - np.max(x))  
    return e_z / e_z.sum(axis=0)
#####



'''
6.  Dimensions of Wxh, Whh, and Why
'''
print("6.  Dimensions of Wxh is 16x4, Whh is 4x4, and Why is 4x3 \n")
Wxh = np.ones((16, 4))  
Whh = np.ones((4, 4))
Why = np.ones((4, 3))   
h0 = np.array([0, 0, 0, 0])


'''
5.  One-hot encoding for each word
'''
print("5.  One-hot encoding for each word")
for word_index in word_to_index:
    x = np.zeros((16,)) 
    x[word_to_index[word_index]] = 1
    h = relu(np.dot(x, Wxh) + np.dot(h0, Whh))
    print(x)
print()

y = np.dot(h, Why)


'''
7.  Application of softmax on example case y = [1.3,0.8,-0.1]
'''
x5 = [1.3,0.8,-0.1]
print("7.   Example softmax = ", softmax(x5))
print()

'''
8.  Applying softmax to y to get y_pred
'''
y_pred = softmax(y)
print("8.  Applying softmax to y to get y_pred: ", y_pred, "\n")

#example
y_true = np.array([1, 0, 0])

'''
9. Computing the cross-entropy loss.
'''
loss = -np.sum(y_true * np.log(y_pred))


print("9.   Cross-entropy loss (loss):", loss, "\n")





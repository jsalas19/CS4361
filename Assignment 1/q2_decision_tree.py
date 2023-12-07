import pandas as pd
import math

# To Do: Calculate Information Entropy Loss
def entropy(labels):
    s = 0
    for label in labels:
        curr_node = labels[label]
        total = sum(curr_node)
        s -= sum([((p/total) * math.log(p/total)) for p in curr_node])
    return s

# To Do: Calculate Misclassification Loss
def misclassification(labels):
    # your implementation here
    return

# To Do: Calculate Gini Impurity Loss
def gini(labels):
    s = 0
    for label in labels:
        curr_node = labels[label]
        total = sum(curr_node)
        s -= sum([- math.pow(p/total, 2) for p in curr_node])
    return 1 - s

# To Do: Calculate gain = starting_loss - calculated_loss
def calculate_gain(starting_labels, split_labels):
    return starting_labels + split_labels

## To Do: Fill split(dataset, feature) function
## Should split the dataset on a feature
def split(dataset, column):
    d = {}
    for i,val in enumerate(dataset[column]):
        if val not in d:
            d[val] = [0,0]
        if dataset["poisonous"][i] == 'e':
            d[val][0] += 1
        else:
            d[val][1] += 1
    return d

def initial_split(dataset, init_label):
    d = {init_label: [0,0]}
    for i in dataset[init_label]:
        if i == 'e':
            d[init_label][0] += 1
        else:
            d[init_label][1] += 1
    return d

## To Do: Fill find_best_split(dataset, label) function
## Should find the best feature to split the dataset on
## Should return best_feature, best_gain
def find_best_split(dataset, label):
    # for each split calculate gain, return best feature to split and the gain
    init_split = initial_split(dataset, label)
    print(init_split)
    initial_entropy = entropy(init_split)
    initial_gini = gini(init_split)

    best_feature, best_gain = "", 1

    for feat in dataset:
        feat_entropy = entropy(split(dataset, feat))
        gain = calculate_gain(initial_entropy, feat_entropy)
        if gain < best_gain:
            best_feature = feat
            best_gain = gain

    return best_feature, best_gain



if __name__ == '__main__':
    # read file
    data = pd.read_csv("C:/Users/Joshua Salas/Downloads/lab1_dataset/lab1_dataset/agaricus-lepiota.csv")
    #print(data)
    
    # learn best feature
    best_feature, best_gain = find_best_split(dataset=data, label="poisonous")

    # write to output file
    f = open("output_agaricus.txt", "w")
    f.write("The Best Feature is {} with a Gain of : {}".format(best_feature, best_gain))
    f.close()

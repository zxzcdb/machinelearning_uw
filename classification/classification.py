import graphlab
import math
import string

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

def get_classification_accuracy(model, data, true_labels):
    prediction = model.predict(data)

    correct_count = 0
    for i in range(len(data)):
        if prediction[i] == true_labels[i]:
            correct_count = correct_count + 1

    accuracy = float(correct_count) / float(len(data))
    return accuracy

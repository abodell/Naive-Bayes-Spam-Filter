import numpy as np
from sklearn.metrics import confusion_matrix

def cleantext(text):
    text = text.lower()
    text = text.strip()
    for letters in text:
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters, " ")
    return text 

def countwords(words, is_spam, counted):
    for each_word in words:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1]=counted[each_word][1] + 1
            else:
                counted[each_word][0]=counted[each_word][0] + 1
        else:
            if is_spam == 1:
                counted[each_word] = [0,1]
            else:
                counted[each_word] = [1,0]
    return counted


def make_percent_list(k, theCount, spams, hams):
    for each_key in theCount:
        theCount[each_key][0] = (theCount[each_key][0] + k)/(2*k+hams)
        theCount[each_key][1] = (theCount[each_key][1] + k)/(2*k+spams)
    return theCount

def NaiveBayes(testWords, percentHam, percentSpam, vocab):
    ham_prob = 0
    spam_prob = 0
    for key in vocab:
        if key in testWords:
            ham_prob += np.log(vocab[key][0])
            spam_prob += np.log(vocab[key][1])
        else:
            ham_prob += np.log(1 - vocab[key][0])
            spam_prob += np.log(1 - vocab[key][1])
    ham_prob = np.exp(ham_prob)
    spam_prob = np.exp(spam_prob)
    totalProb = 1 / (1 + np.exp(np.log(ham_prob * percentHam) - np.log(spam_prob * percentSpam)))
    if totalProb < .5:
        return 0
    return 1


def main():
    spam = 0
    ham = 0
    counted = dict()
    # get all of the stop words into a list 
    file = input("Enter the name of the spam-ham file: ")
    fin = open(file, 'r', encoding = 'unicode-escape')
    stopFile = input("Enter the name of the stop words file: ")
    stopWords = open(stopFile).read().split()

    textline = fin.readline()
    while textline != "":
        is_spam = int(textline[:1])
        if is_spam == 1:
            spam += 1
        else:
            ham += 1
        textline = cleantext(textline[1:])
        words = textline.split()
        # remove stop words from the file
        for word in words:
            if word in stopWords:
                words.remove(word)
        words = set(words)
        counted = countwords(words, is_spam, counted)
        textline = fin.readline()
    vocab = (make_percent_list(1, counted, spam, ham))
    fin.close()

    percentHam = (ham) / (ham + spam)
    percentSpam = (spam) / (ham + spam)

    # now we have to read in the test file
    testFile = input("Enter the name of the labeled test file: ")
    test = open(testFile, 'r', encoding = 'unicode-escape')
    testSpam = 0
    testHam = 0
    actualClassification = []
    predictedClassification = []

    testLine = test.readline()
    while testLine != "":
        is_spam = int(testLine[:1])
        actualClassification.append(is_spam)
        if is_spam == 1:
            testSpam += 1
        else:
            testHam += 1
        testLine = cleantext(testLine[1:])
        # contains the current subject line
        testWords = testLine.split()
        for word in testWords:
            if word in stopWords:
                testWords.remove(word)
        # Bayes calculation happens here 
        # change it to a set just to remove duplicates in case
        testWords = set(testWords)
        predictedClassification.append(NaiveBayes(testWords, percentHam, percentSpam, vocab))
        testLine = test.readline()
    test.close()
    print('Number of Ham emails in the test file: ', testHam)
    print('Number of Spam emails in the test file: ', testSpam)
    # calculate the confusion matrix
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predictedClassification)):
        if predictedClassification[i] == 1 and actualClassification[i] == 1:
            TP += 1
        elif predictedClassification[i] == 0 and actualClassification[i] == 0:
            TN += 1
        elif predictedClassification[i] == 1 and actualClassification[i] == 0:
            FP += 1
        else:
            FN += 1
    
    print(confusion_matrix(actualClassification, predictedClassification))
    
    print("True Positive: ", TP)
    print("True Negative: ", TN)
    print("False Positive: ", FP)
    print("False Negative: ", FN)

    accuracy_score = (TP + TN) / (TP + TN + FP + FN)
    precision_score = TP / (TP + FP)
    recall_score = TP / (TP + FN)
    f1_score = 2 * (1 / ((1/precision_score) + (1/recall_score)))
    print("Accuracy: ", accuracy_score)
    print("Precision: ", precision_score)
    print("Recall: ", recall_score)
    print("F1 Score: ", f1_score)


        
if __name__ == '__main__':
    main()


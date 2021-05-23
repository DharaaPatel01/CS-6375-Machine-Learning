"""
Created on Wed Mar 17

@author: Dhara Patel
"""
import sys
import os 
import re
import codecs
import math
import collections

train_path = "./train"
test_path = "./test"

# if(len(sys.argv) == 3):
#     train_path = sys.argv[1]
#     test_path = sys.argv[2]    
# else:
#    sys.exit("Please give right number of arguments- path for 'train' folder containing both spam and ham folder> \
                                                    # path for 'train' folder containing both spam and ham folder>")                                       
spam_path = train_path + '/spam'
ham_path = train_path + '/ham'

#Find out all the words in ham folder and spam folder and find there counts  
count_ham, count_spam = 0, 0
voc_spam = []
voc_ham = []
stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are",
              "aren't","as","at","be","because","been","before","being","below","between",
              "both","but","by","can't","cannot","could","couldn't","did","didn't","do","does",
              "doesn't","doing","don't","down","during","each","few","for","from","further",
              "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's",
              "her","here","here's","hers","herself","him","himself","his","how","how's","i",
              "i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
              "let's","me","more","most","mustn't","my","myself","no","nor","not","of","off",
              "on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
              "same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
              "such","than","that","that's","the","their","theirs","them","themselves","then",
              "there","there's","these","they","they'd","they'll","they're","they've","this",
              "those","through","to","too","under","until","up","very","was","wasn't","we","we'd",
              "we'll","we're","we've","were","weren't","what","what's","when","when's","where",
              "where's","which","while","who","who's","whom","why","why's","with","won't","would",
              "wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself",
              "yourselves"]

def ReadFile(file,filepath):
    fileHandler = codecs.open(filepath + "/" + file,'rU','latin-1')
    WordFinder = re.findall('[A-Za-z0-9\']+', fileHandler.read())
    allwords = list()
    for word in WordFinder:
        word = word.lower()
        allwords+=[word]
    fileHandler.close()    
    return allwords

def GetListsAndNumOfFiles(filepath):
    wordList = list()
    NumberOfFiles = 0
    for files in os.listdir(filepath):
        if files.endswith(".txt"):
            wordList += ReadFile(files,filepath)
            NumberOfFiles+=1
    return wordList, NumberOfFiles

# Vocab and its count for Spam, Ham and all files together
voc_spam,count_spam = GetListsAndNumOfFiles(spam_path)
voc_ham,count_ham = GetListsAndNumOfFiles(ham_path) 
vocab = voc_ham + voc_spam

# Distinct words and its count for Spam, Ham and all files
SpamDict = dict(collections.Counter(w.lower() for w in voc_spam))
HamDict = dict(collections.Counter(w.lower() for w in voc_ham))
vocabDict = collections.Counter(vocab)

def RemoveStopWords():
    for word in stop_words:
        if word in HamDict:
            del HamDict[word]
        if word in SpamDict:
            del SpamDict[word]
        if word in vocabDict:
            del vocabDict[word]               
# RemoveStopWords()
                    
def AddMissingWords(AllWords,HamSpamWords):
    for words in AllWords:
        if words not in HamSpamWords:
            HamSpamWords[words] = 0
            
#getting missing words in each Ham and Spam list and adding them and intializing their count= 0
AddMissingWords(vocabDict,SpamDict)
AddMissingWords(vocabDict,HamDict)
RemoveStopWords()

# Function to find out P(spam)
#(the number of spam documents / total number of documents)
def FindPSpam():
    p_spam = count_spam/(count_spam + count_ham)
    return p_spam
        
# Function to find out P(ham)
#(the number of ham documents / total number of documents)
def FindPHam():
    p_ham = count_ham/(count_spam + count_ham)
    return p_ham

# P(word1|ham) = (count of word1 in ham + 1)/
#              (total number of words in ham + number of distinct words in training database)
# P(word1|spam) = (count of word1 in spam + 1)/
#                 (total number of words in spam + number of distinct words in training database)  
# distinctCount = Words in Spam/Ham + Missing Words

            
prob_word_spam = dict()
def FindProbWordInSpam():
    distinctCount = 0
    for word in SpamDict:
        distinctCount += (SpamDict[word] + 1)
    for word in SpamDict:
        prob_word_spam[word] = math.log((SpamDict[word] + 1)/distinctCount,2) 
       
prob_word_ham = dict()
def FindProbWordInHam():
    distinctCount = 0                  
    for word in HamDict:
        distinctCount += (HamDict[word] + 1)
    for word in HamDict:
        prob_word_ham[word] = math.log((HamDict[word] + 1)/distinctCount,2)
        
# Caluculating probability for each word in spam and ham folders 
FindProbWordInSpam() 
FindProbWordInHam()

# Function to classify the files as spam or ham    
def PredictSpamOrHam(file_path, classifier):
    P_spam = 0 
    P_ham = 0 
    InCorrectlyClassified = 0
    NumberOfFiles = 0
    
    for fileName in os.listdir(file_path):
        words = ReadFile(fileName,file_path)
        
        # find actual P(spam) and P(ham)
        # log(P(spam|text)) = log(P(spam)) + log(P(word1|spam)) + log(P(word2|spam)) + .... 
        P_spam = math.log(FindPSpam(),2)
        P_ham = math.log(FindPHam(),2)
        
        for word in words:
            if word in prob_word_spam:
                P_spam += prob_word_spam[word]
            if word in prob_word_ham:
                P_ham += prob_word_ham[word]
        NumberOfFiles +=1
        
        if(classifier == "spam" and P_ham >= P_spam):
                InCorrectlyClassified+=1
        if(classifier == "ham" and P_ham <= P_spam):
                InCorrectlyClassified+=1
    return InCorrectlyClassified,NumberOfFiles 
                
spam_testPath = test_path + '/spam'        
ham_testPath = test_path + '/ham'

IC_Spam,TotalSpamFiles = PredictSpamOrHam(spam_testPath, "spam")
IC_Ham,TotalHamFiles = PredictSpamOrHam(ham_testPath, "ham")
TotalIncorrect = IC_Ham + IC_Spam

Accuracy_Spam = ((TotalSpamFiles -  IC_Spam )/(TotalSpamFiles))*100
Accuracy_Ham = ((TotalHamFiles - IC_Ham)/(TotalHamFiles))*100

AllClassifiedFiles = TotalHamFiles + TotalSpamFiles
Accuracy = ((AllClassifiedFiles  - TotalIncorrect)/AllClassifiedFiles)*100

print("Executing Naive Bayes after removing stop words")

print("\nTotal number of files: ", AllClassifiedFiles)

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("\nCalculating Accuracy over Spam Files")
print("Total number of Spam Files: ", TotalSpamFiles)
print("Number of Files Classified as Spam: ", TotalSpamFiles - IC_Spam)
print("Number of Files Classified as Ham: ",IC_Spam)
print("\nNaive Bayes Accuracy For SPAM Files Classification:",
      (TotalSpamFiles -  IC_Spam ),"/",(TotalSpamFiles),"=",round(Accuracy_Spam,2),"%")

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("\nCalculating Accuracy over Ham Files")
print("Total number of Ham Files: ", TotalHamFiles)
print("Number of Files Classified as Ham: ", TotalHamFiles - IC_Ham)
print("Number of Files Classified as Spam: ",IC_Ham)
print("\nNaive Bayes Accuracy For HAM Files Classification:",
      (TotalHamFiles - IC_Ham),"/",TotalHamFiles,"=",round(Accuracy_Ham,2),"%") 

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("\nNaive Bayes Accuracy for TEST Files Classification:",
      (AllClassifiedFiles  - TotalIncorrect),"/",AllClassifiedFiles,"=",round(Accuracy,2),"%")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

print("\n")
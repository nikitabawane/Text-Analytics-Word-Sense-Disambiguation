# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:26:15 2020

@author: nikit
"""
import pandas as pd
import nltk
import string
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  
from string import digits 
import numpy as np
import operator

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

### Baseline Model ###
#rename columns and ignore the index column
def rename_columns(dataset):
    dataset_new = dataset.rename(columns = {0:"Target_Word", 1:"Sense_ID", 2:"Sentence"})
    dataset_new = dataset_new.reset_index(drop=True)
    return dataset_new


def clean(sentence):
    remove_these = string.punctuation.replace('%', '')
    sentence=sentence.lower()
    sentence=sentence.translate(str.maketrans('', '', remove_these ))
    remove_digits = str.maketrans('', '', digits) 
    sentence = sentence.translate(remove_digits)
    return sentence


def retreive_pos_wordnet(sentence):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    list_words = sentence
    final_list = []
    for i in range (len(list_words)):
        tag = nltk.pos_tag(list_words)[i][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        final_tag = tag_dict.get(tag, wordnet.NOUN)
        lemmatized_word = lemmatizer.lemmatize(list_words[i],final_tag)
        final_list.append([final_tag,lemmatized_word])
    return final_list


df_train= pd.read_csv (r'C:\Users\nikit\Desktop\IDS_566\HW2\train.data',header=None,delimiter = "|")
#uncomment when combining train dataset with validation for test prediction
#df_train=df_train.reset_index()
df_test= pd.read_csv (r'C:\Users\nikit\Desktop\IDS_566\HW2\test.data',header=None,delimiter = "|")
df_test=df_test.reset_index()
df_validation= pd.read_csv (r'C:\Users\nikit\Desktop\IDS_566\HW2\validate.data',header=None,delimiter = "|")
df_validation=df_validation.reset_index()

#rename columns of the table
df_train = rename_columns(df_train)
df_test= rename_columns(df_test)
df_validation= rename_columns(df_validation)

#renaming the index column to unique ids
#uncomment when combining train dataset with validation for test prediction
#df_train= df_train.rename(columns = {'index':"UniqueIDs"})
df_test= df_test.rename(columns = {'index':"UniqueIDs"})
df_validation= df_validation.rename(columns = {'index':"UniqueIDs"})

#remove trailing spaces in the target word columns
df_train['Target_Word']=df_train.Target_Word.str.replace(' ', '') 
df_test['Target_Word']=df_test.Target_Word.str.replace(' ', '') 
df_validation['Target_Word']=df_validation.Target_Word.str.replace(' ', '')

#computing baseline accuracy on 
df_baseline=df_validation

#retrieve list of unique target words
UniqueTargetWords=df_baseline['Target_Word'].value_counts()
UniqueTargetWords=pd.DataFrame(UniqueTargetWords)
UniqueTargetWords=UniqueTargetWords.reset_index()
UniqueTargetWords.columns=['Words','Count']
targets=UniqueTargetWords.Words.tolist()

#creating a table to store predictions
Predictions=pd.DataFrame(columns=['Sense_Id_Predicted','Sentence','ActualSenseId','UniqueIDs'])
cntr=0

#compute model for eac target word
for target_word in targets:
    print("Processed ", cntr,"Target words out of ", len(targets))    
    cntr=cntr+1
                
    test_instances=df_baseline.loc[df_baseline['Target_Word']==target_word]
    test_instances=test_instances.reset_index()
                    
    ids=test_instances['UniqueIDs'].tolist()
    actualsenses=test_instances['Sense_ID'].tolist()
                    
    #Training_dict,PriorProb=CreateMyTrainData(df_train,target_word)
    train_instances=df_train.loc[df_train['Target_Word']==target_word]

    senses=pd.DataFrame(train_instances.Sense_ID.value_counts())
    senses=senses.reset_index()

    #assign mode of the sense_id as a prediction of the target word
    predicted=senses[senses.Sense_ID == senses.Sense_ID.max()]['index'].tolist()[0]

    k=0
    for sent in test_instances['Sentence']:
        to_append = [predicted, sent,actualsenses[k],ids[k]]
        k=k+1
        df_length = len(Predictions)
        Predictions.loc[df_length] = to_append

Final_Predictions=Predictions

Final_Predictions['Acc']=Final_Predictions.Sense_Id_Predicted == Final_Predictions.ActualSenseId
accuracy=Final_Predictions['Acc'].value_counts()/len(Final_Predictions['Acc'])
print("Baseline accuracy is %s ",str(accuracy[1]))
#accuracy of baseline model is 81%

####################### Supervised WSD ###################################

#set window size
N=1

#######process of test instance ################
#In this function, we select features based on the window size specified above.
def Process(test_instance):
    #test_instance=Sentence
    test_instance=clean(test_instance)
    listofwords=test_instance.split()

    listofwords=retreive_pos_wordnet(listofwords)
    
    stop = stopwords.words('english')
    listofwords=[x for x in listofwords if x[1] not in stop]   

    index=[x[1] for x in listofwords].index('%%')
    if index-N>0:
        listofwords=listofwords[index-N:index+N+3]
    else:
        listofwords=listofwords[:index+N+1]

    index=[x[1] for x in listofwords].index('%%')
    del listofwords[index:index+3]

    return listofwords

#####################################################################

#This function is used to generate a model for each target word.
#Prior probabilites and feature probabilities
def CreateMyTrainData(df_train,target):
    Filtered=df_train.loc[df_train['Target_Word']==target_word]
    Filtered=Filtered.reset_index()
    
    Filtered['Processed'] = Filtered.apply(lambda x : Process(x['Sentence']),axis=1)
    Filtered['CountofFV']= Filtered.apply(lambda x : len(x['Processed']),axis=1)    

    ###### Calculating prior Probabilities ###############
    Senses_df=pd.DataFrame(Filtered.Sense_ID.value_counts())
    Senses_df['PriorProb']=Senses_df['Sense_ID']/sum(Senses_df['Sense_ID'])        
    list1=list(Senses_df.index)
    list2=list(Senses_df.PriorProb)
    PriorProb=dict(zip(list1,list2))

    ######## count of words in a sense ############
    Prob=pd.DataFrame(Filtered.groupby(by='Sense_ID').sum())
    Prob=Prob[['CountofFV']]
    Prob=Prob.reset_index()

    list1=list(Prob.Sense_ID)
    list2=list(Prob.CountofFV)
    CountofWordsInaSense=dict(zip(list1,list2))

    #######################################################
    #bring dict1 to the format pos-word-sense_id:count
    train_corpus=Filtered['Processed'].tolist()
    train_corpus[0]
    Biglist=list(zip(Filtered['Sense_ID'].tolist(),train_corpus))
    dict1={}
    for temp in Biglist:
        for j in temp[1]:
            key=j[0]+'-'+j[1]+'-'+str(temp[0])
            if key not in dict1:
                dict1[key]=1
            else:
                dict1[key]=dict1[key]+1

    #####################################################
    
    df=pd.DataFrame(columns=['Sense_ID','Word','Numerator','Denominator']) 
    for k in dict1.keys():
            Sense_Id=k.split('-')[-1]
            Word=k.split('-')[:2]
            Word="-".join(Word)

            to_append = [Sense_Id, Word,dict1[k],CountofWordsInaSense[int(Sense_Id)]]
            df_length = len(df)
            df.loc[df_length] = to_append

    ######################################################################    
    #Set vocubulary as distinct count of words 
    V=len(dict1.keys())
    #uncomment this line for lambda =1
    #lambda1 = 1
    #comment this line for lambda =1
    lambda1 = 0.01
    df['SmoothedCompute']=(df['Numerator']+lambda1)/(df['Denominator']+(V*lambda1))
    df['Word_Sense_ID']=df['Word']+'-'+df['Sense_ID']
    Training_dict=dict(zip(df['Word_Sense_ID'],df['SmoothedCompute']))

    return Training_dict,PriorProb

#########################################################################

#Uncomment this line to run on validation dataset
df_test=df_validation
#uncomment to combine training and validation, to run the model on test dataset
#df_train =  df_train.append(df_validation)

UniqueTargetWords=df_test['Target_Word'].value_counts()
UniqueTargetWords=pd.DataFrame(UniqueTargetWords)
UniqueTargetWords=UniqueTargetWords.reset_index()
UniqueTargetWords.columns=['Words','Count']
##############################0 to 100
targets=UniqueTargetWords.Words.tolist()
Predictions=pd.DataFrame(columns=['Sense_Id_Predicted','Sentence','Target_word','UniqueIDs'])
###########################
cntr=0
for target_word in targets:
        print("Processed ", cntr,"Target words out of ", len(targets))    
        cntr=cntr+1
        test_instances=df_test.loc[df_test['Target_Word']==target_word]
        test_instances=test_instances.reset_index()
        #Run model for every target word
        Training_dict,PriorProb=CreateMyTrainData(df_train,target_word)

        ids=test_instances['UniqueIDs'].tolist()
        k=0
        for sent in test_instances['Sentence']:
            ##############################
            eachsentence=sent
            eachsentence=Process(eachsentence)
            ##############################
            results={}
            UniquesenseIds=PriorProb.keys()

            for SenseId in UniquesenseIds:
                    listofvals=[]
                    listoffv=[]
                    for fv in eachsentence:
                                ask=fv[0]+'-'+fv[1]+'-'+str(SenseId)
                                if ask in Training_dict:
                                    listofvals.append(Training_dict[ask])
                                    listoffv.append(ask)

                    results[SenseId]=PriorProb[SenseId]*np.prod(listofvals)


            MaxVal_SenseId=max(results.items(), key=operator.itemgetter(1))[0]
            to_append = [MaxVal_SenseId, sent,target_word,ids[k]]
            k=k+1
            df_length = len(Predictions)
            Predictions.loc[df_length] = to_append

#Uncomment these lines for validation
#Final_Predictions['Acc']=Final_Predictions.Sense_ID == Final_Predictions.Sense_Id_Predicted
#accuracy=Final_Predictions['Acc'].value_counts()/len(Final_Predictions['Acc'])

Final_Predictions.to_csv(r'C:\Users\nikit\Desktop\IDS_566\HW2\Result\test_prediction_with trainvalidate.csv')

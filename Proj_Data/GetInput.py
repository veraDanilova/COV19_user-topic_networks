#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""

This class opens datasets for the corresponding week

"""

import pandas as pd
import re
import string

from gensim.models.wrappers import LdaMallet


def DFvalueToList(value):
    sent = re.sub(r' list', ', list', value)
    sent = re.sub('\n', '', sent)
    sent = eval(sent)
    return sent

def makeMask(start_date, end_date, dataset, column):
    mask = (dataset[column] > start_date) & (dataset[column] <= end_date)
    return mask

class OpenPreprocess:
    """
    
    create an instance of OpenPreprocess
    
    set parameters:
    
    network name net: LJ  or Twi, str
    week number wnum: w*, str
    root folder name, str
    
    output: week-specific LDA, Account Data
    
    """
    wnums = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']
    
    def __init__(self, net, wnum, root):
        
        self.net = net
        self.wnum = wnum
        self.root = root
    
    def prepareData(self): #LJ, Twi
        
        week_list = [
        ("2020-03-22","2020-03-29"), 
        ("2020-03-29","2020-04-05"),
        ("2020-04-05","2020-04-12"),
        ("2020-04-12","2020-04-19"),
        ("2020-04-19","2020-04-26"),
        ("2020-04-26","2020-05-03"),
        ("2020-05-03","2020-05-10"),
        ("2020-05-10","2020-05-17"),
        ("2020-05-17","2020-05-24"),
        ("2020-05-24","2020-06-01")
        ]
        
        if self.net == 'Twi':
            
            dataset = pd.read_csv(self.root+"COR_TWI_resized.csv", engine = 'python', index_col = 0)
            dataset['unigramsC'] = dataset['unigramsC'].apply(lambda x: DFvalueToList(x))
            
            masklist = [makeMask(w[0], w[1], dataset, "time") for w in week_list]

            AN = pd.read_csv(self.root+"Account_Names_Clu_Twi.csv", index_col = 0)
            AN = AN.rename(columns={"cluster_vals": "cluster_11"})
            
            week_clus = [['none',2],['none'],['none',3],['none',6],['none',1],['none',4],['none',7],['none',5],['none',8],['none',9]]
            
            topic_names = []
            
            topic_names = pd.read_csv(self.root+'TopicCorNames_Twi_manual.csv', index_col = 0)
            topic_names.columns = ['tnos']+self.wnums
            """w2v topic names"""
            # topic_names = pd.read_csv(self.root+'TopicCorNames_TWIGen.csv', index_col = 0)
            # topic_names.columns = ['tnos']+self.wnums
            
            model_load = 'model_twi_'

        elif self.net == 'LJ':
          
            dataset = pd.read_csv(self.root+'COR_LJ_resized.csv', engine = 'python')
            dataset['unigramsC'] = dataset['unigramsC'].apply(lambda x: DFvalueToList(x))
            
            masklist = [makeMask(w[0], w[1], dataset, "date") for w in week_list]

            AN = pd.read_csv(self.root+"Account_Names_id.csv", index_col = 0)
            
            week_clus = [['none',7],['none',8],['none',10],['none',0],['none',1],['none',5],['none',6],['none',3],['none',4],['none',9]]
            

            topic_names = []

            topic_names = pd.read_csv(self.root+'TopicCorNames_LJ_manual.csv', index_col = 0)
            topic_names.columns = ['tnos']+self.wnums
            """w2v topic names"""
#             topic_names = pd.read_csv(self.root+'TopicCorNames_LJGen.csv', index_col = 0)
#             topic_names.columns = ['tnos']+self.wnums
            
            model_load = 'model_'
            
        return dataset, AN, model_load, masklist, topic_names, week_clus
    
    def getWeekDataLDA(self, dataset, model_load, masklist):
        
        #create topic column
        model = LdaMallet.load(self.root+model_load+self.wnum)
        
        if self.net == 'LJ':
            model.prefix = self.root+'ldaLJ/'+self.net+self.wnum
        else:
            model.prefix = self.root+'ldaTwi/'+self.net+self.wnum

        data = dataset.loc[masklist[self.wnums.index(self.wnum)]] 
        topics_M = []
        for M in model.load_document_topics():
            topics_M.append([m[1] for m in M].index(max([m[1] for m in M])))
        data["topic"] = topics_M
        
        #set w2v topic names (manually created set of names)
#         data["topic"] = data["topic"].apply(lambda tn: topic_names[self.wnum][tn])
        return data, model
    
    def getAllInputData(self):
        dataset, AN, model_load, masklist, topic_names, week_clus = self.prepareData()
        datasetWeek, modelWeek = self.getWeekDataLDA(dataset, model_load, masklist)
        return dataset, datasetWeek, modelWeek, AN, model_load, masklist




#!/usr/bin/env python
# coding: utf-8

# In[29]:


#!/usr/bin/env python
# coding: utf-8



"""
BIDIRECTED GRAPH, UNIFIED TOPIC TABLE

"""

# general
import re
import math
import string
import gzip
import os
import itertools
import sklearn.preprocessing

# data
import pandas as pd
import json
import gensim
import gensim.corpora as corpora


# LDA
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
import pyLDAvis

# Plotting tools
import matplotlib.pyplot as plt
# %matplotlib inline
import networkx as nx
import plotly.io as pio
pio.renderers.default='notebook'
import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
init_notebook_mode(connected=True)

from GetInput import OpenPreprocess
# from RelCount import TermRelevance

# def DFvalueToList(value):
#     sent = re.sub(r' list', ', list', value)
#     sent = re.sub('\n', '', sent)
#     sent = eval(sent)
#     return sent


class GraphType2D:
    """
    net (network) should be either LJ  or Twi
    """
    wnums = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']
    
    def __init__(self, net, wnum, cl_id, root):
        
        self.net = net
        self.wnum = wnum
        self.cl_id = cl_id
        self.root = root
        
        self.week_clus = []
        self.masklist = []
        self.model_load = ''
        self.AN = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.topic_names = []
        
        self.ByTopicByDisp = pd.DataFrame()
        self.fname = ''
        
        self.accounts = []
        self.N_posts_by_topic = []
        self.N_posts_by_topic_index = []
        self.N_posts_week = []
        self.perc_posts = []
        self.medpostsperuser_inT = []
    
    def prepareGraphdata(self): #LJ, Twi  
        
        op = OpenPreprocess(self.net, '', self.root)
        self.dataset, self.AN, self.model_load, self.masklist, self.topic_names, self.week_clus = op.prepareData()
        
        return 
    
    def getWeekDataLDA(self):
        
        self.prepareGraphdata()
        
        #get week's data and model
        op = OpenPreprocess(self.net, self.wnum, self.root)
        
#         print(len(self.dataset),self.dataset.shape,self.model_load,self.masklist, self.wnum)
        data, _ = op.getWeekDataLDA(self.dataset, self.model_load, self.masklist)
#         print(len(data),data.shape)
#         assign topic_names
        data["topic"] = data["topic"].apply(lambda tn: self.topic_names[self.wnum][tn])
        
        # account stats
        T_Acc_Counts = data.groupby(["acc_name", "topic"]).size().reset_index(name="Time")
        # Time is the number of texts an author wrote about a topic
        T_Acc_Counts_S = T_Acc_Counts.sort_values(['Time', 'acc_name'], ascending = (False, True))

        self.accounts = T_Acc_Counts_S.groupby('acc_name')['Time'].sum().index.tolist()#list of account names
        # print(len(self.accounts))
        per_user_posts = T_Acc_Counts_S.groupby('acc_name')['Time'].sum().tolist()#total number of posts per user
        self.N_posts_by_topic = T_Acc_Counts_S.groupby('topic')['Time'].sum().values #total number of posts per topic
        self.N_posts_by_topic_index = T_Acc_Counts_S.groupby('topic')['Time'].sum().index.values.tolist()
        self.N_posts_week = sum(self.N_posts_by_topic)#total number of posts per week
        #  % постов по теме в неделю от общего числа постов за неделю 
        # self.perc_posts = [round(int(b) / self.N_posts_week * 100,2) for b in self.N_posts_by_topic]
        #+ 2) среднее число постов на пользователя по этой теме
        self.medpostsperuser_inT = [round(b/m) for b,m in zip(self.N_posts_by_topic, T_Acc_Counts_S.groupby(["topic"])['acc_name'].size().values.tolist())]
#         group n-topic accounts without relevance
        Dispersion = pd.DataFrame()
        Dispersion['accs'] = T_Acc_Counts_S['acc_name'].value_counts().index.tolist()
        Dispersion['values'] = T_Acc_Counts_S['acc_name'].value_counts().tolist()

        def set_dispersion(acc):
            for i in Dispersion[Dispersion['accs'] == acc]['values'].values.tolist():
                return i
        T_Acc_Counts_S['Dispersion'] = T_Acc_Counts_S['acc_name'].apply(lambda x: set_dispersion(x))
        
        #here sum() does nothing, because they are already grouped by topic
        self.ByTopicByDisp = T_Acc_Counts_S.groupby(['acc_name','topic','Dispersion'])['Time'].sum().reset_index(name = 'Size')
        
        return T_Acc_Counts_S
    
    def getWeekClusterData(self):
        
        self.getWeekDataLDA()
        
        ByTopicByDisp = self.ByTopicByDisp
        
        week_id = self.wnums.index(self.wnum)
        week_cluster = self.week_clus[week_id]
        
        
        if self.cl_id == 1:
            # print("cluster 1")
            Claccs = self.AN[self.AN['cluster_11'] == week_cluster[self.cl_id]][self.AN[self.wnums[week_id]] >= 1]['names'].values.tolist()
            # print(len(Claccs))
            self.ByTopicByDisp = ByTopicByDisp[ByTopicByDisp['acc_name'].isin(Claccs)]
            self.fname = f"NW_{self.net}_{self.wnum}_clu_{week_cluster[self.cl_id]}_TopicGroups.html"
        else:
            if len(week_cluster) > 1:
                # print("cluster 0")
                Claccs = self.AN[self.AN['cluster_11'] != week_cluster[1]][self.AN[self.wnums[week_id]] >= 1]['names'].values.tolist()
                # print(len(Claccs))
                self.ByTopicByDisp = ByTopicByDisp[ByTopicByDisp['acc_name'].isin(Claccs)]
                self.fname = f"NW_{self.net}_{self.wnum}_clu_none_TopicGroups.html"
            else:     
                Claccs = self.AN[self.AN[self.wnums[week_id]] >= 1]['names'].values.tolist()
                # print(len(Claccs))
#                 self.ByTopicByDisp = ByTopicByDisp[ByTopicByDisp['acc_name'].isin(Claccs)]
                self.fname = f"NW_{self.net}_{self.wnum}_clu_none_TopicGroups.html"
        
        return
    

    def getGraphDataDF(self):
        
        self.getWeekClusterData()

        ByTopicByDisp = self.ByTopicByDisp

        D = {}
        
        for a,b in ByTopicByDisp.groupby('acc_name')['topic', 'Dispersion']:
            # dict entry: per-user list of topics, list of number of texts per topic: [0,1] [10,2]
            D[a] = [str(b['topic'].values.tolist()), b['Size'].values.tolist()]
            #     print(D[a], '\n new \n')

        df = pd.DataFrame.from_dict(D, orient = 'index', columns = ['topics','ntexts'])
        df['users'] = df.index
        df['dispersion'] = df['ntexts'].apply(lambda tps: len(tps))
        # dataframe user - topic - ntexts
        
        listGraph = []

        for a,b in df.groupby('dispersion'):
            #     print(b[])#,'\n next \n',b,'\n new \n')
            tlists = b['topics'].apply(lambda tlist: eval(tlist)).values
            numTlists = b['ntexts'].values
            users = b['users'].values
            #             print(len(b))
            for items in zip(users, tlists, numTlists):
                for t, num in zip(items[1], items[2]):
                    listGraph.append((a, t, num, len(b), items[0]))

        GraphData_v1 = pd.DataFrame()
        GraphData_v1 = pd.DataFrame.from_records(listGraph, columns = ['dispersion','topic','n_texts','n_accs','acc_name'])

        return GraphData_v1


    def GraphType2(self):
        
        GraphData = self.getGraphDataDF()
        
        accounts = self.accounts

        """"""
        # calculate percentage of texts per topic in a given week and cluster
        self.N_posts_by_topic = self.ByTopicByDisp.groupby(['topic'])['Size'].sum().values.tolist()
        self.N_posts_by_topic_index = self.ByTopicByDisp.groupby(['topic'])['Size'].sum().index.tolist()
        #  % постов по теме в неделю от общего числа постов за неделю 
        self.perc_posts = [round(int(b) / self.N_posts_week * 100,2) for b in self.N_posts_by_topic]
        """"""

        N_posts_week = self.N_posts_week
        
        GraphDataNew  = GraphData.groupby(['dispersion','topic'])['n_texts'].sum().reset_index(name = 'sumTexts')
        
        #n_accs_topic
        GraphDataNew['n_accs_topic'] = GraphData.groupby(['dispersion','topic'])['acc_name'].size().values.tolist() #reset_index(name = 'n_accs_topic')

        def calc_text_ratio(df):
            #             NoPostsWeek = N_posts_by_topic[N_posts_by_topic_index.index(df['topic'])]
            #             ratio = df['sumTexts'] / NoPostsWeek # num of posts by topic by week
            ratio = round((df['sumTexts'] / N_posts_week) * 100, 1) #ratio to the total number of posts per week
            return ratio

        GraphDataNew['textratio'] = GraphDataNew.apply(lambda df: calc_text_ratio(df), axis = 1)

        def calc_acc_ratio(df):
            AccsPerDisp = GraphData[GraphData['dispersion'] == df['dispersion']]['n_accs'].values[0]
            NoAccsWeek = len(accounts)
#             print(NoAccsWeek, AccsPerDisp)

            return AccsPerDisp, round((AccsPerDisp / NoAccsWeek) * 100,1) #ratio to the total number of accounts per week

        GraphDataNew['DispAccratio'] = GraphDataNew.apply(lambda df: calc_acc_ratio(df)[1], axis = 1)
        GraphDataNew['n_accs_disp'] = GraphDataNew.apply(lambda df: calc_acc_ratio(df)[0], axis = 1)
        
        """'dispersion','topic','n_texts','n_accs','acc_name'"""
        GraphDataNew['TopicAccratio'] = GraphDataNew['n_accs_topic'].apply(lambda Nacc: round((Nacc / len(accounts)) * 100,1))
        
        return GraphDataNew # dataframe dispersion topic sumTexts (by topic by disp) n_accs_topic textratio DispAccratio TopicAccratio

    
    def plotgraphType2(self):
        
        W_one_max = self.GraphType2()
        
        fname = self.fname
        accounts = self.accounts

        H = nx.DiGraph()

        G = nx.from_pandas_edgelist(W_one_max, 'dispersion', 'topic', edge_attr = True, create_using=H)#edge_attr = 'weight'edge_attr = True

        topic_nodes = [node for node in G if node in W_one_max['topic'].values]

        edges, edge_weights = zip(*nx.get_edge_attributes(G, 'textratio').items())

        pos = nx.bipartite_layout(G, topic_nodes)
        nx.set_node_attributes(G, pos, 'pos')

        cols = ["lightpink",
                "lightsalmon",
                "lightseagreen",
                "lightskyblue",
                "lightslategrey",
                "lightsteelblue",
                "lightblue",
                "lime",
                "limegreen",
                "mediumorchid",
                "mediumpurple",
                "gold",
                "goldenrod"] 


        for u,v,d in G.edges(data = True):
            d['col'] = cols[list(set(topic_nodes)).index(v)]
            d['textratio'] = d['textratio']/2

        
        middle_nodes_list = []
        edge_list = []

        Transparent_x_list = []
        Transparent_y_list = []
        Transparent_hover_list = []

        for e in G.edges(data = True):
            x = [pos[e[0]][0],pos[e[1]][0], None]
            y = [pos[e[0]][1],pos[e[1]][1], None]
            edge_list.append(dict(type = 'scatter', x = x, y = y, mode = 'lines', line = dict(width = e[2]['textratio'], color = e[2]['col'])))
            #transparent nodes
            x0, y0 = G.nodes[e[0]]['pos']
            x1, y1 = G.nodes[e[1]]['pos']

            d = math.sqrt((x1-x0)**2 + (y1 - y0)**2) #distance
            # print(d)
            r = (d*9.3/10)/ d #segment ratio

            Transparent_x = r * x1 + (1 - r) * x0 #find point that divides the segment
            Transparent_y = r * y1 + (1 - r) * y0 #into the ratio (1-r):r

            Transparent_hover = f'Contributing group: {e[0]}-topic users.Ratio of texts to all week texts - {e[2]["textratio"]*2} %, number of texts - {e[2]["sumTexts"]}'
            Transparent_x_list.append(Transparent_x)
            Transparent_y_list.append(Transparent_y)
            Transparent_hover_list.append(Transparent_hover)

            middle_nodes_list.append(dict(type='scatter', x=Transparent_x_list, y=Transparent_y_list,hovertext = Transparent_hover_list, hoverinfo = 'text', mode='markers',marker=go.Marker(
            opacity=0.4, color = 'aliceblue')))


        """TOPIC NODES"""

        topic_node_x = []
        topic_node_y = []
        for node in topic_nodes:
            x, y = G.nodes[node]['pos']
            topic_node_x.append(x)
            topic_node_y.append(y)

        topic_node_labels = [node for node in G.nodes if node in topic_nodes]
        #     print(fname, len(topic_node_labels), set(W_one_max['topic'].values.tolist()), len(set(W_one_max['topic'].values.tolist())))

#         T_counts = W_one_max.groupby('topic')['n_accs'].sum().tolist()
#         T_list = W_one_max.groupby('topic')['n_accs'].sum().index.tolist()
#         topic_node_size = [T_counts[T_list.index(node)]/len(accounts)*100*5 for node in topic_nodes]
#         print(sum(W_one_max[W_one_max['topic'] == 'Countries']['TopicAccratio']))
        topic_node_size = [sum(W_one_max[W_one_max['topic'] == node]['TopicAccratio'])*5 for node in topic_nodes]
        #topic node size is the ratio of accs in this topic in this cluster to the total num of accs

        # =============================================================================
        # [f"<b>{index}) {node}</b>"

        #         description = [f"<b>{node}</b>"
        #                        "<br><br>Per n-topic accounts: " +
        #                        "<br> 1-topic accs:" + str(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == 1]['n_accs'].values[0]) +
        #                        "<br> 2-topic accs:" + str(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == 2]['n_accs'].values[0]) +
        #                        "<br> 3-topic accs:" + str(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == 3]['n_accs'].values[0]) +
        #                        "<br> 4-topic accs:" + str(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == 4]['n_accs'].values[0]) +
        #                        "<br> 5>=topic accs:" + str(sum(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] >= 5]['n_accs'].values))
        #                        for index, node in (topic_node_labels)]
        # dataframe dispersion topic sumTexts (by topic by disp) n_accs_topic textratio DispAccratio TopicAccratio
        description = []

        stats = []

        for node in topic_node_labels:

            stats.append((node, self.N_posts_by_topic[self.N_posts_by_topic_index.index(node)], sum(W_one_max[W_one_max['topic'] == node]['n_accs_topic'])))

            # stats.append((node, self.perc_posts[self.N_posts_by_topic_index.index(node)], round(sum(W_one_max[W_one_max['topic'] == node]['n_accs_topic'])/len(self.accounts)*100,2)))
            dispfrom1to4 = []
            for disp in range(1,5):
                if len(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs_topic'].values) > 0:
                    value = W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs_topic'].values[0]
                    value_perc = W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['TopicAccratio'].values[0]
#                 if len(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values) > 0:
#                     value = W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values[0]
#                     all_active = sum(W_one_max['n_accs'].values)
#                     value_perc = round(value/all_active * 100, 1)
                else:
                    value = 0
                    value_perc = 0
                desc_tp = (value, value_perc)
                dispfrom1to4.append(desc_tp)
            description.append(f"<b>{node}</b>"+
                                "<br> Perc. posts per week: " + str(self.perc_posts[self.N_posts_by_topic_index.index(node)]) + '%' +
                                "<br> Avg posts per account: " + str(self.medpostsperuser_inT[self.N_posts_by_topic_index.index(node)]) +
                               "<br> Ratio No. of topic accs to unique accs: " + str(round(sum(W_one_max[W_one_max['topic'] == node]['n_accs_topic'])/len(self.accounts)*100,2)) + '%' +
                               "<br><br>Perc. n-topic accounts: " +
                               "<br> 1-topic accs: " + str(dispfrom1to4[0][0]) + " , " + str(dispfrom1to4[0][1]) + '%' +
                               "<br> 2-topic accs: " + str(dispfrom1to4[1][0]) + " , " + str(dispfrom1to4[1][1]) + '%' +
                               "<br> 3-topic accs: " + str(dispfrom1to4[2][0]) + " , " + str(dispfrom1to4[2][1]) + '%' +
                               "<br> 4-topic accs: " + str(dispfrom1to4[3][0]) + " , " + str(dispfrom1to4[3][1]) + '%' +
                               "<br> 5>=topic accs: " + str(sum(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] >= 5]['n_accs_topic'].values)) + " , " + 
                               str(round(sum(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] >= 5]['n_accs_topic'].values) / len(accounts) * 100,1))+'%')
            # for disp in range(1,5):
            #     if len(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values) > 0:
            #         value = W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values[0]
            #     else:
            #         value = 0
            #     dispfrom1to4.append(value)
            # description.append(f"<b>{node}</b>"
            #                    "<br><br>Per n-topic accounts: " +
            #                    "<br> 1-topic accs:" + str(dispfrom1to4[0]) +
            #                    "<br> 2-topic accs:" + str(dispfrom1to4[1]) +
            #                    "<br> 3-topic accs:" + str(dispfrom1to4[2]) +
            #                    "<br> 4-topic accs:" + str(dispfrom1to4[3]) +
            #                    "<br> 5>=topic accs:" + str(sum(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] >= 5]['n_accs'].values)))
        # =============================================================================
        topic_node_trace = go.Scatter(x=topic_node_x, y=topic_node_y, mode='markers', hoverinfo='text',hovertext = description, marker=dict(
        size = topic_node_size,
        showscale=True, 
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Rainbow', reversescale=True, color='teal',
        colorbar=dict(
        thickness=15,
        title='',
        xanchor='left',
        titleside='right'), line_width=2))


        """ OTHER NODES """
        other_nodes = list(set(G.nodes()) - set(topic_nodes))
        other_node_x = []
        other_node_y = []
        for node in other_nodes:
            x, y = G.nodes[node]['pos']
            other_node_x.append(x)
            other_node_y.append(y)

        other_node_labels = [node for node in G.nodes if node in other_nodes]
        other_node_size = [W_one_max[W_one_max['dispersion'] == node]['DispAccratio'].values[0]*5 for node in other_nodes]


        # =============================================================================
        # [f"<b>{index}) {node}</b>"
        description_other = [f"<b>{node}-topic accounts:</b>"
        "<br><br> Number of accounts: " + str(W_one_max[W_one_max['dispersion'] == node]['n_accs_disp'].values[0]) +
        "<br><br> Ratio to all accounts: " + str(W_one_max[W_one_max['dispersion'] == node]['DispAccratio'].values[0])+'%'+
        "<br><br> Number of topics covered: " + str(W_one_max[W_one_max['dispersion'] == node]['topic'].nunique())
        for index, node in enumerate(other_node_labels)]
        # =============================================================================
        
        other_node_trace = go.Scatter(x=other_node_x, y=other_node_y, mode='markers', hoverinfo='text', hovertext = description_other,
        marker=dict(size = other_node_size, showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Viridis',
        reversescale=True,
        color='slategray',
        colorbar=dict(
        thickness=15,
        title='',
        xanchor='left',
        titleside='right'),
        line_width=2))


        fig = go.Figure(data=edge_list+middle_nodes_list+[topic_node_trace, other_node_trace],layout=go.Layout(
        autosize=False,
        width=1500,
        height=1000,
        paper_bgcolor="LightSteelBlue",
        plot_bgcolor = "darkslategray",
        title='<br>',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=5,l=5,r=5,t=40),
        annotations=[ dict(
        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
        # py.offline.plot(fig, filename = fname, auto_open=False)
        return fig, stats


#     def runBuildGraph(self):
#         fig = self.plotgraphType2()
#         dataset, AN, week_clus, topic_names, model_load, masklist = self.prepareGraphdata()
#         T_Acc_Counts, T_Acc_Counts_S, accounts, per_user_posts, N_posts_by_topic, N_posts_by_topic_index, N_posts_week, ByTopicByDisp = self.getWeekDataLDA(dataset, topic_names, model_load, masklist)
#         dataByTopicByDisp, fname = self.getWeekClusterData(week_clus, AN, ByTopicByDisp)
#         GraphData = self.getGraphDataDF(dataByTopicByDisp)
#         GraphDataType2 = self.GraphType2(GraphData, N_posts_by_topic, N_posts_by_topic_index, accounts, N_posts_week)
#         fig = self.plotgraphType2(GraphDataType2, accounts, fname)
#         return accounts, week_clus, AN, ByTopicByDisp


#         #This produces a dataframe where 'node' is the list of topics (not dispersion)
#         D = {}
#         for a,b in ByTopicByDisp.groupby('acc_name')['topic', 'Dispersion']:
#             # dict entry: per-user list of topics, list of number of texts per topic: [0,1] [10,2]
#             D[a] = [str(b['topic'].values.tolist()), b['Size'].values.tolist()]

#         df = pd.DataFrame.from_dict(D, orient = 'index', columns = ['topics','ntexts'])
#         df['users'] = df.index
#         # dataframe user - topic - ntexts

#         listGraph = []
#         #df.groupby('topics')
#         for a,b in df.groupby('topics'):

#             numTlists = b['ntexts'].values.tolist()

#             tlist = eval(a)
#             numTsums = []
#             if len(numTlists) > 1:
#                 sumT = [sum(x) for x in zip(*numTlists)]
#                 numTsums = sumT
#             else:
#                 for numTlist in numTlists:
#                     numTsums = numTlist
#             for t,s in zip(tlist, numTsums):
#                 #                 print(t)
#                 listGraph.append([a, t, s, len(b), sum(numTsums), len(tlist)])
#         GraphData = pd.DataFrame()
#         GraphData = pd.DataFrame.from_records(listGraph, columns = ['node_id','topic','n_texts','n_accs','texts_per_id', 'dispersion'])
#         #             print(set(GraphData['topic'].values.tolist()))
#         return GraphData


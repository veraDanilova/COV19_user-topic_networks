#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[122]:


"""
BIDIRECTED GRAPH, UNIFIED TOPIC TABLE

"""

# general
import re
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
    def prepareGraphdata(self): #LJ, Twi
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
        def DFvalueToList(value):
            sent = re.sub(r' list', ', list', value)
            sent = re.sub('\n', '', sent)
            sent = eval(sent)
            return sent
      
        if self.net == 'Twi':
            
            dataset = pd.read_csv(self.root+"COR_TWI_preprocessed_below.csv", engine = 'python', index_col = 0)
            dataset['unigramsC'] = dataset['unigramsC'].apply(lambda x: DFvalueToList(x))
            
            def makeMask(start_date, end_date):
                mask = (dataset['time'] > start_date) & (dataset['time'] <= end_date)
                return mask
            masklist = [makeMask(w[0],w[1]) for w in week_list]

            AN = pd.read_csv(self.root+"Account_Names_Clu_Twi.csv", index_col = 0)
            AN = AN.rename(columns={"cluster_vals": "cluster_11"})
            week_clus = [['none',2],['none'],['none',3],['none',6],['none',1],['none',4],['none',7],['none',5],['none',8],['none',9]]
            topic_names = pd.read_csv(self.root+'TopicCorNames_Twi.csv', index_col = 0)
            topic_names.columns = ['tnos']+self.wnums
            model_load = 'model_twi_'

        elif self.net == 'LJ':
          
            dataset = pd.read_csv(self.root+'data_cutbelowabove.csv', engine = 'python')
            dataset['unigramsC'] = dataset['unigramsC'].apply(lambda x: DFvalueToList(x))

            def makeMask(start_date, end_date):
                mask = (dataset['date'] > start_date) & (dataset['date'] <= end_date)
                return mask
            masklist = [makeMask(w[0],w[1]) for w in week_list]

            AN = pd.read_csv(self.root+"Account_Names_id.csv", index_col = 0)
            week_clus = [['none',7],['none',8],['none',10],['none',0],['none',1],['none',5],['none',6],['none',3],['none',4],['none',9]]
            topic_names = pd.read_csv(self.root+'TopicCorNames_LJ.csv', index_col = 0)
            topic_names.columns = ['tnos']+self.wnums
            model_load = 'model_'
        return dataset, AN, week_clus, topic_names, model_load, masklist
    
    def getWeekDataLDA(self, dataset, topic_names, model_load, masklist):
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

        # account stats
        T_Acc_Counts = data.groupby(["acc_name", "topic"]).size().reset_index(name="Time")
        T_Acc_Counts_S = T_Acc_Counts.sort_values(['Time', 'acc_name'], ascending = (False, True))

        accounts = T_Acc_Counts_S.groupby('acc_name')['Time'].sum().index.tolist()#list of account names
        per_user_posts = T_Acc_Counts_S.groupby('acc_name')['Time'].sum().tolist()#total number of posts per user
        N_posts_by_topic = T_Acc_Counts_S.groupby('topic')['Time'].sum().values #total number of posts per topic
        N_posts_by_topic_index = T_Acc_Counts_S.groupby('topic')['Time'].sum().index.values.tolist()
        N_posts_week = sum(N_posts_by_topic)#total number of posts per week

        # group n-topic accounts
        Dispersion = pd.DataFrame()
        Dispersion['accs'] = T_Acc_Counts_S['acc_name'].value_counts().index.tolist()
        Dispersion['values'] = T_Acc_Counts_S['acc_name'].value_counts().tolist()

        def set_dispersion(acc):
            for i in Dispersion[Dispersion['accs'] == acc]['values'].values.tolist():
                return i
        T_Acc_Counts_S['Dispersion'] = T_Acc_Counts_S['acc_name'].apply(lambda x: set_dispersion(x))

        ByTopicByDisp = T_Acc_Counts_S.groupby(['acc_name','topic','Dispersion'])['Time'].sum().reset_index(name = 'Size')

        return T_Acc_Counts, T_Acc_Counts_S, accounts, per_user_posts, N_posts_by_topic, N_posts_by_topic_index, N_posts_week, ByTopicByDisp
    def getWeekClusterData(self, week_clus, AN, ByTopicByDisp):
        week_cluster = week_clus[self.wnums.index(self.wnum)]

        if self.cl_id == 1:
            Claccs = AN[AN['cluster_11'] == week_cluster[self.cl_id]][AN[self.wnums[self.wnums.index(self.wnum)]] > 1]['names'].values.tolist()
            ByTopicByDisp = ByTopicByDisp[ByTopicByDisp['acc_name'].isin(Claccs)]
            fname = f"NW_{self.net}_{self.wnum}_clu_{week_cluster[self.cl_id]}_TopicGroups.html"
        else:
            if len(week_cluster) > 1:
                Claccs = AN[AN['cluster_11'] != week_cluster[1]][AN[self.wnums[self.wnums.index(self.wnum)]] > 1]['names'].values.tolist()
                ByTopicByDisp = ByTopicByDisp[ByTopicByDisp['acc_name'].isin(Claccs)]
                fname = f"NW_{self.net}_{self.wnum}_clu_none_TopicGroups.html"
            else:
                ByTopicByDisp = ByTopicByDisp
                fname = f"NW_{self.net}_{self.wnum}_clu_none_TopicGroups.html"
        return ByTopicByDisp, fname

    def getGraphDataDF(self, ByTopicByDisp):

        D = {}
        for a,b in ByTopicByDisp.groupby('acc_name')['topic', 'Dispersion']:
            # dict entry: per-user list of topics, list of number of texts per topic: [0,1] [10,2]
            D[a] = [str(b['topic'].values.tolist()), b['Size'].values.tolist()]

        df = pd.DataFrame.from_dict(D, orient = 'index', columns = ['topics','ntexts'])
        df['users'] = df.index
        # dataframe user - topic - ntexts

        listGraph = []
        #df.groupby('topics')
        for a,b in df.groupby('topics'):

            numTlists = b['ntexts'].values.tolist()

            tlist = eval(a)
            numTsums = []
            if len(numTlists) > 1:
                sumT = [sum(x) for x in zip(*numTlists)]
                numTsums = sumT
            else:
                for numTlist in numTlists:
                    numTsums = numTlist
            for t,s in zip(tlist, numTsums):
                #                 print(t)
                listGraph.append([a, t, s, len(b), sum(numTsums), len(tlist)])
        GraphData = pd.DataFrame()
        GraphData = pd.DataFrame.from_records(listGraph, columns = ['node_id','topic','n_texts','n_accs','texts_per_id', 'dispersion'])
        #             print(set(GraphData['topic'].values.tolist()))
        return GraphData

    def GraphType2(self, GraphData, N_posts_by_topic, N_posts_by_topic_index, accounts, N_posts_week):
        GraphData['dispersion'] = GraphData['node_id'].apply(lambda node: len(eval(node)))

        GraphDataNew = GraphData.groupby(['dispersion','topic'])['n_texts'].sum().reset_index(name = 'sumTexts')

        #     print(GraphDataNew.head(20))

        def calc_text_ratio(df):
            #             NoPostsWeek = N_posts_by_topic[N_posts_by_topic_index.index(df['topic'])]
            #             ratio = df['sumTexts'] / NoPostsWeek # num of posts by topic by week
            ratio = round((df['sumTexts'] / N_posts_week) * 100, 1) #ratio to the total number of posts per week
            return ratio

        GraphDataNew['textratio'] = GraphDataNew.apply(lambda df: calc_text_ratio(df), axis = 1)

        SumAccs = pd.DataFrame()
        SumAccs = GraphData.groupby(['dispersion','topic'])['n_accs'].sum().tolist()
        GraphDataNew['n_accs'] = SumAccs

        def calc_acc_ratio(df):
            AccsPerDisp = sum(GraphDataNew[GraphDataNew['dispersion'] == df['dispersion']]['n_accs'].values)
            NoAccsWeek = len(accounts)

            return round((AccsPerDisp / NoAccsWeek) * 100,1) #ratio to the total number of accounts per week

        GraphDataNew['DispAccratio'] = GraphDataNew.apply(lambda df: calc_acc_ratio(df), axis = 1)

        return GraphDataNew # dataframe dispersion  topic  sumTexts  textratio n_accs accratio


    def nameTopics(self, GraphDataType2, topic_names):
        GraphDataType2['topic'] = GraphDataType2['topic'].apply(lambda tname: f"All: {topic_names[self.wnum][topic_names['tnos'] == tname].values[0]}")
        return GraphDataType2
    
    def plotgraphType2(self, GraphDataType2, accounts, fname):
        #     print(len(set(GraphDataType2['topic'].values.tolist())))
        W_one_max = GraphDataType2
#         print(W_one_max.head())

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

        import math
        edge_list = [dict(type = 'scatter', x=[pos[e[0]][0],pos[e[1]][0], None], y = [pos[e[0]][1],pos[e[1]][1], None],
                          mode = 'lines', hovertext='algo', hoverinfo='name', marker=dict(color='rgb(125,125,125)', size=1),
                          line = dict(width = e[2]['textratio'], color = e[2]['col'])) for e in G.edges(data = True)]    



        middle_nodes_list = []
        edge_list = []

        Transparent_x_list = []
        Transparent_y_list = []
        Transparent_hover_list = []

        for e in G.edges(data = True):
            x = [pos[e[0]][0],pos[e[1]][0], None]
            y = [pos[e[0]][1],pos[e[1]][1], None]
            edge_list.append(dict(type = 'scatter', 
                                  x = x, y = y,
                                  mode = 'lines',
                                  line = dict(width = e[2]['textratio'], color = e[2]['col'])))
			#transparent nodes

			x0, y0 = G.nodes[e[0]]['pos']
			x1, y1 = G.nodes[e[1]]['pos']

			d = math.sqrt((x1-x0)**2 + (y1 - y0)**2) #distance
			r = (d*9.3/10)/ d #segment ratio

			Transparent_x = r * x1 + (1 - r) * x0 #find point that divides the segment
			Transparent_y = r * y1 + (1 - r) * y0 #into the ratio (1-r):r

			Transparent_hover = f'Ratio of texts to all week texts - {e[2]["textratio"]*2} %, number of texts - {e[2]["sumTexts"]}'
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

        T_counts = W_one_max.groupby('topic')['n_accs'].sum().tolist()
        T_list = W_one_max.groupby('topic')['n_accs'].sum().index.tolist()
        topic_node_size = [T_counts[T_list.index(node)]/len(accounts)*100*5 for node in topic_nodes]
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
        description = []
        for node in topic_node_labels:
            dispfrom1to4 = []
            for disp in range(1,5):
                if len(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values) > 0:
                    value = W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] == disp]['n_accs'].values[0]
                else:
                    value = 0
                dispfrom1to4.append(value)
            description.append(f"<b>{node}</b>"
                               "<br><br>Per n-topic accounts: " +
                               "<br> 1-topic accs:" + str(dispfrom1to4[0]) +
                               "<br> 2-topic accs:" + str(dispfrom1to4[1]) +
                               "<br> 3-topic accs:" + str(dispfrom1to4[2]) +
                               "<br> 4-topic accs:" + str(dispfrom1to4[3]) +
                               "<br> 5>=topic accs:" + str(sum(W_one_max[W_one_max['topic'] == node][W_one_max['dispersion'] >= 5]['n_accs'].values)))
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
        "<br><br> Number of accounts: " + str(sum(W_one_max[W_one_max['dispersion'] == node]['n_accs'].values)) +
        "<br><br> Ratio to all accounts: " + str(W_one_max[W_one_max['dispersion'] == node]['DispAccratio'].values[0])+'%'
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
        #                             plot_bgcolor = "aliceblue",
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
        py.offline.plot(fig, filename = fname, auto_open=False)
        return fig


    def runBuildGraph(self):
        dataset, AN, week_clus, topic_names, model_load, masklist = self.prepareGraphdata()
        T_Acc_Counts, T_Acc_Counts_S, accounts, per_user_posts, N_posts_by_topic, N_posts_by_topic_index, N_posts_week, ByTopicByDisp = self.getWeekDataLDA(dataset, topic_names, model_load, masklist)
        dataByTopicByDisp, fname = self.getWeekClusterData(week_clus, AN, ByTopicByDisp)
        GraphData = self.getGraphDataDF(dataByTopicByDisp)
        GraphDataType2 = self.GraphType2(GraphData, N_posts_by_topic, N_posts_by_topic_index, accounts, N_posts_week)
        GraphDataType2 = self.nameTopics(GraphDataType2, topic_names)
        print('done', GraphDataType2)
        
        return self.plotgraphType2(GraphDataType2, accounts, fname)


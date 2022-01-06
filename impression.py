from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('bert-base-nli-mean-tokens')
import pandas as pd
from collections import defaultdict
import re
import numpy as np
import sys
from sklearn.cluster import KMeans
import numpy
import nltk
from bert_serving.client import BertClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# bc = BertClient(check_length=False)
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from collections import defaultdict
import re
import numpy as np
import sys
from sklearn.cluster import KMeans
import numpy
import nltk
from bert_serving.client import BertClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# bc = BertClient(check_length=False)
from sklearn.cluster import KMeans
import collections
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from numpy import dot
from numpy.linalg import norm
from sklearn import metrics
def read_df_rel(based_dir, file_input_name):
    file_input = based_dir + file_input_name    
    ff = open(file_input)
    delim=","
    df = pd.read_csv(file_input,delimiter=delim,header=0)        
    return df
def is_any_entities_present(sent, entity_list):
    for ent in entity_list:
        if ent.lower() in nltk.word_tokenize(sent.lower()):
            return True
    return False
def getheadWord(s):
    res=str(s).split('{')
    if len(res)==1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]
def findRelevantSentences(char):
    res=[]
    tokens=nltk.word_tokenize(char)
    for i,row in df.iterrows():
        if is_any_entities_present(row['text'], tokens):
            res.append(i)
    return res
def findAllrels(s,d,g_ext,refs):
    res=set()
    s_m=refs[s]
    d_m=refs[d]
    if not s_m or not d_m:
        return []
    for s_0 in s_m:
        for d_0 in d_m:
            res_0=g_ext.get_edge_data(s_0,d_0)
            if res_0:
                for a in res_0:
                    res.add(res_0[a]['label'])
    return res

def is_any_entities_present(sent, entity_list):
    for ent in entity_list:
        if ent.lower() in nltk.word_tokenize(sent.lower()):
            return ent
    return None

def elbow_plot(data, maxK=10, seed_centroids=None,ShoulPlot=True):
    if len(data)<3:
        return 0
    sse = {}
    maxK=min(maxK,len(data)-1)
    for k in range(1, maxK):
        if ShoulPlot:
            pass
#             print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
        sse[k] = kmeans.inertia_
    if ShoulPlot:
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.show()
    y=list(sse.values())
    x1 = range(1, len(y)+1)
    from kneed import KneeLocator
    if len(y)<3:
        return 0
    kn = KneeLocator(x1,y , curve='convex', direction='decreasing')
    if not kn.knee:
        return 0
    return int(kn.knee)

def findBests(edges,grTruth):
    if not edges:
        return
    rel_truth_embed=bc.encode([grTruth])[0]
    X=bc.encode(edges)
    ind=0
    if len(X)>0:
        scores=[]
        for i in range(len(X)):
            a=X[i]
            b=rel_truth_embed
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            scores.append(cos_sim)
        ind=scores.index(max(scores)) 
    num_class=elbow_plot(X, maxK=10, seed_centroids=None,ShoulPlot=False)
    if num_class==0:
        return edges,edges[ind]
    km = KMeans(n_clusters=min(num_class,len(edges)-1)).fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    res=[]
    for i in closest:
        res.append(edges[i])
    return res,edges[ind]
def findBests2(edges,grTruth):
    if not edges:
        return
    rel_truth_embed=bc.encode([grTruth])[0]
    X=bc.encode(edges)
    num_class=elbow_plot(X, maxK=10, seed_centroids=None,ShoulPlot=False)
    km = KMeans(n_clusters=min(num_class,len(edges)-1)).fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    res=[]
    for i in closest:
        res.append(edges[i])
    return res
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
def findEntites(arg,ent_groups,ent_id):
    arg=arg.replace('}',' ').replace('{',' ')
    res=[]
    for ent in ent_id:
        if ent in nltk.word_tokenize(arg.lower()):
            res.append(ent_id[ent])
        elif " "+ent in arg.lower():
            res.append(ent_id[ent])
    return list(set(res))

def findEdges(i,j,d):
    edges=[]
    edges_ids=[]
    for res in d:
        if i in res['arg1_id'] and j in res['arg2_id']:
            edges.append(res['rel'].replace('}','').replace('{',''))
    return edges
def findEdgesID(i,j,d):
    edges_ids=[]
    for res in d:
        if i in res['arg1_id'] and j in res['arg2_id']:
            edges_ids.append(res['row_number'])
    return edges_ids

def findBests3(edges):
    if not edges:
        return
    X=bc.encode(edges)
    ind=0
    num_class=elbow_plot(X, maxK=10, seed_centroids=None,ShoulPlot=False)
    if num_class==0:
        return edges,edges[ind]
    if num_class<len(edges)/10:
        num_class=min(int(len(edges)/10),num_class+5)
    km = KMeans(n_clusters=min(num_class,len(edges)-1)).fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    res=[]
    for i in closest:
        res.append(edges[i])
    return res
def findBests3(edges):
    if not edges:
        return
    X=embedder.encode(edges)
    ind=0
    num_class=elbow_plot(X, maxK=20, seed_centroids=None,ShoulPlot=False)
    if num_class==0:
        return np.zeros(len(edges))
    if num_class<len(edges)/10:
        num_class=min(int(len(edges)/10),num_class+5)
    km = KMeans(n_clusters=min(num_class,len(edges)-1)).fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    res=[]
    for i in closest:
        res.append(edges[i])
        
    return km.labels_
def which_entities_present(arg, entity_list,isAppos=False,sent=""):
    if 'novel' in sent or 'book' in sent or 'classis' in sent or 'read' in sent:
        return set()
    res=set()
    if len(arg)>50:
        return res
    if isAppos and len(nltk.word_tokenize(arg.lower()))>5:
#         print(arg)
        return res
        
    for c in nltk.word_tokenize(arg.lower()):
        if c.lower() in entity_list:
            res.add(c)
            if isAppos and sent:
                if str(", "+c+",") in sent.lower() or str(" and "+c) in sent.lower() or sent.count(',')>2:
                    res.remove(c)
    
    return res
import hdbscan
from sklearn.cluster import DBSCAN
def findBests4(edges):
    if not edges or len(edges)==1:
        return edges
    num=int(min(5,max(2,len(edges)/2)))
#     clusterer = hdbscan.HDBSCAN()
    X=embedder.encode(edges)
#     clusterer.fit(X)
    clusterer = DBSCAN(eps=2, min_samples=num).fit(X)
    return clusterer.labels_
import pandas as pd
edges_path="/Users/user/Documents/StoryMiner/goodreads/groundtruth/Tim_gold-standard_summaries-Hobbit_edges.csv"
nodes_path="/Users/user/Documents/StoryMiner/goodreads/groundtruth/Tim_gold-standard_summaries-Hobbit_nodes.csv"
delim = "\n"
p="/Users/user/Downloads/Data_raw_goodreads/hobbit.txt"
rel_path= "/Users/user/Documents/StoryMiner/goodreads/df_extractions_with_ner.csv"
df_edges=pd.read_csv(edges_path)
df_nodes=pd.read_csv(nodes_path)
df = pd.read_csv(p,delimiter=delim,header=0,error_bad_lines=False)
entity_versions=entity_versions_hobbit
book=pd.read_csv("/Users/user/Documents/StoryMiner/hobbit.txt",delimiter=delim,header=0,engine='python')
df_rels=read_df_rel("",rel_path)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(list(df['text']))
from nltk import SnowballStemmer
stemmer = SnowballStemmer('english', ignore_stopwords=False)

class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def __init__(self, stemmer, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = stemmer
        
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))
vectorizer_stem_u = StemmedTfidfVectorizer(stemmer=stemmer, sublinear_tf=True)
X = vectorizer_stem_u.fit_transform(list(df['text']))
word2tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
word2tfidf_hobbit=word2tfidf
import collections
inv={}
for e in entity_versions_hobbit:
    for e1 in entity_versions_hobbit[e]:
        inv[e1]=e
final_terms=list(inv.keys())
all_svcop=collections.defaultdict(dict)
for e in final_terms:
    all_svcop[e]=[]
for i,row in df_rels.iterrows():
    if row['type'][0:4]=='SVCop' or  '{is}' in row['rel'] or '{was}' in row['rel'] or '{be}' in row['rel']:
        tmp1=which_entities_present(str(row['arg1_orig']), final_terms,True,df_rels['sentence'][i]) #you can also run one for and
        tmp2=which_entities_present(str(row['arg2_orig']), final_terms,True,df_rels['sentence'][i])#another idea: if len is 1
        if tmp1:
            for a in tmp1:
                all_svcop[a].append(i)
char_desc={}
for e in entity_versions_hobbit:
    char_desc[e]=[]
    for e1 in entity_versions_hobbit[e]:
        char_desc[e].extend(all_svcop[e1])
res_des={}
res_d={}
for e in char_desc:
    tmp=[]
    for i in char_desc[e]:
        tmp.append(str(df_rels['arg2'][i]).replace('}','').replace('{',''))
    if tmp:
        labels=findBests4(tmp)
    #     print(labels)
        d=defaultdict(list)
        for i in range(len(labels)):
            d[labels[i]].append(tmp[i])
        #res_des[e]=res
        res_d[e]=d

from nltk.corpus import stopwords
from scipy.stats import skew 
import collections

def IsItnoisy(s):
    d=defaultdict(int)
    for w in s:
        ws=w.split(' ')
        for ww in ws:
            if ww not in list(stopwords.words('english')):
                if ww in word2tfidf:
                    d[lemmatizer.lemmatize(stemmer.stem(ww.lower()))]+=word2tfidf[ww] 

    if d and (skew(list(d.values()))>1 or np.mean(list(d.values()))>7 ):
        return True
    return False
res_final_hobbit=defaultdict(list)
for e in res_d:
    for i in res_d[e]:
        if len(res_d[e][i])>2 and i!=-1:
            if  IsItnoisy(res_d[e][i]):
                print(len(res_d[e][i]))
                res_final_hobbit[e].append(res_d[e][i])

from numpy import dot
from numpy.linalg import norm
def calculateScore(s1,s2):
    v1=embedder.encode(s1)
    v2=embedder.encode(s2)
    res=0
    for a in v1:
        for b in v2:
            tmp=dot(a,b)#/(norm(a)*norm(b))
            res+=tmp
    return res/(len(s1)*len(s2))
from numpy import dot
from numpy.linalg import norm
def calculateScore(s1,s2):
    v1=bc.encode(s1)
    v2=bc.encode(s2)
    res=0
    r2=0
    for a in v1:
        aa=set()
        for b in v2:
            tmp=dot(a,b)/(norm(a)*norm(b))
#             res+=tmp
            aa.add(tmp)
        res+=max(aa)
    return res/(len(s1))
from numpy import dot
from numpy.linalg import norm
def calculateScorematch(v1,v2):
    res=0
    r2=0
    for a in v1:
        aa=set()
        for b in v2:
            tmp=dot(a,b)/(norm(a)*norm(b))
            aa.add(tmp)
        res+=max(aa)
            #res+=tmp
    return res/(len(v1))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
import numpy as np
# import umap
pca = PCA(n_components=4)
def plotPCA(s,s2,showO=True):
    labels=[]
    x=[]
    for i in range(len(s)):
        labels.extend([i for j in range(len(s[i]))])
        x.extend(s[i])
    for i in range(len(s2)):
        labels.extend([i+len(s) for j in range(len(s2[i]))])
        x.extend(s2[i])
    v=embedder.encode(x)
    X=pca.fit(v).transform(v)
    classes=["first-"+str(i) for i in range(len(s))]
    classes.extend(["second-"+str(i) for i in range(len(s2))])
#     print(classes)
    if showO:
        plt.figure(figsize=(12,6))
        scatter=plt.scatter(X[:, 0], X[:, 1],  c = labels)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)

        plt.show()
    tmp=X[:,0:4]
    ss1=[]
    ss2=[]
    i0=0
    t0=[]
    for j0 in range(len(labels)):
        if labels[j0]==i0:
            t0.append(tmp[j0,:])
        else:
            if labels[j0-1]<len(s):
                ss1.append(t0)
                i0+=1
                t0=[]
            else:
                ss2.append(t0)
                i0+=1
                t0=[]
    if t0:
        ss2.append(t0)
    d_c={}
    for i in range(len(ss1)):
        for j in range(len(ss2)):
            d_c[(i,j)]=calculateScorematch(ss1[i],ss2[j])+calculateScorematch(ss2[j],ss1[i])
    return d_c
all_comps={}
for n1 in final_res_all:
    res_1=final_res_all[n1]
    for n2 in final_res_all:
        if (n1,n2) not in all_comps and (n2,n1) not in all_comps:
            res_2=final_res_all[n2]
            d_all={}
            for e in res_1:                
                for e2 in res_2:
                    d_c=plotPCA(res_1[e],res_2[e2],False)
                    d_all[(e,e2)]=d_c
            all_comps[(n1,n2)]=d_all
from nltk.corpus import stopwords
def getLabel(s):
    d=defaultdict(int)
    for w in s:
        ws=w.split(' ')
        for ww in ws:
            if ww not in list(stopwords.words('english')) and len(ww)>1:
                d[ww.lower()]+=1
    vals=sorted(d.values())
    vals=vals[::-1]
    final_label=s[0]
    for w in d:
        if d[w]==vals[0]:
            final_label=w+","
    if len(vals)<2:
        return final_label
    for w in d:
        if d[w]==vals[1]:
            final_label+=w
            return final_label
    return final_label
import pandas as pd
import seaborn as sns
def plotHeatmap(dict_sim_scores,s1=[],s2=[],save=False,n1="one",n2="two"):
    d2={}
    if s1 and s2:
        for (i,j) in dict_sim_scores:
            if 'girl,young' not in  getLabel(s1[i]) and 'girl,young' not in getLabel(s2[j]):
                if 'classic' not in  getLabel(s1[i]) and 'classic' not in getLabel(s2[j]):
                    d2[getLabel(s1[i]),getLabel(s2[j])]=dict_sim_scores[(i,j)]
    else:
        d2=dict_sim_scores
    ser = pd.Series(list(d2.values()),
                      index=pd.MultiIndex.from_tuples(d2.keys()))
    df = ser.unstack().fillna(0)
    df.shape
    plt.figure()
    sns.clustermap(df,cmap='coolwarm',vmin=-2, vmax=2);
    if save:
        plt.savefig("figs/"+n1+'_ '+n2+'_heamap.png', bbox_inches='tight', dpi=500)
    plt.show()
    

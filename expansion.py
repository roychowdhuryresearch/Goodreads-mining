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



def getheadWord(s):
    res=str(s).split('{')
    if len(res)==1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]
#********** INPUT ********** 
edges_path="/Users/user/Documents/StoryMiner/goodreads/groundtruth/Tim_gold-standard_summaries-MiceMen_edges.csv"
nodes_path="/Users/user/Documents/StoryMiner/goodreads/groundtruth/Tim_gold-standard_summaries-MiceMen_nodes.csv"
delim = "\n"
p="/Users/user/Downloads/Data_raw_goodreads/mice.txt"
rel_path= "/Users/user/Downloads/Results_for_resubmission-2/mice_and_men/mice_relatios_1.csv"
df_ent = pd.read_csv("/Users/user/Downloads/Results_for_resubmission-2/mice_and_men/df_ent_final_ranking.csv")
res_final=load_obj("micemen4")['res_final'] #output of previous paper

df_rels=pd.read_csv(rel_path)
df_edges=pd.read_csv(edges_path)
df_nodes=pd.read_csv(nodes_path)
df = pd.read_csv(p,delimiter=delim,header=0,error_bad_lines=False)
entity_versions=entity_versions_micemen


from nltk import SnowballStemmer
from collections import defaultdict
ent_id={}
ent_groups=collections.defaultdict(set)
k=0
for ent in entity_versions:
    ent_id[ent]=k
    ent_groups[k]=set()
    ent_groups[k].add(ent)
    for ent_2 in entity_versions[ent]:
        ent_id[ent_2]=k
        ent_groups[k].add(ent_2)
    k+=1    
for i,row in df_nodes.iterrows():
    name=str(row['character']).lower()
    if name not in ent_id:
        ent_id[name]=k
        ent_groups[k]=set()
        ent_groups[k].add(name)
        k+=1

stemmer = SnowballStemmer('english', ignore_stopwords=False)
import nltk
from nltk.corpus import stopwords
stps=set(stopwords.words('english'))
cnds=defaultdict(int)
versions=defaultdict(set)
for ind,row in df_ent.iterrows():
    c=nltk.word_tokenize(str(row['entity']))
    for cc in c:
        cnds[stemmer.stem(cc)]+=row['frequency_score_sum_NER_arg']
        versions[stemmer.stem(cc)].add(cc)
keys=set()
dups=set()
for ind,row in df_ent.iterrows():
    if len(str(row['entity']))>2:
        if row['frequency_score_sum_NER_arg']>10 and row['type'] in {'PERSON', 'OTHER(ARG)'}:
            c=nltk.word_tokenize(str(row['entity']))
            if len(c)>1:
                for cc in c:
                    if len(cc)>2:
                        if cc in keys:
                            keys.remove(cc)
                            dups.add(cc)
                        elif cc not in keys and cc not in keys:
                            keys.add(cc)
ents=set()
for ent in entity_versions: 
    ents.add(ent)
    for ent_2 in entity_versions[ent]:
        ents.add(ent_2)
for w in versions:
    for ww in versions[w]:
        if ww in ents:
            for ww1 in versions[w]:
                ents.add(ww1)
#                 print(ww1)
            ents.add(w)
for m in list(df_nodes['character']):
    for mm in m.split(' '):
        ents.add(mm.lower())
tmp=list(cnds.values())
tmp.sort()
tmp=tmp[::-1]
seen=set()
candidates=[]
for i in range(len(tmp)):
    for w, score in cnds.items():
        if score == tmp[i] and w not in seen:
            seen.add(w)
            if score>200 and len(w)>2 and w not in ents:
#                 print(w,versions[w],score,i)
                candidates.append(w) 
            
to_csv_id={}
for m in list(df_nodes['character']):
    for mm in m.split(' '):
        if mm.lower() in ent_id:
            to_csv_id[ent_id[mm.lower()]]=m
        elif m.lower() in ent_id:
            to_csv_id[ent_id[m.lower()]]=m
d_s_candids={}
for w in candidates:
    d2=[]
    for i,row in df_rels.iterrows():
        if is_any_entities_present(row['sentence'].lower(),versions[w]) and len(str(row['arg1']))<50 and len(str(row['arg2']))<50:
            tmp1=findEntites(str(row['arg1']), ent_groups,ent_id)
            tmp2=findEntites(str(row['arg2']), ent_groups,ent_id)
            if tmp1 or tmp2:
                if tmp1!=tmp2:
                    res_tmp={}
                    res_tmp['row_number']=i
                    res_tmp['arg1']=str(row['arg1'])
                    res_tmp['arg2']=str(row['arg2'])
                    res_tmp['rel']=row['rel']
                    res_tmp['arg1_id']=tmp1
                    res_tmp['arg2_id']=tmp2
                    d2.append(res_tmp)
    d_s_candids[w]=d2

nss={}
with open('story.txt', 'a') as the_file:
    the_file.write("s\tt\te1\te2")
    the_file.write('\n')
    for i in range(len(res_final)):
        row=res_final[i]
        edge_c=4
        th=4
        if 'all_edges_1' not in row :
            row['all_edges_1']=[]
        if 'all_edges_2' not in row :
            row['all_edges_2']=[]
        th=4
        if len(row['all_edges_1'])>0:
            the_file.write(df_nodes['character'][row['source']-1]+"\t"+df_nodes['character'][row['target']-1]+"\t"+str(1)+"\t"+row['closest_source_to_target'])
            nss[df_nodes['character'][row['source']-1]]=1
            nss[df_nodes['character'][row['target']-1]]=1
            the_file.write('\n')
        elif len(row['all_edges_2'])>0:
            the_file.write(df_nodes['character'][row['target']-1]+"\t"+df_nodes['character'][row['source']-1]+"\t"+str(1)+"\t"+row['closest_target_to_source'])
            the_file.write('\n')
            nss[df_nodes['character'][row['source']-1]]=1
            nss[df_nodes['character'][row['target']-1]]=1
    for w in candidates:
        okaay=True
        for ww in versions[w]:
            if ww  in stps:
                okaay=False
        if  okaay:
            d=d_s_candids[w]
            all_edges_from2=defaultdict(list)
            all_edges_to2=defaultdict(list)
            c1=0
            c2=0
            for row in d:
                if len(row['arg1'])<20 and len(row['arg2'])<20 and 'hobbit' not in row['arg1'].lower() and 'hobbit' not in row['arg2'].lower():
                    if not row['arg1_id'] and is_any_entities_present(row['arg1'],versions[w]) and not is_any_entities_present(row['arg2'],versions[w]):
                        all_edges_from2[row['arg2_id'][0]].append(getheadWord(row['rel']))
                    if not row['arg2_id'] and is_any_entities_present(row['arg2'],versions[w]) and not is_any_entities_present(row['arg1'],versions[w]):
                        all_edges_to2[row['arg1_id'][0]].append(getheadWord(row['rel']))
            for i in all_edges_from2:
                if len(all_edges_from2[i])>5:
                    c1+=1
#                     the_file.write(list(versions[w])[0]+"\t"+to_csv_id[i]+"\t"+str(2))
#                     the_file.write('\n')
            for j in all_edges_to2:
                if len(all_edges_to2[j])>5:
                    c1+=1
#                     the_file.write(to_csv_id[j]+"\t"+list(versions[w])[0]+"\t"+str(2))
#                     the_file.write('\n')
            for i in all_edges_from2:
                if len(all_edges_from2[i])>5 and c1>2:
                    c1+=1
                    nss[list(versions[w])[0]]=2
                    the_file.write(list(versions[w])[0]+"\t"+to_csv_id[i]+"\t"+str(2)+"\t"+','.join(sorted(all_edges_from2[i])[-3:]))
                    the_file.write('\n')
            for j in all_edges_to2:
                if len(all_edges_to2[j])>5 and c1>2:
                    c2+=1
                    nss[list(versions[w])[0]]=2
                    the_file.write(to_csv_id[j]+"\t"+list(versions[w])[0]+"\t"+str(2)+"\t"+','.join(sorted(all_edges_to2[j])[-3:]))
                    the_file.write('\n')

with open('names.txt', 'a') as the_file:
    the_file.write("n\tc")
    the_file.write('\n')
    for n in nss:
        the_file.write(n+"\t"+str(nss[n]))
        the_file.write('\n')
the_file.close()





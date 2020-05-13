import re
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

file = open("usnewshealth.txt","r+",encoding="utf8")
data=[]
i=0
for line in file:
    row = line.split("|")
    tweet=row[-1]
    tweet=' '.join(filter(lambda x:x[0]!='@', tweet.split()))
    tweet = ' '.join(map(lambda x:x.replace('#',''), tweet.split()))
    tweet=' '.join(map(lambda x:'' if 'http://' in x else x , tweet.split()))
    data.append(tweet.lower())
file.close()

class K_Means:
    def __init__(self, k=2, tol=0.0001, max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
        
    def jaccard_Distance(self, a, b):
        union_count=0
        intersect_count=0
        for word1 in a.split():
            for word2 in b.split():
                if word1 == word2:
                    union_count+=1
                    intersect_count+=1
                else:
                    union_count+=2
        jaccard_dist= (union_count-intersect_count)/union_count
        return jaccard_dist

    def find_centroid(self, tweets):
        min_dist=None
        centroid=None
        for tweet1 in tweets:
            distance=0
            for tweet2 in tweets:
                distance+=self.jaccard_Distance(tweet1, tweet2)
            if min_dist == None or distance< min_dist:
                min_dist=distance
                centroid=tweet1
        return centroid

        

    
    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] =data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            for tweet in data:
                distances=[]
                for centroid in self.centroids:
                    distances.append(self.jaccard_Distance(tweet,self.centroids[centroid]))
                classification = distances.index(min(distances))
                self.classifications[classification].append(tweet)
            
            
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification]=self.find_centroid(self.classifications[classification])
                

            optimized = True

            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                if self.jaccard_Distance(original_centroid,current_centroid) > self.tol:
                    optimized = False
                 
            if optimized:
                break
              
    def predict(self,tweet):
        distances=[]
        for centroid in self.centroids:
            distances.append(self.jaccard_Distance(tweet,self.centroids[centroid]))
        classification = distances.index(min(distances))
        return classification
    
    def SSE(self):
      sse=0
      for centroid in self.centroids:
        for tweet in self.classifications[centroid]:
          sse=sse+(self.jaccard_Distance(self.centroids[centroid], tweet)**2)
      return sse

            
k_values=[3,5,10,15,20]
SSE_values=[]
Size_Of_Cluster=[]
for k in k_values:
  clf= K_Means(k=k)
  clf.fit(data)
  SSE_values.append(clf.SSE()) 
  size_of_cluster={}
  for classification in clf.classifications:
    size_of_cluster[classification+1]=len(clf.classifications[classification])
  Size_Of_Cluster.append(size_of_cluster)
  print(k)
d={'Value of K':k_values, 'SSE':SSE_values, 'Size of each cluster':Size_Of_Cluster}
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
output=pd.DataFrame(data=d,index=None)

print(output)
#print(output.style.set_properties(**{'background-color': 'white','color': 'black'}))

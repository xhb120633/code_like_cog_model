# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:37:26 2023

@author: 51027
"""

import os
import pandas as pd
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import contractions
import openai
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from wordcloud import WordCloud
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
from sklearn.metrics import confusion_matrix

nltk.download('punkt')
nltk.download('wordnet')

openai.api_key = 'XXX' #Your Own API Key

dir_name= 'C:/Users/51027/Documents/GitHub/sorting_algorithm_text_analysis/data'

strategy_dic = ['Unidentified','Gnome Sort','Selection Sort','Insertion Sort','Bubble Sort','Comb Sort','Modified Selection Sort','Shaker Sort','Successive Sweeps','Backward Gnome Sort','Backward Selection Sort','Backward Insertion Sort','Backward Bubble Sort','Backward Comb Sort','Backward Modified Selection Sort','Backward Shaker Sort','Backward Successive Sweeps']

def get_embeddings(text, model): 
    response = openai.Embedding.create(
            input=text,
            engine= model) 
    return response


behavioral_data = pd.read_csv(dir_name+ '/participants.csv')
behavioral_data = np.array(behavioral_data)

#filter out the duplicate trails
behavioral_data = behavioral_data[behavioral_data[:,5]==False,:]
behavioral_data = behavioral_data[behavioral_data[:,9]==False,:]

column_names = [ 'participant_id','network_id','replication','generation', 'condition','cloned','mean_trial_score','algorithm','algorithm_description','exclusion_flag'
]
behavioral_data = pd.DataFrame(behavioral_data, columns=column_names)


network_data = pd.read_csv(dir_name+ '/networks.csv')
network_data = np.array(network_data)


strategy_data = pd.read_excel(dir_name+ '/sorting_algorithm_description.xlsx')
strategy_data = np.array(strategy_data)

for i in range(strategy_data.shape[0]):
    tmp_strategy_name = strategy_data[i,0]
    index = strategy_dic.index(tmp_strategy_name)
    strategy_data[i,0] = index

strategy_len = np.ones((strategy_data.shape[0],1))
for i in range(strategy_data.shape[0]):
    strategy_len[i] = len(strategy_data[i,1])

behavioral_len = np.ones((behavioral_data.shape[0],1))
for i in range(behavioral_data.shape[0]):
    behavioral_len[i] = len(behavioral_data[i,8])   

# #Put text data into GPT-3 ada-002 with embeddings
# model_name = 'text-embedding-ada-002'
# embeddings = np.empty((behavioral_data.shape[0],1536))
# for i in range(behavioral_data.shape[0]):
#     tmp_embeddings = get_embeddings(behavioral_data[i,8],model_name)
#     embeddings [i,:] = tmp_embeddings['data'][0]['embedding']

# model_name = 'text-embedding-ada-002'
# embeddings = np.empty((strategy_data.shape[0],1536))
# for i in range(strategy_data.shape[0]):
#     tmp_embeddings = get_embeddings(strategy_data[i,1],model_name)
#     embeddings [i,:] = tmp_embeddings['data'][0]['embedding']
# strategy_text_embeddings = np.hstack([strategy_data,embeddings])
# # #save data after embedding
# np.save('strategy_text_embeddings.npy', strategy_text_embeddings)

# behavioral_text_embeddings = np.hstack([behavioral_data,embeddings])
# # #save data after embedding
# np.save('behavioral_text_embeddings.npy', behavioral_text_embeddings)

behavioral_text_embeddings = np.load('data/behavioral_text_embeddings.npy', allow_pickle=True)
embeddings = behavioral_text_embeddings[:,10:1546]

strategy_text_embeddings = np.load('data/strategy_text_embeddings.npy', allow_pickle=True)
strategy_embeddings = strategy_text_embeddings[:,2:1538]

text_embeddings = np.vstack([embeddings, strategy_embeddings])

# Create a t-SNE object
tsne = TSNE(n_components=2)

# Fit the mapping data
tsne.fit(embeddings)
#t-SNE transformed data
t_SNE_transformed_data = tsne.fit_transform(embeddings)  
plt.scatter(t_SNE_transformed_data[:,0],t_SNE_transformed_data[:,1])    
plt.ylabel('x2')
plt.xlabel('x1')
plt.savefig('t-SNE.png')

b = np.unique(behavioral_text_embeddings[:,7])
#subset on different questions
for i in b :
    plt.scatter(t_SNE_transformed_data[:,0], t_SNE_transformed_data[:,1], c ='grey')  
    tmp_pos = (strategy_data[:,0] == i)
    tmp_data = t_SNE_transformed_data[tmp_pos,:] 
    plt.scatter(tmp_data[:,0],tmp_data[:,1], c= 'orange')          
    plt.ylabel('x2')
    plt.xlabel('x1')
    tmp_name = 'strategy_'+ str(int(i)) +'_t-SNE.png'
    plt.savefig(tmp_name)
    plt.close()

behavioral_text_t_sne = np.hstack([behavioral_data,t_SNE_transformed_data[0:3408,:]])
behavioral_text_t_sne = behavioral_text_t_sne[behavioral_text_t_sne[:, 7].argsort()]  # sort by day
np.save('behavioral_text_t_SNE.npy', behavioral_text_t_sne)

strategy_text_t_sne = np.hstack([strategy_data,t_SNE_transformed_data[3408:3427,:]])
strategy_text_t_sne = strategy_text_t_sne[strategy_data[:, 0].argsort()]  # sort by day
np.save('strategy_text_t_SNE.npy', strategy_text_t_sne)

behavioral_text_t_sne = np.load('data/behavioral_text_t_SNE.npy', allow_pickle=True)
strategy_text_t_sne = np.load('data/strategy_text_t_SNE.npy', allow_pickle=True)

fig, ax = plt.subplots()
scatter = ax.scatter(behavioral_text_t_sne[:,10], behavioral_text_t_sne[:,11], c=behavioral_text_t_sne[:,6], cmap='viridis')
# Add a color bar
cbar = fig.colorbar(scatter)
# Show plot
plt.show()


# ##find the dimension
# # Perform PCA on the data
# pca = PCA(n_components=None)
# pca.fit(embeddings)

# # Get the explained variances
# explained_variances = pca.explained_variance_ratio_

# # Plot the explained variances
# plt.plot(np.cumsum(explained_variances))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.show()
# plt.close()

# # Plot the explained variances
# plt.plot(explained_variances)
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.show()

# # Calculate the number of dimensions
# cumulative_explained_variances = np.cumsum(explained_variances)
# num_dimensions = np.argmax(cumulative_explained_variances >= 0.95) + 1
# print("Number of Dimensions:", num_dimensions)

###option 1: two representative algorithms
behavioral_data = np.array(behavioral_data)
subset_pos = np.logical_or(behavioral_data[:,7] == 1, behavioral_data[:,7] == 2)
tmp_embeddings = embeddings[subset_pos,:]

##option 2: Merged (forward + backwards algorithms)
tmp_behavioral_data = behavioral_data.copy()
for i in range(tmp_behavioral_data.shape[0]):
    if tmp_behavioral_data[i,7] > 7:
        tmp_behavioral_data[i,7] = tmp_behavioral_data[i,7] - 8
        
subset_pos = (tmp_behavioral_data[:,7]!=0)
tmp_embeddings = embeddings[subset_pos,:]

##option 3: All rough algorithm (except unidentified ones)
subset_pos = (behavioral_data[:,7]!=0)
tmp_embeddings = embeddings[subset_pos,:]





pca = PCA(n_components=200)
embedding_PCA = pca.fit_transform(tmp_embeddings)
strategy_label = behavioral_data[subset_pos,7]
strategy_label = np.array(strategy_label,dtype = 'int')


# Create a StratifiedKFold object
n_fold = 5
skf = StratifiedKFold(n_splits= n_fold)

# Create the SVM model
clf = SVC(kernel='linear', class_weight='balanced')

train_accuracies = []
test_accuracies = []

confusion_matrices = []

for train_index, test_index in skf.split(embedding_PCA, strategy_label):
    X_train, X_test = embedding_PCA[train_index], embedding_PCA[test_index]
    y_train, y_test = strategy_label[train_index], strategy_label[test_index]
    
    # Train the SVM model
    clf.fit(X_train, y_train)
    
    # Predict the responses for test dataset
    y_pred = clf.predict(X_test)

    # Compute the confusion matrix for this fold and add it to the list
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

    # Evaluate the model on the train data
    train_accuracy = clf.score(X_train, y_train)
    train_accuracies.append(train_accuracy)
    
    # Evaluate the model on the test data
    test_accuracy = clf.score(X_test, y_test)
    test_accuracies.append(test_accuracy)

# Assuming confusion_matrices is a list of confusion matrices from each fold
average_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Since we're averaging counts, which are integers, the result might not be integer.
# It's usually okay to round them for display purposes.
average_confusion_matrix = np.round(average_confusion_matrix).astype(int)

labels = [strategy_dic[i] for i in range(1,len(average_confusion_matrix))]

plt.figure(figsize=(12, 10))
sns.heatmap(average_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


accuracy = train_accuracies + test_accuracies
# Create a sample dataframe
data = {'Category': ['A']*2*n_fold,
        'Class': ['Train']*n_fold + ['Test']*n_fold,
        'Values': accuracy }
df = pd.DataFrame(data)

# Reorder the dataframe by Class before plotting
df = df.sort_values('Class', ascending=False)
# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(3, 4))


# Calculate standard deviation for each group
std = df.groupby(['Category', 'Class']).std()['Values'].unstack()
# This part does the grouping and bar plot creation, with error bars
df.groupby(['Category', 'Class']).mean()['Values'].unstack().plot(ax=ax, kind='bar', yerr=std,legend = False)

plt.axhline(y=1/16, color='r', linestyle='--', label='Random Level')
# This part is optional: It just changes the legend location

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Disable the x-axis ticks
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# Modify the x-axis label
plt.xlabel('All Algorithms')
plt.ylabel('Accuracy')
plt.show()



####generate word cloud based on the strategy
# Define the text to generate the word cloud
for i in np.unique(behavioral_data[:,7]):
    tmp_data = behavioral_data[behavioral_data[:,7]==i,:]
    for s in range(tmp_data.shape[0]):
        if s == 0:
            text = tmp_data[s,8]
        else:
            text = text + tmp_data[s,8]

    # Generate the word cloud
    wordcloud = WordCloud().generate(text)
    
    # Plot the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    tmp_name = 'strategy_'+ str(int(i)) +'_word_cloud.png'
    plt.savefig(tmp_name)
    plt.close()
    

# Initialize a list to store the WCSS for each value of k
wcss = []
max_cluster = 30
# Loop over different values of k
for k in range(1, max_cluster):
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k)
    # Fit the model to the data
    kmeans.fit(behavioral_text_t_sne[:,[10,11]])
    # Append the WCSS for this value of k to the list
    wcss.append(kmeans.inertia_)

# Plot the WCSS as a function of k
plt.plot(range(1, max_cluster), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.show()

# create KMeans object
kmeans = KMeans(n_clusters=8, random_state=0)

# fit the embeddings to KMeans object
kmeans.fit(behavioral_text_t_sne[:,10:12])

# get the cluster labels
labels = kmeans.labels_

# get the cluster centers
centers = kmeans.cluster_centers_

# get the distances to other points
distances = kmeans.transform(behavioral_text_t_sne[:,10:12])
dist_list = []
behavioral_text_k_mean = behavioral_text_t_sne

for i in range(behavioral_text_k_mean.shape[0]):
    tmp_label = labels[i]
    dist_list.append(distances[i,tmp_label])

dist_list = np.array(dist_list)
dist_list = np.reshape(dist_list,(len(dist_list),1))
labels = np.reshape(labels,(len(labels),1))
behavioral_text_k_mean = np.hstack([behavioral_text_k_mean,labels,dist_list])

behavioral_text_k_mean = behavioral_text_k_mean[behavioral_text_k_mean[:, 13].argsort()]  # sort by day
behavioral_text_k_mean = behavioral_text_k_mean[behavioral_text_k_mean[:, 12].argsort(kind='mergesort')]  # sort by month
np.save('data/behavioral_text_k_mean.npy', behavioral_text_k_mean)   

behavioral_text_k_mean = pd.DataFrame(behavioral_text_k_mean)
behavioral_text_k_mean.to_csv('data/behavioral_text_k_mean.csv',index=False)


###explore network trajectory
#roughly network text embedding distribution
net_list = np.unique(behavioral_text_t_sne[:,1])
for i in net_list :
    plt.scatter(behavioral_text_t_sne[:,10], behavioral_text_t_sne[:,11], c ='grey')  
    tmp_pos = (behavioral_text_t_sne[:,1] == i)
    tmp_data = behavioral_text_t_sne[tmp_pos,:] 
    plt.scatter(tmp_data[:,10],tmp_data[:,11], c= 'orange')          
    plt.ylabel('x2')
    plt.xlabel('x1')
    tmp_name = 'network_'+ str(int(i)) +'_t-SNE.png'
    plt.savefig(tmp_name)
    plt.close()



for tmp_net in net_list:    
#now visualize the generation-to-generation trajectory
    #filter out a specific network
    tmp_net = 10
    tmp_pos = (behavioral_text_t_sne[:,1] == tmp_net)
    tmp_text_data = behavioral_text_t_sne[tmp_pos,:] 
    tmp_pos = (network_data[:,0] == tmp_net)
    tmp_network_data = network_data[tmp_pos,:] 
    
    tmp_network_data = np.hstack([tmp_network_data,np.zeros((tmp_network_data.shape[0],1))])
    for s in range(tmp_network_data.shape[0]):
        if np.isnan(tmp_network_data[s,2]):
            tmp_network_data[s,4] = 0
        else:
            tmp_parent_pos = (tmp_network_data[:,1] == tmp_network_data[s,2])
            tmp_parent_data = tmp_network_data[tmp_parent_pos,:]
            tmp_network_data[s,4] = tmp_parent_data[0,4] + 1
            
    tmp_text_data = np.hstack([tmp_text_data,np.zeros((tmp_text_data.shape[0],2))])
    for s in range(tmp_text_data.shape[0]):
        tmp_id = tmp_text_data[s,0]
        tmp_data = tmp_network_data[(tmp_network_data[:,1] == tmp_id),:]
        tmp_text_data [s,12] = tmp_data[0,2]
        tmp_text_data [s,13] = tmp_data[0,4]
    
    #let's first start plot the text embeddings with colors assigned to different generations
    tmp_text_data = tmp_text_data[tmp_text_data[:,7]==1,:]
    #find out a line of evoluution
    latest_data = tmp_text_data[tmp_text_data[:,13]==11,:]
    line_id_list = np.zeros((latest_data.shape[0],12))
    t = 0
    for s in range(line_id_list.shape[0]):
        tmp_son_data = latest_data[s,:]
        line_id_list[s,t] =  latest_data[s,13]
        while 1:
            t = t+1
            tmp_parent_id = tmp_son_data[12]
            tmp_parent_data = tmp_text_data[tmp_text_data[:,0]==tmp_parent_id,:]  
            if tmp_parent_data.size == 0 or np.isnan(tmp_parent_id):
                break
            else:
                tmp_parent_data = np.reshape(tmp_parent_data,(tmp_parent_data.shape[1]))
                tmp_son_data = tmp_parent_data
                line_id_list[s,t] = tmp_parent_id
                
    
    for s in range(tmp_text_data.shape[0]):
        if np.isnan(tmp_text_data[s,12])==False:
            tmp_kid_data = tmp_text_data[s,:]
            tmp_kid_data = np.reshape(tmp_kid_data,(tmp_kid_data.shape[0]))
            tmp_parent_data = tmp_text_data[(tmp_text_data[:,0] == tmp_kid_data[12]),:]
            if len(tmp_parent_data)==0:
                continue
            tmp_parent_data = np.reshape(tmp_parent_data,(tmp_parent_data.shape[1]))
            dx = tmp_kid_data[10] - tmp_parent_data[10]
            dy = tmp_kid_data[11] - tmp_parent_data[11]
            plt.arrow( tmp_parent_data[10], tmp_parent_data[11], dx, dy, width=0.005, length_includes_head=True)
    
    
    scatter_plot = plt.scatter(tmp_text_data[:,10],  tmp_text_data[:,11], c= tmp_text_data[:,13], cmap='viridis')
    plt.colorbar(scatter_plot)
    
    tmp_name = 'network'+ str(int(tmp_net)) +'_evolution.png'
    plt.savefig(tmp_name)
    plt.close()

# Show plot
plt.show()

#strategy cluster-wise distribution
behavioral_text_k_mean = np.load('data/behavioral_text_k_mean.npy', allow_pickle=True) 
b = np.unique(behavioral_text_t_sne[:,7])
cluster_list = np.unique(behavioral_text_k_mean[:,12])
cluster_strategy_ratio = -999* np.ones((len(b),len(cluster_list)))
for s in range(len(b)):
    subset_data = behavioral_text_k_mean[behavioral_text_k_mean[:,7]==b[s],:]
    for k in range(len(cluster_list)):
        tmp_cluster_data = subset_data[(subset_data[:,12] ==cluster_list[k]),:]
        tmp_ratio = tmp_cluster_data.shape[0]/subset_data.shape[0]
        cluster_strategy_ratio[s,k] = tmp_ratio 

for s in range(len(b)):
    bar_width = 0.35
    x_pos = np.arange(len(cluster_list))
    plt.bar(x_pos+1, cluster_strategy_ratio[s,:], bar_width, color='orange')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion of Strategy ' + str(int(b[s])))
    tmp_name = 'Strategy'+ str(int(s)) +'_Cluster_Distribution.png'
    plt.savefig(tmp_name)
    plt.close()
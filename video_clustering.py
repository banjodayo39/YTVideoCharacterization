from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import seaborn as sns

from textblob import TextBlob
from tqdm import tqdm

from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from google.colab import drive
from glob import iglob
import glob

import numpy as np
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
import imageio
from google.colab import drive
from PIL import Image

from sklearn.cluster import KMeans
import json
import os

def plot_pie(input_data, labels, title):
    input_data_ = []
    for item in input_data:
#         print(np.argmax(item))
        input_data_.append(np.argmax(item))
    input_data_freq = [input_data_.count(x) for x in range(len(labels))]
    print(input_data_freq)
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.size'] = 13
#     labels = 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)[:len(labels)]  # only "explode" the 5th slice
    print(explode)
    print(labels)
    
    # Experimental, save 
    
    plt.pie(
        input_data_freq, 
        explode=explode, 
        labels=labels, 
        autopct='%1.1f%%',
#         colors=colors_,
        shadow=True, 
        startangle=90, 
        rotatelabels=45
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.tight_layout()
#     plt.title(title)
    plt.savefig("ClusteringResults" + ".png")
#     joblib.dump("/Users/receperol/Desktop/LabeledPlots3D/joblib_plots/" + title+".joblib", plt)
    plt.show()


# def plot_pie(input_data, labels, title):
#     input_data_ = []
#     for item in input_data:
# #         print(np.argmax(item))
#         input_data_.append(np.argmax(item))
#     input_data_freq = [input_data_.count(x) for x in range(len(labels))]
#     fig, ax = plt.subplots(figsize=(16, 9), subplot_kw=dict(aspect="equal"))

#     recipe = []
#     for freq, label in zip(input_data_freq, labels):
#         recipe.append(f"{freq} {label}")

#     data = [float(x.split()[0]) for x in recipe]
#     ingredients = [x.split()[-1] for x in recipe]

#     def func(pct, allvals):
#         absolute = int(pct/100.*np.sum(allvals))
#         return "{:.1f}%\n({:d})".format(pct, absolute)

#     wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
#                                       textprops=dict(color="w"))

#     ax.legend(wedges, ingredients,
#               title="Emotions",
#               loc="center left",
#               bbox_to_anchor=(1, 0, 0.5, 1))

#     plt.setp(autotexts, size=12, weight="bold")
#     ax.set_title(title)
#     plt.savefig(f"ClusteringResults/{title}"+".png")
#     plt.show()
    
    
# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


# Plot silhouette score 
def plot_silhouette(input_data, title):
    
    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(input_data)
        labels = kmeans.labels_
        sil.append(silhouette_score(input_data, labels, metric = 'euclidean'))

    plt.figure(figsize=(16, 9))
    plt.plot([x for x in range(len(sil))], sil)
    plt.title(title, fontsize=16)
    plt.xlabel("Number of clusters", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ClusteringResults"+".png")
    plt.show()
    
    
# plot elbox score for the input data
def plot_wss(input_data, title):
    
    if type(input_data) is not np.array:
#         print("here")
        input_data = np.array(input_data)
    
    wss = calculate_WSS(input_data, 10)

    plt.figure(figsize=(16, 9))
    plt.plot([x for x in range(len(wss))], wss)
    plt.title(title, fontsize=16)
    plt.xlabel("Number of clusters", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
# Visualize the result of Kmeans algorithm on 2D scatter plot
def plot_2D_Scatter(input_data, title, kmeans):
    plt.figure(figsize=(16, 9))
    plt.scatter(input_data[:,0], 
                input_data[:,1], 
                c=kmeans.labels_.astype(float), 
                s=300, cmap="inferno", 
#                 colors=colors_
               )
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"ClusteringResults"+".png")
    plt.show()
    
    
# Visualize the result of K-Means clustering on 3D scatter plot
def plot_3D_scatter(input_data, kmeans, title):
    plt.rcParams["figure.figsize"] = (16, 9)
    
    pca = PCA(n_components=3).fit(input_data)
    input_data_3d = pca.transform(input_data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams["figure.figsize"] = (16, 9)

    unique_elements, counts_elements = np.unique(kmeans.labels_, return_counts=True)

    x =input_data_3d[:,0]
    y =input_data_3d[:,1]
    z =input_data_3d[:,2]

    # ax.scatter(x, y, z, c='r', marker='o')
    scatter = ax.scatter(x, y, z, c=kmeans.labels_.astype(float), 
                s=300, cmap="inferno", 
#                colors=colors
              )
        
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", 
                        title="Clusters")
    ax.add_artist(legend1)
    
    legend2 = ax.legend(handles=scatter.legend_elements()[0], 
                        labels=[str(x) for x in counts_elements], loc="upper right", title="Counts")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
#     ax.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("ClusteringResults"+".png")
    plt.show()

    
# Visualize the result of K-Means clustering on 3D scatter plot with class value highlighted
def plot_3D_scatter_w_text(input_data, kmeans, title, class_labels):
    plt.rcParams["figure.figsize"] = (16, 9)
    
    pca = PCA(n_components=3).fit(input_data)
    input_data_3d = pca.transform(input_data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams["figure.figsize"] = (16, 9)
    
    unique_elements, counts_elements = np.unique(kmeans.labels_, return_counts=True)

    x =input_data_3d[:,0]
    y =input_data_3d[:,1]
    z =input_data_3d[:,2]

    # ax.scatter(x, y, z, c='r', marker='o')
    scatter = ax.scatter(x, y, z, c=kmeans.labels_.astype(float), 
                         s=300, cmap="inferno",
                         )
    
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", 
                        title="Clusters")
    ax.add_artist(legend1)
    
    legend2 = ax.legend(handles=scatter.legend_elements()[0], 
                        labels=[str(x) for x in counts_elements], loc="upper right", title="Counts")
    
    for x_, y_, z_, l_ in zip(x, y, z, class_labels):
#         ax.text(x_, y_, z_, str(l_) , size=16, zorder=1)
        ax.text(x_*1.1, y_*1.1, z_*1.1, str(l_) , size=10, zorder=1) # Add some flavor, class name, and elevate the position of text

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
#     ax.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("ClusteringResults"+".png")
    plt.show()
    

# read metadata file and load as dictionary
def read_metadata(filename):
    assert filename, str
    
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist!")
    
    with open(filename, "r") as f:
        metadata = json.load(f)
        
    return metadata


# read video collection folder and list of video collections and return titles, and descriptions of each video
def file2metadata(foldername, video_collection):
    assert video_collection, list
    
    if len(video_collection) == 0:
        raise ValueError("Provided video collection is empty!")
    
    titles = []
    descriptions = []
    
    for vid in video_collection:
        filename = foldername + vid + "/" + vid + "_metadata.json"
        metadata = read_metadata(filename=filename)
        if "video_title" in metadata.keys():
            titles.append(metadata["video_title"])
        else:
            titles.append(metadata["title"])
        descriptions.append(metadata["description"])
        
    return titles, descriptions


# remove url from provided text
# reference: https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


# Preprocess text by removing punctuations and urls then lower all letters
def normalize(text):
    if type(text) is float:
        text = " "
        return text

    if len(text) == 0 or len(text) == 1:
        print(f"Provided text is empty!")
        return text
        
    # remove urls
    text_noUrl = remove_URL(text)
    
    no_punctuation= text_noUrl.translate(punctuations)
    output = no_punctuation.lower()
    
    # remove single characters
    output = ' '.join( [w for w in output.split() if len(w)>1] )
    
    return output.strip()


# clean list of text
def clean_List_of_text(list_of_text):
    assert list_of_text, list
    
    if len(list_of_text) == 0:
        raise ValueError(f"Provided text is empty!")
        
    cleaned = []
    
    for line in tqdm(list_of_text):
        cleaned.append(normalize(line))
    
    return cleaned


# Calculate sentiment score for the list of text
def sentiment_score_calc(myList):
    assert myList, list
    
    output = []
    
    for idx in tqdm(range(len(myList))):
        if myList[idx] == "" or myList[idx] == " " or myList[idx] == [' ']:
            output.append([0, 0])
        else:
            output.append(TextBlob(myList[idx]))
        
    return output


# Calculate emotion score for the list of text
def emotion_score_calc(myList):
    assert myList, list
    
    es = EmoNet()
    output = []
    
    for idx in tqdm(range(len(myList))):
        if myList[idx] == "" or myList[idx] == " " or myList[idx] == [' ']:
            output.append([0, 0, 0, 0, 0, 0, 0, 0])
        else:            
#             print(myList[idx])
            temp = []
            pred_ = es.predict(myList[idx], with_dist=True)[0][2]
            temp = [
                pred_["anger"],
                pred_["anticipation"],
                pred_["disgust"],
                pred_["fear"],
                pred_["joy"],
                pred_["sadness"],
                pred_["surprise"],
                pred_["trust"]
            ]
            output.append(temp)
        
    return output


# Calculate toxicity score for the list of text
def toxicity_score_calc(myList):
    assert myList, list
    
    output = []
    
    for idx in tqdm(range(len(myList))):
        if myList[idx] == " " or myList[idx] == "":
            output.append([0, 0, 0, 0, 0, 0, 0])
        else:
            output.append(toxicity(myList[idx]))
        
    return output


# Load video transcripts
def load_transcript(data_folder, ids):
    
    output = []
    
    for idx in tqdm(ids):
        filename = data_folder + idx + "/" + idx + "_transcript.txt"
        
        if not os.path.exists(filename):
            output.append([" "])
        else:
            with open(filename, "r") as f:
                content = f.read()

            output.append(' '.join([line for line in content.split("\n")]))
        
    return output

def unique(myList):
    unique_list = []
    for idx in myList:
        if idx not in unique_list:
            unique_list.append(idx)
            
    return unique_list
    
    unique_elements, counts_elements = np.unique(kmeans.labels_, return_counts=True)

    x =input_data_3d[:,0]
    y =input_data_3d[:,1]
    z =input_data_3d[:,2]

    # ax.scatter(x, y, z, c='r', marker='o')
    scatter = ax.scatter(x, y, z, c=kmeans.labels_.astype(float), 
                         s=300, cmap="inferno",
                         )
    
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", 
                        title="Clusters")
    ax.add_artist(legend1)
    
    legend2 = ax.legend(handles=scatter.legend_elements()[0], 
                        labels=[str(x) for x in counts_elements], loc="upper right", title="Counts")
    
    for x_, y_, z_, l_ in zip(x, y, z, class_labels):
#         ax.text(x_, y_, z_, str(l_) , size=16, zorder=1)
        ax.text(x_*1.1, y_*1.1, z_*1.1, str(l_) , size=10, zorder=1) # Add some flavor, class name, and elevate the position of text
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
#     ax.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    

# read metadata file and load as dictionary
def read_metadata(filename):
    assert filename, str
    
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist!")
    
    with open(filename, "r") as f:
        metadata = json.load(f)
        
    return metadata


# read video collection folder and list of video collections and return titles, and descriptions of each video
def file2metadata(foldername, video_collection):
    assert video_collection, list
    
    if len(video_collection) == 0:
        raise ValueError("Provided video collection is empty!")
    
    titles = []
    descriptions = []
    
    for vid in video_collection:
        filename = foldername + vid + "/" + vid + "_metadata.json"
        metadata = read_metadata(filename=filename)
        if "video_title" in metadata.keys():
            titles.append(metadata["video_title"])
        else:
            titles.append(metadata["title"])
        descriptions.append(metadata["description"])
        
    return titles, descriptions


def transform_with_PCA(images):
    images_array = np.array(images)
    #n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
    barcode_pca = PCA(n_components=0.8)
    barcode_pca.fit(images_array)
    transform_barcode_pca = barcode_pca.transform(images)
    return transform_barcode_pca


def run_Kmeans(images):
    transform_barcode_pca = transform_with_PCA(images)
    kmeans = KMeans(n_clusters=6, random_state=0).fit(transform_barcode_pca)
    return kmeans

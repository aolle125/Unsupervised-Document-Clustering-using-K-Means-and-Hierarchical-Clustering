# Unsupervised-Document-Clustering-using-K-Means-and-Hierarchical-Clustering

1.	Reading and Procurement of Data

I read a 1000 House Bills from the 2020 Session which were HB1-1000 for document clustering. Jong and Seung ran their script and sent me the text file for all the bills. I read the text files into a list and extracted the Bill Number and the Full Text of the Bills and disregarded the metadata as it was not necessary for clustering of documents. After extraction, the data was stored into a Pandas Dataframe. Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns. Pandas DataFrame consists of three principal components, the data, rows, and columns.


2.	Data Cleaning and Preparation

The most important step for data cleaning was to check if there were any null fields in the data. Fortunately there were no Null fields in the data which were checked by df.isna().count() . After this was checked then I performed the main task of removing the non-alphanumeric characters from the full text fields. This was important as we only want the words and numbers related to the bills and not the special characters. 
df.Text = df.Text.str.replace('[^a-zA-Z ]', '')

Example: (Bill HB1)

Before Cleaning:
'Introduced:2020 SESSION20101095DHOUSE BILL NO. 1Offered January 8, 2020Prefiled November 18, 2019A BILL to amend and reenact Â§Â§ 24.2-416.1, 24.2-452, 24.2-612,24.2-700, 24.2-701, 24.2-701.1, 24.2-702.1, 24.2-703.1, 24.2-703.2, 24.2-705.1,24.2-705.2, 24.2-706, 24.2-709, and 24.2-1004 of the Code of Virginia, relatingto absentee voting; no excuse required.----------Patrons-- Herring, Carroll Foy, Carter, Filler-Corn, Gooditis, Guzman,Keam, Kory, Levine, Lopez, McQuinn, Tyler, Watts and Willett; Senator:McClellan-------

After Cleaning:
‘Introduced SESSIONDHOUSE BILL NO Offered January  Prefiled November A BILL to amend and reenact and of the Code of Virginia relatingto absentee voting no excuse requiredPatrons Herring Carroll Foy Carter FillerCorn Gooditis GuzmanKeam Kory Levine Lopez McQuinn Tyler Watts and Willett SenatorMcClellanReferred to Committee on Privileges and ElectionsBe it enacted by the General Assembly of Virginia That’

I also worked on removing data fields that had less than three characters as they were not important for document clustering. The NLTK Package consists a list of stop words. A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. We used this package to remove stop words from all the text fields in the Full Text Data as only the important words summarizing the bills were more important for clustering.





3.	Conversion of Full Text to TF-IDF

We used the sklearn.feature_extraction.text.TfidfVectorizer which is used to convert a collection of raw documents to a matrix of TF-IDF features. TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. TF-IDF for a word in a document is calculated by multiplying two different metrics:

The term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.
The inverse document frequency of the word across a set of documents. This means, how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.

So, if the word is very common and appears in many documents, this number will approach 0. Otherwise, it will approach 1. Multiplying these two numbers results in the TF-IDF score of a word in a document. The higher the score, the more relevant that word is in that particular document. As hyper parameters to TfidfVectorizer, we used min_df = 5 i.e. the word must appear in atleast 5 documents and max_df = .75 which means that the word must appear in less than 75% of the documents. 


4.	Clustering of Documents

We use K-Means and Mini-Batch K-Means for finding the optimum set of clusters.
 Eg: MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data)

As we can see in the figure below, the sum of squared errors is minimum for clusters 10 and thus we have divided the documents into 10 clusters. 

I have used Elbow Method and the Silhouette Score to choose the k for k-means

After clustering is performed I used PCA and T-SNE to embed the clusters to two-dimensions which would enhance visibility of the clusters. 

The most important revelation using the code helped me find the important key words for every cluster and we divided the documents into clusters with the results given below.

Example:
Cluster 0
unlaw,test,duti,secretari,divis

Cluster 1
offici,regist,registr,vote,elect

Cluster 2
parent,educ,divis,student,school

Cluster 3
civil,counti,commission,penalti,violat

 

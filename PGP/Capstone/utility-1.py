import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

def find_unique_values_count(dataframe) :
    for column in dataframe :
         print("Unique value in column ", column , " is ", len(dataframe[column].unique()))
         print("Unique Values are :", dataframe[column].value_counts()) 
         print("############################################################################")


def createPieChart(df, column_name,title):
  return px.pie(df,names=column_name,labels=df[column_name].unique(),title=title)

def createCountPlotByAccidentLevel(df, col1):
    fig = plt.figure(figsize = (15, 7.2))
    ax = fig.add_subplot(121)
    sns.countplot(x = col1, data = df, ax = ax, orient = 'v',
                  hue = 'Accident Level').set_title(col1.capitalize() +' count plot by Accident Level', 
                                                                      fontsize = 13)
    plt.legend(labels = df['Accident Level'].unique())
    plt.xticks(rotation = 90)
    
    ax = fig.add_subplot(122)
    sns.countplot(x = col1, data = df, ax = ax, orient = 'v', 
                  hue = 'Potential Accident Level').set_title(col1.capitalize() +' count plot by Potential Accident Level', 
                                                                      fontsize = 13)
    plt.legend(labels = df['Potential Accident Level'].unique())
    plt.xticks(rotation = 90)
    return plt.show()

def createCountPlot(df, col1, hue):
    fig = plt.figure(figsize = (15, 7.2))
    sns.countplot(x = col1, data = df,hue = hue).set_title(col1.capitalize() +' count plot by ' + hue, 
                                                                      fontsize = 13)
    plt.legend(labels = df[hue].unique())
    plt.xticks(rotation = 90)

    return plt.show()

# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
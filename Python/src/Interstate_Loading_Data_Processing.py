#Interstate Loading Data Processing:
import os
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.chdir('/Users/heqiaoruan/Documents/Github/OnlineExtension_RFPerm')
from model_registry_class import ModelRegistry


'''
Data-Processing for the NYC taxi data, you may want to set the batch-size as 1.
'''
dirs = Path('/Users/heqiaoruan/Documents/Github/OnlineExtension_RFPerm')


'''
#Extracting the embeddings followed by the PCA with the first 10 PCs: first 10 of them
accounts for 95% of the variance.

##################################################################
import os
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA, TruncatedSVD
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df['weather_main'] = df['weather_main'].fillna('unknown').astype(str)
df['weather_description'] = df['weather_description'].fillna('').astype(str)
df['holiday'] = df['holiday'].fillna('Not').astype(str)

#Extracting the embeddings followed by the PCA with the first 20 PCs
#For simplicity, we leverage the light-weighted contextual word embedding model for faster computation:
texts = []
for _, row in df.iterrows():
    s = f"Holiday: {row['holiday']}. Weather: {row['weather_main']}. Description: {row['weather_description']}."
    texts.append(s)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size = 128, show_progress_bar = True)
embeddings = embeddings.astype(np.float32)
#Perform PCA on this embedding:
pca = PCA()
embeddings_pca = pca.fit_transform(embeddings)
explained = pca.explained_variance_ratio_
#explained.cumsum()#Top10 explained 94% variance.
embeddings_pca = embeddings_pca[:, :10]
pd.DataFrame(embeddings_pca).to_csv("embeddings_intersate.csv") -> return the 'embeddings_interstate.csv' here.
#We run it on the colab and save:
##################################################################

For simplicity, we leverage the light-weighted contextual word embedding model for faster computation:
Additionally we build a couple of other features:
We leverage the temperature as well as the time information to predict the traffic volumne.
- temperature
- rain amount
- day_of_week
- month
- temperature
'''

df_metro = pd.read_csv("Real_Data/Load_Traffic_Data/Metro_Interstate_Traffic_Volume.csv")
df_metro['tempC'] = df_metro['temp'] - 273.15
df_metro['date_time'] = pd.to_datetime(df_metro['date_time'])
df_metro['hour'] = df_metro['date_time'].dt.hour
df_metro['day_of_week'] = df_metro['date_time'].dt.dayofweek
df_metro['month'] = df_metro['date_time'].dt.month
numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'day_of_week', 'hour', 'month']
for cols in numerical_cols:
    df_metro[cols] = df_metro[cols].values.astype(np.float32)
df_numeric = df_metro[numerical_cols]
df_embedding = pd.read_csv("Real_Data/Load_Traffic_Data/embeddings_intersate.csv").drop(['Unnamed: 0'], axis = 1)
df_embedding.columns = ["X" + str(i) for i in np.arange(10)]

df_interstate = pd.DataFrame(np.hstack([df_numeric, df_embedding, np.array(df_metro['traffic_volume'].rename({'traffic_volume': 'Y'})).reshape(-1,1)]))
df_interstate['date_time'] = df_metro['date_time']
df_interstate.to_csv("interstate_loading_data.csv")


'''
As you suggested before, let's save it for later.
Extract the dataset into the lists of the 
Dataframes(Reference) + Following batches of Datasets 
'''
def extract_minibatch_df(df_X, df_Y, time_ref, ref_size = 3000, batch_size = 200):
    rows = []
    n = df_X.shape[0]
    df_X_ref = df_X[:ref_size, :]
    df_Y_ref = df_Y[:ref_size]
    time_col_ref = time_ref[:ref_size]
    rows.append([time_col_ref, df_X_ref, df_Y_ref])
    for i in range(ref_size, n, batch_size):
        if i + batch_size >= n:
            time_col_ref = time_ref[i: ]
            df_X_eval_batch = df_X[i:, :]
            df_Y_eval_batch = df_Y[i:]
        else:
            time_col_ref = time_ref[i:(i + batch_size)]
            df_X_eval_batch = df_X[i:(i + batch_size)]
            df_Y_eval_batch = df_Y[i:(i + batch_size)]
        rows.append([time_col_ref, df_X_eval_batch, df_Y_eval_batch])
    return rows

rows_result = extract_minibatch_df(df_X, df_Y, time_ref = time_col_index, ref_size = 3000, batch_size = 200)

#Running on that dataset:

onlinePermOOB_stream(row_result, time_ref = 200, ref_size = 3000, batch_size = 1)
onlinePermOOB_stream(row_result, time_ref = 200, ref_size = 3000, batch_size = 1)





















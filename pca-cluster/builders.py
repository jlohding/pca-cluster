import os
import glob
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class tsBuilder:
    def __init__(self, fpath, start_date="2010-01-01", end_date="2015-01-01"):
        self.fpath = fpath # absolute path to directory of .txt data
        self.dtformat = "%Y%m%d"
        self.start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")
        self.data = pd.DataFrame()

    def log(self, msg):
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

    def get_ts(self, symbol):
        try:
            return self.datas[symbol]
        except KeyError as e:
            raise Exception(f"dataframe {symbol} is not in data")

    def get_paths(self, extension=""):
        all_files = glob.glob(self.fpath + f"/*{extension}") # grab all files in directory
        return {os.path.basename(v):v for v in all_files}

    def compare_etfs(self, symbol1, symbol2):
        dfs = []
        for symbol in [symbol1, symbol2]:
            df = pd.read_csv(self.fpath + symbol)
            df["Date"] = pd.to_datetime(df["Date"], format=self.dtformat)
            df = df.set_index("Date", drop=True) # format date and set as datetimeindex
            df[symbol] = df["Close"]
            dfs.append(df[symbol][self.start_date:self.end_date])

        cat = pd.concat(dfs, axis=1)

        fig, ax = plt.subplots(2)
        cat[symbol1].plot(ax=ax[0], title=symbol1, color="red")
        cat[symbol2].plot(ax=ax[1], title=symbol2, color="blue")
        plt.show()

    def check_cointegration(self, symbol1, symbol2):
        pass

    def is_in_interval(self, df):
        '''
        Checks if the dataframe has data throughout the entire test/train interval
        '''
        return (df.index[0] > self.start_date or df.index[-1] < self.end_date)

    def build_frame(self, paths):
        frames = []
        for symbol, path in paths.items():

            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"], format=self.dtformat)
            df = df.set_index("Date", drop=True) # format date and set as datetimeindex

            if self.is_in_interval(df): # exclude dataframes not in test/train interval
                continue
            else:
                df = df[self.start_date:self.end_date] # keep it within the date range
                df[f"{symbol}"] = np.log(df["Close"]) - np.log(df["Close"].shift(1)) # get logarithmic returns 
                daily_log_returns = df[f"{symbol}"]
                df = df.dropna()
            
                frames.append(daily_log_returns)
        
        cat = pd.concat(frames,axis=1).transpose()
        cat = cat.fillna(0, axis=1)        
        self.data = cat
        self.log(f"build_frame completed with {len(self.data)} valid frames")

    def feature_scaling(self):
        if len(self.data) == 0:
            raise Exception("No data available")
        else:
            normalised = StandardScaler().fit_transform(self.data.values) # assume lognormal price distribution
            return normalised
    
    def pca(self, normalised, n):
        pca = PCA(n_components=n)
        components = pca.fit_transform(normalised)

        pc_df = pd.DataFrame(components, columns=[f"PC{k}" for k in range(1,n+1)], index=[self.data.index])

        '''
        plotting for 2 dimensional subspace only (n=2), delete later 
        fig, ax = plt.subplots()
        df.plot(x="PC1", y="PC2", kind="scatter", ax=ax)
        for k, _ in df.iterrows():
            ax.annotate(k,_)
        
        plt.show()
        '''
        return pc_df

    def cluster_kmeans(self, pc_df, n):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(pc_df)
        y_kmeans = kmeans.predict(pc_df)
        pc_df["Cluster"] = y_kmeans

        '''
        2 dimensionsal subspace only, delete later
        plt.scatter(pc_df["PC1"], pc_df["PC2"], c=y_kmeans, cmap='plasma')
        plt.show()
        '''

        clusters = pc_df.groupby("Cluster").indices.values()
        groupings = []

        for members in clusters:
            group = pc_df.iloc[members].index.to_list()
            group = [x[0] for x in group]
            groupings.append(group)

        return groupings
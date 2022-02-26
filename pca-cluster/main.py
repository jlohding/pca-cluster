from builders import tsBuilder

def main(ts, pca_components, clusters):
    paths_d = ts.get_paths() # get paths to data
    ts.build_frame(paths_d) # build dataframe of daily log returns 
    norm = ts.feature_scaling() # normalised features
    pca_df = ts.pca(norm, n=pca_components) # apply PCA
    cluster_result = ts.cluster_kmeans(pca_df, clusters) # kmeans cluster

    print(cluster_result) # list of groups of similar stocks

    with open("cluster_results.txt", "w") as f:
        f.write(str(cluster_result))

if __name__ == "__main__":
    ts = tsBuilder("/home/jerry/pca-cluster/stock_data/") # replace with absolute path to 
    main(ts, pca_components=3, clusters=50)

    #ts.compare_etfs("DTO", "SCO") # plot pairs of stocks



# Dimensionality Reduction and Clustering for Pairs Trading Selection

#### Authors: Jerry Loh [@jlohding](https://github.com/jlohding), Lian Kah Seng [@sunroofgod](https://github.com/sunroofgod)

### About
- View the source code [here](main.ipynb)
- View the project report [here](project_report.pdf)
- In this project, we attempt to search for pairs of securities suitable for a mean-reversion pairs trading strategy using machine learning methods. In particular, we apply unsupervised dimensionality reduction and clustering techniques to historical daily price data for a set of 4088 ETFs. 
  - These methods include Principal Component Analysis (PCA), k-Means Clustering, Density-Based Spatial Clustering of Applications with Noise (DBSCAN), and Hierarchical Clustering. 
- This project is part of our graded coursework in the National University of Singapore module IT1244: Artificial Intelligence: Technology and Impact.

### Setup
```git clone https://github.com/jlohding/pca-cluster.git```

Make sure you have the following dependencies in ```requirements.txt```
```
python3==3.9.2
pandas==1.3.0
numpy==1.20.3
sklearn==1.0.2
```

Run the Jupyter Notebook ```main.ipynb``` on your local machine or Google Colab 

### Project Report
![](/img/project_report/project_report-1.png)
![](/img/project_report/project_report-2.png)
![](/img/project_report/project_report-3.png)
![](/img/project_report/project_report-4.png)
![](/img/project_report/project_report-5.png)
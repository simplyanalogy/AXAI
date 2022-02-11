# AXAI
Analogical Explainable AI

- Regarding UCI datasets, please go to the web site https://archive.ics.uci.edu/ml/index.php to get the data. Duplicated rows and rows with unknown values were removed.

- Regarding artificial datasets, they are created by python code.

- To get the AR indexes, please run kr22-AIR-index.ipynb
  For artificial data, change the function in Action 5. To use f4, set f=f4. And set the dataset "dataset.csv", eg filename="dataset.csv"
  For UCI data, change the filename in Action 5. To use dataset "datasets/breast-cancer.csv", then change the dataset to "datasets/breast-cancer.csv"

To get the explanation, please run kr22-Explanations.ipynb
  The dimension of the data and the mask for relevant attributes are currently hardcoded in Section 4. For the mask, 1 means relevant and 0 means irrelevant.
  For artificial data, change the function in Action 4. To use f4, set f=f4

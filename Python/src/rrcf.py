'''
rrcf(robust random cut forest) for online/streaming anomaly detection:
'''
import rrcf


X, y = fetch_openml('titanic', version = 1, as_frame = True, return_X_y = True)
#X = titanic.frame.drop('survived', axis = 1)
#y = titanic.frame['survived']


cate_features = ['embarked', 'sex', 'pclass']
cate_transformer = Pipeline(
    steps = [
    ('encoded': OneHotEncoder(handle_unknown = 'ignore')),
    ('selector': SelectPercentile(chi2, percentile = 50)),
    ]
)
cate_transformer 
preprocessor = ColumnTransformer(
    transformers = [
    ('num': numeric_transformer, numeric_features),
    ('cat': categorical_transformer, categorical_features)
])
preprocessor.fit(X, y)
clf = Pipeline(
    steps = [('preprocesor', preprocessor),
    ('rf': RandomForestClassifier(random_state = 1,
        max_depth = 4))]
    )

#value -> value towards to end -> then calculate whether you can reach the 
#end-points.

#How would you make it to 


#map reduce method to find the median?

#making a histogram followed by the proportion of values that
#accounts in that bin.
#then project each of the subset into.

import numpy as np
import rrcf
X = np.random.randn(100, 2)
tree = rrcf.RCTree(X)
tree = rrcf.RCTree()

tree = rrcf.RCTree()
for i in range(10):
    X = np.random.randn(2)
    tree.insert_point(X, index = i)


tree.insert_point(X, index = i)
#calculate the anomaly score:
X = np.random.randn(100, 2)
tree = rrcf.RCTree()
inlier = np.array([0, 0])
outlier = np.array([4, 4])
tree.insert_point(inlier, index = 'inlier')
tree.insert_point(outlier, index = 'outlier')
tree.codisp('inlier')
tree.codisp('outlier')

tree.insert_point(inlier, index = 'inlier')


#pd.merge_asof(X1, X2, by = , direction = 'backward',
#left_on = '', right_on = '')












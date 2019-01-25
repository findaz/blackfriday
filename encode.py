import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class PipelineFriendlyLabelEncoder(TransformerMixin):

    '''
    Pipeline assumes LabelEncoder's fit_transform method is defined 
    to take three positional arguments, while it is defined to take 
    only two. This class wraps LabelEncoder to solve the problem
    '''

    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)

    def fit(self, X, y = 0):
        # unique returns sorted order and thus
        # maintains natural order where there is one
        self.encoder.fit( np.unique(X) )
        return self

    def transform(self, X, y=0):
        res = self.encoder.transform(X)
        return np.reshape(res, (-1, 1) )

class PipelineFriendlyBinarizer(TransformerMixin):

    '''
    Pipeline assumes LabelBinarizer's fit_transform method is defined 
    to take three positional arguments, while it is defined to take 
    only two. This class wraps LabelEncoder to solve the problem
    '''

    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, X, y = 0):
        self.encoder.fit( X )
        return self

    def transform(self, X, y=0):
        res = self.encoder.transform(X)
        return res #np.reshape(res, (-1, 1) )

class PandasBinarizer(TransformerMixin):

    def __init__(self, *args, **kwargs):
        self.dummy_na = kwargs.get('dummy_na', False) 
        self.columns = None
        
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        res = pd.get_dummies(X, dummy_na = self.dummy_na)
        self.columns = res.columns
        return res
    
class AttributeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.key]
        
if __name__ == "__main__":

    data = pd.read_csv('./data/blackfriday-train.csv')

    tonumber = ['Age', 'Gender', 'Stay_In_Current_City_Years']

    pipe_1 = [ (name, Pipeline([
        ( 'selector', AttributeSelector(key = name) ),
        ( 'encoder', PipelineFriendlyLabelEncoder() )
    ]) ) for name in tonumber ]


    toonehot = ['Occupation', 'City_Category', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    pipe_2 =  [ (name, Pipeline([
        ( 'selector', AttributeSelector(key = name) ),
        ( 'encoder', PandasBinarizer(dummy_na = False) )
    ]) ) for name in toonehot[0:3] ]

    pipe_2 =  pipe_2 + [ (name, Pipeline([
        ( 'selector', AttributeSelector(key = name) ),
        ( 'encoder', PandasBinarizer(dummy_na = True) )
    ]) ) for name in toonehot[3:] ]

    pipelines = pipe_1 + pipe_2
    
    feats = FeatureUnion(pipelines)

    f = feats.fit_transform(data)

    newcols = ['Age', 'Gender', 'Stay_In_Current_City_Years']
    for i, name in enumerate(toonehot):
        n = i + len(tonumber)
        dummies = feats.transformer_list[n][1].named_steps['encoder'].columns
        for dum in dummies:
            newcols.append(name + '_' + str(dum).replace('.0', ''))


    dataset = pd.DataFrame(data = f, columns = newcols)

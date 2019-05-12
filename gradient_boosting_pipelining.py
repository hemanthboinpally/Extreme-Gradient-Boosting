""" Pipelining"""

"""
1)

Encoding categorical columns I: LabelEncoder

First, you will need to fill in missing values - as you saw previously, the column LotFrontage has many missing values.
 Then, you will need to encode any categorical columns in the dataset
 using one-hot encoding so that they are encoded numerically.




"""

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


"""

      MSZoning PavedDrive Neighborhood BldgType HouseStyle
    0       RL          Y      CollgCr     1Fam     2Story
    1       RL          Y      Veenker     1Fam     1Story
    2       RL          Y      CollgCr     1Fam     2Story
    3       RL          Y      Crawfor     1Fam     2Story
    4       RL          Y      NoRidge     1Fam     2Story
       MSZoning  PavedDrive  Neighborhood  BldgType  HouseStyle
    0         3           2             5         0           5
    1         3           2            24         0           2
    2         3           2             5         0           5
    3         3           2             6         0           5
    4         3           2            15         0           5
"""


###################

"""
2)

Encoding categorical columns II: OneHotEncoder

Natural Ordering Problem. 

Okay - so you have your categorical columns encoded numerically. Can you now move onto using pipelines and XGBoost?
Not yet! In the categorical columns of this dataset, there is no natural ordering between the entries. 
As an example: Using LabelEncoder, the CollgCr Neighborhood was encoded as 5, 
while the Veenker Neighborhood was encoded as 24, and Crawfor as 6. Is Veenker "greater" than Crawfor and CollgCr? 
No - and allowing the model to assume this natural ordering may result in poor performance.

There is another step needed: You have to apply a one-hot encoding to create binary, or "dummy" variables. 

"""

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features = categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)


"""
OUTPUT:

(1460, 21)
(1460, 62)

After one hot encoding, which creates binary variables out of the categorical variables, there are now 62 columns.

"""


#################


"""
3)

Encoding categorical columns III: DictVectorizer

Using a DictVectorizer on a DataFrame that has been converted to a dictionary allows you to
get label encoding as well as one-hot encoding in one go.

"""

# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)

"""
OUTPUT:

    {'MSSubClass': 23, 'LotFrontage': 22, 'LotArea': 21, 'OverallQual': 55, 'OverallCond': 54, 'YearBuilt': 61, 
    'Remodeled': 59, 'GrLivArea': 11, 'BsmtFullBath': 6, 'BsmtHalfBath': 7, 'FullBath': 9, 'HalfBath': 12, 
    'BedroomAbvGr': 0, 'Fireplaces': 8, 'GarageArea': 10, 'MSZoning=RL': 27, 'PavedDrive=Y': 58, 
    'Neighborhood=CollgCr': 34, 'BldgType=1Fam': 1, 'HouseStyle=2Story': 18, 'SalePrice': 60, 
    'Neighborhood=Veenker': 53, 'HouseStyle=1Story': 15, 'Neighborhood=Crawfor': 35, 'Neighborhood=NoRidge': 44, 
    'Neighborhood=Mitchel': 40, 'HouseStyle=1.5Fin': 13, 'Neighborhood=Somerst': 50, 'Neighborhood=NWAmes': 43, 
    'MSZoning=RM': 28, 'Neighborhood=OldTown': 46, 'Neighborhood=BrkSide': 32, 'BldgType=2fmCon': 2, 
    'HouseStyle=1.5Unf': 14, 'Neighborhood=Sawyer': 48, 'Neighborhood=NridgHt': 45, 'Neighborhood=NAmes': 41,}
    
    vocabulary_ which maps the names of the features to their indices. 


"""

#############

"""
4) Preprocessing pipeline 

Ames housing data, let's use the much cleaner and more succinct DictVectorizer approach and put it alongside an 
XGBoostRegressor inside of a scikit-learn pipeline.

"""

# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from  sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)
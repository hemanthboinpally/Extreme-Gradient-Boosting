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
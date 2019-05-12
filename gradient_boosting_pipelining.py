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
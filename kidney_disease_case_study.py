import xgboost as xgb

"""
Kidney disease case study I: Categorical Imputer

sklearn-pandas:

you'll be able to impute missing categorical values directly using the Categorical_Imputer() class in sklearn_pandas,
and the DataFrameMapper() class to apply any arbitrary sklearn-compatible transformer on DataFrame columns, 
where the resulting output can be either a NumPy array or DataFrame.


Dictifier that encapsulates converting a DataFrame using .to_dict("records") without you having to do it 
explicitly (and so that it works in a pipeline). Finally, we've also provided the list of feature names 
in kidney_feature_names, the target name in kidney_target_name, the features in X, and the target in y.

your task is to apply the CategoricalImputer to impute all of the categorical columns in the dataset.
You can refer to how the numeric imputation mapper was created as a template. Notice the keyword arguments 
input_df=True and df_out=True? This is so that you can work with DataFrames instead of arrays.
By default, the transformers are passed a numpy array of the selected columns as input, 
and as a result, the output of the DataFrame mapper is also an array. 
Scikit-learn transformers have historically been designed to work with numpy arrays, not pandas DataFrames, 
even though their basic indexing interfaces are similar.

"""

# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )

"""
OUTPUT:

age        9
    bp        12
    sg        47
    al        46
    su        49
    bgr       44
    bu        19
    sc        17
    sod       87
    pot       88
    hemo      52
    pcv       71
    wc       106
    rc       131
    rbc      152
    pc        65
    pcc        4
    ba         4
    htn        2
    dm         2
    cad        2
    appet      1
    pe         1
    ane        1
"""


############################


"""
2) Kidney disease case study II: Feature Union


Having separately imputed numeric as well as categorical columns, your task is now to use scikit-learn's FeatureUnion
to concatenate their results, which are contained in two separate transformer objects - numeric_imputation_mapper, 
and categorical_imputation_mapper, respectively.
"""

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])


###############################

"""
3) Kidney disease case study III: Full Pipeline

It's time to piece together all of the transforms along with an XGBClassifier to build the full pipeline!

Besides the numeric_categorical_union that you created in the previous exercise, 
there are two other transforms needed: the Dictifier() transform which we created for you, and the DictVectorizer().

"""

# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier())
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))



"""
OUTPUT:

3-fold AUC:  0.998637406769937

"""
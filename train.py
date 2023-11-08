import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# read in data
df = pd.read_csv('dataset.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

cat = ['marital_status', 'application_mode', 'course', 'daytime/evening_attendance'
      ,'previous_qualification','nationality',  "mother's_qualification", "father's_qualification",
       "mother's_occupation", "father's_occupation", 'displaced', 'educational_special_needs',
       'debtor', 'uptodate_fees', 'gender', 'scholarship_holder', 'international']

num = ['application_order',
 'enrollment_age',
 '1stsem_credited',
 '1stsem_enrolled',
 '1stsem_evaluations',
 '1stsem_approved',
 '1stsem_grade',
 '1stsem_without_evaluations',
 '2ndsem_credited',
 '2ndsem_enrolled',
 '2ndsem_evaluations',
 '2ndsem_approved',
 '2ndsem_grade',
 '2ndsem_without_evaluations',
 'unemployment_rate',
 'inflation_rate',
 'gdp']

for s in cat:
    df[s]  = df[s].astype("category")

enrolled_df = df[df['target'] == "enrolled"]
df = df[df['target'] != "enrolled"]

df.target = (df.target == "dropout").astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=5)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_test = df_test.target.values
del df_test["target"]

features_to_drop = ['1stsem_credited', '1stsem_without_evaluations', '2ndsem_credited', '2ndsem_without_evaluations','unemployment_rate',
 'inflation_rate','gdp', ]
df = df.drop(features_to_drop, axis=1)

def logistic_train(df_train, y_train, columns, C=1):
 dicts = df_train[columns].to_dict(orient="records")

 dv = DictVectorizer(sparse=False)
 X_train = dv.fit_transform(dicts)

 model = LogisticRegression(C=C, max_iter=1000)
 model.fit(X_train, y_train)

 return dv, model


def logistic_predict(df, dv, model, columns):
 dicts = df[columns].to_dict("records")
 X = dv.transform(dicts)
 pred = model.predict_proba(X)[:, 1]
 return pred


dv, model = logistic_train(df_full_train, df_full_train.target,df_full_train.columns)
y_pred =logistic_predict(df_test, dv, model,df_test.columns)
auc = roc_auc_score(y_test, y_pred)

print(auc)
output_file = f"dropout1.bin"
with open(output_file, "wb") as f_out:
    pickle.dump((dv,model), f_out)
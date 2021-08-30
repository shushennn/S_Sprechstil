# include the MLD directory here, if the script is not run in this directory
# import sys
# sys.path.append("MYPATHTO/mld/src")

# requires, that additionally xgboost and audb are installed.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import audb
import midlevel_descriptors as mld

# load database
db = audb.load('emodb', version="1.0.1", format='wav')
df_emo = db["emotion"].df

# extract MLDs
fex = mld.MLD()
df_feat = fex.extract_from_index(df_emo, num_jobs=None,
                                 cache_path="/tmp/emodb_mld.pkl")

# emotion classifier
y = df_emo["emotion"].to_numpy()
X = df_feat.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
mod = XGBClassifier()
mod.fit(X_train, y_train)
ans = mod.predict(X_test)
print("accuracy:", accuracy_score(ans, y_test))

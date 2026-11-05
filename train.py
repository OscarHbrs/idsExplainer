from xgboost import XGBClassifier

from data import load_data

X_train, y_train, X_test, y_test, attacks = load_data()

model = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=4)
model.fit(X_train, y_train)
model.save_model("model.ubj")

print("accuracy:", model.score(X_test, y_test))

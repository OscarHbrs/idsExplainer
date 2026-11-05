import time

import pandas as pd
import shap
from xgboost import XGBClassifier

from data import load_data
from names import NAMES

X_train, y_train, X_test, y_test, attacks = load_data()
raw_test = pd.read_csv("data/test.csv", encoding="utf-8-sig")

model = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=4)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)

right = 0
total = 0

while True:
    i = X_test.sample(1).index[0]
    row = X_test.loc[[i]]
    real = y_test[i]

    percents = model.predict_proba(row)[0]
    guess = percents.argmax()

    reasons = explainer.shap_values(row)[0, :, guess]

    total += 1
    if guess == real:
        right += 1

    print("\n" + "=" * 50)
    print(f"connection {i}   right so far: {right}/{total}\n")

    for a in range(len(attacks)):
        bar = "#" * int(percents[a] * 30)
        mark = ""
        if a == guess:
            mark += " <- ai"
        if a == real:
            mark += " <- real"
        print(f"{attacks[a]:15} {percents[a] * 100:5.1f}%  {bar}{mark}")

    print(f"\nwhy {attacks[guess]}?")
    order = abs(reasons).argsort()[::-1]
    for f in order[:3]:
        feature = X_test.columns[f]
        if reasons[f] > 0:
            word = "for"
        else:
            word = "against"
        print(f"  {word:8} {NAMES[feature]}: {raw_test.loc[i, feature]}")

    time.sleep(3)

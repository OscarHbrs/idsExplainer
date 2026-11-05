from xgboost import XGBClassifier

from data import load_data

X_train, y_train, X_test, y_test, attacks = load_data()

model = XGBClassifier()
model.load_model("model.ubj")

guesses = model.predict(X_test)

for a in range(len(attacks)):
    rows = y_test == a
    percent = (guesses[rows] == a).mean() * 100
    bar = "#" * int(percent / 3)
    print(f"{attacks[a]:15} {percent:5.1f}%  {bar}")

print(f"\ntotal           {(guesses == y_test).mean() * 100:5.1f}%")

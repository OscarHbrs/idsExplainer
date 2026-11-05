import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def load_data():
    train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    test = pd.read_csv("data/test.csv", encoding="utf-8-sig")

    words = ["proto", "service", "state"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[words] = encoder.fit_transform(train[words])
    test[words] = encoder.transform(test[words])

    labels = LabelEncoder()
    y_train = labels.fit_transform(train["attack_cat"])
    y_test = labels.transform(test["attack_cat"])

    X_train = train.drop(columns=["id", "attack_cat", "label"])
    X_test = test.drop(columns=["id", "attack_cat", "label"])

    return X_train, y_train, X_test, y_test, labels.classes_

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = ["proto", "service", "state"]
DROP = ["id", "attack_cat", "label"]


def load_data():
    train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    test = pd.read_csv("data/test.csv", encoding="utf-8-sig")

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[CATEGORICAL] = encoder.fit_transform(train[CATEGORICAL])
    test[CATEGORICAL] = encoder.transform(test[CATEGORICAL])

    return (
        train.drop(columns=DROP),
        train["label"],
        test.drop(columns=DROP),
        test["label"],
    )

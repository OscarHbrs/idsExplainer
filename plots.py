import matplotlib.pyplot as plt
import shap
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier

from data import load_data
from names import NAMES

X_train, y_train, X_test, y_test, attacks = load_data()

model = XGBClassifier()
model.load_model("model.ubj")

#rows are the real attack, columns are what the ai guessed
fig, ax = plt.subplots(figsize=(10, 9))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=attacks,
                                      xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax)
plt.title("real vs guessed")
plt.tight_layout()
plt.savefig("pics/confusion.png", dpi=120)
plt.close()

#shap for 3000 random connections, longer bar = feature mattered more overall
rows = X_test.sample(3000, random_state=0)
values = shap.TreeExplainer(model).shap_values(rows)
per_attack = [values[:, :, a] for a in range(len(attacks))]
names = [NAMES[feature] for feature in X_test.columns]

shap.summary_plot(per_attack, rows, feature_names=names, class_names=attacks,
                  plot_type="bar", max_display=10, show=False, plot_size=(13, 6))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("how much it changed the decision on average")
plt.title("what the ai looks at the most")
plt.tight_layout()
plt.savefig("pics/shap.png", dpi=120)

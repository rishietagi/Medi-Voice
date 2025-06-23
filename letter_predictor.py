import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from xgboost import XGBClassifier

df = pd.read_csv("ecapa_embeddings_with_labels.csv")

X = df.drop(columns=["filename", "age", "gender", "generated_name", "first_letter"])
y = df["first_letter"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, "first_letter_label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=7, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

joblib.dump(clf, "first_letter_predictor.pkl")



y_pred = clf.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print(classification_report(y_test_labels, y_pred_labels))

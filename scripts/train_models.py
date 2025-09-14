# Import libraries
import pandas as pd
import xgboost as xgb

# Import these libraries from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_churn_classification(X_train, X_test, y_train, y_test, model_type="all"):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }

    valid_models = list(models.keys())
    results = []

    # Scaling setup (for Logistic Regression only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply scaling to logistic regression only
    def get_data(name):
        return (X_train_scaled, X_test_scaled) if name == "logistic_regression" else (X_train, X_test)

    if model_type == "all":
        for name, model in models.items():
            Xtr, Xte = get_data(name)
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

            results.append({
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            })

        results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        best_model = results_df.loc[results_df['f1_score'].idxmax()]

        print("\nModel Performance Summary:")
        print(results_df)
        print(f"\nBest Model: {best_model['model']} "
              f"(F1-score: {best_model['f1_score']:.4f}, Accuracy: {best_model['accuracy']:.4f})")

        return results_df

    elif model_type in valid_models:
        model = models[model_type]
        Xtr, Xte = get_data(model_type)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"accuracy : {accuracy:.4f}, f1_score : {f1:.4f}")
        print(f"\nClassification Report for {model_type}:")
        print(classification_report(y_test, y_pred))

        return {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred
        }

    else:
        raise ValueError(f"Invalid model_type. Choose from {valid_models} or 'all'.")


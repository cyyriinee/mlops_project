import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def prepare_data(train_path, test_path):
    """Charge et prétraite les données comme dans le notebook."""

    # Charger les fichiers CSV
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Concaténer les datasets pour un traitement uniforme
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Supprimer la colonne "State"
    if "State" in df.columns:
        df = df.drop(columns=["State"])

    # Définir les colonnes catégorielles à encoder
    categorical_cols = ["International plan", "Voice mail plan"]
    encoder = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])

    # Séparer X et y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Séparer en train et test (20% test comme dans le notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un modèle de classification."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy


def save_model(model, filename="models/model.joblib"):
    """Sauvegarde le modèle entraîné."""
    joblib.dump(model, filename)


def load_model(filename="models/model.joblib"):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)

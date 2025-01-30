from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


def main():
    train_path = "churn-bigml-80.csv"
    test_path = "churn-bigml-20.csv"

    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)

    print("🔄 Entraînement du modèle...")
    model = train_model(X_train, y_train)

    print("📊 Évaluation du modèle sur les données de test...")
    evaluate_model(model, X_test, y_test)

    print("💾 Sauvegarde du modèle...")
    save_model(model)

    print("♻️ Chargement du modèle et nouvelle évaluation...")
    loaded_model = load_model()
    evaluate_model(loaded_model, X_test, y_test)


if __name__ == "__main__":
    main()
# Modification pour test CI

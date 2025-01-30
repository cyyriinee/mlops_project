# Variables
PYTHON = python3
VENV = venv
ACTIVATE = . $(VENV)/bin/activate

# Installation des dépendances
install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install -r requirements.txt

# Vérification du code
check:
	$(ACTIVATE) && flake8 --max-line-length=100 --exclude=venv .
	$(ACTIVATE) && black --line-length=100 --exclude=venv .

# Préparation des données
prepare:
	$(ACTIVATE) && $(PYTHON) main.py --prepare

# Entraînement du modèle
train:
	$(ACTIVATE) && $(PYTHON) main.py --train

# Évaluation du modèle
evaluate:
	$(ACTIVATE) && $(PYTHON) main.py --evaluate

# Exécuter les tests
test:
	$(ACTIVATE) && pytest tests/

# Tout exécuter
all: install check prepare train evaluate test


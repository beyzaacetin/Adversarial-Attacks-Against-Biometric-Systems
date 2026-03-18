# Adversarial Attacks Against Biometric Systems + Continuous Authentication

## Computer Security - Graduate Term Project

### Project Overview

This project combines two security research areas:

1. **Continuous Authentication Using Behavioral Biometrics**: A system that continuously verifies users based on typing patterns (keystroke dynamics), mouse movements, and touchscreen gestures.

2. **Adversarial Attacks Against Biometric Systems**: Investigating how adversarial machine learning techniques (FGSM, PGD) can fool the trained biometric authentication models, and implementing defense mechanisms.

### Architecture

```
React Frontend (Demo UI)
    ↓ REST API
FastAPI Backend
    ├── Authentication Models (SVM, RF, LSTM, Autoencoder)
    ├── Adversarial Attack Module (FGSM, PGD, Statistical Mimicry)
    ├── Defense Module (Adversarial Training, Input Smoothing)
    └── Data Pipeline (Keystroke, Mouse, Touch preprocessing)
```

### Datasets

| Modality | Source | Users | Samples |
|----------|--------|-------|---------|
| Keystroke | CMU Benchmark (simulated format) | 51 | 20,400 |
| Mouse | Balabit Challenge (simulated format) | 10 | 60 sessions |
| Touch | Synthetic generator | 15 | 3,000 gestures |

**Note**: When using real datasets, replace CSV paths:
- Keystroke: https://www.cs.cmu.edu/~keystroke/ or Kaggle
- Mouse: https://github.com/balabit/Mouse-Dynamics-Challenge

### Project Structure

```
biometric-auth-project/
├── backend/
│   ├── app/              # FastAPI application
│   ├── data/             # Data generation & preprocessing
│   │   ├── generate_keystroke_data.py
│   │   ├── keystroke_preprocessor.py
│   │   ├── mouse_preprocessor.py
│   │   ├── touch_preprocessor.py
│   │   └── run_pipeline.py
│   ├── models/           # ML authentication models
│   ├── attacks/          # Adversarial attack implementations
│   └── utils/            # Shared utilities
├── frontend/             # React demo interface
│   └── src/
├── datasets/             # Generated/downloaded data
│   ├── keystroke/
│   ├── mouse/
│   └── touch/
├── notebooks/            # Jupyter analysis notebooks
└── docs/                 # Report and presentation
```

### Quick Start

```bash
# 1. Generate datasets and run preprocessing
cd backend/data
python run_pipeline.py

# 2. Start backend (after model training)
cd backend
uvicorn app.main:app --reload

# 3. Start frontend
cd frontend
npm install && npm start
```

### Key Security Concepts

- **Equal Error Rate (EER)**: The point where FAR = FRR
- **False Accept Rate (FAR)**: Probability of accepting an impostor
- **False Reject Rate (FRR)**: Probability of rejecting a genuine user
- **FGSM Attack**: Single-step gradient-based perturbation
- **PGD Attack**: Iterative projected gradient descent
- **Adversarial Training**: Retraining with adversarial examples

### Author
- Beyza - Graduate Student, Computer Security

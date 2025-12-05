# Codex Test Instructions

**Objective:**
Run a comprehensive "Test Pilot" simulation for the JetX Prediction System using the provided standalone script.

**Context:**
This is a complex ensemble machine learning system designed to predict the outcome of the "JetX" game.
- **Core Goal:** Predict if the multiplier will exceed **1.50x**.
- **Models:** It uses 7 different models (CatBoost, k-NN, LSTM, LightGBM, MLP, Transformer, HMM) and a Meta-Learner.
- **Constraints:** The system strictly adheres to chronological data splitting to prevent data leakage.

**Your Task:**
1.  **Locate the File:** Find the file named `standalone_runner.py` in the root directory.
2.  **Execute:** Run this script using Python.
    ```bash
    python standalone_runner.py
    ```
3.  **Monitor Process:**
    - The script will automatically load data from `jetx.db`.
    - It will train ALL models from scratch (this may take a few minutes).
    - It will save these temporary test models to a folder named `models_standalone`.
4.  **Analyze Results:**
    - Once training is complete, the script will run a short simulation on the last 500 games.
    - **Report back on:**
        - Did all models train successfully?
        - What was the final "Profit/Loss" for the simulation?
        - Were there any errors?

**Important Rules:**
- **DO NOT modify** any files in the `jetx_project` directory.
- **DO NOT modify** `app.py`.
- You are only authorized to run `standalone_runner.py` and report the output.
- If `jetx.db` is missing, you cannot proceed. Ensure the database is present.

**Expected Output:**
I expect to see a log confirming the training of each model (1/9 to 9/9) and a final summary table showing the simulation performance.

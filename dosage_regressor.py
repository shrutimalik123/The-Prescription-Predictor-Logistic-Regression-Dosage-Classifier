import math

def logistic_regression_pharmacy_game():
    # 1. Scenario: Intensive Care Unit Clinical Decision Support
    print("--- 🩺 THE PRESCRIPTION PREDICTOR: LOGISTIC REGRESSION SIM 🩺 ---")
    print("Mission: Evaluate patient drug concentration levels to predict toxicity risk.")
    print("Goal: Use a Sigmoid function to map continuous lab inputs into an overdose probability.")

    # 2. Historical Training Reference Points (Continuous Lab Vectors)
    # Concentration Levels (mg/dL) -> Label: 0 = Safe, 1 = Overdose Hazard
    history = [
        {"concentration": 12.0, "label": 0, "status": "Safe Baseline"},
        {"concentration": 18.0, "label": 0, "status": "Safe Baseline"},
        {"concentration": 25.0, "label": 0, "status": "Safe Baseline"},
        {"concentration": 38.0, "label": 1, "status": "Overdose Incident"},
        {"concentration": 45.0, "label": 1, "status": "Overdose Incident"},
    ]

    print("\n--- 🖥️ HISTORICAL ICU PATIENT RECORDS ---")
    for idx, case in enumerate(history):
        print(f"Record {idx+1}: Drug Concentration = {case['concentration']} mg/dL -> [{case['status']}]")

    # 3. Game Inputs: Simulating Trained Parameters (Weight and Bias)
    print("\n--- STEP 1: CALIBRATE THE LINEAR CLASSIFICATION WEIGHTS ---")
    print("Logistic Regression wraps a linear equation inside a Sigmoid gate: z = (Concentration * Weight) + Bias")
    try:
        weight = float(input("Enter Weight parameter (Recommended: 0.4): "))
        bias = float(input("Enter Bias parameter (Recommended: -12.0): "))
    except ValueError:
        weight, bias = 0.4, -12.0

    # 4. Incoming Intercepted Prescription Telemetry
    # A doctor enters an emergency prescription with a high concentration value
    test_concentration = 32.0
    print(f"\n--- 🚨 CLINICAL QUEUE: EMERGENCY ORDER INTERCEPTED ---")
    print(f"Patient Lab Profile -> Detected Drug Concentration: {test_concentration} mg/dL")

    # 5. The Math: Forward Log-Odds Matrix
    print("\n--- 🔄 COMPUTING SIGMOID ACTIVATION ALGEBRA ---")
    
    # Step A: Compute the linear log-odds combination (z)
    z = (test_concentration * weight) + bias
    print(f"Linear Log-Odds Score (z): {z:.4f}")

    # Step B: Pass z through the Sigmoid function to calculate a probability between 0 and 1
    # Sigmoid formula: P = 1 / (1 + e^(-z))
    overdose_probability = 1.0 / (1.0 + math.exp(-z))
    
    print(f"Calculated Overdose Probability P(Overdose): {overdose_probability:.2%}")

    # 6. Probability Thresholding Decision Step
    threshold = 0.50
    print(f"System Safety Gate Threshold: {threshold * 100:.1f}%")

    if overdose_probability >= threshold:
        prediction = 1
        verdict = "❌ CRITICAL WARNING: ORDER BLOCKED DUE TO TOXICITY RISK"
    else:
        prediction = 0
        verdict = "✅ ORDER CONFIRMED: DOSAGE PROFILE CLEARED FOR DISPENSING"

    print(f"\nAutomated System Verdict: {verdict}")

    # 7. Validation Verification
    actual_truth = 1 # A concentration level of 32 mg/dL crosses heavily into toxicity space
    if prediction == actual_truth:
        print("\n🏆 SUCCESS: Your Logistic Regression engine successfully caught the clinical hazard!")
        print("The prescription was flagged and redirected for supervising physician review.")
    else:
        print("\n💥 PATIENT COMPLIANCE EMERGENCY: Critical misclassification! Review your model parameters.")

if __name__ == "__main__":
    logistic_regression_pharmacy_game():

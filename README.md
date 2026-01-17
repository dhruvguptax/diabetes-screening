Diabetes Risk Screening AI

A machine learning based diabetes **risk screening** system built using Python and scikit-learn.  
This project uses basic health parameters to predict whether a person may be at high or low risk of diabetes.

> Disclaimer: This project is for educational purposes only.  
> It is not a medical diagnosis tool and should not be used as a substitute for professional medical advice.

---

Project Overview

Early risk screening can help in timely medical consultation and lifestyle changes.  
This project demonstrates how machine learning can be used to build a simple, interpretable health risk screening system.

The model is trained on a public medical dataset and outputs:
- High Risk of Diabetes
- Low Risk of Diabetes

---

Dataset

- Name:Pima Indians Diabetes Dataset  
- Type: Public medical dataset  
- Features used:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

- Target Variable:
  - `0` → No diabetes
  - `1` → Diabetes

---

Model Used

- Algorithm: Logistic Regression
- Reason: Simple, interpretable, and suitable for binary classification in screening tasks

---

Workflow

1. Load dataset
2. Clean invalid medical values (e.g. zero glucose, BMI)
3. Split data into training and testing sets
4. Train a Logistic Regression model
5. Evaluate model accuracy
6. Predict diabetes risk for new inputs

Results

- Model Accuracy: ~75%
- The model performs reasonably well for a basic screening system.

Tech Stack

- Python
- Google Colab
- pandas
- numpy
- scikit-learn




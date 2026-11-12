# DiaTrust-NX: Model Card

## Model Overview

- **Model Name**: DiaTrust-NX
- **Version**: 2.0.0
- **Model Release Date**: November 2026
- **Architecture**:
  - **Early Stage Model**: $L_2$-Regularized Logistic Regression with Clinical Interaction Feature Engineering and Platt Probability Calibration.
  - **CDC Model**: Balanced Gradient Boosted Decision Trees (CatBoost) with Validation Decision Threshold Optimization.
- **Frameworks**: Python 3.14+, scikit-learn 1.9+, CatBoost 1.2+, PyTorch 2.12+, FastAPI 0.128+, Pydantic 2.12+.
- **License**: Research and Educational Use.

---

## 1. Intended Use and Clinical Context

DiaTrust-NX provides two independent, specialized predictive endpoints designed for distinct healthcare settings:

### 1.1 DiaTrust-NX Early Stage (UCI ID 529)
- **Clinical Setting**: Primary care, urgent care, or outpatient clinics evaluating patients presenting with potential early symptoms of diabetes mellitus.
- **Target Population**: Adults presenting with non-specific or metabolic complaints (frequent urination, persistent thirst, unexplained weight loss, fatigue).
- **Intended Purpose**: Assist clinicians in identifying patients who urgently require formal laboratory diagnostic workups (e.g., Fasting Plasma Glucose, Oral Glucose Tolerance Test, or Glycated Hemoglobin HbA1c).
- **Out of Scope**: Self-diagnosis at home without clinical supervision; replacing laboratory blood tests; differentiating between Type 1, Type 2, LADA, or gestational diabetes.

### 1.2 DiaTrust-NX CDC (UCI ID 891)
- **Clinical Setting**: Public health screening initiatives, health maintenance organizations (HMOs), occupational health programs, and epidemiological risk stratification.
- **Target Population**: General adult population assessed via annual health surveys, community screening kiosks, or health risk assessments (HRAs).
- **Intended Purpose**: High-sensitivity population-level risk triage to identify asymptomatic individuals with elevated lifestyle and cardiometabolic risk profiles who would benefit from preventive lifestyle interventions or primary care follow-ups.
- **Out of Scope**: Acute clinical diagnosis; prescribing anti-diabetic medications; evaluating pediatric patients.

---

## 2. Input Specifications and Schema

### 2.1 Early Stage Input Schema
All inputs are categorical (Yes/No, Male/Female) except `Age` (integer):
- `Age`: Chronological age in years (16–90).
- `Gender`: Patient sex (`Male` or `Female`).
- `Polyuria`: Excessive or abnormally frequent urination (`Yes` / `No`).
- `Polydipsia`: Excessive or persistent thirst (`Yes` / `No`).
- `sudden_weight_loss`: Rapid, unexplained weight loss (`Yes` / `No`).
- `weakness`: Generalized fatigue or lethargy (`Yes` / `No`).
- `Polyphagia`: Excessive hunger or increased appetite (`Yes` / `No`).
- `Genital_thrush`: Recurrent fungal or candidal genital infections (`Yes` / `No`).
- `visual_blurring`: Episodes of blurred or altered vision (`Yes` / `No`).
- `Itching`: Pruritus without primary dermatological cause (`Yes` / `No`).
- `Irritability`: Mood changes, restlessness, or irritability (`Yes` / `No`).
- `delayed_healing`: Slow healing of minor cuts, wounds, or skin abrasions (`Yes` / `No`).
- `partial_paresis`: Muscle weakness or localized numbness (`Yes` / `No`).
- `muscle_stiffness`: Joint or muscle rigidity (`Yes` / `No`).
- `Alopecia`: Significant scalp hair loss (`Yes` / `No`).
- `Obesity`: Clinical obesity indicator (`Yes` / `No`).

### 2.2 CDC Input Schema
Standardized survey indicators from the CDC BRFSS:
- `HighBP`: High blood pressure diagnosis (0 = No, 1 = Yes).
- `HighChol`: High cholesterol diagnosis (0 = No, 1 = Yes).
- `CholCheck`: Cholesterol check within 5 years (0 = No, 1 = Yes).
- `BMI`: Body Mass Index in $\text{kg/m}^2$ (integer, 12–98).
- `Smoker`: Smoked at least 100 cigarettes in lifetime (0 = No, 1 = Yes).
- `Stroke`: History of diagnosed stroke (0 = No, 1 = Yes).
- `HeartDiseaseorAttack`: History of coronary heart disease or myocardial infarction (0 = No, 1 = Yes).
- `PhysActivity`: Physical activity in past 30 days outside regular job (0 = No, 1 = Yes).
- `Fruits`: Consumes fruit 1 or more times per day (0 = No, 1 = Yes).
- `Veggies`: Consumes vegetables 1 or more times per day (0 = No, 1 = Yes).
- `HvyAlcoholConsump`: Heavy alcohol consumption (men >14 drinks/week, women >7 drinks/week; 0 = No, 1 = Yes).
- `AnyHealthcare`: Any health insurance coverage (0 = No, 1 = Yes).
- `NoDocbcCost`: Could not see doctor in past 12 months due to cost (0 = No, 1 = Yes).
- `GenHlth`: Self-rated general health (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor).
- `MentHlth`: Days of poor mental health in past 30 days (0–30).
- `PhysHlth`: Days of poor physical illness/injury in past 30 days (0–30).
- `DiffWalk`: Serious difficulty walking or climbing stairs (0 = No, 1 = Yes).
- `Sex`: Biological sex (0 = Female, 1 = Male).
- `Age`: 13-level age category (1 = 18–24, 2 = 25–29, ..., 9 = 60–64, ..., 13 = 80+).
- `Education`: Highest education completed (1 = Never/kindergarten, 6 = 4+ years college).
- `Income`: Annual household income scale (1 = <$10k, 8 = >$75k).

---

## 3. Verified Performance Metrics

### 3.1 Early Stage Model (Selected: Regularized Logistic Regression)
- **Validation-Optimized Decision Threshold**: $\tau = 0.1682$
- **Probability Calibration**: Platt scaling (Sigmoid)
- **Performance Across Splits**:

| Metric | Training Split | Validation Split | Untouched Test Split |
| :--- | :---: | :---: | :---: |
| Accuracy | 87.57% | 97.70% | **96.55%** |
| Balanced Accuracy | 84.68% | 97.06% | **97.22%** |
| Precision | 84.84% | 96.36% | **100.0%** |
| Recall (Sensitivity) | 97.18% | 100.0% | **94.44%** |
| Specificity | 72.18% | 94.12% | **100.0%** |
| $F_1$ Score | 0.9059 | 0.9815 | **0.9714** |
| ROC-AUC | 0.9790 | 0.9845 | **0.9927** |
| PR-AUC | 0.9888 | 0.9890 | **0.9959** |
| Brier Score | 0.0568 | 0.0729 | **0.0738** |

### 3.2 CDC Model (Selected: Balanced CatBoost)
- **Validation-Optimized Decision Threshold**: $\tau = 0.5136$
- **Class Imbalance Strategy**: Loss function inverse frequency class weighting
- **Performance Across Splits**:

| Metric | Training Split | Validation Split | Untouched Test Split |
| :--- | :---: | :---: | :---: |
| Accuracy | 73.24% | 72.93% | **73.06%** |
| Balanced Accuracy | 75.79% | 75.42% | **75.43%** |
| Precision | 31.67% | 31.36% | **31.18%** |
| Recall (Sensitivity) | 79.32% | 78.87% | **78.70%** |
| Specificity | 72.25% | 71.96% | **72.15%** |
| $F_1$ Score | 0.4526 | 0.4488 | **0.4466** |
| ROC-AUC | 0.8359 | 0.8318 | **0.8336** |
| PR-AUC | 0.4461 | 0.4366 | **0.4344** |
| Brier Score | 0.1737 | 0.1757 | **0.1735** |

---

## 4. Explainability and Key Predictors

### 4.1 Early Stage Top Drivers (TreeSHAP & Logistic Coefficients)
1. **Polyuria**: Strongest positive predictor of acute hyperglycemia.
2. **Polydipsia**: Second strongest predictor, reflecting osmotic dehydration.
3. **Classic_Triad**: Engineered interaction indicating co-occurrence of polyuria, polydipsia, and rapid weight loss.
4. **Gender (Male)**: Modulates baseline symptom severity presentation.
5. **Age & Total Symptoms**: Cumulative symptom burden strongly elevates post-test probability.

### 4.2 CDC Top Drivers (TreeSHAP Summary)
1. **GenHlth (General Health)**: Monotonically increasing risk from excellent (1) to poor (5).
2. **HighBP (Hypertension)**: Key component of metabolic syndrome.
3. **BMI (Body Mass Index)**: Non-linear escalation of risk above $28.0\ \text{kg/m}^2$.
4. **Age Cohort**: Older age brackets (age > 55) exhibit elevated baseline incidence.
5. **Cardiometabolic_Risk**: Composite indicator of co-occurring hypertension, hyperlipidemia, and cardiovascular events.

---

## 5. Limitations, Biases, and Ethical Considerations

1. **Self-Reported Survey Bias**: The CDC dataset relies entirely on telephonic survey self-reports, which are susceptible to recall bias, misclassification of hypertension/cholesterol, and under-reporting of BMI and alcohol intake.
2. **Lack of Glycemic Biomarkers**: Neither dataset contains continuous laboratory biomarkers (e.g., fasting plasma glucose, continuous glucose monitoring traces, or HbA1c values). Predictions reflect symptom or survey risk, not direct glycemic measurement.
3. **Cohort Representation**: The Early Stage dataset (520 cases) was collected from Sylhet Diabetes Hospital, Bangladesh. Clinical presentation thresholds may vary across different geographical and ethnic populations.
4. **Precision-Recall Trade-off in Population Screening**: To achieve 78.70% recall on the CDC cohort, precision is 31.18%. For every 10 positive screening alerts, approximately 3 individuals have diabetes and 7 represent false alerts who require reassurance and standard primary care evaluation.

---

## 6. Mandatory Medical Research Disclaimer

> **IMPORTANT CLINICAL AND REGULATORY NOTICE**:
> 
> The **DiaTrust-NX** system is a research prototype engineered strictly for academic, scientific, and experimental evaluation purposes.
> 
> - **NOT A MEDICAL DEVICE**: DiaTrust-NX has not been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other national or regional medical regulatory agency.
> - **NOT A DIAGNOSTIC TOOL**: Under no circumstances should the outputs, probabilities, risk tiers, or feature attributions provided by DiaTrust-NX be used to establish a clinical diagnosis, replace certified clinical judgment, dictate medical therapy, or defer necessary medical consultation.
> - **MANDATORY LABORATORY CONFIRMATION**: Definitive diagnosis of diabetes mellitus must always be confirmed through certified laboratory testing, including Fasting Plasma Glucose (FPG), 2-hour Oral Glucose Tolerance Test (OGTT), or standardized Glycated Hemoglobin (HbA1c) assays, interpreted by a licensed healthcare professional.


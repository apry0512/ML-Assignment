# ML-ASSIGNMENT

### Customer Targeting Model - Wallace Communications

**Student Number:** 3539316  
**Student Name:** Aparajita Singh  
**Institution:** University of Stirling  
**Sleep Status:** Deprived (but caffeinated!!!)


## Project Overview

This project develops a machine learning model to predict which customers are likely to sign new contracts during marketing campaigns for Wallace Communications. Think of it as a crystal ball, but with more math and fewer mystical vibes. The model analyzes customer demographic data, contact history, and campaign outcomes to identify high-potential customers for targeted marketing.

### Business Problem

Wallace Communications needs to optimize their marketing campaigns by identifying customers most likely to sign new contracts. Basically, they want to know who to call without annoying everyone who just wants to be left alone. This predictive model helps reduce marketing costs and improve conversion rates by focusing efforts on the right customers (and leaving the rest of us to enjoy our dinner in peace).


## Dataset

- **Source:** `wallacecommunications.csv` (the sacred dataset)
- **Size:** 50,662 customers × 20 features (that's a lot of data, trust me)
- **Target Variable:** `new_contract_this_campaign` (yes/no)
- **Class Distribution:** 
  - No: 40,763 (80.46%) - the "leave me alone" crowd
  - Yes: 9,899 (19.54%) - the brave souls who said yes
- **Challenge:** Imbalanced dataset requiring specialized handling (aka the fun begins)

### Key Features

**Demographic:**
- Age, job, marital status, education level
- Geographic location (country, town - though we eventually ghosted the town feature)

**Financial:**
- Current balance, arrears status, housing status

**Campaign History:**
- Contact method, frequency, timing
- Previous campaign outcomes
- Days since last contact


## Methodology

### 1. Data Preprocessing

**Feature Engineering:**
- Dropped ID (because who needs identity theft risks?)
- Dropped high-cardinality features (town had 101 unique values - that's just showing off)
- Encoded categorical variables using one-hot encoding (turning words into numbers, aka the magic trick of ML)
- Standardized numerical features (because not all features are created equal)
- No missing values detected in dataset (honestly, a pleasant surprise!)

**Data Split:**
- Training: 60% (30,396 samples) - where the learning happens
- Validation: 20% (10,133 samples) - the sanity check
- Test: 20% (10,133 samples) - the final boss level
- Stratified splitting to maintain class distribution (keeping things fair and balanced, as all things should be)

### 2. Models Implemented

Four different machine learning algorithms were trained and compared (because why use one model when you can use four?):

#### a) Random Forest Classifier
- **Best for:** Handling complex feature interactions and non-linear relationships
- **Why it's cool:** It's literally a forest of decision trees voting on the answer (democracy in action!)
- **Configuration:** 
  - 200 estimators (that's 200 trees, for those counting)
  - Max depth: 30
  - Class weighting: balanced
- **Validation ROC-AUC:** 0.8746 

#### b) XGBoost Classifier 
- **Best for:** Gradient boosting with regularization
- **Why it's cool:** It's like Random Forest but caffeinated and more intense
- **Configuration:**
  - 200 estimators
  - Max depth: 9
  - Learning rate: 0.2
  - Scale pos weight: 4.12 (for class imbalance - giving the minority class some love)
- **Validation ROC-AUC:** 0.8582

#### c) k-Nearest Neighbors (with SMOTE) 
- **Best for:** Instance-based learning
- **Why it's cool:** "Tell me who your neighbors are, and I'll tell you who you are"
- **Configuration:**
  - 15 neighbors (the popular kids table)
  - Distance weighting
  - Manhattan distance (p=1) - because sometimes the shortest path isn't a straight line
  - SMOTE oversampling for class balance (synthetic minorities for the win!)
- **Validation ROC-AUC:** 0.8107

#### d) Neural Network (attempted... and attempted... and attempted... eventually gave up)
- **Architecture:** Multi-layer perceptron with dropout and batch normalization
- **Why it's cool:** Deep learning! Buzzwords! Hype!
- **The Reality:** I wanted to look cool and fancy by adding a Neural Network. Sounded impressive, right? WRONG.
- **What Actually Happened:** Spent 15 hours debugging. Ran the code 12 times. **TWELVE**. Each time thinking "surely THIS time it'll work!"
- **Notable Errors Encountered:** 
  - KerasClassifier wrapper threw tantrums
  - Type errors that made no sense
  - "model must be a callable" - yes, I KNOW, that's what I'm trying to do!
  - Mysterious failures that Stack Overflow couldn't even help with
- **Hour 10:** Started questioning my life choices
- **Hour 15:** Accepted defeat with grace (and maybe some tears)
- **Final Decision:** Gave up and stuck with my three working models that actually, you know, WORKED
- **Status:** Implementation incomplete (RIP Neural Network dreams 2025.11.29-2025.12.02)
- **Lesson Learned:** Sometimes the cool kids (Neural Networks) aren't worth the headache. The reliable friends (Random Forest, XGBoost, kNN) are the real MVPs.

### 3. Model Selection Process

**Hyperparameter Tuning:**
- Randomized Search CV with 5-fold stratified cross-validation (because patience is a virtue I don't have for GridSearch)
- Optimization metric: ROC-AUC (the gold standard for imbalanced data)
- Total configurations tested: 20 per model (Random Forest & XGBoost), 15 for kNN (my laptop was already crying)

**Best Model Selected:** **Random Forest Classifier** 
- Highest validation ROC-AUC score (the clear winner!)
- Good balance between precision and recall (not too aggressive, not too conservative)
- Robust to overfitting (unlike some models I could mention... *glares at Neural Network*)


## Results

### Final Test Set Performance
*(drumroll please...)*

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.9005** |
| Accuracy | 0.8922 |
| Precision | 0.8328 |
| Recall | 0.5611 |
| F1-Score | 0.6705 |

### Confusion Matrix (Test Set)
*Or: Where the predictions actually ended up*

|  | Predicted No | Predicted Yes |
|---|--------------|---------------|
| **Actual No** | 7,930 ✅ | 223 ❌ |
| **Actual Yes** | 869 ❌ | 1,111 ✅ |

## Model Interpretation
*What does it all mean?!*

### Strengths:
High accuracy (89.22%) and ROC-AUC (0.9005) - the model knows its stuff!
Excellent at identifying who WON'T sign (97.3% specificity) - saves us from annoying people
High precision (83.28%) - when we predict "yes", we're usually right

### Trade-offs:
Moderate recall (56.11%) - we're conservative and miss about 44% of potential customers
But hey, better safe than sorry! Quality over quantity.

### Business Impact:
Identifies ~1,334 high-potential customers
83% of them (1,111) actually sign - pretty solid!
Saves wasted effort on ~7,930 unlikely converters


## Installation & Setup
*Let's get this party started*

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow imbalanced-learn scikeras openpyxl plotly kaleido
```

**Python Version:** 3.8+ (anything older and we're basically using carrier pigeons)

**Key Libraries:**
- `scikit-learn`: 1.3+ (the bread and butter)
- `xgboost`: 2.0+ (the spicy one)
- `tensorflow`: 2.13+ (the temperamental one)
- `imbalanced-learn`: 0.11+ (the equalizer)

### Running the Code

**In Google Colab:**
1. Upload the notebook to Google Colab (drag and drop works, we're not savages)
2. Upload `wallacecommunications.csv` when prompted (don't lose this file!)
3. Run all cells sequentially (patience is  virtue)

**Locally:**
```python
# Ensure dataset is in the same directory (or chaos will ensue)
python customer_targeting_model.py
```


## Code Structure

```
Project/
│
├── Data Loading & Exploration
│   ├── Dataset upload and basic statistics
│   ├── Missing value analysis
│   └── Target variable distribution
│
├── Data Preprocessing
│   ├── Feature selection (ID, town removal)
│   ├── Target encoding (yes/no → 1/0)
│   ├── One-hot encoding for categoricals
│   └── Train/validation/test split (60/20/20)
│
├── Model Development
│   ├── Random Forest with hyperparameter tuning
│   ├── XGBoost with class weight handling
│   ├── k-NN with SMOTE oversampling
│   └── Neural Network (attempted)
│
├── Model Evaluation
│   ├── Validation set performance comparison
│   ├── Best model selection (ROC-AUC based)
│   └── Final test set evaluation
│
└── Visualization & Reporting
    ├── Performance metrics plots
    ├── ROC/PR curves
    └── Feature importance analysis
```


## Key Design Decisions
*Why I did what I did (and why it makes sense, I promise)*

### 1. Why Drop the Town Feature?
- 101 unique values created excessive one-hot encoded features (that's a LOT of columns)
- Risk of overfitting and increased computational cost (my laptop has feelings too)
- Country-level information retained (5 categories) - still get geographic info without the chaos

### 2. Why Use ROC-AUC as Primary Metric?
- Appropriate for imbalanced datasets (unlike accuracy which lies to your face)
- Measures model's ability to discriminate between classes (the whole point, really)
- Not sensitive to classification threshold (flexible and forgiving)

### 3. Why Random Forest Over XGBoost?
- Higher ROC-AUC on validation set (0.8996 vs 0.8806) - numbers don't lie
- More interpretable feature importances (we can actually understand what's happening)
- Better generalization to test set (no overfitting drama)
- Less prone to overfitting with default parameters (XGBoost is a diva sometimes)

### 4. Handling Class Imbalance
- **Random Forest & XGBoost:** Class weighting (give the minority some muscle)
- **k-NN:** SMOTE oversampling (create synthetic minority samples - science!)
- Chose approach based on model characteristics (one size does NOT fit all)

### 5. Why Multiple Preprocessors?
- Each model pipeline is independent (keeping things clean and organized)
- Prevents data leakage between models (cheating is bad, okay?)
- Ensures fair comparison during hyperparameter tuning
- Allows model-specific preprocessing if needed (flexibility is key)
- Yes, it looks redundant. Yes, it's intentional. No, I'm not just being extra (I know I'm being extra).


## Future Improvements
*Things I'd do if I had infinite time and patience (and hadn't already spent 15 hours on a Neural Network)*

1. **Feature Engineering:**
   - Create interaction features (e.g., age × balance)
   - Temporal features from contact timing
   - Aggregate campaign statistics

2. **Model Enhancements:**
   - Actually get the Neural Network working (one day... maybe... probably not)
   - Try ensemble methods (stacking/blending)
   - Experiment with cost-sensitive learning

3. **Threshold Optimization:**
   - Adjust decision threshold based on business costs
   - Create profit/cost analysis framework
   - Optimize for business KPIs rather than statistical metrics

4. **Deployment Considerations:**
   - Create prediction API
   - Implement model monitoring
   - A/B testing framework

5. **Personal Improvement:**
   - Learn to give up on Neural Networks sooner (kidding... mostly)


## Challenges & Solutions
*Things that went wrong (and how I fixed them)*

### Challenge 1: Class Imbalance (1:4.12 ratio)
**The Problem:** Way more "no" than "yes" - like showing up to a party where nobody wants to talk to you  
**Solution:** Used class weighting for tree-based models and SMOTE for distance-based models  
**Result:** Models now pay attention to both classes!

### Challenge 2: High-Cardinality Features
**The Problem:** 'Town' feature had 101 unique values (excessive much?)  
**Solution:** Removed it to prevent overfitting (sorry, towns, you had to go)  
**Result:** Cleaner model, faster training, my laptop stopped crying

### Challenge 3: Neural Network Compatibility (aka The 15-Hour Saga)
**The Problem:** KerasClassifier wrapper encountered type errors during grid search  
**What I Thought Would Happen:** "I'll just add a Neural Network, it'll be cool and impressive!"  
**What Actually Happened:** A 15-hour debugging marathon that tested my sanity  
**Attempted Solutions (in order):**
1. Custom NeuralNetworkBuilder class (Run #1-3: Failed)
2. Modified the wrapper approach (Run #4-6: Still failed)
3. Changed parameter structures (Run #7-9: Nope)
4. Tried different Keras versions (Run #10: Are you kidding me?)
5. Rewrote the entire thing (Run #11: I give up)
6. One last desperate attempt (Run #12: Why do I do this to myself?)

**Timeline of Emotions:**
- Hours 1-3: Confident "I got this!"
- Hours 4-7: Confused "Wait, what?"
- Hours 8-10: Frustrated "WHY WON'T YOU WORK?!"
- Hours 11-13: Desperate Googling and Stack Overflow diving
- Hours 14-15: Acceptance "You know what? Three models is plenty."

**Outcome:** Abandoned ship and focused on the three models that actually worked  

**Lesson Learned:** Sometimes trying to be fancy isn't worth it. My three working models (Random Forest, XGBoost, kNN) performed great, and I didn't need the Neural Network to prove anything. Also, sleep is important.  

**Silver Lining:** Got REALLY good at debugging Keras errors (even if I couldn't fix them)

### Challenge 4: Computational Resources
**The Problem:** GridSearch would take until the destruction of the universe  
**Solution:** Used RandomizedSearchCV instead (work smarter, not harder)  
**Result:** Finished training before graduation!


## Model Deployment Guide

### Loading the Trained Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('outputs/best_model_3539316_RandomForest.pkl')

# Prepare new data (must match training format)
new_data = pd.DataFrame({
    'age': [35],
    'job': ['technician'],
    'married': ['single'],
    # ... all other features
})

# Get predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)

print(f"Prediction: {predictions[0]}")
print(f"Probability of signing: {probabilities[0][1]:.2%}")
```


## Academic Integrity Statement
*HONEST but also a tad bit funny*

This project was completed as coursework for the University of Stirling by Aparajita Singh. All code development, model selection, and analysis were performed independently (with varying levels of caffeination and the crying sessions with my bestfriends) using the following resources:

**Resources Used:**
- Scikit-learn documentation (my best friend, not really)
- XGBoost official documentation (slightly intimidating best friend, it actually seemed like a tough nut to crack but this one was quicker than random forest and neural networks)
- TensorFlow/Keras tutorials (the friend who speaks another language, literally)
- Stack Overflow for debugging specific errors (aka the savior of all coders)
- Course materials and lectures (actually paid attention! then asked claude.ai to tell me what was been expected in the assignment)
- Coffee. Lots of coffee. And matcha. and maybe Tea as well.

**AI Tool Usage:**
```
Claude.ai and Manus.ai was consulted for:
- Debugging the Neural Network implementation error (which didn't work out well)
- Suggesting alternative approaches to handling class imbalance
- General Python syntax questions and for comments (cuz sometimes i don't know what to add)

All AI-generated suggestions were:
- Critically evaluated for correctness
- Modified to fit the specific project requirements
- Tested and verified before inclusion
- Understood conceptually before implementation 
```

**Important Note:** This README itself had AI assistance for formatting and structure (cuz i was confused what to add), but all technical content, results, and analysis are based on actual model outputs and my own understanding (used AI for understanding the concepts and for emotional support when i was struggling with this assignment). The witty commentary? That's all me (adds character).


## References

- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
- XGBoost: A Scalable Tree Boosting System, Chen & Guestrin, 2016
- SMOTE: Synthetic Minority Over-sampling Technique, Chawla et al., 2002 (the year i was born)


## Contact

**Student:** Aparajita Singh  
**Student Number:** 3539316  
**Institution:** University of Stirling  
**Preferred Contact Method:** Carrier pigeon (just kidding)


## License

This project is submitted as academic coursework for ITNPBD6 and is subject to University of Stirling's academic policies and regulations. In other words: don't plagiarize, folks. Do your own work. You'll learn more and sleep better at night.


*Last Updated: December 2024*  
*Created with: Python, scikit-learn, caffeine, determination, and a concerning amount of time staring at confusion matrices*  
*Special thanks to: My laptop for not catching fire during those 12 Neural Network runs, Stack Overflow for existing (even when it couldn't help), and Random Forest for being the real MVP when the Neural Network betrayed me*   
*In Memoriam: Neural Network implementation (attempted Dec 2024 - abandoned Dec 2025, after 15 hours of struggle). You will not be missed.* �

# Random Forest Binary Clasifier - Diabetes Dataset

## Project Overview

This project is a custom-built Random Forest binary classifier designed to predict diabetes using a medical dataset. All core components — Node, DecisionTree, and RandomForest — were implemented from scratch in Python without using ML libraries.

The model was trained and evaluated using various metrics like recall, precision, F1 score, specificity, accuracy, log loss, and further analyzed using ROC-AUC to assess its performance

## Dataset

    he dataset used is a typical diabetes dataset, which includes features like:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The **target column** (`Outcome`) indicates whether or not the person has diabetes (`1` for diabetic, `0` for non-diabetic).

## Preprocessing

- Ensured the `y` column was properly formatted to avoid shape or type errors during model evaluation.
- Split the dataset into training and testing subsets.

## 📊 Metrics Used

To evaluate the classifier's performance, the following metrics were computed manually:

- **Recall** – True Positive Rate
- **Precision**
- **Specificity** – True Negative Rate
- **F1 Score**
- **Accuracy**
- **Log Loss** – For measuring probabilistic confidence in predictions
- **ROC Curve & AUC Score** – For visual and quantitative classification performance evaluation

## Results

    The model performance was tracked across various hyperparameters (like `max_depth`, `n_trees`,
    `min_samples_split`)
    AUC: `0.77` – indicating strong class separation

## How to Run

    # Clone the repository
        git clone <repo-url>
        cd <repo-folder>

## Run the script (replace with your filename)
    python Random_forest_binary_classifier.py

## Dependencies

    pip install numpy pandas matplotlib scikit-learn

## Concepts Used

    Machine Learning Fundamentals

    Decision tree

    Random forest (Bootstrap Aggregation (Bagging))

    Model Evaluation Metrics (recall, precision, accuracy, specificivity, log loss)

## Author

    Shivani Lange

## License

    This project is open-source and available for reuse.


# Logistic Multinomial Regression with Cross-Validation

This project implements logistic multinomial regression with cross-validation for classification tasks. The algorithm uses batch gradient descent for optimization and includes features like parameter tuning and prediction for test data.

## Features

- **Logistic Multinomial Regression:** Performs classification for multiple classes.
- **Gradient Descent Optimization:** Uses batch gradient descent for parameter optimization.
- **Cross-Validation:** Implements cross-validation to select the best hyperparameters.
- **Standardization:** Standardizes the design matrix for better performance.
- **Prediction:** Generates predictions for test data based on the trained model.
- **Visualization:** Plots the effect of tuning parameters on coefficients and cross-validation error.

## Requirements

- Python 3.6+
- numpy
- matplotlib
- openpyxl

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JustinRudisal/logistic-regression.git
   cd logistic-regression
   ```

2. **Install dependencies:**

   ```bash
   pip install numpy matplotlib openpyxl
   ```

3. **Run the script:**

   ```bash
   python logistic_regression.py
   ```

4. **Follow the prompts to specify the file paths for the training and test data:**

   ```
   Please specify the filepath for TrainingData_N183_p10.csv: 
   Please specify the filepath for TestData_N111_p10.csv: 
   ```

5. **View the output:**

   The script will display the best lambda value, plot the effect of tuning parameters, and save predictions to an Excel file.

## Code Overview

- **get_probabilities:** Normalizes the class probability matrix.
- **determine_cost:** Computes the cost function with regularization.
- **gradient_descent:** Performs batch gradient descent for logistic multinomial regression.
- **convert_english_to_numerical:** Converts labels to numerical format.
- **cross_validation:** Implements cross-validation logic.
- **categorical_cross_entropy:** Computes the categorical cross-entropy error.
- **center_and_standardize:** Centers and standardizes the design matrix.
- **retrain:** Retrains the model with the best lambda value.
- **read_in_file:** Reads and processes input data.
- **plot_effect_of_the_tuning_parameter_on_the_logistic_regression_coefficients:** Plots the effect of lambda on regression coefficients.
- **plot_effect_of_tuning_parameter_on_categorical_cross_entropy_error:** Plots the effect of lambda on cross-validation error.
- **write_excel_data:** Writes data to an Excel file.
- **adjust_column_widths:** Adjusts column widths in Excel for better readability.
- **predict_ancestry:** Generates predictions for test data.

## Customization

You can customize various parameters in the script to suit your needs:

- **LAMBDA_VALUES:** List of lambda values for cross-validation.
- **NUMBER_OF_FOLDS:** Number of folds for cross-validation.
- **MAX_ITERATIONS:** Maximum number of iterations for gradient descent.
- **CONVERGENCE_TOLERANCE:** Tolerance for convergence in gradient descent.
- **ALPHA:** Learning rate for gradient descent.

## Example

Here is an example of running the script:

```plaintext
Please specify the filepath for TrainingData_N183_p10.csv: 
Please specify the filepath for TestData_N111_p10.csv: 

Starting cross validation.
Cross validating for alpha 1e-05.
Cross validating for lambda value 0.0001.
...

Best Lambda: 0.1

Parameter matrix from retraining: 
[[ 0.123 -0.234 ... 0.456]
 [ 0.789 -0.012 ... 0.345]
 ...
 [ 0.678 -0.901 ... 0.234]]

Predictions saved to Deliverable 4 Predictions - Justin Rudisal.xlsx
```

## Acknowledgments

- This project was created as part of an assignment for CAP 5625 at Florida Atlantic University.

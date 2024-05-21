# Author: Justin Rudisal
# Assignment 4 CAP 5625
import csv
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook

NAMES = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]
LAMBDA_VALUES = [10**exp for exp in range(-4, 5)] 
NUMBER_OF_FOLDS = 5
MAX_ITERATIONS = 10000
CONVERGENCE_TOLERANCE = 1e-6
ALPHA = 0.00001


def get_probabilities(matrix):
    normalized_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return normalized_matrix / np.sum(normalized_matrix, axis=1, keepdims=True)


def determine_cost(design_matrix, response_vector, parameter_matrix, lambda_value):
    size = design_matrix.shape[0]
    probabilities = get_probabilities(np.dot(design_matrix, parameter_matrix))

    # I learned the hard way that this is needed in order to avoid log(0)
    log_probability = -np.sum(response_vector * np.log(probabilities + 1e-5))
    cost = (1/size) * log_probability

    reg_cost = cost + (lambda_value/(2*size)) * np.sum(np.square(parameter_matrix[1:]))
    return reg_cost


def gradient_descent(design_matrix, response_vector, lambda_value, number_of_classes):
    """
    Using algorithm 1 from our professors provided assignment guide to perform batch gradient descent for logistic multinomial regression.
    """
    number_of_features = design_matrix.shape[1]

    # Step 3) Initialize the parameter matrix
    parameter_matrix  = np.zeros((number_of_features, number_of_classes))

    total_samples = design_matrix.shape[0]
    cost_history = [0] * MAX_ITERATIONS

    # Step 8) Repeat Steps 4 to 7 for 10,000 iterations
    for iteration in range(MAX_ITERATIONS):
        # Step 4) Compute unnormalized class probability matrix
        unnormalized_class_probabilities = np.dot(design_matrix, parameter_matrix )

        # Step 5) Compute normalized class probability matrix
        class_probabilities = get_probabilities(unnormalized_class_probabilities)

        # Steps 6 and 7) Update the parameter matrix
        prediction_error = class_probabilities - response_vector
        gradient = np.dot(design_matrix.T, prediction_error) / total_samples
        parameter_matrix -= ALPHA * (gradient + (lambda_value / total_samples) * parameter_matrix)

        current_cost = determine_cost(design_matrix, response_vector, parameter_matrix, lambda_value)
        cost_history[iteration] = current_cost
        
        if iteration > 0 and abs(cost_history[iteration] - cost_history[iteration - 1]) < CONVERGENCE_TOLERANCE:
            break

    # Step 9) Set the last updated parameter matrix
    return parameter_matrix


def convert_english_to_numerical(vector, number_of_labels):
    """
    Convert an array of labels to values that are udnerstandable in a bianry format.
    """
    size = vector.size
    numerical = np.zeros((size, number_of_labels))
    numerical[np.arange(size), vector] = 1
    return numerical


def cross_validation(design_matrix, converted_response_vector, lambda_values, number_of_folds, number_of_classes):
    """
    Cross validation logic that is almost exactly the same as what I wrote for the previous assignments. 
    """
    print("Starting cross validation.")
    number_of_observations = design_matrix.shape[0]
    categorical_cross_entropy_error_values = []
    parameter_matrix_values = []

    data_in_each_fold = np.arange(number_of_observations) % number_of_folds

    print(f"Cross validating for alpha {ALPHA}.")
    for lambda_value in lambda_values:
        print(f"Cross validating for lambda value {lambda_value}.")
        categorical_cross_entropy_error_per_fold = []
        parameter_matrices_per_fold = []

        for fold_number in range(number_of_folds):
            # Set up the training and validation data 
            training_design_matrix = design_matrix[data_in_each_fold != fold_number]
            validation_design_matrix = design_matrix[data_in_each_fold == fold_number]
            training_response_vector_converted = converted_response_vector[data_in_each_fold != fold_number]
            validation_response_vector_converted = converted_response_vector[data_in_each_fold == fold_number]

            # Center and standardize the training data
            standardized_training_design_matrix, centering_values, standardization_values = center_and_standardize(training_design_matrix)

            # Apply the same centering and standardization values to the validation data
            validation_design_matrix = (validation_design_matrix - centering_values) / standardization_values

            # Do algorithm 1 (similar concept to our other assignments)
            parameter_matrix = gradient_descent(standardized_training_design_matrix, training_response_vector_converted, lambda_value, number_of_classes)
            parameter_matrices_per_fold.append(parameter_matrix)

            # Categorical cross entropy error calculation for the current fold
            predicted_response_vector = get_probabilities(np.dot(validation_design_matrix, parameter_matrix))
            categorical_cross_entropy_error = categorical_cross_entropy(validation_response_vector_converted, predicted_response_vector)
            categorical_cross_entropy_error_per_fold.append(categorical_cross_entropy_error)

        # Average categorical cross entropy across folds to get the CVE
        categorical_cross_entropy_error = sum(categorical_cross_entropy_error_per_fold) / number_of_folds
        categorical_cross_entropy_error_values.append(categorical_cross_entropy_error)

        # Average parameter vector across folds
        parameter_matrix = sum(parameter_matrices_per_fold) / number_of_folds
        parameter_matrix_values.append(parameter_matrix)  

    return categorical_cross_entropy_error_values, parameter_matrix_values


def categorical_cross_entropy(validation_response_vector_converted, predicted_response_vector):
    # Need a small value like 1e-5 to prevent log(0) which causes issue
    epsilon = 1e-5
    cross_entropy = -np.sum(validation_response_vector_converted * np.log(predicted_response_vector + epsilon), axis=1)
    return np.mean(cross_entropy)


def center_and_standardize(design_matrix):
    centering_values = np.mean(design_matrix, axis=0)
    standardization_values = np.std(design_matrix, axis=0)

    # Handle the zero standard deviation situation that was giving me runtime warnings and causing issues :(
    standardization_values[standardization_values == 0] = 1

    # Standardize the design matrix 
    standardized_design_matrix = (design_matrix - centering_values) / standardization_values
    return standardized_design_matrix, centering_values, standardization_values


def retrain(design_matrix, response_vector, best_lambda_value, number_of_classes):

    # Center and standardize
    standardized_training_design_matrix, centering_values, standardization_values = center_and_standardize(design_matrix)

    response_vector_converted = convert_english_to_numerical(response_vector, number_of_classes)
    
    # Run the algorithm 1 again on it
    retrained_parameter_matrix = gradient_descent(standardized_training_design_matrix, response_vector_converted, best_lambda_value, number_of_classes)
    
    print("Parameter matrix from retraining: \n" + str(retrained_parameter_matrix))
    return retrained_parameter_matrix, centering_values, standardization_values


def read_in_file():
    print("Reading in the input file.")
    filepath = input("Please specify the filepath for TrainingData_N183_p10.csv: ")

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = list(reader)
    
    # Read the rest of the data
    data = np.array(data, dtype=object)
    
    # The features are all the columns except the last one
    design_matrix = data[:, :-1].astype(float)
    
    # The ancetry classes are in the last column
    response_vector = data[:, -1]
    
    # Map ancestry classes 
    classes, converted_response_vector = np.unique(response_vector, return_inverse=True)
    
    return design_matrix, converted_response_vector, classes


def plot_effect_of_the_tuning_parameter_on_the_logistic_regression_coefficients(design_matrix, parameter_matrix, classes):
    number_of_features = design_matrix.shape[1] - 1 
    number_of_classes = len(classes)
    for class_idx in range(number_of_classes):
        plt.figure(figsize=(10, 6))
        for feature_idx in range(1, number_of_features + 1): 
            feature_coefficients = [param[feature_idx, class_idx] for param in parameter_matrix]
            plt.plot(np.log10(LAMBDA_VALUES), feature_coefficients, label=NAMES[feature_idx - 1])

        plt.xticks(np.log10(LAMBDA_VALUES), [f"{np.log10(lv):.2f}" for lv in LAMBDA_VALUES])
        plt.xlabel("Log10 of Tuning Parameter (Lambda)")
        plt.ylabel(f"Inferred Coefficients for {classes[class_idx]} Ancestry")
        plt.title(f"Effect of Tuning Parameter on Coefficients for {classes[class_idx]} Ancestry")
        plt.legend()
        plt.grid(True)
        plt.show()
    write_logistic_regression_coefficients_plots_to_excel(parameter_matrix, "Deliverable 1 Plots - Justin Rudisal.xlsx")


def plot_effect_of_tuning_parameter_on_categorical_cross_entropy_error(cve_values):
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(LAMBDA_VALUES), cve_values, marker="o", linestyle="-", color="b")
    plt.xticks(np.log10(LAMBDA_VALUES), [f"{np.log10(lv):.2f}" for lv in LAMBDA_VALUES])
    plt.xlabel("Log10 of Tuning Parameter (Lambda)")
    plt.ylabel("Cross-Validation Error")
    plt.title("Tuning Parameter Effect on Cross Validation Error")
    plt.grid(True)
    plt.show()
    write_cve_plots_to_excel(cve_values, "Deliverable 2 Plots - Justin Rudisal.xlsx")


def write_cve_plots_to_excel(plots, excel_filename):  
    header = ["Lambda Values (X-Axis)", "Cross-Validation Error Values (Y-Axis)"]
    write_excel_data(header, plots, excel_filename)
    

def write_logistic_regression_coefficients_plots_to_excel(plots, excel_filename):
    header = ["Lambda Values (X-Axis)"] + [name + " (Y-Axis)" for name in NAMES]
    write_excel_data(header, plots, excel_filename)


def write_excel_data(header, plots, excel_filename):
    """
    Writes data to an Excel file, skipping the first element of each row (intercept term).
    """
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = f"Alpha {ALPHA}"
    worksheet.append(header)
    for lambda_value, parameter_values in zip(LAMBDA_VALUES, plots):
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()
        elif isinstance(parameter_values, float):
            parameter_values = [parameter_values]
        parameter_values_str = [str(val) for val in parameter_values[:]]
        row = [lambda_value] + parameter_values_str
        worksheet.append(row)
    adjust_column_widths(worksheet)
    workbook.save(excel_filename)


def adjust_column_widths(worksheet):
    """
    I'm a bit OCD when it comes to code I create and the outputs it generates... and it really annoyed me
    that the excel columns weren't autoadjusting their width because of the header names. This method fixes that.
    """
    for column in worksheet.columns:
        max_length = 0
        column = [cell for cell in column]
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        width = (max_length + 2)
        worksheet.column_dimensions[cell.column_letter].width = width


def predict_ancestry(retrained_parameter_matrix, centering_values, standardization_values, classes):
    """
    This will take in a second input file and output the predictions needed for deliverable 4. 
    """
    filepath = input("Please specify the filepath for TestData_N111_p10.csv: ")
    print("Predicting Ancestry for the second file.")

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        data = list(reader)
    data = np.array(data, dtype=object)
    input_data = data[:, :-1].astype(float)

    # Add a column of ones
    input_data = np.hstack([np.ones((input_data.shape[0], 1)), input_data])

    # Do the standardization
    input_data[:, 1:] = (input_data[:, 1:] - centering_values[1:]) / standardization_values[1:]

    # Get the expected ancestries from the last column
    expected_ancestries = data[:, -1]

    # Calculate probabilities for each individual
    probabilities = get_probabilities(np.dot(input_data, retrained_parameter_matrix))
    
    # Find the most probable ancestry class for each individual
    predicted_classes = np.argmax(probabilities, axis=1)
    
    # Do the same kind of excel magic as above
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Ancestry Predictions'
    header = ['Individual'] + [f'Probability_{ancestry}' for ancestry in classes] + ['Known Ancestry', 'Predicted Ancestry']
    sheet.append(header)
    for index, (prob, expected, label) in enumerate(zip(probabilities, expected_ancestries, predicted_classes)):
        row = [f'Individual {index+1}'] + list(prob) + [expected, classes[label]]
        sheet.append(row)
    adjust_column_widths(sheet)
    workbook.save("Deliverable 4 Predictions - Justin Rudisal.xlsx")
    print(f"Predictions saved to Deliverable 4 Predictions - Justin Rudisal.xlsx")


def main():
    # Reading in the file to get the data and then setting up the design matrix
    design_matrix, response_vector, classes = read_in_file()

    # Add a column of ones to design_matrix for the intercept term
    design_matrix = np.hstack([np.ones((design_matrix.shape[0], 1)), design_matrix])

    converted_response_vector = convert_english_to_numerical(response_vector, len(classes))

    # Cross validation logic
    cve_values, parameter_matrix = cross_validation(design_matrix, converted_response_vector, LAMBDA_VALUES, NUMBER_OF_FOLDS, len(classes))

    # Find which lambda gave the smallest mean squared error during cross validation
    best = np.argmin(cve_values)
    best_lambda = LAMBDA_VALUES[best]
    print(f"Best Lambda: {best_lambda}")

    # Plotting logic
    plot_effect_of_the_tuning_parameter_on_the_logistic_regression_coefficients(design_matrix, parameter_matrix, classes)
    plot_effect_of_tuning_parameter_on_categorical_cross_entropy_error(cve_values)

    # Now that we know the best lambda value, let's retrain with it
    retrained_parameter_matrix, centering_values, standardization_values = retrain(design_matrix, response_vector, best_lambda, len(classes))

    # Predict Ancestry for deliverable 4
    predict_ancestry(retrained_parameter_matrix, centering_values, standardization_values, classes)


if __name__ == "__main__":
    main()
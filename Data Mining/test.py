from random import randint
import random
import numpy as np
import pandas as pd
import sys
import subprocess

def root_mean_square_error(predicted_values, true_values):
    rmse = 0
    for i in range(0, len(predicted_values)):
        rmse += (float(predicted_values[i]) - float(true_values[i])) ** 2
    return np.sqrt(rmse / len(predicted_values))
    
#Spearman’s correlation
def rank_correlation(predicted_values, true_values):
    # Sort the predicted values
    predicted_values.sort()
    # Get the rank of the true values
    true_values_rank = []
    for i in range(0, len(true_values)):
        true_values_rank.append(float(true_values[i][2]))
    # Get the rank of the predicted values
    predicted_values_rank = []
    for i in range(0, len(predicted_values)):
        predicted_values_rank.append(float(predicted_values[i]))
    # Get the rank correlation
    rank_corr = 0
    for i in range(0, len(predicted_values)):
        rank_corr += (float(predicted_values_rank[i]) - float(true_values_rank[i])) ** 2
    return np.sqrt(rank_corr / len(predicted_values))


def test_success_rate(patient_id, condition_id, n_samples):
    # Import csv
    df = pd.read_csv("outputs/output_" + patient_id + "_" + condition_id + ".csv").set_index("ids")

    # Selecting n non 0 success rates as the test set
    selected_values =  []
    y = []
    while len(selected_values) != n_samples:
        random_therapy = df.sample(n=1).index.to_list()
        random_condition = random.sample(df.columns.to_list(), 1)
        success_rate = df.loc[random_therapy[0], random_condition[0]]
        if success_rate.all() != 0:
            selected_values.append([random_therapy[0], random_condition[0], success_rate])
            y.append(success_rate)

    print(selected_values)

    # Use find_success_rate_of_therapy(...) to get the predicted value
    y_pred = []
    for i in range(0, len(selected_values)):
        print('py code.py datasetB.json ' +  str(selected_values[i][0]) + ' ' +  str(condition_id) + ' ' + str(selected_values[i][1]))
        proc = subprocess.Popen(['py', 'code.py',  'datasetB.json', str(selected_values[i][0]), str(condition_id), str(selected_values[i][1])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = proc.communicate()[0].strip().decode('utf-8')
        print("Result: " + output)
        y_pred.append(output)

    # Print Results
    for i in range(len(y_pred)):
        print(str(y_pred[i]) + " - " + str(y[i]))

    # Root-mean-square-error
    rmse = root_mean_square_error(y_pred, y)
    print("RMSE: " + str(rmse))

    # Rank correlation
    rank_corr = rank_correlation(y_pred, selected_values)
    print("Rank correlation: " + str(rank_corr))

if __name__ == "__main__":
    # Rquirement: csv file must exist!

    test_success_rate("6", "Cond248", 10)
from random import randint
from numpy.linalg import cond
from scipy.sparse.extract import find
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import json
import datetime
import sys
import re
import pandas as pd

class Condition:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type

# New class to support the extra attributes between class "Condition" and the asked attributes in patent.conditions
class Patient_condition: 
    def __init__(self, id, diagnosed, cured, kind):
        self.id = id
        self.diagnosed = diagnosed 
        self.cured = cured
        self.kind = kind

class Therapy:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type

# New class to support the extra attributes between class "Terapy" and the asked attributes in "trials"
class Patient_therapy:
    def __init__(self, id, start, end, condition, therapy, successful):
        self.id = id
        self.start = start
        self.end = end
        self.condition = condition
        self.therapy = therapy
        self.successful = successful
  
class Patient:
    def __init__(self, id, name, conditions, trials):
        self.id = id
        self.name = name
        self.conditions = conditions
        self.trials = trials

def load_demo_patients():
    c1 = Condition("Cond1", "High Blood Pressure", "Blood Pressure")
    c2 = Condition("Cond2", "Heart Arrhythmia", "Heath Condition")
    t1 = Therapy("Th1", "Acetoxybenzoic Acid (Aspirin)", "Acetoxybenzoic Acid (Aspirin)")
    t2 = Therapy("Th2", "Cough Syrup Quibron", "Sugar")
    p_c1 = Patient_condition("pc1", 20210915, 20210915, "Cond1")
    p_c2 = Patient_condition("pc2", 20210602, 20210930, "Cond2")
    p_c3 = Patient_condition("pc3", 20210603, 20210905, "Cond2")
    p_t1 = Patient_therapy("tr1", 20210915, 20211215, "pc1", "Th1", 10)
    p_t2 = Patient_therapy("tr2", 20210102, 20210130, "pc2", "Th2", 100)
    p_t3 = Patient_therapy("tr3", 20210102, 20210130, "pc2", "Th3", 90)
    patient1 = Patient(1, "John", [p_c1.__dict__, p_c2.__dict__], [p_t1.__dict__, p_t2.__dict__])
    patient2 = Patient(1, "Peter", [p_c3.__dict__], [p_t3.__dict__])

    conditions = [condition.__dict__ for condition in [c1, c2]]
    therapies = [therapy.__dict__ for therapy in [t1, t2]]
    patients = [patient.__dict__ for patient in [patient1, patient2]]
    return conditions, therapies, patients

def export_demo_dataset(file, conditions, therapies, patients):
    with open(file, 'w') as outfile:
        json.dump({"Conditions": conditions, "Therapies": therapies, "Patients": patients}, outfile)
    outfile.close()

def export_dataset(file, conditions, therapies, patients):
    printable_patients = []
    for patient in patients:
        printable_patients.append(Patient(patient.id, patient.name, [condition.__dict__ for condition in patient.conditions], [trial.__dict__ for trial in patient.trials]))

    with open(file, 'w') as outfile:
        json.dump({"Conditions": [condition.__dict__ for condition in conditions], "Therapies": [therapy.__dict__ for therapy in therapies], "Patients": [patient.__dict__ for patient in printable_patients]}, outfile)
    outfile.close()

def import_dataset(file):
    conditions = []
    therapies = []
    patients = []
    patient_conditions = []
    patient_therapies = []
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        for condition in data["Conditions"]:
            obj = json.loads(json.dumps(condition), object_hook=lambda d: Condition(**d))
            conditions.append(obj)
        for therapy in data["Therapies"]:
            obj = json.loads(json.dumps(therapy), object_hook=lambda d: Therapy(**d))
            therapies.append(obj)
        for patient in data["Patients"]:
            current_patient_conditions = []
            current_patient_therapies = []
            for patient_condition in patient["conditions"]:
                for attribute in patient_condition:
                    if attribute == "id":
                        id = patient_condition[attribute]
                    if attribute == "diagnosed":
                        diagnosed = patient_condition[attribute]
                    if attribute == "cured":
                        cured = patient_condition[attribute]
                    if attribute == "kind":
                        kind = patient_condition[attribute]
                p_c = Patient_condition(id, diagnosed, cured, kind)
                patient_conditions.append(p_c)
                current_patient_conditions.append(p_c)
            for patient_therapy in patient["trials"]:
                for attribute in patient_therapy:
                    if attribute == "id":
                        id = patient_therapy[attribute]
                    if attribute == "condition":
                        condition = patient_therapy[attribute]
                    if attribute == "end":
                        end = patient_therapy[attribute]
                    if attribute == "start":
                        start = patient_therapy[attribute]
                    if attribute == "therapy":
                        therapy = patient_therapy[attribute]
                    if attribute == "successful":
                        successful = patient_therapy[attribute]
                p_c = Patient_therapy(id, start, end, condition, therapy, successful)
                patient_therapies.append(p_c)
                current_patient_therapies.append(p_c)
            p = Patient(patient["id"], patient["name"], current_patient_conditions, current_patient_therapies)
            patients.append(p)
    json_file.close()
    return conditions, therapies, patients
 
def generate_condtions(soucre_file):
    values = []
    with open(soucre_file, 'r') as conditions:
        i = 0
        for condition in conditions.read().split("\n"):
            cond = condition.split(' ')
            if len(cond) > 1:
                c = Condition("Cond" + str(i), condition, cond[len(cond) - 1])
            else:
                c = Condition("Cond" + str(i), condition, cond[0])
            values.append(c)
            i += 1
    conditions.close()
    return values
    
def generate_therapies(soucre_file):
    values = []
    with open(soucre_file, 'r') as conditions:
        i = 0
        for therapy in conditions.read().split("\n"):
            the = therapy.split(' ')
            t = Therapy("Th" + str(i), therapy, the[0])
            values.append(t)
            i += 1
    conditions.close()
    return values

def get_random_dates_with_interval(beginning=20000101, fixed_start_date=False):
    date = datetime.datetime.strptime(str(beginning), "%Y%m%d")
    if fixed_start_date:
        start_date = datetime.date(date.year, date.month, date.day)
    else:
        start_date = datetime.date(randint(date.year, 2021), randint(date.month, 12), randint(date.day, 28))
    end_date = datetime.date(2021, 12, 31)
    random_days = randint(0, (end_date - start_date).days)
    random_date = start_date + datetime.timedelta(days = random_days)
    return start_date.strftime("%Y%m%d"), random_date.strftime("%Y%m%d")

def get_random_condition(conditions, threated_conditions = []):
    conditions = [condition for condition in conditions if condition not in threated_conditions]
    return conditions[randint(0, len(conditions) - 1)]

def get_random_therapy(therapies):
    return therapies[randint(0, len(therapies) - 1)]

def generate_patients(source_file, conditions, therapies):
    values = []
    condition_id_counter = 0
    therapy_id_counter = 0
    with open(source_file, 'r', encoding="utf-8") as patient_names:
        i = 0
        for name in patient_names.read().split("\n"):
            n_patient_condition = randint(1, 3)
            n_patient_therapy = randint(1, n_patient_condition)
            patient_conditions = []
            patient_therapies = []
            for j in range(n_patient_condition):
                start_date, end_date = get_random_dates_with_interval()
                p_c = Patient_condition("pc" + str(condition_id_counter), start_date, None, get_random_condition(conditions).__dict__["id"])
                patient_conditions.append(p_c)
                condition_id_counter += 1

            threated_conditions = []
            for k in range(n_patient_therapy):
                condition = get_random_condition(patient_conditions, threated_conditions)
                threated_conditions.append(condition)

                #Generate therapies maximum 3 times until the condition is cured
                start_date_of_condition = condition.__dict__["diagnosed"]
                start_date, end_date = get_random_dates_with_interval(start_date_of_condition, True)

                same_condition_counter = 0
                max_number_of_therapies_per_condition = randint(1, 3)
                while condition.__dict__["cured"] == None:
                    p_t = Patient_therapy("tr" + str(therapy_id_counter), start_date, end_date, condition.__dict__["id"], get_random_therapy(therapies).__dict__["id"], randint(0, 100))

                    #If Therapy Worked (<75%), cure the condition
                    if p_t.__dict__["successful"] >= 75:
                        for cond in patient_conditions:
                            if cond.__dict__["id"] == p_t.__dict__["condition"]:
                                cond.__dict__["cured"] = p_t.__dict__["end"]
                    patient_therapies.append(p_t)
                    therapy_id_counter += 1
                    same_condition_counter += 1
                    start_date, end_date = get_random_dates_with_interval(end_date, True)

                    # Break if the condition is cured or the maximum number of therapies per condition is reached
                    if same_condition_counter == max_number_of_therapies_per_condition: break
                    
            p = Patient(i, name.capitalize(), patient_conditions, patient_therapies)
            values.append(p)
            i += 1
    patient_names.close()
    return values

def generate_new_dataset(dataset_name):
    conditions = generate_condtions("source_data/conditions.txt")
    therapies = generate_therapies("source_data/therapies.txt")
    patients = generate_patients("source_data/names_lot.txt", conditions, therapies)
    export_dataset(dataset_name, conditions, therapies, patients)

def find_condition_by_id(conditions, id):
    for condition in conditions:
        if condition.__dict__["id"] == id:
            return condition

def find_condition_by_kind(conditions, id):
    for condition in conditions:
        if condition.__dict__["kind"] == id:
            return condition
    return None

def check_if_condition_was_attempoted_to_cure(patient_therapies, condition_id):
    for patient_therapy in patient_therapies:
        if patient_therapy.__dict__["condition"] == condition_id:
            return True
    return False

def patient_analysis(patient, condtion):
    condtion = condtion.__dict__
    patient_conditions = patient.__dict__["conditions"]
    patient_therapies = patient.__dict__["trials"]
    patient_conditions_kinds = [condition.__dict__["kind"] for condition in patient_conditions]
    patient_therapy_ids = [therapy.__dict__["therapy"] for therapy in patient_therapies]

    print("==== Patient Analysis ====")
    print("Name: " + patient.__dict__["name"] + " (id: " + str(patient.__dict__["id"]) + ")\n")
    print("Conditions of the Patient: ")
    for p_c in patient_conditions:
        print(p_c.__dict__["kind"] + " | Cured? -> " + str(p_c.__dict__["cured"] != None))
    print("\nTherapies of the Patient: ")
    print(patient_therapy_ids)
    print("\nRequested condition: " + condtion["name"] + " (id: " + condtion["id"] + ")" + " | Type: " + condtion["type"])

    # Check if patient has condition
    patient_has_condition = condtion["id"] in patient_conditions_kinds
    print("\nDoes the patient have condition (" + condtion["id"] + ")? -> " + str(patient_has_condition))

    # Check if condition is cured
    condition_by_kind = find_condition_by_kind(patient_conditions, condtion["id"])
    if condition_by_kind != None:
        patient_has_condition_already_cured = condition_by_kind.__dict__["cured"] != None
        print("\nIs condition (" + condtion["id"] + ") already cured? -> " + str(patient_has_condition_already_cured))
    
    print("==========================\n")

def print_patients(patients):
    for patient in patients:
        print(patient.__dict__["name"] + " (id: " + str(patient.__dict__["id"]) + ")")

def find_patients_by_condition(patients, condition):
    patients_by_condition = []
    for patient in patients:
        for patient_condition in patient.__dict__["conditions"]:
            if patient_condition.__dict__["kind"] == condition.__dict__["id"]:
                patients_by_condition.append(patient)
    return patients_by_condition

def find_patient_by_id(patients, id):
    for patient in patients:
        if patient.__dict__["id"] == id:
            return patient

# Resolve condition id by secondary key of therapy
def get_condition_kind_by_trial_condition_id(patient, trial_condition_id):
    for patient_therapies in patient.__dict__["trials"]:
        if patient_therapies.__dict__["condition"] == trial_condition_id:
            for patient_condition in patient.__dict__["conditions"]:
                if patient_condition.__dict__["id"] == patient_therapies.__dict__["condition"]:
                    return patient_condition.__dict__["kind"]

def get_all_applied_therapies(patients):
    therapies = set()
    for patient in patients:
        for trial in patient.__dict__["trials"]:
            therapies.add(trial.__dict__["therapy"])
    return sorted(therapies)

def get_success_rate_of_therapy(patient, therapy, trial_needed = True):
    for trial in patient.__dict__["trials"]:
        if trial.__dict__["therapy"] == therapy:
            if trial_needed:
                return trial.__dict__["successful"], trial.__dict__["condition"]
            else:
                return trial.__dict__["successful"]

def is_zero_vector(vector):
    for v in vector:
        if v != 0:
            return False
    return True
    
def generate_vectors(patients, condition_id, patient_id, norm_vectos=True):
    vectors = []
    therapy_ids = get_all_applied_therapies(patients)  

    for patient in patients:
        vector = []
        for therapy_id in therapy_ids:
            if therapy_id in [th.__dict__["therapy"] for th in patient.__dict__["trials"]] or patient_id == patient.__dict__["id"]:
                if patient_id == patient.__dict__["id"]:
                    success_rate = get_success_rate_of_therapy(patient, therapy_id, False)
                    if success_rate == None:
                        success_rate = 0
                    vector.append(success_rate)
                else:
                    success_rate, trial_condition_id = get_success_rate_of_therapy(patient, therapy_id, True)
                    # Check if therapy is applied for the given condition
                    if get_condition_kind_by_trial_condition_id(patient, trial_condition_id) == condition_id:
                        vector.append(success_rate)
                    else: vector.append(0)
            else: vector.append(0)
        vector.append(patient.__dict__["id"])
        vectors.append(vector)

    therapy_ids.append("ids")
    df = pd.DataFrame(vectors, columns=therapy_ids)
    df.set_index('ids', inplace=True)

    if norm_vectos:
        # Normalize Scores by subtracting row means
        df = df.apply(lambda x: x - df.mean(axis = 1))
    return df

def get_biggest_patient_condition_id(patients):
    biggest_condition_id = 0
    for patient in patients:
        for condition in patient.__dict__["conditions"]:
            id_num = re.sub('\D', '', condition.__dict__["id"])
            if int(id_num) > biggest_condition_id:
                biggest_condition_id = int(id_num)
    return biggest_condition_id + 1

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
   
def get_most_similar_patients(n_patients, patient, patients, condition_id):
    patients_vectors = generate_vectors(patients, condition_id, patient.__dict__["id"])
    patient_vector = patients_vectors.loc[patient.__dict__["id"]]
    patients_vectors = patients_vectors.drop(patient.__dict__["id"])

    patient_similarities = []
    for index, row in patients_vectors.iterrows():
        if is_zero_vector(row):
            if index in patients_vectors.index:
                patients_vectors = patients_vectors.drop(index)
        else:
            patient_similarities.append(cosine_sim(row, patient_vector))
  
    ids =  patients_vectors.index[np.argsort(patient_similarities)[::-1]][:n_patients].to_list() 
    return [find_patient_by_id(patients, id) for id in ids]
    
# Remove vectors where all the values are 0 
def remove_vectors(df):
    for (columnName, columnData) in df.iteritems():
        if is_zero_vector(columnData.values):
            df = df.drop(columnName, axis=1)
    return df

# Find most recommended 5 therapies
def find_best_therapy(df):
    n_patients = df.shape[0]
    n_therapies = df.shape[1]
    recommendation_rates = []
    for i in range(n_therapies):
        recommendation_rates.append((1 / n_patients) * df.iloc[:,i].sum())

    return df.columns[np.argsort(recommendation_rates)[::-1]][:5].to_list()

def print_therapy_by_therapy_id(therapy_id):
    for therapy in therapies:
        if therapy.__dict__["id"] == therapy_id:
            return "Therapy: (id: " + therapy.__dict__["id"] + ") " + therapy.__dict__["name"] + " - Type: " + therapy.__dict__["type"]
    
def print_recommended_therapies(best_therapies, patient_id, condition_id):
    print("\nRecommended therapies for patient with id " + patient_id + " for condition with id " + condition_id + ":")
    for i in range(0, len(best_therapies)):
        print(str(i + 1) + ". " + print_therapy_by_therapy_id(best_therapies[i]))

if __name__ == "__main__":
    """ 
    Input 1: A set P of patients, their conditions, and the ordered list of trials each patient has done for each of his/her conditions (i.e, his/her medical history)
    Input 2: A specific patient P[q'], his/her conditions, the ordered list of trials he/she has done for each of these conditions (i.e, his/her medical history). 
    Input 3: A condition c[q]
    Output: A therapy th[ans]

    This means that to run the program you need to provide 3 arguments: 
    The dataset, The patient id, The condition id
    > code.py dataset.json JohnID headacheID 
    output: Recommended Therapy for the patient with id JohnID for the condition with id headacheID
            The output of the program will be an ordered list of 5 recommended therapies
    """
    #generate_new_dataset("dataset_new.json")

    # Check number of parameters
    if len(sys.argv) != 4:
        print("Usage: \"py code.py dataset.json JohnID headacheID\"")
        sys.exit()

    # Import dataset
    conditions, therapies, patients = import_dataset(sys.argv[1])
    patient_id = sys.argv[2]
    condition_id = sys.argv[3]

    # Check if patient exists
    if patients[int(patient_id)] == None:
        print("Patient with id " + patient_id + " does not exist.")
        sys.exit()
    
    # Check if condition exists
    if find_condition_by_id(conditions, condition_id) == None:
        print("Condition with id " + condition_id + " does not exist.")
        sys.exit()
    
    patient = patients[int(patient_id)]
    condition = find_condition_by_id(conditions, condition_id)

    # Patient Analysis (Optional)
    #patient_analysis(patient, condition)

    # If Requested Patient doesn't have the given condition then write it to the screen and add the condition as newly diagnosed condition
    if condition_id not in [cond.__dict__["kind"] for cond in patient.__dict__["conditions"]]:
        start_date, end_date = get_random_dates_with_interval()
        p_c = Patient_condition("pc" + str(get_biggest_patient_condition_id(patients)), start_date, end_date, condition_id)
        patient.__dict__["conditions"] = patient.__dict__["conditions"] + [p_c]
        print("\nPatient with id " + patient_id + " has newly diagnosed condition: " + condition.__dict__["name"] + " (id: " + str(condition.__dict__["id"]) + ")")

    # Get all patients who has the given condition cured
    patients_with_condition = find_patients_by_condition(patients, condition)

    # Generate vectors for all patients
    df = generate_vectors(patients_with_condition, condition_id, patient.__dict__["id"], True)
    #print(df)

    # Use the vectors to find the most similar patients
    similar_patients = get_most_similar_patients(20, patient, patients_with_condition, condition_id)
    #print_patients(similar_patients)

    # Generate vectors from the 20 most similar patients patients
    similar_patients_vectors = generate_vectors(similar_patients, condition_id, patient.__dict__["id"], False)
    #print(similar_patients_vectors)

    similar_patients_vectors = remove_vectors(similar_patients_vectors)
    print(similar_patients_vectors)

    # For each therapy, predict the success rate of each therapies, and find the best ones
    best_therapies = find_best_therapy(similar_patients_vectors)
    #print(best_therapies)

    print_recommended_therapies(best_therapies, patient_id, condition_id)

    # Main TODOs
    # Check if success rate is set always by the most recently used therapy?

    # TODOs
    # Check and print only the max number of recommended therpaies found!!!
    # Check if print if there is no recommended therapy found
    # Check if recommendations are really good - Recommended users are really similar (Should be good)
    # Predict each therapy for our patient and find the best one
    # Take outputs as examples for a step by step walkthrough of the program
    # Add error handling in case of no similar patients
    # Add error handling in case of no good therapies can be found
from random import randint
import json
import datetime

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
                obj = json.loads(json.dumps(patient_condition), object_hook=lambda d: Patient_condition(**d))
                patient_conditions.append(obj)
                current_patient_conditions.append(obj)
            for patient_therapy in patient["trials"]:
                obj = json.loads(json.dumps(patient_therapy), object_hook=lambda d: Patient_therapy(**d))
                patient_therapies.append(obj)
                current_patient_therapies.append(obj)
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
            if len(the) > 1:
                t = Therapy("Th" + str(i), therapy, the[0])
                values.append(t)
            i += 1
    conditions.close()
    return values

def get_random_dates_with_interval(beginning=20000101):
    date = datetime.datetime.strptime(str(beginning), "%Y%m%d")
    start_date = datetime.date(randint(date.year, 2021), randint(date.month, 12), randint(date.day, 28))
    end_date = datetime.date(2021, 12, 31)
    random_days = randint(0, (end_date - start_date).days)
    random_date = start_date + datetime.timedelta(days = random_days)
    return start_date.strftime("%Y%m%d"), random_date.strftime("%Y%m%d")

def get_random_condition(conditions):
    return conditions[randint(0, len(conditions) - 1)]

def get_random_therapy(therapies):
    return therapies[randint(0, len(therapies) - 1)]

def generate_patients(source_file, conditions, therapies):
    values = []
    condition_id_counter = 0
    therapy_id_counter = 0
    with open(source_file, 'r') as patient_names:
        i = 0
        for name in patient_names.read().split("\n"):
            n_patient_condition = randint(1, 5)
            n_patient_therapy = randint(1, 5)
            patient_conditions = []
            patient_therapies = []
            for j in range(n_patient_condition):
                start_date, end_date = get_random_dates_with_interval()
                p_c = Patient_condition("pc" + str(condition_id_counter), start_date, None, get_random_condition(conditions).__dict__["id"])
                patient_conditions.append(p_c)
                condition_id_counter += 1

            for k in range(n_patient_therapy):
                condition = get_random_condition(patient_conditions)
                start_date_of_condition = condition.__dict__["diagnosed"]
                start_date, end_date = get_random_dates_with_interval(start_date_of_condition)
                p_t = Patient_therapy("tr" + str(therapy_id_counter), start_date_of_condition, end_date, condition.__dict__["id"], get_random_therapy(therapies).__dict__["id"], randint(0, 100))
                #If Therapy Worked (<70%), cure the condition
                if p_t.__dict__["successful"] >= 75:
                    for cond in patient_conditions:
                        if cond.__dict__["id"] == p_t.__dict__["condition"]:
                            cond.__dict__["cured"] = p_t.__dict__["end"]
                patient_therapies.append(p_t)
                therapy_id_counter += 1
            p = Patient(i, name, patient_conditions, patient_therapies)
            values.append(p)
            i += 1
    patient_names.close()
    return values

if __name__ == "__main__":
    """ 
    Input 1: A set P of patients, their conditions, and the ordered list of trials each patient has done for each of his/her conditions (i.e, his/her medical history)
    Input 2: A specific patient p[q'], his/her conditions, the ordered list of trials he/she has done for each of these conditions (i.e, his/her medical history). 
    Input 3: A condition c[q]
    Output: A therapy th[ans]

    This means that to run the program you need to provide 3 arguments: 
    The dataset, The patient id, The condition id
    > code.py dataset.json JohnID headacheID 
    output: Recommended Therapy for the patient with id JohnID for the condition with id headacheID
    """
    conditions, therapies, patients = load_demo_patients()
    export_demo_dataset("dataset.json", conditions, therapies, patients)
    conditions, therapies, patients = import_dataset("dataset.json")
    export_dataset("dataset_new.json", conditions, therapies, patients)

    conditions = generate_condtions("source_data/conditions.txt")
    therapies = generate_therapies("source_data/therapies.txt")
    patients = generate_patients("source_data/names.txt", conditions, therapies)
    export_dataset("dataset_new.json", conditions, therapies, patients)

 

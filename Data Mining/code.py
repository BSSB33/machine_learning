import json

class Condition:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type

# New class to support the extra attributes between class "Condition" and the asked attributes in patent.conditions
class Patent_condition: 
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
    p_c1 = Patent_condition("pc1", 20210915, 20210915, "Cond1")
    p_c2 = Patent_condition("pc2", 20210602, 20210930, "Cond2")
    p_c3 = Patent_condition("pc3", 20210603, 20210905, "Cond2")
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
                obj = json.loads(json.dumps(patient_condition), object_hook=lambda d: Patent_condition(**d))
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

def generate_dataset(file, n_patients, n_conditions, n_therapies, n_trials):
    conditions = []
    therapies = []
    patients = []
    patient_conditions = []
    patient_therapies = []
    for i in range(n_patients):
        for j in range(n_conditions):
            c = Condition(str(i) + str(j), "Condition " + str(i) + str(j), "Condition")
            conditions.append(c)
        for k in range(n_therapies):
            t = Therapy(str(i) + str(k), "Therapy " + str(i) + str(k), "Therapy")
            therapies.append(t)
        for l in range(n_trials):
            p_c = Patent_condition(str(i) + str(l), 20200101 + l, 20200101 + l, str(i) + str(l))
            patient_conditions.append(p_c)
            p_t = Patient_therapy(str(i) + str(l), 20200101 + l, 20200101 + l, str(i) + str(l), str(i) + str(l), 10)
            patient_therapies.append(p_t)
        p = Patient(i, "Patient " + str(i), patient_conditions, patient_therapies)
        patients.append(p)
        patient_conditions = []
        patient_therapies = []
    export_dataset(file, conditions, therapies, patients)
    
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

    generate_dataset("dataset_tmp.json", 1, 1, 1, 2)

 

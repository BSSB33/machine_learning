import json

class Condition:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type

# New class to support the extra attributes between class "Condition" and the asked attributes in patent.conditions
class Patent_condition: 
    def __init__(self, id, diganosed, cured, id_of_condition):
        self.id = id
        self.diagnosed = diganosed
        self.cured = cured
        self.kind = id_of_condition

class Therapy:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type

# New class to support the extra attributes between class "Terapy" and the asked attributes in "trials"
class Patient_therapy:
    def __init__(self, id, start, end, id_of_condition, id_of_therapy, successful):
        self.id = id
        self.start = start
        self.end = end
        self.condition = id_of_condition
        self.therapy = id_of_therapy
        self.successful = successful
  
class Patient:
    def __init__(self, id, name, patient_conditions, patient_therapies):
        self.id = id
        self.name = name
        self.conditions = patient_conditions
        self.trials = patient_therapies


def load_demo_patients():
    c1 = Condition("Cond1", "High Blood Pressure", "Blood Pressure")
    c2 = Condition("Cond2", "Heart Arrhythmia", "Heath Condition")
    t1 = Therapy("Th1", "Acetoxybenzoic Acid (Aspirin)", "Acetoxybenzoic Acid (Aspirin)")
    t2 = Therapy("Th2", "Cough Syrup Quibron", "Sugar")
    p_c1 = Patent_condition("pc1", 20210915, 20210915, "Cond1")
    p_c2 = Patent_condition("pc2", 20210602, 20210930, "Cond2")
    p_t1 = Patient_therapy("tr1", 20210915, 20211215, "pc1", "Th1", 10)
    p_t2 = Patient_therapy("tr2", 20210102, 20210130, "pc2", "Th2", 100)
    patient1 = Patient(1, "John", [p_c1.__dict__, p_c2.__dict__], [p_t1.__dict__, p_t2.__dict__])
    patient2 = Patient(1, "Peter", [p_c1.__dict__], [p_t1.__dict__])

    conditions = [condition.__dict__ for condition in [c1, c2]]
    therapies = [therapy.__dict__ for therapy in [t1, t2]]
    patients = [patient.__dict__ for patient in [patient1, patient2]]
    return conditions, therapies, patients

def export_dataset(conditions, therapies, patients):
    with open('dataset.json', 'w') as outfile:
        json.dump({"Conditions": conditions, "Therapies": therapies, "Patients": patients}, outfile)
    outfile.close()

def import_dataset():
    conditions = []
    with open('dataset.json', 'r') as json_file:
        data = json.load(json_file)
        for condition in data["Conditions"]:
            obj = json.loads(json.dumps(condition), object_hook=lambda d: Condition(**d))
            conditions.append(obj)
            print(obj.__dict__)
        for therapy in data["Therapies"]:
            obj = json.loads(json.dumps(therapy), object_hook=lambda d: Therapy(**d))
            conditions.append(obj)
            print(obj.__dict__)
    json_file.close()
    
if __name__ == "__main__":
    """ 
    Input 1: A set P of patients, their conditions, and the ordered list of trials each patient has done for each of his/her conditions (i.e, his/her medical history)
    Input 2: A specific patient p[q'], his/her conditions, the ordered list of trials he/she has done for each of these conditions (i.e, his/her medical history). 
    Input 3: A condition c[q]
    Output: A therapy th[ans]
    """
    #conditions, therapies, patients = load_demo_patients()
    #export_dataset(conditions, therapies, patients)
    import_dataset()


 

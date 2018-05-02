import pandas as pd
import numpy as np
import random as rnd
import math

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp

def cutBasedGrade(event):
    if event["invcurvature_s"] > 0: return 0
    else:
        muon_one = abs(event["end_z1"]) < 5 and event["end_rho1"] < 2
        muon_two = abs(event["end_z2"]) < 5 and event["end_rho2"] < 2
        if muon_one and muon_two: return 1
        elif muon_one or muon_two: return 2
        else: return 3

def invariantMass(event):
    sum_E = event["E1"] + event["E2"]
    sum_px = event["px1"] + event["px2"]
    sum_py = event["py1"] + event["py2"]
    sum_pz = event["pz1"] + event["pz2"]
    return math.sqrt(sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2)

def dict_to_matrix(data_dict, target_col_name = "Grade", truth = True):
    matrix = []
    target = []
    for entry in data_dict:
        cache = []
        for col in use_columns:
            cache.append(entry[col])
        matrix.append(cache)
        if truth: target.append(cutBasedGrade(entry))
        else: target.append(int(entry[target_col_name]))
    return matrix, target

headerNames = ["Event", "Grade", \
               "E1", "px1", "py1", "pz1", "start_x1", "start_y1", "start_z1", \
               "end_x1", "end_y1", "end_z1", "invcurvature_s1", "invcurvature_e1", \
               "E2", "px2", "py2", "pz2", "start_x2", "start_y2", "start_z2", \
               "end_x2", "end_y2", "end_z2", "invcurvature_s2", "invcurvature_e2"]

use_columns = ["invcurvature_s","invcurvature_e", "end_x1", "end_y1", "end_z1", "end_x2", "end_y2", "end_z2",\
               "start_x1", "start_y1", "start_z1", "start_x2", "start_y2", "start_z2"]

headerNames_allevents = ["Event", \
               "E1", "px1", "py1", "pz1", "start_x1", "start_y1", "start_z1", \
               "end_x1", "end_y1", "end_z1", "invcurvature_s1", "invcurvature_e1", \
               "E2", "px2", "py2", "pz2", "start_x2", "start_y2", "start_z2", \
               "end_x2", "end_y2", "end_z2", "invcurvature_s2", "invcurvature_e2"]
alleventsfile = pd.read_table("AllEvents.txt", header = None, usecols=range(25))
alleventsfile.columns = headerNames_allevents
alleventsfile["invcurvature_s"] = alleventsfile["invcurvature_s1"] * alleventsfile["invcurvature_s2"]/1000
alleventsfile["invcurvature_e"] = alleventsfile["invcurvature_e1"] * alleventsfile["invcurvature_e2"]/1000
alleventsfile["start_rho1"] = (alleventsfile["start_x1"]**2 + alleventsfile["start_y1"]**2)**0.5
alleventsfile["start_rho2"] = (alleventsfile["start_x2"]**2 + alleventsfile["start_y2"]**2)**0.5
alleventsfile["end_rho1"] = (alleventsfile["end_x1"]**2 + alleventsfile["end_y1"]**2)**0.5
alleventsfile["end_rho2"] = (alleventsfile["end_x2"]**2 + alleventsfile["end_y2"]**2)**0.5

allevents_dict = []
for row in alleventsfile.as_matrix():
    res = {}
    for column, name in zip(row, alleventsfile.columns):
        res[name] = column
    allevents_dict.append(res)
    
allevents_dict_truthsorted = [[e for e in allevents_dict if cutBasedGrade(e) == grade] for grade in range(4)]
allevents_truthsorted_mat = [dict_to_matrix(mat) for mat in allevents_dict_truthsorted]

class JPsiMLClassifier(object):
    def __init__(self, filename):
        self.name = filename
        self.trainfile = pd.read_table(filename, header = None, usecols = range(26))
        self.trainfile.columns = headerNames
        self.trainfile["invcurvature_s"] = self.trainfile["invcurvature_s1"] * self.trainfile["invcurvature_s2"]/1000
        self.trainfile["invcurvature_e"] = self.trainfile["invcurvature_e1"] * self.trainfile["invcurvature_e2"]/1000
        self.trainfile["start_x"] = abs(self.trainfile["start_x1"] * self.trainfile["start_x2"])
        self.trainfile["start_y"] = abs(self.trainfile["start_y1"] * self.trainfile["start_y2"])
        self.trainfile["start_z"] = abs(self.trainfile["start_z1"] * self.trainfile["start_z2"])
        self.trainfile["end_x"] = abs(self.trainfile["end_x1"] * self.trainfile["end_x2"])
        self.trainfile["end_y"] = abs(self.trainfile["end_y1"] * self.trainfile["end_y2"])
        self.trainfile["end_z"] = abs(self.trainfile["end_z1"] * self.trainfile["end_z2"])
        self.trainfile["start_rho1"] = (self.trainfile["start_x1"]**2 + self.trainfile["start_y1"]**2)**0.5
        self.trainfile["start_rho2"] = (self.trainfile["start_x2"]**2 + self.trainfile["start_y2"]**2)**0.5
        self.trainfile["start_rho"] = self.trainfile["start_rho1"] * self.trainfile["start_rho2"]
        self.trainfile["end_rho1"] = (self.trainfile["end_x1"]**2 + self.trainfile["end_y1"]**2)**0.5
        self.trainfile["end_rho2"] = (self.trainfile["end_x2"]**2 + self.trainfile["end_y2"]**2)**0.5
        self.trainfile["end_rho"] = self.trainfile["end_rho1"] * self.trainfile["end_rho2"]
        self.trainfile["start_z"] = self.trainfile["start_z1"] * self.trainfile["start_z2"]
        self.trainfile["end_z"] = self.trainfile["end_z1"] * self.trainfile["end_z2"]
        
        self.error_classifier_not_initialised = AttributeError("Classifier has not been initialised. Initialise using start_train() method.")
        
        self.train_dict = []
        for row in self.trainfile.as_matrix():
            res = {}
            for column, name in zip(row, self.trainfile.columns):
                res[name] = column
            self.train_dict.append(res)
        
        self.train_dict_truthsorted = [[e for e in self.train_dict if cutBasedGrade(e) == grade] for grade in [0, 1, 2, 3]]
        self.train_truthsorted_mat = [dict_to_matrix(mat, truth = False) for mat in self.train_dict_truthsorted]
        
        self.classifier_ready = False
        self.predict_mass_ready = False
        self.direct_mass_ready = False
    
    def start_train(self):
        self.stratified_dataset = []
        for grade_set in self.train_truthsorted_mat:
            _train_x, _test_x, _train_y, _test_y = train_test_split(grade_set[0]*2, grade_set[1]*2, \
                                                                    test_size = 0.4, stratify = grade_set[1]*2, random_state = 42)
            self.stratified_dataset.append([_train_x, _test_x, _train_y, _test_y])
            
        self.classifier_set = [RandomForestClassifier() for i in range(4)]
        for grade in range(4):
            self.classifier_set[grade].fit(self.stratified_dataset[grade][0], self.stratified_dataset[grade][2])
        self.classifier_ready = True
    
    def set_name(self, name):
        self.name = name

    def generate_student_confusion(self):
        self.student_confusion = np.zeros([4,4])
        for t in [(int(row["Grade"]), cutBasedGrade(row)) for row in self.train_dict]:
            self.student_confusion[t[0], t[1]] += 1
    
    def plot_student_confusion(self):
        self.generate_student_confusion()
        plt.figure(figsize = (10,7))
        self.student_confusion_df = pd.DataFrame(
            self.student_confusion, 
            index = [idx for idx in ['zero', 'one', 'two', 'three']],
            columns = [col for col in ['zero', 'one', 'two', 'three']])
        sns.heatmap(self.student_confusion_df, annot=True, fmt='g')
        plt.ylabel("Graded")
        plt.xlabel("Truth")
        plt.title("{}'s confusion matrix".format(self.name))

    def get_test_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = []
        for grade in range(4):
            score.append(self.classifier_set[grade].score(self.stratified_dataset[grade][1], self.stratified_dataset[grade][3]))
        return score
    
    def print_test_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = self.get_test_score()
        print("         zero    one    two  three")
        print("Grade: ", end='')
        for grade in range(4): print("{:6.2f}".format(score[grade]), end = ' ')
        print()
    
    def get_train_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = []
        for grade in range(4):
            score.append(self.classifier_set[grade].score(self.stratified_dataset[grade][0], self.stratified_dataset[grade][2]))
        return score
    
    def print_train_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = self.get_train_score()
        print("         zero    one    two  three")
        print("Grade: ", end='')
        for grade in range(4): print("{:6.2f}".format(score[grade]), end = ' ')
        print()
    
    def get_test_train_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = []
        for grade in range(4):
            score.append(self.classifier_set[grade].score(self.stratified_dataset[grade][0]+self.stratified_dataset[grade][1], \
                                                          self.stratified_dataset[grade][2]+self.stratified_dataset[grade][3]))
        return score
    
    def print_test_train_score(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        score = self.get_test_train_score()
        print("         zero    one    two  three")
        print("Grade: ", end='')
        for grade in range(4): print("{:6.2f}".format(score[grade]), end = ' ')
        print()
    
    def print_scoring(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        print("Test dataset")
        self.print_test_score()
        print("Train dataset")
        self.print_train_score()
        print("Test + Train dataset")
        self.print_test_train_score()
        
    def plot_direct_mass(self):
        self.invariant_mass_direct_list = []
        for event in self.train_dict:
            if event["Grade"] == 3:
                self.invariant_mass_direct_list.append(invariantMass(event))
        plt.hist(self.invariant_mass_direct_list, range=(2,5), bins = 15)
        plt.title("J/Psi invariant mass with grade 3 from {} (Direct histogram)".format(self.name))
        self.direct_mass_ready = True
    
    def plot_predict_mass(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        self.invariant_mass_list = []
        for grade in range(4):
            for event, event_dict in zip(self.train_truthsorted_mat[grade][0], self.train_dict_truthsorted[grade]):
                if self.classifier_set[grade].predict([event])[0] == 3:
                    self.invariant_mass_list.append(invariantMass(event_dict))
        plt.hist(self.invariant_mass_list, range=(2,5), bins = 15)
        plt.title("J/Psi invariant mass with grade 3 from {} (by scikit-learn)".format(self.name))
        self.predict_mass_ready = True
    
    def plot_predict_mass_all(self):
        if not self.classifier_ready:
            raise self.error_classifier_not_initialised
        self.invariant_mass_all_list = []
        for grade in range(4):
            for event, event_dict in zip(allevents_truthsorted_mat[grade][0], allevents_dict_truthsorted[grade]):
                if self.classifier_set[grade].predict([event])[0] == 3:
                    self.invariant_mass_all_list.append(invariantMass(event_dict))
        plt.hist(self.invariant_mass_all_list, range=(2,5), bins = 15)
        plt.title("J/Psi invariant mass with grade 3 from {} (2000 events)".format(self.name))
    
    def print_ks_test(self):
        if not self.predict_mass_ready:
            raise AttributeError("Predicted invariant mass histogram has not been generated. Generate using plot_predict_mass() method.")
        if not self.direct_mass_ready:
            raise AttributeError("Direct invariant mass histogram has not been generated. Generate using plot_direct_mass() method.")
        print("Kolmogorov-Smirnov test \non predicted/direct invariant mass histogram")
        print("--------------------------------------")
        ks_res = ks_2samp(self.invariant_mass_direct_list, self.invariant_mass_list)
        print("KS stats: {:.20f}".format(ks_res.statistic))
        print("p-value:  {:.20f}".format(ks_res.pvalue))
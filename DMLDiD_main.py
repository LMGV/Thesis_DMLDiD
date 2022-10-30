from DMLDiD_fns import *

# define directories
inputs_dir = ".\\Inputs"
results_dir = ".\\Results"

explanatory_variable = "FOODWASTE"

variables = ["Male", "Age", "Self_employed", "Employee", "Manual_worker", "Student", 
             "Rural", "Urban"] + [explanatory_variable] + ["Date"]

# import data
for i in range(1, 5):
    globals()[f"x_cluster{str(i)}"] = pd.read_csv(f"{inputs_dir}\\X_cluster{str(i)}.csv")
    globals()[f"x_cluster{str(i)}"] = globals()[f"x_cluster{str(i)}"][variables]
    globals()[f"d_cluster{str(i)}"] = pd.read_csv(f"{inputs_dir}\\y_cluster{str(i)}.csv")
    

# defining variables
#random.seed(2616)
save = False     # whether to write estimation results to .csv or not
K = 2            # number of folds. Set to 2 following Chang's (2020) simulation.
                 # Note that the code is fully written for K = 2. The variable is 
                 # defined specifically only to explicitly mark that two folds are used
run = False      # indicates whether to run the estimation or not. If not, then 
                 # previously saved results will be loaded. Note that running
                 # can take a couple of hours
                
# estimation

if run == True:
    atet_1, atet_2, atet_3, atet_4  = [], [], [], []
    for j in range(500):
        atet_1.append(ATET(1, x_cluster1, d_cluster1, explanatory_variable, save = save))
        atet_2.append(ATET(2, x_cluster2, d_cluster2, explanatory_variable, save = save))
        atet_3.append(ATET(3, x_cluster3, d_cluster3, explanatory_variable, save = save))
        atet_4.append(ATET(4, x_cluster4, d_cluster4, explanatory_variable, save = save))
else:
    atet_1 = pd.read_csv(f"{results_dir}\\atet_cluster1_{explanatory_variable}.csv")
    atet_2 = pd.read_csv(f"{results_dir}\\atet_cluster2_{explanatory_variable}.csv")
    atet_3 = pd.read_csv(f"{results_dir}\\atet_cluster3_{explanatory_variable}.csv")
    atet_4 = pd.read_csv(f"{results_dir}\\atet_cluster4_{explanatory_variable}.csv")
    
    
display(pd.DataFrame(data = {"Cluster 1": [np.round(np.mean(atet_1), 3),
                                           np.round(np.std(atet_1), 3)],
                             "Cluster 2": [np.round(np.mean(atet_2), 3),
                                           np.round(np.std(atet_2), 3)],
                             "Cluster 3": [np.round(np.mean(atet_3), 3),
                                           np.round(np.std(atet_3), 3)],
                             "Cluster 4": [np.round(np.mean(atet_4), 3),
                                           np.round(np.std(atet_4), 3)],},
                     index = ["ATET", "std"]))

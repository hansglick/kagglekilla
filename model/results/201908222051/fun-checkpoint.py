import pandas as pd
import os
import numpy as np
from sklearn import *
from datetime import datetime
from shutil import copyfile

# miscellaneous
def define_settings_id():
    dateTimeObj = datetime.now()
    informations = [dateTimeObj.year,dateTimeObj.month,dateTimeObj.day,dateTimeObj.hour,dateTimeObj.minute]
    informations = [str(item).zfill(2) for item in informations]
    informations = "".join(informations)
    return informations

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
            
def listfiles_fromfolders(folders_list):
    files_list = [list(absoluteFilePaths(item)) for item in folders_list]
    files_list = [item for subl in files_list for item in subl ]
    return files_list

def copyfiles_to_folder(files_to_copy,folderpath):
    for f in files_to_copy:
        copyfile(f,os.path.join(folderpath,os.path.basename(f)))
    return None

def send_list_of_submissions(all_datasets_for_modeling,list_tuples_id,rootpathfile,verbose=True):
    
    for idr,r in enumerate(list_tuples_id):

        run_settingID,modelID = r
        p = pd.Series(all_datasets_for_modeling[run_settingID][model_ID]["predictions"])
        i = pd.Series(all_datasets_for_modeling[run_settingID][model_ID]["id_predictions"])
        submission = pd.concat((i,p),axis = 1).rename(columns = {0 : "datetime",1 : "count"})
        namesub = "submission_" + str(idr) +".csv"
        submission.to_csv(namesub,index = False)
        pathfile = os.path.join(rootpathfile,namesub)

        commentaires = "'runID:" + run_settingID +", modelID:"+modelID+"'"

        commandbash = "kaggle competitions submit -c bike-sharing-demand -f " + pathfile + " -m " + commentaires
        
        if verbose:
            print(commandbash)

        subprocess.call(commandbash,shell = True)

    return None

# Features part



def import_clean_save(pathfile):
    pathfile = os.path.basename(pathfile)
    filename, file_extension = os.path.splitext(pathfile)
    df = pd.read_csv(pathfile)
    df.reset_index(inplace = True, drop=True)
    df.datetime = pd.to_datetime(df.datetime)
    df["month"] = df.datetime.map(lambda a : a.month_name())
    df["hour"] = df.datetime.map(lambda a : a.hour)
    df["day"] = df.datetime.map(lambda a : a.day_name())
    df["year"] = df.datetime.map(lambda a : a.year)
    df.to_pickle(filename+".pkl")
    return None

def to_ohe(df,feature):
    le = preprocessing.OneHotEncoder(handle_unknown='ignore')
    le.fit(np.expand_dims(df[feature].values,axis=1))
    dftoadd = pd.DataFrame(le.transform(np.expand_dims(df[feature].values,axis=1)).toarray()).astype(int)
    dftoadd.columns = ["OHE_"+feature+"_"+str(item) for item in list(dftoadd)]
    df.reset_index(inplace=True, drop=True)
    dftoadd.reset_index(inplace=True, drop=True)
    df = pd.concat((df,dftoadd),axis = 1)
    return df

def to_le(df,feature):
    le = preprocessing.LabelEncoder()
    le.fit(df[feature])
    df["LE_"+feature] = le.transform(df[feature])
    return df

def to_ore(df,feature):
    le = preprocessing.OrdinalEncoder()
    le.fit(np.expand_dims(df[feature].values,axis=1))
    df["ORE_"+feature] = le.transform(np.expand_dims(df[feature].values,axis=1)).astype(int)
    return df

def extract_target_features(build_train_test_for_model_parameters):
    L = []
    L.append(build_train_test_for_model_parameters["target"]["name"])
    if "subs" in build_train_test_for_model_parameters["target"]:
        L = L+build_train_test_for_model_parameters["target"]["subs"]
    return L

def encode_and_exclude(df,build_train_test_for_model_parameters):

    for k,v in build_train_test_for_model_parameters.items():

        if k == "oheencoding":
            for feature in v:
                df = to_ohe(df,feature)

        if k == "labelencoding":
            for feature in v:
                df = to_le(df,feature)

        if k == "ordinalencoding":
            for feature in v:
                df = to_ore(df,feature)

        if k == "exclusion":
            df = df[[item for item in list(df) if item not in v]]

    return df

def extract_strates(build_train_test_for_model_parameters):
    strates = ["global"]
    if "strata" in build_train_test_for_model_parameters:
        strates = strates+build_train_test_for_model_parameters["strata"]
    return strates


def build_official_train_and_test(df,
                                  build_train_test_for_model_parameters,
                                  strate,
                                  targetname,
                                  pathfiletrain="../data/train.pkl"):
    
    df.sort_values(by=[build_train_test_for_model_parameters["id"]],inplace = True)
    train = pd.read_pickle(pathfiletrain)
    train.sort_values(by=[build_train_test_for_model_parameters["id"]],inplace = True)
    targetdf = train[extract_target_features(build_train_test_for_model_parameters)]
    

    sol = {}

    if strate == "global":
        mytrain = df[[item for item in list(df) if item not in build_train_test_for_model_parameters["id"]]][df.dataset=="train"]
        mytest = df[[item for item in list(df) if item not in build_train_test_for_model_parameters["id"]]][df.dataset=="test"]
        
        mytrainid = df[build_train_test_for_model_parameters["id"]][df.dataset=="train"].values
        mytestid = df[build_train_test_for_model_parameters["id"]][df.dataset=="test"].values
        
        mytrain.drop(columns = ["dataset"],inplace=True)
        mytest.drop(columns = ["dataset"],inplace=True)
        ytrain = targetdf[targetname]
        
        if "target_transformation" in build_train_test_for_model_parameters:
            ytrain = build_train_test_for_model_parameters["target_transformation"]["transform"](ytrain)
            
        
        sol["global"] = {"x_train" :mytrain.values ,
                         "y_train" : ytrain.values ,
                         "x_test" :mytest.values,
                         "features" : list(df),
                         "x_train_id": mytrainid,
                         "x_test_id" : mytestid}

    else:
        
        for value in df[strate].unique():
            mytrain = df[[item for item in list(df) if item not in build_train_test_for_model_parameters["id"]]][(df.dataset=="train") & (df[strate]==value)]
            mytest = df[[item for item in list(df) if item not in build_train_test_for_model_parameters["id"]]][(df.dataset=="test") & (df[strate]==value)]
            
            mytrainid = df[build_train_test_for_model_parameters["id"]][(df.dataset=="train") & (df[strate]==value)].values
            mytestid = df[build_train_test_for_model_parameters["id"]][(df.dataset=="test") & (df[strate]==value)].values
            
            mytrain.drop(columns = ["dataset"],inplace=True)
            mytest.drop(columns = ["dataset"],inplace=True)
            ytrain = targetdf[targetname][df[strate]==value]
            
            if "target_transformation" in build_train_test_for_model_parameters:
                    ytrain = build_train_test_for_model_parameters["target_transformation"]["transform"](ytrain)
            
            sol[strate+"_"+str(value)] = {"x_train" : mytrain.values,
                                          "y_train" : ytrain.values,
                                          "x_test" : mytest.values,
                                          "features" : list(df),
                                          "x_train_id": mytrainid,
                                          "x_test_id" : mytestid}

    return sol


def build_all_datasets(build_train_test_for_model_parameters,df):
    
    targets = extract_target_features(build_train_test_for_model_parameters)
    strates = extract_strates(build_train_test_for_model_parameters)
    d = {}
    
    for s in strates:
        for t in targets:
            d[(s,t)] = build_official_train_and_test(df,build_train_test_for_model_parameters,s,t)
    
    return d


# Model part
def is_hyperparam_for_holdout(hyparam):
    
    L = []
    for k,v in hyparam.items():
        
        try:
            size = len(v)
        except:
            size = 1
        L.append(size)

    holdout_decision = np.all(np.array(L)==1)

    return holdout_decision


def correct_hyparam_for_cv(hyparam):
    
    hyparam_copy = hyparam.copy()
    
    for k,v in hyparam_copy.items():
        try:
            len(v)
        except :
            hyparam_copy[k] = [v]

    return hyparam_copy








def clean_hyparam_up(dic,keyname,defaultval):
    dic_copy = dic.copy()
    
    if keyname in dic_copy:
        solution = dic_copy[keyname]
        del dic_copy[keyname]
    else:
        solution = defaultval
        
    return dic_copy,solution





def clean_up_combinaison_define_search(combinaison,prefix):
    L = []
    combinaison_copy = combinaison.copy()
    search_dic = {}
    for k,v in combinaison_copy.items():
        if k.split("_")[0] == prefix:
            L.append(k)
            search_dic[k.split(prefix+"_",1)[1]] = v
    
    for k in L:
        del combinaison_copy[k]
            
    return combinaison_copy,search_dic

def set_model_according_parameters(modelfamilly,combinaison):

    holdout_decision = is_hyperparam_for_holdout(combinaison)

    if not holdout_decision :
        
        # Clean Combinaison and Define Search Dic
        combinaison,search_dic = clean_up_combinaison_define_search(combinaison,"search")
        combinaison = correct_hyparam_for_cv(combinaison)

        if "n_iter" not in search_dic:
            
            # GridSearch Case
            model = set_model_for_gridsearch(modelfamilly,combinaison,search_dic)
            
        else:
            # RandomSearch Case
            model = set_model_for_randomsearch(modelfamilly,combinaison,search_dic)
            
        toreturn = (model,search_dic,holdout_decision)
    else:
        # HoldOut Case
        combinaison,split_dic = clean_up_combinaison_define_search(combinaison,"split")
        model = modelfamilly
        model.set_params(**combinaison)
        toreturn = (model,split_dic,holdout_decision)
        
    return toreturn

def set_model_for_gridsearch(model,combinaison,search_dic):
    
    d = {}
    d["estimator"] = model
    d["param_grid"] = combinaison.copy()
    z = {**d, **search_dic.copy()}

    gsmodel = model_selection.GridSearchCV(estimator = model,
                                           param_grid = combinaison.copy())
    gsmodel.set_params(**z)

    
    return gsmodel



def set_model_for_randomsearch(model,combinaison,search_dic):
    
    d = {}
    d["estimator"] = model
    d["param_distributions"] = combinaison.copy()
    z = {**d, **search_dic.copy()}
    
    rsmodel = model_selection.RandomizedSearchCV(estimator = model,
                                                 param_distributions = combinaison.copy())
    rsmodel.set_params(**z)
    
    return rsmodel




def run_all_settings(all_datasets_for_modeling,
                     dic_of_models,
                     dic_of_hyparam,
                     build_train_test_for_model_parameters,
                     customscoring):
    
    d = {}

    for k,v in all_datasets_for_modeling.items():
        keyslist = list(v.keys()).copy()

        for idata in keyslist:
            dictrip = all_datasets_for_modeling[k][idata]

            for idmodel,modelfamilly in dic_of_models.items():

                for idcombo,combinaison in enumerate(dic_of_hyparam[idmodel]):

                    idofmodel = str(k) + "_" + str(idata) + "_" + str(idmodel) + "_" + str(idcombo)
                    print("Feature Strate : ", k[0],"\n",
                          "Target : ", k[1],"\n",
                          "Dataset : ", idata,"\n")
                    current_model,informations_dic,holdout_decision = set_model_according_parameters(modelfamilly,combinaison)
                    
                    if holdout_decision:
                        res = save_result_model_holdout(current_model,
                                                        dictrip,
                                                        informations_dic,
                                                        build_train_test_for_model_parameters,
                                                        customscoring)
                    else:
                        res = save_result_model_non_holdout(current_model,
                                                            dictrip,
                                                            informations_dic,
                                                            build_train_test_for_model_parameters,
                                                            customscoring)

                    d[idofmodel] = res
                    
    return d


def save_result_model_non_holdout(current_model,
                                  dictrip,
                                  informations_dic,
                                  build_train_test_for_model_parameters,
                                  customscoring,
                                  verbose=True):

    typename = type(current_model).__name__
    modelname = type(current_model.estimator).__name__
    current_model.fit(dictrip["x_train"],dictrip["y_train"])
    dicsave = {}

    # Predictions part
    if "target_transformation" in build_train_test_for_model_parameters:
        transformed_predictions = current_model.predict(dictrip["x_test"])
        transformed_predictions[np.where(transformed_predictions<0)] = 0
        predictions = build_train_test_for_model_parameters["target_transformation"]["reciproque"](transformed_predictions)
        dicsave["transformed_predictions"] = transformed_predictions
    else:
        predictions = current_model.predict(dictrip["x_test"])

    dicsave["typename"] = typename
    dicsave["modelname"] = modelname
    dicsave["predictions"] = predictions
    dicsave["features"] = dictrip["features"]
    dicsave["id_predictions"] = dictrip["x_test_id"]
    dicsave["model"] = current_model
    dicsave["best_score"] = current_model.best_score_
    dicsave["informations"] = informations_dic
    dicsave["results"] = pd.DataFrame(current_model.cv_results_)
    
    if verbose:
        print("Type : ",typename,"\n",
              "Model : ",modelname,"\n",
              "Infos : ",informations_dic,"\n",
              "Features : ", len(dictrip["features"]),"\n",
              "Nombre de modèles testés : ", len(dicsave["results"]),"\n",
              "Score : ",abs(dicsave["best_score"]),"\n",
              "","\n",
              "* * * * * * * * * * * * * * * * * * * * * * * * * ","\n")

    return dicsave




def save_result_model_holdout(current_model,
                              dictrip,
                              informations_dic,
                              build_train_test_for_model_parameters,
                              customscoring,
                              verbose=True):

    typename = "holdout"
    modelname = type(current_model).__name__
    dicsave = {}

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dictrip["x_train"],
                                                                        dictrip["y_train"],
                                                                        **informations_dic)
    current_model.fit(X_train,y_train)
    
    # Predictions part
    if "target_transformation" in build_train_test_for_model_parameters:
        transformed_predictions = current_model.predict(dictrip["x_test"])
        transformed_predictions[np.where(transformed_predictions<0)] = 0
        predictions = build_train_test_for_model_parameters["target_transformation"]["reciproque"](transformed_predictions)
        dicsave["transformed_predictions"] = transformed_predictions
    else:
        predictions = current_model.predict(dictrip["x_test"])

    # Score part
    transformed_predictions_holdout = current_model.predict(X_test)
    transformed_predictions_holdout[np.where(transformed_predictions_holdout<0)] = 0
    # predictions_holdout = build_train_test_for_model_parameters["target_transformation"]["reciproque"](transformed_predictions_holdout)
    score = customscoring(y_test,transformed_predictions_holdout)
    
    

    # saving part
    dicsave["typename"] = typename
    dicsave["modelname"] = modelname
    dicsave["predictions"] = predictions
    dicsave["features"] = dictrip["features"]
    dicsave["id_predictions"] = dictrip["x_test_id"]
    dicsave["model"] = current_model
    dicsave["score"] = score
    dicsave["informations"] = informations_dic
    
    
    
    if verbose:
        print("Type : ",typename,"\n",
              "Model : ",modelname,"\n",
              "Infos : ",informations_dic,"\n",
              "Features : ", len(dictrip["features"]),"\n",
              "Score : ",abs(dicsave["score"]),"\n",
              "","\n",
              "* * * * * * * * * * * * * * * * * * * * * * * * * ","\n")

    return dicsave
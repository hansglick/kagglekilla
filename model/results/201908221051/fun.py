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




def set_model_according_parameters(modelfamilly,combinaison):

    holdout_decision = is_hyperparam_for_holdout(combinaison)

    if not holdout_decision :
        
        # valeurs implicites
        combinaison,niter = clean_hyparam_up(combinaison,"implicit_niter",None)
        combinaison,kfold = clean_hyparam_up(combinaison,"implicit_kfold",5)
        combinaison,njobs = clean_hyparam_up(combinaison,"implicit_njobs",-1)
        combinaison,myscoringfun = clean_hyparam_up(combinaison,"implicit_scoring",None)
        
        hyparam = correct_hyparam_for_cv(combinaison)


        if niter is None:
            # GRIDSEARCH CASE
            
            model = set_model_for_gridsearch(modelfamilly,hyparam,kfold,myscoringfun,njobs)
            #print(model)
            hyparam["implicit_kfold"] = kfold
            hyparam["implicit_njobs"] = njobs
            hyparam["implicit_scoring"] = myscoringfun
            #print(model)
        else:
            # RANDOMSEARCH CASE
            model = set_model_for_randomsearch(modelfamilly,hyparam,kfold,myscoringfun,njobs,niter)
            hyparam["implicit_niter"] = niter
            hyparam["implicit_kfold"] = kfold
            hyparam["implicit_njobs"] = njobs
            hyparam["implicit_scoring"] = myscoringfun
    else:
        # HOLDOUT CASE
        hyparam = combinaison
        hyparam,ptest = clean_hyparam_up(hyparam,"implicit_ptest",0.3)
        hyparam,rtest = clean_hyparam_up(hyparam,"implicit_rtest",1)
        model = modelfamilly
        model.set_params(**hyparam)
        hyparam["implicit_ptest"] = ptest
        hyparam["implicit_rtest"] = rtest
        

    #print(model)
    #print("")
    #print("")
    return model,hyparam,holdout_decision




def clean_hyparam_up(dic,keyname,defaultval):
    dic_copy = dic.copy()
    
    if keyname in dic:
        solution = dic_copy[keyname]
        del dic_copy[keyname]
    else:
        solution = defaultval
        
    return dic_copy,solution



def set_model_for_gridsearch(model,hyparam,kfold,myscoringfun,njobs):
    
    hyparam_copy = hyparam.copy()
    gridsearchparam = {}
    gridsearchparam["estimator"] = model
    gridsearchparam["param_grid"] = hyparam_copy
    gridsearchparam["cv"] = kfold
    gridsearchparam["scoring"] = myscoringfun
    gridsearchparam["iid"] = False
    gridsearchparam["n_jobs"] = njobs
    gsmodel = model_selection.GridSearchCV(estimator = gridsearchparam["estimator"],
                                           param_grid = gridsearchparam["param_grid"])
    gsmodel.set_params(**gridsearchparam)

    
    
    return gsmodel




def set_model_for_randomsearch(model,hyparam,kfold,myscoringfun,njobs,niter):
    
    hyparam_copy = hyparam.copy()
    randsearchparam = {}
    randsearchparam["estimator"] = model
    randsearchparam["param_distributions"] = hyparam_copy
    randsearchparam["cv"] = kfold
    randsearchparam["scoring"] = myscoringfun
    randsearchparam["iid"] = False
    randsearchparam["n_jobs"] = njobs
    randsearchparam["n_iter"] = niter
    rsmodel = model_selection.RandomizedSearchCV(estimator = randsearchparam["estimator"],
                                                 param_distributions = randsearchparam["param_distributions"])
    rsmodel.set_params(**randsearchparam)
    
    return rsmodel





def save_result_model_non_holdout(current_model,dictrip,build_train_test_for_model_parameters,customscoring,verbose=True):

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
    
    if verbose:
        print(typename,", ",modelname,", ",len(dictrip["features"])," features. Best score sur données transformées : ", dicsave["best_score"])

    return dicsave




def save_result_model_holdout(current_model,dictrip,build_train_test_for_model_parameters,hyparam,customscoring,verbose=True):

    typename = "holdout"
    modelname = type(current_model).__name__
    dicsave = {}

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dictrip["x_train"],
                                                                        dictrip["y_train"],
                                                                        test_size=hyparam["implicit_ptest"],
                                                                        random_state=hyparam["implicit_rtest"])

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
    
    
    if verbose:
        print(typename,", ",modelname,", ",str(len(dictrip["features"]))," features. score : ", str(score))

    return dicsave



def run_all_settings(all_datasets_for_modeling,dic_of_models,dic_of_hyparam,build_train_test_for_model_parameters,customscoring):

    d = {}

    for k,v in all_datasets_for_modeling.items():
        keyslist = list(v.keys()).copy()

        for idata in keyslist:
            dictrip = all_datasets_for_modeling[k][idata]

            for idmodel,modelfamilly in dic_of_models.items():

                for idcombo,combinaison in enumerate(dic_of_hyparam[idmodel]):

                    idofmodel = str(k) + "_" + str(idata) + "_" + str(idmodel) + "_" + str(idcombo)
                    print("id : ",idofmodel)
                    current_model,hyparam,holdout_decision = set_model_according_parameters(modelfamilly,combinaison)
                    if holdout_decision:
                        res = save_result_model_holdout(current_model,dictrip,build_train_test_for_model_parameters,hyparam,customscoring)
                    else:
                        res = save_result_model_non_holdout(current_model,dictrip,build_train_test_for_model_parameters,customscoring)

                    d[idofmodel] = res
                    print("")
    return d
# Load librairies
import os
import sys
import joblib
import dill
import pandas as pd
import sklearn
from flask import Flask, jsonify, request
import json
from sklearn.neighbors import NearestNeighbors
import shap
import git
from importlib.metadata import version
# from custtransformer import CustTransformer
#from P7_functions import CustTransformer
from sklearn.decomposition import PCA


#########################################################
# Loading data and model 
#--------------------------------------------------------
# best model 
bestmodel = joblib.load('model_pret.joblib')

thresh = 0.8
#--------------------------------------------------------
# load data
dt = pd.read_csv('data.csv')
feat_imp_global = pd.read_csv('feat_imp_global.csv')
df = dt.drop(columns = 'TARGET')
#df = df[:500]
dta = df.drop(columns = 'SK_ID_CURR')
x_exple = shap.sample(dta, 2)
explainer = shap.KernelExplainer(bestmodel.predict_proba, x_exple)


###############################################################
# instantiate Flask object
app = Flask(__name__)
app.config["DEBUG"] = True
# view when API is launched
# Test local : http://127.0.0.1:5000

@app.route('/mysite', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('C:/Users/WILSON/Documents/projet7')
        origin = repo.remotes.origin
        origin.pull()
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400

@app.route("/")
def index():
    return "API loaded, models and data loaded, data computed…"

@app.route("/api/v1/resources/feat_imp")
def feat_imp():
    return feat_imp_global.to_json()

@app.route('/api/v1/resources/id_client')
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = pd.Series(list(df.SK_ID_CURR.sort_values()))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return jsonify({
                   'data': sk_ids_json})
                      
# return data of one customer when requested (SK_ID_CURR)
# Test local : http://127.0.0.1:5000/api/v1/resources/client?SK_ID_CURR=100128

@app.route('/api/v1/resources/client/')
def data_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = request.args.get('SK_ID_CURR', type=int)
    # Get the personal data for the customer (pd.Series)
    cust_ser = df.loc[df['SK_ID_CURR'] == sk_id_cust]
    # Convert the pd.Series (df row) of customer's data to JSON
    #cust_json = json.loads(cust_ser.to_json())
    # Return the cleaned data
    return cust_ser.to_json()
                       
                                           

# answer when asking for score and decision about one customer

@app.route('/api/v1/resources/score_client/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = request.args.get('SK_ID_CURR', type=int)
    # Get the data for the customer (pd.DataFrame)
    X_cust = df.loc[df['SK_ID_CURR'] == sk_id_cust]
    X_cust = X_cust.drop(columns = 'SK_ID_CURR')
    # Compute the score of the customer (using the whole pipeline)   
    score_cust = bestmodel.predict_proba(X_cust)[:,1][0]
    # Return score
    return jsonify({
                     'SK_ID_CURR': sk_id_cust,
                    'score': score_cust,
                    'thresh': thresh})

    
@app.get('/api/v1/resources/shap/')
def get_shap():
    """
    Calculates the probability of default for a client.  
  
    Returns:  
    - SHAP values (json).
    """
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = request.args.get('SK_ID_CURR', type=int)
    # Get the data for the customer (pd.DataFrame)
    X_cus = df.loc[df['SK_ID_CURR'] == sk_id_cust]
    X_cus = X_cus.drop(columns = 'SK_ID_CURR')
    #explainer = shap.KernelExplainer(bestmodel.predict_proba, dta)
    #shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust)[1]))
    shap_values = explainer.shap_values(X_cus)
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': X_cus.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)
    return df_shap.to_json()
    #return shap_vals_cust.to_json()

# find 20 nearest neighbors among the training set
def get_df_neigh(sk_id_cust):
    # get data of 20 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(df)
    X_cust = df.loc[df['SK_ID_CURR'] == sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=20,
                           return_distance=False).ravel()
    nearest_cust_idx = list(dt.iloc[idx].index)
    X_neigh_df = dt.loc[nearest_cust_idx, :]
    
    return X_neigh_df

# return data of 20 neighbors of one customer when requested (SK_ID_CURR)

@app.route('/api/v1/resources/neigh_client/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = request.args.get('SK_ID_CURR', type=int)
    # return the nearest neighbors
    X_neigh_df = get_df_neigh(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    # Return the cleaned data jsonified
    return jsonify({
                    'X_neigh': X_neigh_json
                    })

# Return list of features 
@app.route('/api/v1/resources/features')
def feat_clt():
    # Extract list of features
    feat_cl = pd.Series(df.columns.values.tolist())
    # Convert pd.Series to JSON
    feat_cl_json = json.loads(feat_cl.to_json())
    # Returning the processed data
    return jsonify({
                   'data': feat_cl_json})


if __name__ == "__main__" :
    app.run()
# import packages
# ------------------------------------
import requests
import json
import joblib
from pandas import json_normalize
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from PIL import Image
    # -----------------------------------------------
st.set_page_config(page_title='Tableau de bord de demandes de crédit',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
# Display the title
st.title('Tableau de bord de demandes de prêt')
st.subheader("WILSON Adjete - Data Scientist")
# Display the LOGO
img = Image.open("LOGO.png")
st.sidebar.image(img, width=250)
# Local URL: http://localhost:8501
API_URL = "https://wilsonadjete.pythonanywhere.com/api/v1/resources/"
# # Display the loan image
img = Image.open("loan.jpg")
st.image(img, width=100)
    
TIMEOUT = (5, 30)   
bestmodel = joblib.load('model_pret.joblib')
@st.cache_data
def get_id_list():
    # URL of the sk_id API
    id_api_url = API_URL + "id_client"
    # Requesting the API and saving the response
    response = requests.get(id_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of "ID" from the content
    id_customers = pd.Series(content['data']).values
    #id_customers = pd.Series(response['data']).values
    return id_customers

@st.cache_data
def get_selected_cust_data(selected_id):
    # URL of the sk_id API
    data_api_url = API_URL + "client/?SK_ID_CURR=" + str(selected_id)
    # Requesting the API and saving the response
    response = requests.get(data_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    x_custom = pd.DataFrame(content)
    # x_cust = json_normalize(content['data'])
    #y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
    # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
    #print(100*'on affiche x_custom',x_custom)
    return x_custom

@st.cache_data
def get_data_neigh(selected_id):
    # URL of the scoring API (ex: SK_ID_CURR = 100005)
    neight_data_api_url = API_URL + "neigh_client/?SK_ID_CURR=" + str(selected_id)
    # save the response of API request
    response = requests.get(neight_data_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # convert data to pd.DataFrame and pd.Series
    # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
    # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
    # data_all_customers = pd.DataFrame(content['data_all_cust'])
    data_neig = pd.DataFrame(content['X_neigh'])
    #target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
    return data_neig

# Get score (cached)
@st.cache_data
def get_score_model(selected_id):
    # URL of the sk_id API
    score_api_url = API_URL + "score_client/?SK_ID_CURR=" + str(selected_id)
    # Requesting the API and saving the response
    response = requests.get(score_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # Getting the values of "ID" from the content
    score_model = (content['score'])
    threshold = content['thresh']
    return score_model, threshold

@st.cache_data
def values_shap(selected_id):
    # URL of the sk_id API
    shap_values_api_url = API_URL + "shap/?SK_ID_CURR=" + str(selected_id)
    # Requesting the API and saving the response
    response = requests.get(shap_values_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    content = pd.DataFrame(content)
    # Getting the values of "ID" from the content
    print(content)
    return content

@st.cache_data
def feat_imp():
    # URL of the sk_id API
    feat_imp_api_url = API_URL + "feat_imp"
    # Requesting the API and saving the response
    response = requests.get(feat_imp_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    content = pd.DataFrame(content)
    # Getting the values of "ID" from the content
    #print(content)
    return content

@st.cache_data
def feat():
    # URL of the sk_id API
    feat_api_url = API_URL + "features"
    # Requesting the API and saving the response
    response = requests.get(feat_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of "ID" from the content
    features_name = pd.Series(content['data']).values
    return features_name



#Selected id
# list of customer's ID's
cust_id = get_id_list()
# Selected customer's ID
selected_id = st.sidebar.selectbox('Sélectionnez le numéro de client dans la liste:', cust_id, key=18)
st.write('Votre identifiant sélectionné = ', selected_id)

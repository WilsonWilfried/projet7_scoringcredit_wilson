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
from shap.plots import waterfall
import matplotlib.pyplot as plt
from PIL import Image
import streamlit.components.v1 as components

def main():

    # -----------------------------------------------
    st.set_page_config(page_title='Tableau de bord de demandes de crédit',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.title('Tableau de bord de demandes de prêt')
    st.subheader("WILSON Adjete - Data Scientist")
    
    # Local URL: http://localhost:8501
    API_URL = "http://127.0.0.1:5000/api/v1/resources/"
    
    # Display the LOGO
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=250)

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
    
    
    # Gauge Chart
    @st.cache_data
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "green",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'lightgreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig
    
     #                         Customer's data checkbox
        
    if st.sidebar.checkbox("Information sur le client"):
        st.markdown('données du client sélectionné :')
        data_selected_cust = get_selected_cust_data(selected_id)
        # data_selected_cust.columns = data_selected_cust.columns.str.split('.').str[0]
        st.write(data_selected_cust)
      #                         Model's decision checkbox
    if st.sidebar.checkbox("Décision sur le crédit", key=38):
        # Get score & threshold model
        score, threshold_model = get_score_model(selected_id)
        # Display score (default probability)
        st.write('score obtenue : {:.0f}%'.format(score * 100))
        # Display default threshold
        st.write('seuil à atteindre pour accord de credit : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if score >= threshold_model:
            decision = "Prêt accordé"
        else:
            decision = "Prêt rejetté"
        st.write("Decision :", decision)
        
        figure = gauge_plot(score, threshold_model)
        st.write(figure)
        # Add markdown
        st.markdown('_Compteur à jauge pour le client demandeur._')
     
        #                 Display local SHAP waterfall checkbox
        if st.checkbox('les informations qui agissent sur la décision en générale', key=25):
            feat_imp_global = feat_imp()
            fig2, ax = plt.subplots(figsize=(20, 15))
            plt.title("Feature importances", fontsize=30)
            sns.barplot(x=feat_imp_global['score'], y=feat_imp_global['feature'])
            plt.xlabel('Importances', fontsize=26)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.gcf()
            st.pyplot(fig2)
        if st.checkbox('Interpretation de la décision du client', key=30):
            fig3, ax = plt.subplots(figsize=(20, 15))
            with st.spinner('SHAP bar plots displaying in progress..... Please wait.......'):
                nb_features = st.slider("Nombre d'informations à afficher",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # Get Shap values for customer & expected values
                shap_df = values_shap(selected_id) 
                shap.bar_plot(shap_df['SHAP value'], feature_names=shap_df['feature'],                                     max_display=nb_features)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                #plt.xlabel(fontsize=26)
                plt.gcf()
                st.pyplot(fig3)
                st.markdown('_SHAP bar pour le client demendeur._')
                # Add details title
                expander = st.expander("A propos de SHAP bar plot...")
                expander.write("Plus la barre est longue, plus son importance est grande dans la prise de décision.les barres rouges vers la droite contribuent à une décision favorable à l'accord de credit, alors que les barres bleues vers la gauche contribuent à une décision défavorable à l'accord de credit.")
        if st.checkbox('comparer le client aux autres clients similaires', key=20):
            st.header('Comparaison aux autres clients')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner(' creation in progress...please wait.....'):
                # Get features names
                features = feat()
                # Get selected columns
                #disp_box_cols = get_list_display_features(features, 2, key=45)
                box_cols = st.multiselect(
                 'choisis les informations à comparer:',
                  sorted(features),max_selections=2,
                   key=45)
                if len(box_cols) != 2 :
                    st.write('Sélectionner 2 information à comparer')
                else :
                    
                    data_neigh = get_data_neigh(selected_id)
                    x_cus = get_selected_cust_data(selected_id)
                    data_neigh_box = data_neigh.loc[:, box_cols]
                    #box_cols= ['FLAG_OWN_REALTY','AMT_INCOME_TOTAL']
                    x_cus_box = x_cus.loc[:, box_cols]
                    #print(x_cus_box.iloc[:,1])
                    #print(x_cus_box.iloc[:,0])
                    #print(type(x_cus_box))
                    plt.scatter(data_neigh_box.iloc[:,0], data_neigh_box.iloc[:,1], s = 200, c = 'yellow', marker = '*', edgecolors = 'black')
                    plt.scatter(x_cus_box.iloc[:,0], x_cus_box.iloc[:,1], s = 400, c = 'red', marker = 'o', edgecolors = 'red')
                    ax.set_xlabel(box_cols[0], fontsize=24)
                    ax.set_ylabel(box_cols[1], fontsize=24)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.gcf()
                    st.pyplot(fig)
                    expander = st.expander("A propos du graphique...")
                    expander.write("Le client est comparé à d'autres clients similaires, maximum 20.il faudra choisir dans la liste déroulante deux informations sur lesquelles la comparaison a été faite.Sur le graphique le client demandeur est représenté par le point rouge, et les autres clients par les étoiles")
                  
    main()
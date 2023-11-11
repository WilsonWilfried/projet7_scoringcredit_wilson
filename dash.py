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

# # Display the loan image
img = Image.open("loan.jpg")
st.image(img, width=100)
    
TIMEOUT = (5, 30)   

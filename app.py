import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class


@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

url = "https://www.linkedin.com/in/josu%C3%A9-afouda/"

def main():
    st.title("JoshAutoML")
    st.sidebar.write("[Author: Josué AFOUDA ](%s)" % url)
    st.sidebar.markdown(
        "**This web app is a No-Code tool for Exploratory Data Analysis and building Machine Learning model for Regression and Classification tasks.**\n"            
        "1. Load your dataset file (CSV file);\n"            
        "2. Click on *Profile Dataset* button in order to generate the pandas profiling of the dataset;\n"
        "3. Choose your target column;\n"
        "4. Choose the machine learning task (Regression or Classification);\n"
        "5. Click on *Run Modelling* in order to start the training process.\n"
        "When the model is built, you can view the results like the pipeline model, Residuals plot, ROC Curve, Confusion Matrix, Feature importance, etc.\n"
        "\n6. Download the Pipeline model in your local computer."
    )
    st.set_option('deprecation.showfileUploaderEncoding', False)

    file = st.file_uploader("Upload your dataset in CSV format", type=["csv"])
    if file is not None:
        data = load_data(file)
        st.dataframe(data.head())
        profile = st.button("Profile Dataset")
        if profile:
            profile_df = data.profile_report()
            st_profile_report(profile_df)

        target = st.selectbox("Select the target variable", data.columns)

        data = data.dropna(subset=[target]) # au cas où il y aura des valeurs manquantes dans la variable cible

        task = st.selectbox("Select a task", ["Regression", "Classification"])

        if task == "Regression":
            if st.button('Run Modelling'):
                exp_reg = setup_reg(data, target = target)
                model_reg = compare_models_reg()
                save_model_reg(model_reg, 'best_reg_model')
                st.success("Regression Model built Successfully!")

                # Results
                st.write("Residuals")
                plot_model_reg(model_reg, plot = 'residuals', save=True)
                st.image("Residuals.png")
                
                st.write("Feature importance")
                plot_model_reg(model_reg, plot = 'feature', save=True)
                st.image("Feature Importance.png")

                with open('best_reg_model.pkl', 'rb') as f: 
                    st.download_button('Download Model', f, file_name="best_reg_model.pkl")
                    
        
        if task == "Classification":
            if st.button('Run Modelling'):
                exp_class = setup_class(data, target = target)
                model_class = compare_models_class()
                save_model_class(model_class, 'best_class_model')
                st.success("Classification Model built Successfully!")

                # Results
                col5, col6 = st.columns(2)
                with col5:
                    st.write("ROC curve")
                    plot_model_class(model_class, save=True)
                    st.image("AUC.png")
                
                with col6:
                    st.write("Classification Report")
                    plot_model_class(model_class, plot = 'class_report', save=True)
                    st.image("Class Report.png")
                
                col7, col8 = st.columns(2)
                with col7:
                    st.write("Confusion Matrix")
                    plot_model_class(model_class, plot = 'confusion_matrix', save=True)
                    st.image("Confusion Matrix.png")
                
                with col8:
                    st.write("Feature Importance")
                    plot_model_class(model_class, plot = 'feature', save=True)
                    st.image("Feature Importance.png")

                # Download the model
                with open('best_class_model.pkl', 'rb') as f: 
                    st.download_button('Download Model', f, file_name="best_class_model.pkl")
    
    else:
        st.image("https://github.com/JosueAfouda/JoshAutoML/raw/main/home-image.png")

if __name__=='__main__':
    main()
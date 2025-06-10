import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os

st.title("TensoraLabs - Data Preprocessing")
st.write("Data preprocessing without writing any line of code.")

file = st.file_uploader("Upload your CSV File:", type='.csv')

if file:
    file_name = os.path.splitext(file.name)[0]
    df_ = pd.read_csv(file)
    st.write("Original Data:")
    st.write(df_.head(10))  
    
    df = df_.dropna()
    df = pd.get_dummies(df)

    selected_column = st.selectbox("Select Column for Outlier Detection:", df.columns)
    
    btn = st.button("Preprocess Data")
    
    if btn:
        if selected_column in df.columns:
            df['Z_Score'] = stats.zscore(df[selected_column])
            threshold = 3
            
            outliers = df[np.abs(df['Z_Score']) > threshold]
            df_no_outliers_ = df[np.abs(df['Z_Score']) <= threshold]
            
            df_no_outliers = df_no_outliers_.drop(columns=['Z_Score'])
            outliers = outliers.drop(columns=['Z_Score'])
            
            st.write("Detected Outliers:")
            st.write(df_no_outliers_)
            csv_data_ = df_no_outliers_.to_csv(index=False)
            st.download_button(
                label="Download Outlier Data",
                data=csv_data_,
                file_name=f"{file_name}_outlier_data.csv",
                mime="text/csv"

            )
        else:
            st.warning(f"Column '{selected_column}' not found after encoding.")
            df_no_outliers = df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_no_outliers)
        X_scaled = pd.DataFrame(X_scaled, columns=df_no_outliers.columns)
        
        st.write("Preprocessed Data (Outliers Removed):")
        st.write(X_scaled)
        
        st.write(f"Raw Data Shape: {df_.shape}")
        st.write(f"Preprocessed Data Shape: {X_scaled.shape}")
        
        st.success("Preprocessing Completed!")
        
        csv_data = X_scaled.to_csv(index=False)
        
        st.download_button(
            label="Download Preprocessed Data as CSV",
            data=csv_data,
            file_name=f"{file_name}_preprocessed_data.csv",
            mime="text/csv"
        )
        
        st.balloons()

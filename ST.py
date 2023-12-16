import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from streamlit_option_menu import option_menu

with st.sidebar:
    
    selected = option_menu('Cancer Predictor ',
                          
                          ['Home page','Breast Cancer',
                           'Kidney Cancer',
                           'Skin Cancer'],
                          icons=['house','heart','activity','person'],
                          default_index=0)
    

if (selected == 'Home page'):
     st.title("Welcome to Cancer Predictor!")

     st.write("""
    <div style="font-family: 'Calibri', sans-serif; font-size: 32px; border: 2px solid #808080; padding: 5px; border-radius: 5px;">
        <p> This application aims to predict the type of cancer (Breast Cancer, Skin Cancer, Kidney Cancer) based on cell characteristics.
       <br>
       The data used in this application is collected from a database on KAGGLE.
       <br>
       It is important to note that the predictions provided by this application are based on statistical models and should not replace a professional diagnosis.
       <br>
       This application remains a decision support tool that can assist healthcare professionals in the diagnostic process, but should not be used as a substitute for a professional diagnosis.</p>
    </div>
""", unsafe_allow_html=True)
     

     

elif (selected == 'Breast Cancer'):
 def get_clean_data():

    df = pd.read_csv('data/data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # convert Malignant to 1 and benign to 0
    return df

 def add_sidebar(data):

    st.sidebar.header("Cell Nuclei Measurements")
    input_dict = {}
    column_names = data.columns[1:]
    sliders_labels = [(f"{column} (mean)", column) for column in column_names]

    for label, key in sliders_labels:

        input_dict[key] = st.sidebar.slider(

            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())

        )
    return input_dict

 def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():

        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

 def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    values1 = [input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
               input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
               input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
               input_data['fractal_dimension_mean']]

    values2 = [input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
               input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
               input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']]

    values3 = [input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
               input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
               input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
               input_data['fractal_dimension_worst']]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(

        r=values1,
        theta=categories,
        fill='toself',
        name='mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories,
        fill='toself',
        name='standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values3,
        theta=categories,
        fill='toself',
        name='worst value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )
    return fig




 def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)  # Reshape to a 2D array
    scaled_input_array = scaler.transform(input_array)
    
    # Assuming the model.predict() method is used for predictions
    prediction = model.predict(scaled_input_array)
    st.subheader("Cell cluster prediction ")
    st.write("The cell cluster is:")
    if prediction[0]==0:
        st.write("Benign ✅")

    else:
        st.write("Malignant ❌")
        st.write("""
    <div style="border: 2px solid #808080; padding: 5px; border-radius: 5px;">
        <p> The person needs a specific treatment.
       <br>
       Doctors on cancer treatment team might include:
       <br>
       -A breast surgeon or surgical oncologist: a doctor who uses surgery to treat breast cancer.
       <br>
       -A radiation oncologist: a doctor who uses radiation to treat cancer.
       <br>
       -A medical oncologist: a doctor who uses chemotherapy, hormone therapy, immunotherapy, and other medicines to treat cancer.</p>
    </div>
""", unsafe_allow_html=True)

    st.write("In terms of probability :")
    st.write("Probability of being benign:",model.predict_proba(scaled_input_array)[0][0])
    st.write("Probability of being malignant:",model.predict_proba(scaled_input_array)[0][1])
    
 


 def main():
    
 
    with st.container():
        st.title("Breast Cancer")
        st.write("Breast cancer is a type of cancer that originates in the cells of the breast. It is one of the most common cancers affecting women, but it can also occur in men.In 2020, there were approximately 685,000 deaths from breast cancer globally.")

    data = get_clean_data()
    input_data = add_sidebar(data)

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        add_predictions(input_data)
 
 
 if __name__ == "__main__":
    main()




elif (selected == 'Kidney Cancer'):



#loading saved models

 t_model = pickle.load(open('C:/Users/HP/CANCER PRED/kidney cancer/t_model.sav', 'rb'))

    
    # page title
 st.title('Kidney Cancer')
 st.write("""
    <div style="border: 2px solid #808080; padding: 5px; border-radius: 5px;">
        <p>Kidney cancer, also known as renal cell carcinoma (RCC), is a type of cancer that originates in the kidneys, the organs responsible for filtering waste products from the blood and producing urine.Overall, the lifetime risk for developing kidney cancer in men is about 1 in 46 (2.02%) and is about 1 in 80 (1.03%) for women.</p>
    </div>
""", unsafe_allow_html=True)
 st.write("LET'S BEGIN THE TEST")
    
    # getting the input data from the user
 col1, col2, col3 = st.columns(3)
    
 with col1:
        age = st.text_input('Age')
        
 with col2:
        bloodpressure = st.text_input('Blood Pressure')
    
 with col3:
        sg = st.text_input('Specific Gravity')
    
 with col1:
        al = st.text_input('Albumin')
    
 with col2:
        su = st.text_input('Sugar')
    
 with col3:
        pc = st.text_input('Pus Cell')
    
 with col1:
        bgr = st.text_input('Blood Glucose')
    
 with col2:
        bu = st.text_input('Blood Urea')

 with col3:
        sc = st.text_input('Serum Creatinine')

 with col1:
        hemo = st.text_input('Hemoglobin')
    
 with col2:
        pcv = st.text_input('Packed Cell Volume')

 with col3:
        wc = st.text_input('White Blood Cell Count')

 with col1:
        rc = st.text_input('Red Blood Cell Count')
    
 with col2:
        appet = st.text_input('Appetite')

 with col3:
        pe = st.text_input('Peda Edema') 
        
 with col1:
        ane = st.text_input('Anemia')        
    
    
    # code for Prediction
 kidney_diagnosis = ''
    
    # creating a button for Prediction
    
 if st.button('Kidney Cancer Test Result'):
        kidney_prediction = t_model.predict([[age, bloodpressure, sg, al, su, pc, bgr, bu, sc, hemo, pcv, wc, rc, appet, pe, ane]])
        
        if (kidney_prediction[0] == 1):
          kidney_diagnosis = st.markdown("<span style='color:red'>The person have a kidney cancer</span>", unsafe_allow_html=True)
          st.write("""
    <div style="border: 2px solid #808080; padding: 5px; border-radius: 5px;">
        <p>The person needs a specific treatment.
        <br>
        Doctors on cancer treatment team might include:   
        <br>                  
        A urologist: a doctor who specializes in treating diseases of the urinary system.
        <br>
        A radiation oncologist: a doctor who treats cancer with radiation therapy.
        <br>
        A medical oncologist: a doctor who treats cancer with medicines such as chemotherapy, targeted therapy, or immunotherapy.</p>           
    </div>
""", unsafe_allow_html=True)
        else:
          kidney_diagnosis = 'The person do not have a kidney cancer'
          st.success(kidney_diagnosis)


elif (selected == 'Skin Cancer'):
    import streamlit as st
    import numpy as np
    from PIL import Image
    import torch
    from torchvision import transforms
    import torch.nn as nn

    # Load the pre-trained skin cancer classification model
    model_path = "C:/Users/HP/CANCER PRED/kidney cancer/model_danni.pt"  # Replace with the actual path to your model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    


    

    # Transformation for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7630392, 0.5456477, 0.57004845], std=[0.1409286, 0.15261266, 0.16997074])
    ])

    # Streamlit app interface
    st.title("Skin Cancer ")
    st.write("Upload an image for skin cancer classification.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = Image.open(uploaded_file)
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(img)

        # Get predicted class
        predicted_class = torch.argmax(output).item()

        # Display result
        st.image(Image.open(uploaded_file), caption="Uploaded Image.", use_column_width=True)
        st.write("Predicted Class:", predicted_class)
    

     
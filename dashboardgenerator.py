import streamlit as st
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from vertexai.preview.language_models import TextGenerationModel

# Initialize the Streamlit app
st.set_page_config(page_title="ChartGenerator", page_icon=":bar_chart:", layout="wide")

# Apply the theme
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content .sidebar-header {
        background-color: #1e3a8a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: #ffffff;
        border-radius: 4px;
        border: none;
    }
    .stImage {
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.image("https://www.nicesoftwaresolutions.com/logo.png", width=150)
st.sidebar.title("InsightsBoard")

# User input for Google Generative AI API Key
api_key = st.sidebar.text_input("Enter Google Generative AI API Key", type="password")

def configure_api(api_key):
    genai.configure(api_key=api_key)

if api_key:
    configure_api(api_key)

# User input for number of visualizations
num_visuals = st.sidebar.slider("Number of Visualizations", 1, 10, 3)

# Option to upload file or connect to database
data_source = st.sidebar.selectbox("Data Source", ["Upload File", "Connect to Database"])

# Load data based on user input
data = None
if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
else:
    db_url = st.sidebar.text_input("Database URL")
    table_name = st.sidebar.text_input("Table Name")
    if st.sidebar.button("Load Data"):
        try:
            engine = sqlalchemy.create_engine(db_url)
            data = pd.read_sql_table(table_name, engine)
        except Exception as e:
            st.error(f"Error loading data from database: {str(e)}")

# Function to get insights using Google Generative AI
def get_insights(api_key, data):
    genai.configure(api_key=api_key)
    
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    input_prompt = f"Analyze the following data and generate insights:\n\n{data.head().to_string()}"
    response = model.predict(input_prompt, **parameters)
    insights = response.text
    return insights

# Function to generate visualizations
def generate_visuals(data, insights, num_visuals):
    visuals = []
    insights_list = insights.split('\n')
    for i in range(num_visuals):
        kpi = insights_list[i % len(insights_list)]
        fig, ax = plt.subplots()
        if "bar" in kpi:
            sns.barplot(x=data.columns[0], y=data.columns[1], ax=ax)
        elif "line" in kpi:
            sns.lineplot(x=data.columns[0], y=data.columns[1], ax=ax)
        elif "scatter" in kpi:
            sns.scatterplot(x=data.columns[0], y=data.columns[1], ax=ax)
        ax.set_title(kpi)
        visuals.append(fig)
    return visuals

# Main panel for dashboard
if data is not None:
    st.header("Generated Dashboard")
    insights = get_insights(api_key, data)
    visuals = generate_visuals(data, insights, num_visuals)
    for fig in visuals:
        st.pyplot(fig)
else:
    st.info("Please upload a file or connect to a database.")

# Text box for prompting
prompt_text = st.text_area("Customizations Prompt")

# Apply customizations based on user input
if prompt_text:
    st.write(f"Customizations: {prompt_text}")
    # Implement customizations based on prompt_text

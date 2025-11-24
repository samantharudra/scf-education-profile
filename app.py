import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------------------
# Load model objects once
# -------------------------------------------
@st.cache_resource
def load_objects():
    with open("model_objects.pkl", "rb") as f:
        objects = pickle.load(f)
    return objects

objects = load_objects()

model_pipeline = objects['pipeline']
label_encoder = objects['label_encoder']
cluster_summary_median = objects['cluster_summary_median']

# -------------------------------------------
# Category maps
# -------------------------------------------
married_map = {1: "Married/living with partner", 2: "Neither married nor living with partner"}
race_map = {1: "White", 2: "Black", 3: "Hispanic/Latino", 4: "Other"}

# -------------------------------------------
# Cluster descriptions (long paragraphs)
# -------------------------------------------
cluster_paragraphs = {
    1: """
### **Cluster 1: Older, Very Wealthy Households**

Cluster 1 includes households who have built substantial wealth over time.  
Members of this cluster are typically **older**, overwhelmingly **married**, and disproportionately **white**.  
They hold **very high net worth**, **high incomes**, and modest, manageable debt levels.  

Their financial and demographic profiles reflect **long-term stability, accumulated assets, and strong economic security**.
""",
    2: """
### **Cluster 2: Low-Wealth, Lower-Income, Low-Debt Households**

Cluster 2 households have **very low net worth**, **lower incomes**, and **minimal debt**.  
They are more likely to be **unmarried**, **middle-aged**, and disproportionately **Black**.  
Education levels are lower, and these households often face liquidity constraints and structural barriers that inhibit wealth accumulation.  

This cluster reflects households experiencing **persistent economic insecurity**.
""",
    3: """
### **Cluster 3: Younger, Moderate-Income, High-Debt Households**

Cluster 3 consists primarily of **younger** households with **moderate incomes** and **very high debt burdens**, especially **education debt**.  
They are racially diverse, representing both Black and white households.  
These profiles suggest early-career individuals investing in education and careers while carrying substantial debt.  

This cluster highlights **upwardly mobile but financially leveraged households**.
"""
}

# -------------------------------------------
# Helper: Format median table
# -------------------------------------------
def styled_table(series):
    df = pd.DataFrame(series)
    df.columns = ["Median Value"]

    return df.style.format({
        "Median Value": lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and abs(x) > 100 else x
    }).set_properties(**{
        "background-color": "#f0f2f6",
        "color": "#000",
        "border-color": "black",
        "font-size": "16px"
    }).set_table_styles([
        {"selector": "th", "props": [("background-color", "#0047AB"), ("color", "white"), ("font-size", "16px")]}
    ])

# -------------------------------------------
# Helper: Cluster summary renderer
# -------------------------------------------
def render_cluster_summary(cluster_id):
    st.markdown(cluster_paragraphs[cluster_id])

    median_stats = cluster_summary_median.loc[cluster_id].copy()

    # Convert coded medians to readable text
    median_stats['MARRIED'] = married_map.get(round(median_stats['MARRIED']), "Unknown")
    median_stats['RACECL4'] = race_map.get(round(median_stats['RACECL4']), "Unknown")

    median_stats['AGE'] = f"{median_stats['AGE']:.0f}"
    median_stats['EDUC'] = f"{median_stats['EDUC']:.1f} yrs"

    st.markdown("### **Median Household Characteristics**")
    st.dataframe(styled_table(median_stats), use_container_width=True)

# -------------------------------------------
# STREAMLIT PAGE TITLE
# -------------------------------------------
st.title("Education, Debt, and Household Financial Structure in the United States")
st.write("### **By: Samantha Rudravajhala**")
st.markdown("---")

# -------------------------------------------
# TABS
# -------------------------------------------
overview_tab, tab1, tab2, tab3, tab4 = st.tabs([
    "Project Overview",
    "Cluster 1 Overview",
    "Cluster 2 Overview",
    "Cluster 3 Overview",
    "Build Your Profile"
])

# -------------------------------------------
# PROJECT OVERVIEW TAB
# -------------------------------------------
with overview_tab:

    st.header("Project Overview")

    st.markdown("""
This app explores the relationship between **educational attainment**, **household financial structure**, and **socioeconomic mobility**, using data from the **Survey of Consumer Finances (SCF)**.

### **Methods Used**

#### **1. Hierarchical Clustering**
I used hierarchical clustering to group U.S. households into three segments based on:
- Income  
- Net worth  
- Debt  
- Age  
- Race  
- Marital status  

These clusters represent meaningful financial profiles that characterize different economic realities among U.S. households.

#### **2. XGBoost Regression for Education Prediction**
An XGBoost classifier predicts **highest level of education completed** based on:
- Certain financial characteristics  
- Basic demographics  
- Cluster membership

This provides a way to understand how education aligns with broader patterns of financial well-being.

---

### **Motivation**

Higher education is widely seen as a primary driver of economic mobility. Students often take on significant debt with expectations of improved future earnings. Understanding how education interacts with asset accumulation, debt burdens, and household financial profiles can highlight important trends in economic opportunity.

This project:
- Examines education in the context of full financial profiles  
- Highlights structural inequities contributing to persistent wealth gaps  
- Connects educational outcomes to household economic realities  
- Helps inform policies that expand access and reduce disparities in postsecondary attainment  

---

### **How to Use This App**

1. **Browse Clusters:**  
   Explore the three household clusters created from the SCF data.  

2. **Identify Your Profile:**  
   Get a sense of which cluster best matches your financial characteristics or life stage.

3. **Build Your Own Profile:**  
   Enter your own demographic and financial information.

4. **Predict Your Education Category:**  
   The model will predict educational attainment based on patterns from the SCF dataset.

This app is designed to help you explore how education interacts with financial structure and where you may fit within the broader American financial landscape.

""")

# -------------------------------------------
# CLUSTER TABS
# -------------------------------------------
with tab1:
    render_cluster_summary(1)

with tab2:
    render_cluster_summary(2)

with tab3:
    render_cluster_summary(3)

# -------------------------------------------
# BUILD YOUR PROFILE TAB
# -------------------------------------------
with tab4:

    st.header("Build Your Financial Profile & Predict Education Level")

    cluster = st.selectbox("Select Your Cluster", options=[1, 2, 3])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    married = st.selectbox("Marital Status", options=list(married_map.values()))
    race = st.selectbox("Race Category", options=list(race_map.values()))
    income = st.number_input("Annual Income ($)", min_value=0, max_value=5_000_000, value=50_000, step=1000)
    networth = st.number_input("Net Worth ($)", min_value=-2_000_000, max_value=20_000_000, value=100_000, step=1000)

    if st.button("Predict Education Category"):

        married_code = {v: k for k, v in married_map.items()}[married]
        race_code = {v: k for k, v in race_map.items()}[race]

        log_income = np.log(income + 1)
        log_networth = np.log(networth + 1) if networth >= 0 else 0

        input_df = pd.DataFrame({
            'hier_cluster': [cluster],
            'AGE': [age],
            'MARRIED': [married_code],
            'RACECL4': [race_code],
            'log_INCOME': [log_income],
            'log_NETWORTH': [log_networth]
        })

        pred_encoded = model_pipeline.predict(input_df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"🎓 **Predicted Education Category: {pred_label}**")

        st.markdown("### **Why This Category?**")
        st.markdown("""
The model predicts education level based on thousands of real SCF households with similar
demographic and financial characteristics. Your predicted category reflects the statistical
patterns observed among people with comparable **income, wealth, race, marital status, and cluster membership**.
""")

        st.markdown(f"### **Cluster {cluster} Reference Profile**")
        render_cluster_summary(cluster)

import streamlit as st
import pandas as pd
import pymysql

# Set up Streamlit page config
st.set_page_config(page_title="Live Data Dashboard", layout="wide")

# Title
st.title("📊 Live Data Dashboard")

# Database connection
def get_connection():
    return pymysql.connect(host="localhost", user="root", password="", database="system_state")

# Query and load DataFrame
def load_data():
    conn = get_connection()
    query = "SELECT * FROM states1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Load data
df = load_data()

# Display raw data
st.subheader("Raw Data")
st.dataframe(df)

# Clean column names
df.columns = df.columns.str.strip()

# Plot each numeric column with its name
st.subheader("Live Column Charts")

num_cols = df.select_dtypes(include="number").columns

if len(num_cols) == 0:
    st.warning("No numeric columns found.")
else:
    cols = st.columns(2)  # 2-column layout
    for i, col in enumerate(num_cols):
        with cols[i % 2]:
            st.markdown(f"#### {col}")  # 👈 Title above plot
            st.line_chart(df[col])

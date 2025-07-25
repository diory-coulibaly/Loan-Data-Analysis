import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches

st.set_page_config(layout="wide")
st.title("Loan Data Analysis Dashboard")

# 📷 Quick preview of expected CSV format
from PIL import Image
image = Image.open("loanData.JPG")
st.image(image, caption="", use_container_width=True, output_format="JPEG")


@st.cache_data
def optimize_dtypes(df):
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

uploaded_file = st.file_uploader("Upload your loan data CSV file", type=["csv"])
if uploaded_file:
    loan_data = pd.read_csv(uploaded_file)
    loan_data.columns = loan_data.columns.str.replace('.', '_', regex=False)
    loan_data = optimize_dtypes(loan_data)

    st.subheader("Data Preview")
    st.dataframe(loan_data.head())

    # Histogram: int_rate with annotation
    st.subheader("Interest Rate Distribution by Credit Policy")
    fig, axs = plt.subplots(nrows=2, figsize=(10,8), sharex=True)
    for i, policy in enumerate([0, 1]):
        subset = loan_data[loan_data["credit_policy"] == policy]
        sns.histplot(subset["int_rate"], kde=False, ax=axs[i])
        mean_val = subset["int_rate"].mean()
        axs[i].axvline(mean_val, color='r')
        axs[i].set_ylabel("Counts")
        axs[i].text(0.16, axs[i].get_ylim()[1]*0.75,
                    f'{"Risky" if policy==0 else "Healthy"} Loans. Mean={round(mean_val,2)}',
                    fontsize=13, color='black')
    st.pyplot(fig)

    # Histogram: dti with annotation
    st.subheader("Debt-to-Income Ratio Distribution by Credit Policy")
    fig, axs = plt.subplots(nrows=2, figsize=(10,8), sharex=True)
    for i, policy in enumerate([0, 1]):
        subset = loan_data[loan_data["credit_policy"] == policy]
        sns.histplot(subset["dti"], kde=False, ax=axs[i])
        mean_val = subset["dti"].mean()
        axs[i].axvline(mean_val, color='r')
        axs[i].set_ylabel("Counts")
        axs[i].set_xlabel("Debt-to-Income Ratio", fontsize=14)
        axs[i].text(20, axs[i].get_ylim()[1]*0.75,
                    f'{"Risky" if policy==0 else "Healthy"} Loans. Mean={round(mean_val,2)}',
                    fontsize=13, color='black')
    st.pyplot(fig)

    # DTI by purpose
    st.subheader("Debt-to-Income by Loan Purpose and Credit Policy")
    mean_dti = loan_data.groupby(["purpose", "credit_policy"])["dti"].mean().reset_index()
    fig, ax1 = plt.subplots(figsize=(10,8))
    sns.barplot(data=mean_dti, x="purpose", y="dti", hue="credit_policy", ax=ax1)
    plt.xticks(rotation=90, fontsize=14)
    plt.ylabel("Debt to Income Ratio", fontsize=14)
    plt.xlabel("Purpose", fontsize=13)
    plt.title("Purpose vs DTI for Healthy and Risky Loans", fontsize=14)

    ax2 = ax1.twinx().twiny()
    plt.axhline(y=loan_data[loan_data["credit_policy"]==0]["dti"].mean(),
                color='blue', linestyle='--', label="Avg DTI for Risky loans")
    plt.axhline(y=loan_data[loan_data["credit_policy"]==1]["dti"].mean(),
                color='orange', linestyle='--', label="Avg DTI for Healthy loans")
    plt.legend(loc=1)
    st.pyplot(fig)

    # FICO vs Interest Rate
    st.subheader("FICO vs Interest Rate")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=loan_data, x="fico", y="int_rate", hue="credit_policy", ax=ax)
    sns.regplot(data=loan_data, x="fico", y="int_rate", scatter=False, color="b", ax=ax)
    ax.set_title("FICO vs Int. Rate", fontsize=14)
    ax.set_ylabel("Interest Rate")
    ax.set_xlabel("FICO - Credit Score")
    rect = patches.Rectangle((660, 0.055), 100, 0.02, edgecolor="r", fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(625, 0.08, "Poor FICO score,\nLow Interest Rates and healthy", fontsize=12,
            color="r", fontweight="semibold")
    st.pyplot(fig)

    # Duration of Credit Line
    st.subheader("Credit Line Duration vs Credit Policy")
    df_dur = loan_data[["credit_policy", "days_with_cr_line"]].copy()
    df_dur["years"] = (df_dur["days_with_cr_line"] / 365.25).round(2)
    def bucket(y):
        if y <= 3:
            return "Short Term"
        elif y <= 10:
            return "Medium Term"
        elif y <= 20:
            return "Long Term"
        return "Very Long Term"
    df_dur["term"] = df_dur["years"].apply(bucket)
    term_counts = df_dur.groupby(["term", "credit_policy"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=term_counts, x="term", y="count", hue="credit_policy", ax=ax)
    ax.set_title("Loan Duration Buckets vs Credit Policy", fontsize=14)
    st.pyplot(fig)

    # Inquiries vs Credit Policy
    st.subheader("Inquiries in Last 6 Months vs Credit Policy")
    fig, ax = plt.subplots(figsize=(6,6))
    x_vals = loan_data["credit_policy"].map({0:"Risky", 1:"Healthy"})
    ax.scatter(x=x_vals, y=loan_data["inq_last_6mths"], color="b")
    ax.set_title("Credit Policy vs Inquiries", fontsize=14)
    ax.set_xlabel("Credit Policy")
    ax.set_ylabel("# of Inquiries in Last 6 Months")
    avg_risky = np.round(loan_data[loan_data["credit_policy"]==0]["inq_last_6mths"].mean(), 0)
    avg_healthy = np.round(loan_data[loan_data["credit_policy"]==1]["inq_last_6mths"].mean(), 0)
    ax.text(0, 10, f"Avg Inquiries = {avg_healthy}", fontsize=13, color="r")
    ax.text(0.25, 30, f"Avg Inquiries = {avg_risky}", fontsize=13, color="r")
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin analysis.")

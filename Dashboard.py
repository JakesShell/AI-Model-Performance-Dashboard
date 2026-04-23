import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Model Performance Dashboard", layout="wide")

data = [
    {"Model": "Model A", "Accuracy": 91, "Precision": 88, "Recall": 86, "F1 Score": 87},
    {"Model": "Model B", "Accuracy": 87, "Precision": 84, "Recall": 82, "F1 Score": 83},
    {"Model": "Model C", "Accuracy": 94, "Precision": 91, "Recall": 89, "F1 Score": 90},
    {"Model": "Model D", "Accuracy": 89, "Precision": 86, "Recall": 85, "F1 Score": 85},
]

df = pd.DataFrame(data)

st.title("AI Model Performance Dashboard")
st.write(
    "Interactive dashboard for comparing AI model evaluation metrics and reviewing overall performance."
)

st.sidebar.header("Filter Options")
selected_model = st.sidebar.selectbox("Select A Model", df["Model"].tolist())
selected_metric = st.sidebar.selectbox(
    "Select A Metric", ["Accuracy", "Precision", "Recall", "F1 Score"]
)

filtered_data = df[df["Model"] == selected_model]

st.subheader(f"Selected Model Overview: {selected_model}")

col1, col2, col3, col4 = st.columns(4)
row = filtered_data.iloc[0]

col1.metric("Accuracy", f"{row['Accuracy']}%")
col2.metric("Precision", f"{row['Precision']}%")
col3.metric("Recall", f"{row['Recall']}%")
col4.metric("F1 Score", f"{row['F1 Score']}%")

st.subheader("Selected Metric Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df["Model"], df[selected_metric])
ax.set_title(f"{selected_metric} Across Models")
ax.set_xlabel("Model")
ax.set_ylabel(selected_metric)
st.pyplot(fig)

st.subheader("Selected Model Detail")
st.dataframe(filtered_data, width="stretch")

st.subheader("Average Performance Table")
st.dataframe(df, width="stretch")

st.subheader("Overall Accuracy Snapshot")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(df["Model"], df["Accuracy"])
ax2.set_title("Accuracy Across Models")
ax2.set_xlabel("Model")
ax2.set_ylabel("Accuracy")
st.pyplot(fig2)

st.subheader("Key Observations")
best_accuracy_model = df.loc[df["Accuracy"].idxmax(), "Model"]
best_f1_model = df.loc[df["F1 Score"].idxmax(), "Model"]

st.markdown(
    f"""
- **Best Accuracy:** {best_accuracy_model}
- **Best F1 Score:** {best_f1_model}
- **Selected Metric In Focus:** {selected_metric}
- **Use Case:** Quick review of model quality for portfolio, analytics, or reporting demonstrations
"""
)

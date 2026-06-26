import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)
# ===========================
# LOAD MODEL
# ===========================

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl") 

# ===========================
# LOAD DATASETS
# ===========================

amazon_df = pd.read_csv("amazon_sales.csv")
student_df = pd.read_csv("StudentsPerformance.csv.csv")
# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "😊 Sentiment Analysis",
        "📊 Amazon Analytics",
        "🎓 Student Analytics",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Developer")
st.sidebar.write("**Devanshi Singh**")

st.sidebar.info("CODTECH Internship 2025")

# ======================================
# HOME PAGE
# ======================================

if page == "🏠 Home":

    st.title("Sentiment Analysis Dashboard")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This project combines

- Amazon Sales Analytics
- Student Performance Analytics
- NLP Sentiment Analysis

""")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Model Accuracy",
            "81.27%"
        )

    with c2:
        st.metric(
            "Datasets",
            "3"
        )

    with c3:
        st.metric(
            "ML Algorithm",
            "Logistic Regression"
        )

    

# ======================================
# SENTIMENT PAGE
# ======================================

elif page == "😊 Sentiment Analysis":

   st.title("😊 Customer Sentiment Analysis")

st.markdown("---")

# Customer Details
st.subheader("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    customer_name = st.text_input("Customer Name")
    age = st.number_input("Age", 10, 100)

with col2:
    email = st.text_input("Email")
    city = st.text_input("City")

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

st.markdown("---")

# Product Details
st.subheader("🛒 Product Details")

product = st.text_input("Product Name")

category = st.selectbox(
    "Category",
    [
        "Electronics",
        "Books",
        "Clothing",
        "Sports",
        "Home & Kitchen",
        "Beauty",
        "Toys"
    ]
)

rating = st.slider(
    "Product Rating",
    1,
    5,
    3
)

st.markdown("---")

# Review
st.subheader("✍ Customer Review")

review = st.text_area(
    "Write your review here...",
    height=150
)

# Predict Button
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        vector = vectorizer.transform([review])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        positive = probability[0][1] * 100
        negative = probability[0][0] * 100

        confidence = max(positive, negative)

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Review")
            recommendation = "✅ Recommended for Purchase"
        else:
            st.error("😞 Negative Review")
            recommendation = "❌ Not Recommended"

        st.progress(int(confidence))

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Positive %",
                f"{positive:.2f}%"
            )

        with c2:
            st.metric(
                "Negative %",
                f"{negative:.2f}%"
            )

        st.success(recommendation)

        st.markdown("---")

        st.subheader("📋 Customer Summary")

        st.write(f"**Name:** {customer_name}")
        st.write(f"**Email:** {email}")
        st.write(f"**City:** {city}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")

        st.write(f"**Product:** {product}")
        st.write(f"**Category:** {category}")
        st.write(f"**Rating:** ⭐ {rating}/5")

        st.info(review)
# ======================================
# AMAZON PAGE
# ======================================

elif page == "📊 Amazon Analytics":

    st.title("📊 Amazon Sales Analytics")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Orders", len(amazon_df))

    with col2:
        st.metric(
            "Total Revenue",
            f"₹ {amazon_df['TotalAmount'].sum():,.2f}"
        )

    with col3:
        st.metric(
            "Average Order",
            f"₹ {amazon_df['TotalAmount'].mean():,.2f}"
        )

    st.markdown("---")

    st.subheader("Revenue by Category")

    category = amazon_df.groupby("Category")["TotalAmount"].sum().reset_index()

    fig = px.bar(
        category,
        x="Category",
        y="TotalAmount",
        color="Category",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Revenue by Country")

    country = amazon_df.groupby("Country")["TotalAmount"].sum().reset_index()

    fig2 = px.pie(
        country,
        names="Country",
        values="TotalAmount",
        hole=0.4
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Products")

    top = (
        amazon_df.groupby("ProductName")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig3 = px.bar(
        top,
        x="ProductName",
        y="TotalAmount",
        color="ProductName",
        text_auto=True
    )

    st.plotly_chart(fig3, use_container_width=True)
# ======================================
# STUDENT PAGE
# ======================================
elif page == "🎓 Student Analytics":

    st.title("🎓 Student Performance Analytics")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Students", len(student_df))

    with c2:
        st.metric(
            "Average Math",
            round(student_df["math score"].mean(), 2)
        )

    with c3:
        st.metric(
            "Average Reading",
            round(student_df["reading score"].mean(), 2)
        )

    st.markdown("---")

    st.subheader("Gender Distribution")

    fig1 = px.pie(
        student_df,
        names="gender",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("Math Score Distribution")

    fig2 = px.histogram(
        student_df,
        x="math score",
        nbins=20,
        color_discrete_sequence=["green"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Reading Score Distribution")

    fig3 = px.histogram(
        student_df,
        x="reading score",
        nbins=20,
        color_discrete_sequence=["orange"]
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.subheader("Writing Score Distribution")

    fig4 = px.histogram(
        student_df,
        x="writing score",
        nbins=20,
        color_discrete_sequence=["purple"]
    )

    st.plotly_chart(fig4, use_container_width=True)
# ======================================
# ABOUT PAGE
# ======================================
elif page == "ℹ About":

    st.title("ℹ About Project")

    st.write("""
### 👩‍💻 Developer
**Devanshi Singh**

### 🎯 Internship
CODTECH Internship - Task 4

### 📌 Project
Professional NLP and Analytics Dashboard

### 🚀 Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

### 📊 Features
- Amazon Sales Analytics
- Student Performance Analytics
- Machine Learning Sentiment Analysis
- Interactive Dashboard
""")
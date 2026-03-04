import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

# app title
st.title("Machine Learning Streamlit Application")

# task selection
task = st.sidebar.selectbox(
    "Select Machine Learning Task",
    ["Startup Profit Prediction (Regression)", "Loan Approval Classification"]
)

# regression section
if task == "Startup Profit Prediction (Regression)":

    st.header("Startup Profit Predictor")

    uploaded_file = st.file_uploader("Upload Startup Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        st.subheader("Dataset Preview")
        st.write(df.head())

        if "Profit" not in df.columns:
            st.error("Dataset must contain Profit column")
            st.stop()

        # scatter plot
        if st.checkbox("Show Profit vs R&D Spend Plot"):
            fig, ax = plt.subplots()
            ax.scatter(df["R&D Spend"], df["Profit"])
            ax.set_xlabel("R&D Spend")
            ax.set_ylabel("Profit")
            st.pyplot(fig)

        X = df.drop("Profit", axis=1)
        y = df["Profit"]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.sidebar.subheader("Enter Startup Details")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.sidebar.number_input(col, value=0.0)

        input_df = pd.DataFrame([input_data])

        if st.button("Predict Profit"):
            prediction = model.predict(input_df)
            st.success(f"Predicted Profit: ₹ {prediction[0]:,.2f}")

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Model Performance")
        st.write("R² Score:", r2)
        st.write("Mean Squared Error:", mse)

        # actual vs predicted plot
        if st.checkbox("Show Actual vs Predicted Plot"):
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual Profit")
            ax.set_ylabel("Predicted Profit")
            st.pyplot(fig)

# classification section
elif task == "Loan Approval Classification":

    st.header("Loan Approval Classification")

    uploaded_file = st.file_uploader("Upload Loan Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # remove missing values
        df = df.dropna()

        # remove id column
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        st.subheader("Dataset Preview")
        st.write(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        X = df.drop(target, axis=1)
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)

        pred_lr = model_lr.predict(X_test)

        model_knn = KNeighborsClassifier(n_neighbors=3)
        model_knn.fit(X_train, y_train)

        pred_knn = model_knn.predict(X_test)

        st.subheader("Accuracy")

        acc_lr = accuracy_score(y_test, pred_lr)
        acc_knn = accuracy_score(y_test, pred_knn)

        st.write("Logistic Regression Accuracy:", acc_lr)
        st.write("KNN Accuracy:", acc_knn)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, pred_lr)
        st.write(cm)

        st.subheader("Classification Report")
        report = classification_report(y_test, pred_lr, output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("Cross Validation Score")
        cv = cross_val_score(model_lr, X, y, cv=5)
        st.write(cv.mean())
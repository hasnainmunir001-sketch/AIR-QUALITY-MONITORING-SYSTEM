import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing import load_and_clean
from src.feature_engineering import engineer_features, get_feature_target
from src import models, evaluation

st.set_page_config(
    page_title="Air Quality AI Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title { font-size: 2.8rem; font-weight: 800; color: #2c3e50; }
        .subtitle { font-size: 1.1rem; color: #7f8c8d; }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 10px;
        }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #2980b9; }
        .metric-label { font-size: 0.9rem; color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)

UCI_CSV = "data/AirQualityUCI.csv"


@st.cache_data(show_spinner="Fetching & cleaning dataset...")
def get_data():
    df = load_and_clean(UCI_CSV)
    df = engineer_features(df)
    X, y = get_feature_target(df)

    q33, q66 = y.quantile([0.33, 0.66])
    y_class = pd.cut(
        y,
        bins=[-float("inf"), q33, q66, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    return df, X, y, y_class


@st.cache_resource(show_spinner="Training models... wait...")
def get_trained_models(X_train, y_train, X_test, y_test, feature_names):
    sup = models.train_supervised_models(X_train, y_train, feature_names)
    kmeans = models.train_kmeans(X_train, feature_names)
    semi, n_label = models.train_semi_supervised(
        X_train, y_train, feature_names, labeled_frac=0.10
    )
    return sup, kmeans, semi


def align_clusters(y_true, y_pred):
    """Hungarian algorithm se clusters ko labels map kare."""
    labels = sorted(y_true.unique())
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: labels[row_ind[i]] for i in range(len(row_ind))}
    mapped = pd.Series(y_pred).map(mapping)
    return mapped, cm, mapping


def make_input_row(X, date_val, time_val, overrides):
    row = X.median().to_dict()
    dt = pd.Timestamp.combine(date_val, time_val)

    row["Hour"] = dt.hour
    row["DayOfWeek"] = dt.dayofweek
    row["Month"] = dt.month
    row["IsWeekend"] = int(dt.dayofweek >= 5)
    row["HourSin"] = np.sin(2 * np.pi * dt.hour / 24)
    row["HourCos"] = np.cos(2 * np.pi * dt.hour / 24)

    row.update(overrides)
    return pd.DataFrame([row])


def metrics_table(sup_models, semi, X_test, y_test):
    rows = []
    for name, mdl in sup_models.items():
        res = evaluation.evaluate_classifier(mdl, X_test, y_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision_macro"],
                "Recall": res["recall_macro"],
                "F1-score": res["f1_macro"],
            }
        )
    res = evaluation.evaluate_classifier(semi, X_test, y_test)
    rows.append(
        {
            "Model": "Semi-Supervised (Self-Training)",
            "Accuracy": res["accuracy"],
            "Precision": res["precision_macro"],
            "Recall": res["recall_macro"],
            "F1-score": res["f1_macro"],
        }
    )
    return pd.DataFrame(rows)


def main():
    df, X, y, y_class = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, stratify=y_class, random_state=42
    )
    feature_names = list(X.columns)
    sup_models, kmeans, semi = get_trained_models(
        X_train, y_train, X_test, y_test, feature_names
    )

    with st.sidebar:
        st.markdown("## 🌍 Air Quality AI")
        menu = st.radio(
            "Menu",
            [
                "🏠 Home",
                "📊 EDA",
                "🤖 Supervised Models",
                "🔮 Predict",
                "🔬 Unsupervised",
                "🧪 Semi-Supervised",
                "📈 Comparison",
            ],
        )
        st.markdown("---")
        st.caption("UCI Air Quality dataset")

    if menu == "🏠 Home":
        render_home(df, X, y_class, sup_models)

    elif menu == "📊 EDA":
        render_eda(df, X, y_class)

    elif menu == "🤖 Supervised Models":
        render_supervised(sup_models, X_test, y_test, feature_names)

    elif menu == "🔮 Predict":
        render_predict(X, sup_models)

    elif menu == "🔬 Unsupervised":
        render_unsupervised(kmeans, X_train, y_train, X_test, y_test)

    elif menu == "🧪 Semi-Supervised":
        render_semi(semi, X_test, y_test)

    elif menu == "📈 Comparison":
        render_comparison(sup_models, semi, X_test, y_test)


def render_home(df, X, y_class, sup_models):
    st.markdown(
        '<div class="main-title">Advanced Air Quality Monitoring System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Benzene pollution classification using multiple ML paradigms</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Records</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{X.shape[1]}</div><div class="metric-label">Features</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{y_class.nunique()}</div><div class="metric-label">Classes</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(sup_models) + 2}</div><div class="metric-label">Models</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Problem & Objective")
    st.write(
        """
        Air pollution forecasting is essential for public health. This app classifies hourly 
        **benzene (C6H6)** concentration from the **UCI Air Quality dataset** into **Low, Medium, High**.
        
        Three learning paradigms are implemented:
        - **Supervised**: Random Forest, SVM, Neural Network (MLP)
        - **Unsupervised**: K-Means clustering for pattern discovery
        - **Semi-Supervised**: Self-Training with only 10% labeled data
        """
    )
    st.markdown("---")
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)


def render_eda(df, X, y_class):
    st.header("📊 Exploratory Data Analysis")

    t1, t2, t3 = st.tabs(["Overview", "Correlations", "Trends"])

    with t1:
        st.write("**Shape:**", X.shape)
        counts = y_class.value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.pie(
            counts,
            values="Count",
            names="Class",
            hole=0.4,
            color="Class",
            color_discrete_map={
                "Low": "#2ecc71",
                "Medium": "#f1c40f",
                "High": "#e74c3c",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        corr = X.corr().round(2)
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=650
        )
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        df["Hour"] = df["DateTime"].dt.hour
        hourly = (
            df.groupby("Hour")[["CO(GT)", "NOx(GT)", "NO2(GT)"]].mean().reset_index()
        )
        fig = px.line(
            hourly,
            x="Hour",
            y=["CO(GT)", "NOx(GT)", "NO2(GT)"],
            markers=True,
            title="Average pollutant concentration by hour",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_supervised(sup_models, X_test, y_test, feature_names):
    st.header("🤖 Supervised Learning Evaluation")

    model_name = st.selectbox("Choose a model", list(sup_models.keys()))
    mdl = sup_models[model_name]
    res = evaluation.evaluate_classifier(mdl, X_test, y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy']:.3f}")
    c2.metric("Precision (macro)", f"{res['precision_macro']:.3f}")
    c3.metric("Recall (macro)", f"{res['recall_macro']:.3f}")
    c4.metric("F1-score (macro)", f"{res['f1_macro']:.3f}")

    fig = px.imshow(
        res["confusion_matrix"],
        text_auto=True,
        x=res["labels"],
        y=res["labels"],
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(height=450, width=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report_df = evaluation.classification_report_df(mdl, X_test, y_test)
    st.dataframe(report_df.round(3), use_container_width=True)

    if model_name == "Random Forest":
        st.subheader("Feature Importance")
        importances = mdl.named_steps["classifier"].feature_importances_
        feat_imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=True)
        fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)


def render_predict(X, sup_models):
    st.header("🔮 Real-Time Pollution Prediction")

    with st.form("predict_form"):
        st.write("Sensor values set karo. Baaki features dataset median pe default.")

        c1, c2 = st.columns(2)
        with c1:
            d = st.date_input("Date", value=pd.Timestamp("2004-10-01"))
            t = st.time_input("Time", value=pd.Timestamp("12:00:00").time())

            co = st.slider(
                "CO(GT)",
                float(X["CO(GT)"].min()),
                float(X["CO(GT)"].max()),
                float(X["CO(GT)"].median()),
            )
            nmhc = st.slider(
                "NMHC(GT)",
                float(X["NMHC(GT)"].min()),
                float(X["NMHC(GT)"].max()),
                float(X["NMHC(GT)"].median()),
            )
            nox = st.slider(
                "NOx(GT)",
                float(X["NOx(GT)"].min()),
                float(X["NOx(GT)"].max()),
                float(X["NOx(GT)"].median()),
            )
            no2 = st.slider(
                "NO2(GT)",
                float(X["NO2(GT)"].min()),
                float(X["NO2(GT)"].max()),
                float(X["NO2(GT)"].median()),
            )

        with c2:
            temp = st.slider(
                "Temperature (T)",
                float(X["T"].min()),
                float(X["T"].max()),
                float(X["T"].median()),
            )
            rh = st.slider(
                "Relative Humidity (RH)",
                float(X["RH"].min()),
                float(X["RH"].max()),
                float(X["RH"].median()),
            )
            ah = st.slider(
                "Absolute Humidity (AH)",
                float(X["AH"].min()),
                float(X["AH"].max()),
                float(X["AH"].median()),
            )

            with st.expander("Advanced sensor overrides"):
                pt08_co = st.slider(
                    "PT08.S1(CO)",
                    float(X["PT08.S1(CO)"].min()),
                    float(X["PT08.S1(CO)"].max()),
                    float(X["PT08.S1(CO)"].median()),
                )
                pt08_nmhc = st.slider(
                    "PT08.S2(NMHC)",
                    float(X["PT08.S2(NMHC)"].min()),
                    float(X["PT08.S2(NMHC)"].max()),
                    float(X["PT08.S2(NMHC)"].median()),
                )
                pt08_nox = st.slider(
                    "PT08.S3(NOx)",
                    float(X["PT08.S3(NOx)"].min()),
                    float(X["PT08.S3(NOx)"].max()),
                    float(X["PT08.S3(NOx)"].median()),
                )
                pt08_no2 = st.slider(
                    "PT08.S4(NO2)",
                    float(X["PT08.S4(NO2)"].min()),
                    float(X["PT08.S4(NO2)"].max()),
                    float(X["PT08.S4(NO2)"].median()),
                )
                pt08_o3 = st.slider(
                    "PT08.S5(O3)",
                    float(X["PT08.S5(O3)"].min()),
                    float(X["PT08.S5(O3)"].max()),
                    float(X["PT08.S5(O3)"].median()),
                )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        overrides = {
            "CO(GT)": co,
            "NMHC(GT)": nmhc,
            "NOx(GT)": nox,
            "NO2(GT)": no2,
            "T": temp,
            "RH": rh,
            "AH": ah,
            "PT08.S1(CO)": pt08_co,
            "PT08.S2(NMHC)": pt08_nmhc,
            "PT08.S3(NOx)": pt08_nox,
            "PT08.S4(NO2)": pt08_no2,
            "PT08.S5(O3)": pt08_o3,
        }

        input_row = make_input_row(X, d, t, overrides)

        st.subheader("Model Predictions")
        cols = st.columns(len(sup_models))

        for col, (name, mdl) in zip(cols, sup_models.items()):
            pred = mdl.predict(input_row)[0]
            color = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}[pred]

            with col:
                st.markdown(f"**{name}**")
                st.markdown(
                    f"<h2 style='margin:0;color:{color};'>{pred}</h2>",
                    unsafe_allow_html=True,
                )

                if hasattr(mdl, "predict_proba"):
                    prob = mdl.predict_proba(input_row)[0]
                    probs = dict(zip(mdl.classes_, prob))
                    st.json({k: f"{v:.2f}" for k, v in probs.items()})


def render_unsupervised(kmeans, X_train, y_train, X_test, y_test):
    st.header("🔬 Unsupervised Learning: K-Means Clustering")

    clusters_train = kmeans.predict(X_train)
    mapped_train, cm, mapping = align_clusters(y_train, clusters_train)
    ari = evaluation.adjusted_rand_score(y_test, kmeans.predict(X_test))

    st.metric("Adjusted Rand Index (test)", f"{ari:.3f}")
    st.write("Cluster-to-label mapping (Hungarian alignment):", mapping)

    preprocessor = kmeans.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_test)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc)

    clusters = kmeans.predict(X_test)
    mapped, _, _ = align_clusters(y_test, clusters)

    vis_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Mapped Cluster": mapped,
            "True Label": y_test.values,
        }
    )

    fig = px.scatter(
        vis_df,
        x="PC1",
        y="PC2",
        color="Mapped Cluster",
        symbol="True Label",
        opacity=0.7,
        color_discrete_map={
            "Low": "#2ecc71",
            "Medium": "#f1c40f",
            "High": "#e74c3c",
        },
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_semi(semi, X_test, y_test):
    st.header("🧪 Semi-Supervised Learning: Self-Training")

    st.write(
        "Only **10%** of the training data is labeled; the rest is unlabeled."
    )

    res = evaluation.evaluate_classifier(semi, X_test, y_test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy']:.3f}")
    c2.metric("Precision (macro)", f"{res['precision_macro']:.3f}")
    c3.metric("Recall (macro)", f"{res['recall_macro']:.3f}")
    c4.metric("F1-score (macro)", f"{res['f1_macro']:.3f}")

    fig = px.imshow(
        res["confusion_matrix"],
        text_auto=True,
        x=res["labels"],
        y=res["labels"],
        color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_comparison(sup_models, semi, X_test, y_test):
    st.header("📈 Model Comparison")

    df = metrics_table(sup_models, semi, X_test, y_test)

    fig = px.bar(
        df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metrics Table")
    st.dataframe(df.round(3), use_container_width=True)


if __name__ == "__main__":
    main()
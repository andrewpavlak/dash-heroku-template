from dash import dash_table
#https://github.com/andrewpavlak/dash-heroku-template/raw/refs/heads/master/Fuel%20Data%20Clean.xlsx
import numpy as np
import pandas as pd
#from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from scipy.stats import gaussian_kde
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


# Data and model setup
 

#DATA_PATH = Path("Fuel Data Clean.xlsx")
df = pd.read_excel('https://github.com/andrewpavlak/dash-heroku-template/raw/refs/heads/master/Fuel%20Data%20Clean.xlsx')

df = df.dropna(
    subset=[
        "combined_mpg_ft1",
        "manual_or_automatic",
        "tailpipe_co2_in_grams_mile_ft1",
        "engine_displacement",
    ]
)

data = df.copy()

trans_options = sorted(data["manual_or_automatic"].dropna().unique().tolist())
class_options = sorted(data["class"].dropna().unique().tolist())
drive_options = sorted(data["drive"].dropna().unique().tolist())
fuel_type_options = sorted(data["fuel_type"].dropna().unique().tolist())
fuel_type1_options = sorted(data["fuel_type_1"].dropna().unique().tolist())

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

for col in ["vehicle_id", "engine_index"]:
    if col in numeric_cols:
        numeric_cols.remove(col)

data["combined_mpg_ft1_log"] = np.log(data["combined_mpg_ft1"])

data["engine_displacement_log"] = np.log(data["engine_displacement"])
data["tailpipe_log"] = np.log(data["tailpipe_co2_in_grams_mile_ft1"])

RIDGE_X_OPTIONS = [
    "engine_displacement",
    "engine_displacement_log",
    "tailpipe_co2_in_grams_mile_ft1",
    "tailpipe_log",
]

RIDGE_Y_OPTIONS = [
    "combined_mpg_ft1_log",
    "combined_mpg_ft1",
]

X_reg = data[["engine_displacement", "manual_or_automatic", "tailpipe_co2_in_grams_mile_ft1"]]
y_reg = data["combined_mpg_ft1_log"]

reg_cats = ["manual_or_automatic"]
reg_num_log = ["engine_displacement", "tailpipe_co2_in_grams_mile_ft1"]

reg_numeric_log_pipe = Pipeline(
    steps=[
        ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ]
)

reg_preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), reg_cats),
        ("num_log", reg_numeric_log_pipe, reg_num_log),
    ]
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=123
)

reg_model = Pipeline(
    steps=[
        ("preprocess", reg_preprocess),
        ("model", Ridge(alpha=1.0)),
    ]
)

reg_model.fit(Xr_train, yr_train)
yr_pred_log = reg_model.predict(Xr_test)
yr_pred = np.exp(yr_pred_log)
yr_true = np.exp(yr_test)

reg_pred_log_all = reg_model.predict(X_reg)
reg_resid_log = y_reg - reg_pred_log_all

reg_diag = pd.DataFrame(
    {
        "fitted_log_mpg": reg_pred_log_all,
        "actual_log_mpg": y_reg.values,
        "resid_log_mpg": reg_resid_log,
    }
)

reg_rmse = np.sqrt(mean_squared_error(yr_true, yr_pred))
reg_r2 = r2_score(yr_true, yr_pred)

reg_results = pd.DataFrame(
    {
        "actual_mpg": yr_true,
        "predicted_mpg": yr_pred,
    }
)

 
# Logistic: Manual vs Automatic
 

y_trans = data["manual_or_automatic"]
X_trans = data[
    [
        "year",
        "class",
        "drive",
        "engine_cylinders",
        "engine_displacement",
        "fuel_type",
        "fuel_type_1",
        "city_mpg_ft1",
        "highway_mpg_ft1",
        "unadjusted_city_mpg_ft1",
        "unadjusted_highway_mpg_ft1",
        "combined_mpg_ft1",
        "annual_fuel_cost_ft1",
        "save_or_spend_5_year",
        "annual_consumption_in_barrels_ft1",
        "tailpipe_co2_in_grams_mile_ft1",
    ]
].copy()

Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_trans, y_trans, test_size=0.2, random_state=42, stratify=y_trans
)

trans_cats = ["class", "drive", "fuel_type", "fuel_type_1"]
trans_nums = [
    "year",
    "engine_cylinders",
    "engine_displacement",
    "city_mpg_ft1",
    "highway_mpg_ft1",
    "unadjusted_city_mpg_ft1",
    "unadjusted_highway_mpg_ft1",
    "combined_mpg_ft1",
    "annual_fuel_cost_ft1",
    "save_or_spend_5_year",
    "annual_consumption_in_barrels_ft1",
    "tailpipe_co2_in_grams_mile_ft1",
]

trans_num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
trans_cat_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

trans_preprocess = ColumnTransformer(
    transformers=[
        ("num", trans_num_pipe, trans_nums),
        ("cat", trans_cat_pipe, trans_cats),
    ]
)

trans_model = Pipeline(
    steps=[
        ("preprocess", trans_preprocess),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
            ),
        ),
    ]
)

trans_model.fit(Xt_train, yt_train)
yt_pred = trans_model.predict(Xt_test)
yt_proba = trans_model.predict_proba(Xt_test)

trans_acc = accuracy_score(yt_test, yt_pred)

classes = trans_model.named_steps["model"].classes_
pos_class = "Manual" if "Manual" in classes else classes[0]
pos_idx = np.where(classes == pos_class)[0][0]
yt_bin = (yt_test == pos_class).astype(int)
proba_pos = yt_proba[:, pos_idx]

fpr, tpr, _ = roc_curve(yt_bin, proba_pos)
trans_auc = roc_auc_score(yt_bin, proba_pos)

cm = confusion_matrix(yt_test, yt_pred, labels=classes)
cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in classes],
    columns=[f"Pred {c}" for c in classes],
)

 
# Eco logistic
 

eco_features = [
    "year",
    "class",
    "drive",
    "fuel_type",
    "fuel_type_1",
    "engine_cylinders",
    "engine_displacement",
]

eco_df = data.dropna(subset=eco_features + ["tailpipe_co2_in_grams_mile_ft1"]).copy()
eco_df["eco_friendly"] = (eco_df["tailpipe_co2_in_grams_mile_ft1"] <= 400).astype(int)

y_eco = eco_df["eco_friendly"]
X_eco = eco_df[eco_features]

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_eco,
    y_eco,
    test_size=0.2,
    random_state=42,
    stratify=y_eco,
)

eco_nums = ["year", "engine_cylinders", "engine_displacement"]
eco_cats = ["class", "drive", "fuel_type", "fuel_type_1"]

eco_num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
eco_cat_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

eco_preprocess = ColumnTransformer(
    transformers=[
        ("num", eco_num_pipe, eco_nums),
        ("cat", eco_cat_pipe, eco_cats),
    ]
)

eco_model = Pipeline(
    steps=[
        ("preprocess", eco_preprocess),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
            ),
        ),
    ]
)

eco_model.fit(Xe_train, ye_train)

ye_pred = eco_model.predict(Xe_test)
ye_proba = eco_model.predict_proba(Xe_test)

eco_acc = accuracy_score(ye_test, ye_pred)

eco_classes = eco_model.named_steps["model"].classes_
eco_pos_class = 1
eco_pos_idx = list(eco_classes).index(eco_pos_class)

ye_bin = (ye_test == eco_pos_class).astype(int)
eco_proba_pos = ye_proba[:, eco_pos_idx]

eco_fpr, eco_tpr, _ = roc_curve(ye_bin, eco_proba_pos)
eco_auc = roc_auc_score(ye_bin, eco_proba_pos)

eco_cm = confusion_matrix(ye_test, ye_pred, labels=eco_classes)
eco_cm_df = pd.DataFrame(
    eco_cm,
    index=[f"True {c}" for c in eco_classes],
    columns=[f"Pred {c}" for c in eco_classes],
)

 
# Clustering
 

cluster_numeric_cols = [
    'year',
    "engine_displacement",
    "engine_cylinders",
    "combined_mpg_ft1",
    "city_mpg_ft1",
    "highway_mpg_ft1",
    "annual_fuel_cost_ft1",
    "annual_consumption_in_barrels_ft1",
    "tailpipe_co2_in_grams_mile_ft1",
]

cluster_rows = data.dropna(subset=cluster_numeric_cols).index
cluster_data = data.loc[cluster_rows, :].copy()
cluster_features = cluster_data[cluster_numeric_cols].copy()

cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(cluster_features)

 
# KNN regression
 

knn_num_features = [
    "engine_displacement",
    "engine_cylinders",
    "combined_mpg_ft1",
    "city_mpg_ft1",
    "highway_mpg_ft1",
    "annual_fuel_cost_ft1",
    "annual_consumption_in_barrels_ft1",
    "tailpipe_co2_in_grams_mile_ft1",
]

knn_cat_features = ["class", "drive", "fuel_type", "fuel_type_1"]

knn_required = knn_num_features + knn_cat_features + [
    "combined_mpg_ft1",
    "year",
]

knn_df = data.dropna(subset=knn_required).copy()

X_knn = knn_df[knn_num_features + knn_cat_features]
y_knn_mpg = knn_df["combined_mpg_ft1"]
y_knn_year = knn_df["year"]

train_idx, test_idx = train_test_split(
    knn_df.index, test_size=0.2, random_state=42
)

X_knn_train = X_knn.loc[train_idx]
X_knn_test = X_knn.loc[test_idx]

y_knn_mpg_train = y_knn_mpg.loc[train_idx]
y_knn_mpg_test = y_knn_mpg.loc[test_idx]

y_knn_year_train = y_knn_year.loc[train_idx]
y_knn_year_test = y_knn_year.loc[test_idx]

knn_num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
knn_cat_pipe = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

knn_preprocess = ColumnTransformer(
    transformers=[
        ("num", knn_num_pipe, knn_num_features),
        ("cat", knn_cat_pipe, knn_cat_features),
    ]
)

 
# Styling helpers
 

BACKGROUND_COLOR = "#f5f7fb"
CARD_BG = "#ffffff"
ACCENT_COLOR = "#2a6fdb"
TEXT_COLOR = "#1f2933"

CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "padding": "20px",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.08)",
    "marginTop": "20px",
    "marginBottom": "20px",
}

SECTION_TITLE_STYLE = {
    "marginTop": "0",
    "marginBottom": "12px",
    "fontWeight": "600",
    "color": TEXT_COLOR,
}

LABEL_STYLE = {
    "fontWeight": "500",
    "marginBottom": "4px",
    "display": "block",
    "color": "#4b5563",
}

TABS_STYLE = {
    "marginTop": "10px",
}

TAB_STYLE = {
    "padding": "10px 18px",
    "fontWeight": "500",
    "border": "none",
    "backgroundColor": "transparent",
}

TAB_SELECTED_STYLE = {
    "padding": "10px 18px",
    "fontWeight": "600",
    "borderBottom": f"3px solid {ACCENT_COLOR}",
    "backgroundColor": "#e5edff",
    "color": ACCENT_COLOR,
}

SUB_TAB_STYLE = {
    "padding": "8px 14px",
    "fontSize": "14px",
    "border": "none",
    "backgroundColor": "transparent",
}

SUB_TAB_SELECTED_STYLE = {
    "padding": "8px 14px",
    "fontSize": "14px",
    "borderBottom": f"2px solid {ACCENT_COLOR}",
    "backgroundColor": "#eef2ff",
    "color": ACCENT_COLOR,
}

 
# App layout
 

app: Dash = dash.Dash(__name__,suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Fuel Economy ML Dashboard",
                    style={
                        "textAlign": "center",
                        "marginBottom": "4px",
                        "color": TEXT_COLOR,
                    },
                ),
                html.P(
                    "Interactive exploration of fuel economy data and machine learning models.",
                    style={
                        "textAlign": "center",
                        "marginTop": "0",
                        "color": "#6b7280",
                    },
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-eda",
            style=TABS_STYLE,
            children=[
                dcc.Tab(
                    label="README",
                    value="tab-readme", 
                    style=TAB_STYLE, 
                    selected_style=TAB_SELECTED_STYLE
                ),
                dcc.Tab(
                    label="Data Explorer",
                    value="tab-eda",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label="Supervised Models",
                    value="tab-models",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
                dcc.Tab(
                    label="K-Means Clustering with PCA",
                    value="tab-unsup",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                ),
            ],
        ),
        html.Div(id="tab-content"),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "20px 24px 40px 24px",
        "backgroundColor": BACKGROUND_COLOR,
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
)


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab_content(active_tab):
    if active_tab == "tab-eda":
        return html.Div(
            [
                html.Div(
                    [
                        html.H3("Explore the Fuel Dataset", style=SECTION_TITLE_STYLE),
                        html.P(
                            "Use the controls below to visualize distributions and relationships "
                            "between numeric variables in the dataset.",
                            style={"color": "#6b7280", "marginBottom": "16px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Histogram variable (numeric):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="eda-hist-col",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in numeric_cols
                                            ],
                                            value=numeric_cols[0],
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Scatterplot X and Y:",
                                            style=LABEL_STYLE,
                                        ),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="eda-scatter-x",
                                                    options=[
                                                        {"label": c, "value": c}
                                                        for c in numeric_cols
                                                    ],
                                                    value=numeric_cols[0],
                                                    clearable=False,
                                                    style={"marginBottom": "6px"},
                                                ),
                                                dcc.Dropdown(
                                                    id="eda-scatter-y",
                                                    options=[
                                                        {"label": c, "value": c}
                                                        for c in numeric_cols
                                                    ],
                                                    value=numeric_cols[1],
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "marginBottom": "16px",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        dcc.Graph(id="eda-hist"),
                        dcc.Graph(id="eda-scatter"),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        html.H4("Sample of Data", style=SECTION_TITLE_STYLE),
                        dash_table.DataTable(
                            id="eda-table",
                            columns=[{"name": c, "id": c} for c in data.columns],
                            data=data.head(15).to_dict("records"),
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": ACCENT_COLOR,
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            style_cell={
                                "padding": "6px",
                                "fontSize": 12,
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#f9fafb",
                                }
                            ],
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )
    elif active_tab == "tab-readme":
        return html.Div(
            [
                html.H3("Project README", style=SECTION_TITLE_STYLE),

                dcc.Tabs(
                    id="readme-tabs",
                    value="readme-data",
                    children=[
                        dcc.Tab(label="Data", value="readme-data", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED_STYLE),
                        dcc.Tab(label="Linear Regression", value="readme-ridge", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED_STYLE),
                        dcc.Tab(label="Logistic Regression", value="readme-logit", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED_STYLE),
                        dcc.Tab(label="KNN Models", value="readme-knn", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED_STYLE),
                        dcc.Tab(label="K-Means Clustering", value="readme-kmeans", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED_STYLE),
                    ],
                ),

                html.Div(id="readme-content")
            ],
            style=CARD_STYLE
        )    

    elif active_tab == "tab-models":
        return html.Div(
            [
                html.Div(
                    [
                        html.H3("Supervised Learning Models", style=SECTION_TITLE_STYLE),
                        html.P(
                            "Compare regression models and explore predictions "
                            "for custom vehicles.",
                            style={"color": "#6b7280", "marginBottom": "8px"},
                        ),
                        dcc.Tabs(
                            id="model-tabs",
                            value="model-reg",
                            style={"marginTop": "6px"},
                            children=[
                                dcc.Tab(
                                    label="Ridge Regression (MPG)",
                                    value="model-reg",
                                    style=SUB_TAB_STYLE,
                                    selected_style=SUB_TAB_SELECTED_STYLE,
                                ),
                                dcc.Tab(
                                    label="KNN Models",
                                    value="model-knn",
                                    style=SUB_TAB_STYLE,
                                    selected_style=SUB_TAB_SELECTED_STYLE,
                                ),
                                dcc.Tab(
                                    label="Logistic: Manual vs Automatic",
                                    value="model-trans",
                                    style=SUB_TAB_STYLE,
                                    selected_style=SUB_TAB_SELECTED_STYLE,
                                ),
                                dcc.Tab(
                                    label="Logistic: Eco-Friendly",
                                    value="model-eco",
                                    style=SUB_TAB_STYLE,
                                    selected_style=SUB_TAB_SELECTED_STYLE,
                                ),

                            ],
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(id="model-tab-content"),
            ]
        )

    elif active_tab == "tab-unsup":
        return html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Unsupervised Learning: KMeans Clustering",
                            style=SECTION_TITLE_STYLE,
                        ),
                        html.P(
                            "Cluster vehicles based on their specifications and explore how "
                            "clusters differ across variables.",
                            style={"color": "#6b7280", "marginBottom": "16px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Number of clusters (K):",
                                    style=LABEL_STYLE,
                                ),
                                dcc.Slider(
                                    id="kmeans-k",
                                    min=2,
                                    max=8,
                                    step=1,
                                    value=2,
                                    marks={k: str(k) for k in range(2, 9)},
                                ),
                            ],
                            style={"marginBottom": "16px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "X-axis variable:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="cluster-x",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in cluster_numeric_cols
                                            ],
                                            value='tailpipe_co2_in_grams_mile_ft1',
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Y-axis variable:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="cluster-y",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in cluster_numeric_cols
                                            ],
                                            value='combined_mpg_ft1',
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "marginBottom": "12px",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        dcc.Graph(id="cluster-scatter"),
                        html.Div(
                            id="cluster-summary",
                            style={
                                "marginTop": "10px",
                                "fontWeight": "bold",
                                "color": "#374151",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    return html.Div()


 
# Callbacks
 

@app.callback(
    Output("eda-hist", "figure"),
    Output("eda-scatter", "figure"),
    Input("eda-hist-col", "value"),
    Input("eda-scatter-x", "value"),
    Input("eda-scatter-y", "value"),
)
def update_eda(hist_col, x_col, y_col):
    hist_fig = px.histogram(
        data, x=hist_col, nbins=40, title=f"Distribution of {hist_col}"
    )
    hist_fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    scatter_fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        hover_data=["make", "model", "year"],
        title=f"{y_col} vs {x_col}",
    )
    scatter_fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return hist_fig, scatter_fig


@app.callback(
    Output("model-tab-content", "children"),
    Input("model-tabs", "value"),
)
def render_model_tab(active_model_tab):
    if active_model_tab == "model-reg":
        # Defaults for the input form (unchanged)
        default_engine_disp = float(data["engine_displacement"].median())
        default_co2 = float(data["tailpipe_co2_in_grams_mile_ft1"].median())
        default_trans = trans_options[0] if len(trans_options) > 0 else None

        return html.Div(
            [
                # Description + metrics
                html.Div(
                    [
                        html.P(
                            "Linear (Ridge) regression on log MPG using engine displacement, "
                            "tailpipe CO₂, and transmission type. Predictions are reported on "
                            "the original combined MPG scale.",
                            style={"color": "#4b5563"},
                        ),
                        html.Ul(
                            [
                                html.Li(f"Test RMSE (on MPG scale): {reg_rmse:.2f}"),
                                html.Li(f"Test R²: {reg_r2:.3f}"),
                            ],
                            style={"color": "#111827"},
                        ),
                    ],
                    style=CARD_STYLE,
                ),

                # Axis selectors + interactive scatter
                html.Div(
                    [
                        html.H4(
                            "Relationship Between Predictors and Combined MPG",
                            style=SECTION_TITLE_STYLE,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "X-axis variable:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="ridge-x",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in RIDGE_X_OPTIONS
                                            ],
                                            value="tailpipe_co2_in_grams_mile_ft1",
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Y-axis variable:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="ridge-y",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in RIDGE_Y_OPTIONS
                                            ],
                                            value="combined_mpg_ft1",
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "marginBottom": "12px",
                            },
                        ),
                        dcc.Graph(id="ridge-scatter"),
                    ],
                    style=CARD_STYLE,
                ),

                # Prediction form (unchanged)
                html.Div(
                    [
                        html.H4(
                            "Predict Combined MPG for a Custom Vehicle",
                            style=SECTION_TITLE_STYLE,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Engine displacement (liters):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="reg-engine-displacement",
                                            type="number",
                                            value=default_engine_disp,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Tailpipe CO₂ (grams/mile):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="reg-co2",
                                            type="number",
                                            value=default_co2,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Transmission type:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="reg-transmission",
                                            options=[
                                                {"label": t, "value": t}
                                                for t in trans_options
                                            ],
                                            value=default_trans,
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "marginTop": "10px",
                                "marginBottom": "10px",
                            },
                        ),
                        html.Div(
                            id="reg-pred-output",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "18px",
                                "marginTop": "10px",
                                "color": "#111827",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    elif active_model_tab == "model-trans":
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {trans_auc:.3f})",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False,
            )
        )
        roc_fig.update_layout(
            title="ROC Curve: Manual vs Automatic",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        cm_fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix: Manual vs Automatic",
        )
        cm_fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        default_year = int(data["year"].median())
        default_engine_cyl = int(data["engine_cylinders"].median())
        default_engine_disp = float(data["engine_displacement"].median())
        default_city_mpg = float(data["city_mpg_ft1"].median())
        default_hwy_mpg = float(data["highway_mpg_ft1"].median())
        default_unadj_city = float(data["unadjusted_city_mpg_ft1"].median())
        default_unadj_hwy = float(data["unadjusted_highway_mpg_ft1"].median())
        default_combined_mpg = float(data["combined_mpg_ft1"].median())
        default_annual_cost = float(data["annual_fuel_cost_ft1"].median())
        default_save_5yr = float(data["save_or_spend_5_year"].median())
        default_barrels = float(data["annual_consumption_in_barrels_ft1"].median())
        default_co2 = float(data["tailpipe_co2_in_grams_mile_ft1"].median())

        default_class = class_options[0] if class_options else None
        default_drive = drive_options[0] if drive_options else None
        default_fuel_type = fuel_type_options[0] if fuel_type_options else None
        default_fuel_type1 = fuel_type1_options[0] if fuel_type1_options else None

        return html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Logistic regression predicting whether a vehicle has a manual or "
                            "automatic transmission from engine specs and fuel economy metrics.",
                            style={"color": "#4b5563"},
                        ),
                        html.Ul(
                            [
                                html.Li(f"Test Accuracy: {trans_acc:.3f}"),
                                html.Li(
                                    f"ROC AUC (for '{pos_class}'): {trans_auc:.3f}"
                                ),
                            ],
                            style={"color": "#111827"},
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        dcc.Graph(figure=roc_fig),
                        dcc.Graph(figure=cm_fig),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        html.H4(
                            "Predict Transmission Type for a Custom Vehicle",
                            style=SECTION_TITLE_STYLE,
                        ),

                        # Row 1
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Model year:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-year",
                                            type="number",
                                            value=default_year,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Vehicle class:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="log-class",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in class_options
                                            ],
                                            value=default_class,
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Drive type:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="log-drive",
                                            options=[
                                                {"label": d, "value": d}
                                                for d in drive_options
                                            ],
                                            value=default_drive,
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 2
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Engine cylinders:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-engine-cyl",
                                            type="number",
                                            value=default_engine_cyl,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Engine displacement (liters):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-engine-disp",
                                            type="number",
                                            value=default_engine_disp,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 3
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Fuel type:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="log-fuel-type",
                                            options=[
                                                {"label": f, "value": f}
                                                for f in fuel_type_options
                                            ],
                                            value=default_fuel_type,
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Fuel type 1:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="log-fuel-type1",
                                            options=[
                                                {"label": f, "value": f}
                                                for f in fuel_type1_options
                                            ],
                                            value=default_fuel_type1,
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 4
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "City MPG (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-city-mpg",
                                            type="number",
                                            value=default_city_mpg,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Highway MPG (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-hwy-mpg",
                                            type="number",
                                            value=default_hwy_mpg,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Combined MPG (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-combined-mpg",
                                            type="number",
                                            value=default_combined_mpg,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 5
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Unadjusted city MPG (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-unadj-city",
                                            type="number",
                                            value=default_unadj_city,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Unadjusted highway MPG (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-unadj-hwy",
                                            type="number",
                                            value=default_unadj_hwy,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 6
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Annual fuel cost (ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-annual-cost",
                                            type="number",
                                            value=default_annual_cost,
                                            step=10,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "5-year save/spend vs. average:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-save-5yr",
                                            type="number",
                                            value=default_save_5yr,
                                            step=10,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        # Row 7
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Annual consumption (barrels, ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-barrels",
                                            type="number",
                                            value=default_barrels,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Tailpipe CO₂ (g/mile, ft1):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="log-co2",
                                            type="number",
                                            value=default_co2,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        html.Div(
                            id="logit-pred-output",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "18px",
                                "marginTop": "15px",
                                "color": "#111827",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    elif active_model_tab == "model-eco":
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=eco_fpr,
                y=eco_tpr,
                mode="lines",
                name=f"ROC (AUC = {eco_auc:.3f})",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False,
            )
        )
        roc_fig.update_layout(
            title="ROC Curve: Eco-Friendly vs Not",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        cm_fig = px.imshow(
            eco_cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix: Eco-Friendly vs Not",
        )
        cm_fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        default_year_e = int(eco_df["year"].median())
        default_engine_cyl_e = int(eco_df["engine_cylinders"].median())
        default_engine_disp_e = float(eco_df["engine_displacement"].median())

        default_class_e = class_options[0] if class_options else None
        default_drive_e = drive_options[0] if drive_options else None
        default_fuel_type_e = fuel_type_options[0] if fuel_type_options else None
        default_fuel_type1_e = fuel_type1_options[0] if fuel_type1_options else None

        return html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Logistic regression predicting whether a vehicle is eco-friendly "
                            "(tailpipe CO₂ ≤ 400 g/mi) based on basic specs.",
                            style={"color": "#4b5563"},
                        ),
                        html.Ul(
                            [
                                html.Li(f"Test Accuracy: {eco_acc:.3f}"),
                                html.Li(
                                    f"ROC AUC (P(eco-friendly=1)): {eco_auc:.3f}"
                                ),
                            ],
                            style={"color": "#111827"},
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        dcc.Graph(figure=roc_fig),
                        dcc.Graph(figure=cm_fig),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        html.H4(
                            "Predict Eco-Friendly Status for a Custom Vehicle",
                            style=SECTION_TITLE_STYLE,
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Model year:", style=LABEL_STYLE),
                                        dcc.Input(
                                            id="eco-year",
                                            type="number",
                                            value=default_year_e,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Vehicle class:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="eco-class",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in class_options
                                            ],
                                            value=default_class_e,
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Drive type:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="eco-drive",
                                            options=[
                                                {"label": d, "value": d}
                                                for d in drive_options
                                            ],
                                            value=default_drive_e,
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Fuel type:", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="eco-fuel-type",
                                            options=[
                                                {"label": f, "value": f}
                                                for f in fuel_type_options
                                            ],
                                            value=default_fuel_type_e,
                                            clearable=False,
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Fuel type 1:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="eco-fuel-type1",
                                            options=[
                                                {"label": f, "value": f}
                                                for f in fuel_type1_options
                                            ],
                                            value=default_fuel_type1_e,
                                            clearable=False,
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Engine cylinders:",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="eco-engine-cyl",
                                            type="number",
                                            value=default_engine_cyl_e,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "marginRight": "10px",
                                        "minWidth": "0",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Engine displacement (liters):",
                                            style=LABEL_STYLE,
                                        ),
                                        dcc.Input(
                                            id="eco-engine-disp",
                                            type="number",
                                            value=default_engine_disp_e,
                                            step=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "marginTop": "10px"},
                        ),

                        html.Div(
                            id="eco-pred-output",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "18px",
                                "marginTop": "15px",
                                "color": "#111827",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    elif active_model_tab == "model-knn":
        return html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "K-Nearest Neighbors regression models predicting either "
                            "combined MPG or model year. Adjust K and compare performance.",
                            style={"color": "#4b5563"},
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Target variable:",
                                    style=LABEL_STYLE,
                                ),
                                dcc.RadioItems(
                                    id="knn-target",
                                    options=[
                                        {"label": "Combined MPG", "value": "mpg"},
                                        {"label": "Model Year", "value": "year"},
                                    ],
                                    value="year",
                                    inline=True,
                                ),
                            ],
                            style={"marginBottom": "15px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Number of neighbors (K):",
                                    style=LABEL_STYLE,
                                ),
                                dcc.Slider(
                                    id="knn-k",
                                    min=1,
                                    max=25,
                                    step=1,
                                    value=3,
                                    marks={k: str(k) for k in range(1, 26, 4)},
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        dcc.Graph(id="knn-graph"),
                        html.Div(
                            id="knn-metrics",
                            style={
                                "marginTop": "10px",
                                "fontWeight": "bold",
                                "color": "#111827",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    return html.Div()


@app.callback(
    Output("cluster-scatter", "figure"),
    Output("cluster-summary", "children"),
    Input("kmeans-k", "value"),
    Input("cluster-x", "value"),
    Input("cluster-y", "value"),
)
def update_kmeans_clusters(k, x_var, y_var):
    if x_var is None or y_var is None:
        x_var, y_var = cluster_numeric_cols[0], cluster_numeric_cols[1]

    pca = PCA(n_components=2).fit(cluster_scaled)
    pca_coords = pca.transform(cluster_scaled)
    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_df)

    plot_df = cluster_data.copy()
    plot_df["Cluster"] = clusters.astype(str)

    hover_cols = [c for c in ["make", "model", "year"] if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x=x_var,
        y=y_var,
        color="Cluster",
        hover_data=hover_cols,
        title=f"KMeans Clusters (K={k}) on {x_var} vs {y_var}",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    counts = pd.Series(clusters).value_counts().sort_index()
    summary_text = "Cluster sizes: " + ", ".join(
        [f"Cluster {i}: {counts[i]} vehicles" for i in counts.index]
    )

    return fig, summary_text


@app.callback(
    Output("reg-pred-output", "children"),
    Input("reg-engine-displacement", "value"),
    Input("reg-co2", "value"),
    Input("reg-transmission", "value"),
)
def update_reg_prediction(engine_disp, co2, transmission):
    if engine_disp is None or co2 is None or transmission is None:
        return "Enter values above to see a predicted combined MPG."

    X_new = pd.DataFrame(
        {
            "engine_displacement": [engine_disp],
            "tailpipe_co2_in_grams_mile_ft1": [co2],
            "manual_or_automatic": [transmission],
        }
    )

    log_pred = reg_model.predict(X_new)[0]
    mpg_pred = float(np.exp(log_pred))

    return f"Predicted combined MPG: {mpg_pred:.1f}"


@app.callback(
    Output("eco-pred-output", "children"),
    Input("eco-year", "value"),
    Input("eco-class", "value"),
    Input("eco-drive", "value"),
    Input("eco-fuel-type", "value"),
    Input("eco-fuel-type1", "value"),
    Input("eco-engine-cyl", "value"),
    Input("eco-engine-disp", "value"),
)
def update_eco_prediction(
    year,
    cls,
    drive,
    fuel_type,
    fuel_type1,
    engine_cyl,
    engine_disp,
):
    vals = [year, cls, drive, fuel_type, fuel_type1, engine_cyl, engine_disp]
    if any(v is None for v in vals):
        return "Enter values above to see whether the car is predicted to be eco-friendly."

    X_new = pd.DataFrame(
        {
            "year": [year],
            "class": [cls],
            "drive": [drive],
            "fuel_type": [fuel_type],
            "fuel_type_1": [fuel_type1],
            "engine_cylinders": [engine_cyl],
            "engine_displacement": [engine_disp],
        }
    )

    proba = eco_model.predict_proba(X_new)[0]
    pred_label = eco_model.predict(X_new)[0]

    prob_eco = proba[eco_pos_idx]
    label_text = "Eco-friendly" if pred_label == 1 else "Not eco-friendly"

    return f"Predicted class: {label_text} (P(eco-friendly) = {prob_eco:.2f})"

@app.callback(
    Output("readme-content", "children"),
    Input("readme-tabs", "value")
)
def render_readme_section(section):
    variable_list = html.Ul([
        html.Li([
            html.B(col + ": "),
            str(data[col].dtype)
        ])
        for col in data.columns
    ])
    DATA_TEXT = html.Div([
        html.H4("Data Source"),
        html.P("""
Data originally comes from https://www.fueleconomy.gov/feg/ws/ (U.S. Department of Energy).
The data was later compiled into a Kaggle dataset (now removed), but we preserved the originally downloaded file.
        """),

        html.H4("Data Cleaning"),
        html.Ul([
            html.Li("Original data: 38,113 rows × 81 columns"),
            html.Li("Cleaned data: 36,974 rows × 24 columns"),
            html.Li("Many sparse or non-informative variables removed"),
            html.Li("Created is_electric (binary electric indicator)"),
            html.Li("Created manual_or_automatic for transmission type"),
        ]),

        html.H4("Variables in Cleaned Dataset"),
        variable_list
    ])
    fig_resid_fitted = px.scatter(
        reg_diag,
        x="fitted_log_mpg",
        y="resid_log_mpg",
        opacity=0.25,
        title="Residuals vs Fitted (Log Combined MPG)",
    )
    fig_resid_fitted.update_layout(
        template="plotly_white",
        xaxis_title="Fitted log(MPG)",
        yaxis_title="Residual (log MPG)",
        margin=dict(l=40, r=20, t=60, b=40),
        title_x=0.5,
    )
    fig_resid_fitted.add_shape(
        type="line",
        x0=reg_diag["fitted_log_mpg"].min(),
        y0=0,
        x1=reg_diag["fitted_log_mpg"].max(),
        y1=0,
        line=dict(dash="dash"),
    )

    fig_resid_hist = px.histogram(
        reg_diag,
        x="resid_log_mpg",
        nbins=40,
        title="Histogram of Residuals (Log Combined MPG)",
    )
    fig_resid_hist.update_layout(
        template="plotly_white",
        xaxis_title="Residual (log MPG)",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=60, b=40),
        title_x=0.5,
    )

    RIDGE_TEXT = html.Div([
        html.H4("Ridge Regression Model"),

        html.P(
            "Goal: Identify which vehicle characteristics best predict fuel efficiency "
            "(combined MPG) while addressing multicollinearity and checking standard "
            "linear-regression assumptions."
        ),

        html.H5("Response Variable"),
        html.Ul([
            html.Li("combined_mpg_ft1 (modeled on the log scale as combined_mpg_ft1_log)")
        ]),

        html.H5("Predictor Variables"),
        html.Ul([
            html.Li("engine_displacement (log-transformed and scaled)"),
            html.Li("manual_or_automatic (one-hot encoded)"),
            html.Li("tailpipe_co2_in_grams_mile_ft1 (log-transformed and scaled)"),
        ]),

        html.H5("Transformations and Regularization"),
        html.Ul([
            html.Li("Both the response and key numeric predictors were log-transformed to better satisfy linearity."),
            html.Li("Numeric predictors were standardized (mean 0, unit variance) before fitting the model."),
            html.Li("Ridge regularization (L2 penalty) was used to reduce the impact of multicollinearity."),
            html.Li("An alpha value of 1.0 was chosen; larger alphas did not improve test MSE."),
        ]),

        html.H5("Model Performance"),
        html.Ul([
            html.Li("R² (test): 0.97879"),
            html.Li("Test MSE (on log scale): 0.00131"),
            html.Li("On the original MPG scale, errors are small relative to the range of observed MPG values."),
        ]),

        html.H5("Model Assumption Checks"),

        html.H6("1. Linearity"),
        html.Ul([
            html.Li(
                "Scatterplots of log-transformed predictors versus log combined MPG showed approximately "
                "linear relationships over most of the data range."
            ),
            html.Li(
                "Some deviations and outliers are present at extreme values, but no strong systematic curvature "
                "was observed in the bulk of the data."
            ),
        ]),

        html.H6("2. Independence of Errors"),
        html.Ul([
            html.Li("A Durbin–Watson test was computed on the residuals from the Ridge model."),
            html.Li("The Durbin–Watson statistic was approximately 2.0, consistent with little autocorrelation."),
        ]),

        html.H6("3. Homoscedasticity (Constant Variance)"),
        html.Ul([
            html.Li("Residuals-vs-fitted plot (below) shows residuals scattered roughly symmetrically around zero."),
            html.Li("No strong funnel shape or pattern suggesting severe heteroscedasticity."),
        ]),

        html.H6("4. Normality of Residuals"),
        html.Ul([
            html.Li("Histogram of residuals (below) is approximately bell-shaped and centered at zero."),
            html.Li("Minor deviations from perfect normality are expected with real-world data."),
        ]),

        html.H6("5. Multicollinearity"),
        html.Ul([
            html.Li("CO₂ and engine displacement are correlated, producing non-trivial VIF values."),
            html.Li("Ridge regularization was chosen specifically to stabilize coefficients in the presence of this multicollinearity."),
        ]),

        html.P(
            "Overall, diagnostic checks from the assumption notebook (linearity, independence, homoscedasticity, "
            "residual normality, and multicollinearity) support the use of a log-linear Ridge model for "
            "combined MPG in this dataset."
        ),

        html.H5("Diagnostic Plots"),
        dcc.Graph(figure=fig_resid_fitted),
        dcc.Graph(figure=fig_resid_hist),
    ])
    
    co2_data = data["tailpipe_co2_in_grams_mile_ft1"].dropna()

    fig_co2 = px.histogram(
        co2_data,
        x=co2_data,
        nbins=40,
        opacity=0.65,
        title="Distribution of Tailpipe CO₂ Emissions (g/mi)",
    )

    kde = gaussian_kde(co2_data)
    x_range = np.linspace(co2_data.min(), co2_data.max(), 400)
    y_kde = kde(x_range)

    y_kde_scaled = y_kde * len(co2_data) * (x_range[1] - x_range[0])

    fig_co2.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde_scaled,
            mode="lines",
            line=dict(color="blue", width=2),
            name="KDE",
        )
    )

    fig_co2.update_layout(
        template="plotly_white",
        xaxis_title="tailpipe_co2_in_grams_mile_ft1",
        yaxis_title="Count",
        title_x=0.5,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    
    LOGISTIC_TEXT = html.Div(
        [
            html.H4("Logistic Regression Models"),
            html.H3("Eco-Friendly Model"),

            html.P(
                "Goal: Predict whether a vehicle is eco-friendly (1) or not (0) "
                "based on engine, vehicle, and fuel characteristics."
            ),

            html.P(
                "Through researching EDA standards and exploring the dataset, a reasonable "
                "threshold for determining whether a vehicle is eco-friendly is a CO₂ "
                "tailpipe emission level of 400 g/mi."
            ),

            html.Ul(
                [
                    html.Li("≈ 25% of vehicles are eco-friendly (CO₂ ≤ 400 g/mi)."),
                    html.Li("≈ 75% of vehicles are not eco-friendly."),
                    html.Li(
                        "Class imbalance is handled by adjusting the class_weight parameter "
                        "inside the logistic regression pipeline."
                    ),
                ]
            ),

            html.H5("Distribution of Tailpipe CO₂ (g/mi)"),
            html.P(
                "The plot below shows the distribution of tailpipe CO₂ emissions. "
                "A threshold of 400 g/mi was selected based on this distribution "
                "to define whether a vehicle is considered eco-friendly."
            ),

            dcc.Graph(
                figure=fig_co2,
                style={"maxWidth": "700px", "margin": "20px auto"},
            ),

            html.H5("Features Used"),
            html.Ul(
                [
                    html.Li("Engine: cylinders, displacement, is_electric"),
                    html.Li("Vehicle characteristics: year, class, drive"),
                    html.Li("Fuel info: fuel_type, fuel_type_1"),
                ]
            ),

            html.H5("Model Info"),
            html.Ul(
                [
                    html.Li("Numeric variables scaled using StandardScaler()"),
                    html.Li("Categorical variables one-hot encoded"),
                    html.Li("Solver: lbfgs"),
                    html.Li("Regularization: L2"),
                    html.Li("Class imbalance handled via class weights inside the pipeline"),
                ]
            ),

            html.H5("Performance"),
            html.Ul(
                [
                    html.Li("CV Accuracy: 0.887"),
                    html.Li("CV Log Loss: 0.260"),
                    html.Li("Test Accuracy: 0.886"),
                    html.Li("Test Log Loss: 0.258"),
                    html.Li("Test ROC AUC: 0.962"),
                ]
            ),

            html.Hr(),
            html.H3("Transmission Model"),

            html.P(
                "Goal: Predict whether a vehicle has a manual (1) or automatic (0) "
                "transmission based on its specifications."
            ),

            html.H5("Features Used"),
            html.Ul(
                [
                    html.Li("Categorical: class, drive, fuel_type, fuel_type_1"),
                    html.Li(
                        "Numeric: year, engine_cylinders, engine_displacement, "
                        "city_mpg_ft1, highway_mpg_ft1, unadjusted_city_mpg_ft1, "
                        "unadjusted_highway_mpg_ft1, combined_mpg_ft1, "
                        "annual_fuel_cost_ft1, save_or_spend_5_year, "
                        "annual_consumption_in_barrels_ft1, "
                        "tailpipe_co2_in_grams_mile_ft1"
                    ),
                    html.Li("Multicollinearity handled with L2 regularization"),
                ]
            ),

            html.H5("Model Info"),
            html.Ul(
                [
                    html.Li("Numeric variables scaled using StandardScaler()"),
                    html.Li("Categorical variables one-hot encoded"),
                    html.Li("Solver: lbfgs"),
                    html.Li("Regularization: L2"),
                    html.Li("Class imbalance handled via class weights inside pipeline"),
                ]
            ),

            html.H5("Performance"),
            html.Ul(
                [
                    html.Li("CV Accuracy: 0.685"),
                    html.Li("CV Log Loss: 0.560"),
                    html.Li("Test Accuracy: 0.695"),
                    html.Li("Test Log Loss: 0.553"),
                    html.Li("Test ROC AUC: 0.787"),
                ]
            ),
        ],
        style=CARD_STYLE,
    )

    KNN_TEXT = html.Div([
        html.H4("K-Nearest Neighbor (KNN) Models"),

        html.H3("MPG Prediction"),
        html.P("Goal: Predict a vehicle's combined MPG using similarity-based modeling."),
        html.H5("Features"),
        html.Ul([
            html.Li("Engine: cylinders, displacement, is_electric"),
            html.Li("Vehicle: make, class, drive, transmission"),
            html.Li("Fuel: fuel_type, fuel_type_1"),
        ]),
        html.H5("Best Parameters"),
        html.Ul([
            html.Li("k = 5"),
            html.Li("p = 1 (Manhattan)"),
            html.Li("Weights = distance"),
        ]),
        html.H5("Performance"),
        html.Ul([
            html.Li("Test RMSE ≈ 1.20 MPG"),
            html.Li("Test R² ≈ 0.944"),
        ]),

        html.H3("Year Prediction"),
        html.P("Goal: Predict the model year using only vehicle specifications."),
        html.H5("Best Parameters"),
        html.Ul([
            html.Li("k = 3"),
            html.Li("p = 1 (Manhattan)"),
            html.Li("Weights = distance"),
        ]),
        html.H5("Performance"),
        html.Ul([
            html.Li("Test RMSE ≈ 2.55 years"),
            html.Li("Test R² ≈ 0.939"),
        ]),
    ])
    KMEANS_TEXT = html.Div(
        [
            html.H4("K-Means Clustering Model"),

            html.P(
                "Objective: Apply PCA and K-Means to identify natural clusters of vehicles based on "
                "fuel economy, emissions, and engine characteristics, and compare the resulting "
                "structure to insights from logistic regression models."
            ),

            html.H5("Data Preparation"),
            html.Ul(
                [
                    html.Li("Selected engine, MPG, emissions, and fuel-cost features."),
                    html.Li(
                        "Categorical variables were one-hot encoded for the non-PCA K-Means workflow."
                    ),
                    html.Li(
                        "For PCA-based clustering, categorical variables were removed to avoid "
                        "inflating variance in directions unrelated to numeric vehicle behavior."
                    ),
                    html.Li("All numeric features were standardized using StandardScaler()."),
                ]
            ),

            html.H5("K-Means (No PCA)"),
            html.P(
                "The optimal number of clusters was determined to be K = 2, based on:"
            ),
            html.Ul(
                [
                    html.Li(
                        "The Elbow Method: inertia dropped sharply from K=1→2, with diminishing "
                        "returns afterward."
                    ),
                    html.Li(
                        "Silhouette Scores: K=2 produced the highest cohesion and separation, with "
                        "lower scores for K ≥ 3."
                    ),
                ]
            ),
            html.P("Cluster interpretations:"),
            html.Ul(
                [
                    html.Li(
                        "Cluster 0: Vehicles with smaller engines, higher MPG, lower emissions, and "
                        "lower annual fuel cost."
                    ),
                    html.Li(
                        "Cluster 1: Vehicles with larger engines, lower MPG, higher emissions, and "
                        "higher annual fuel cost."
                    ),
                ]
            ),
            html.P(
                "These clusters strongly align with known vehicle class distinctions—compact cars "
                "versus trucks/SUVs—indicating the clustering reveals meaningful structure in the data."
            ),

            html.H5("PCA + K-Means"),
            html.P(
                "Principal Component Analysis (PCA) was applied to reduce multicollinearity and "
                "capture the major sources of variation in MPG, emissions, and engine characteristics."
            ),
            html.Ul(
                [
                    html.Li(
                        "PC1 captured the efficiency–inefficiency continuum "
                        "(approximately 60% of total variance). Vehicles with high MPG loaded "
                        "negatively, while high CO₂ and large engines loaded positively."
                    ),
                    html.Li(
                        "PC2 captured secondary variation (around 10% of variance), including "
                        "differences related to vehicle class and drive configuration."
                    ),
                ]
            ),
            html.P(
                "K-Means applied to the 2-component PCA space again produced two cleanly separated "
                "clusters, which closely mirrored the non-PCA clustering results."
            ),

            html.H5("Key Insights"),
            html.Ul(
                [
                    html.Li(
                        "Fuel efficiency is the dominant underlying structure in the dataset. "
                        "Both PCA and K-Means independently identify an efficiency-based split."
                    ),
                    html.Li(
                        "Cluster identities remain consistent across PCA and non-PCA models, "
                        "strengthening confidence in the stability and interpretability of the results."
                    ),
                    html.Li(
                        "Findings reinforce conclusions from logistic regression models: "
                        "engine size, emissions, and fuel cost strongly influence vehicle categorization."
                    ),
                    html.Li(
                        "The natural partitions discovered by clustering align with real-world "
                        "vehicle categories (e.g., compact/economy cars vs. SUVs/trucks)."
                    ),
                ]
            ),
        ],
        style=CARD_STYLE,
    )

    if section == "readme-data":
        return DATA_TEXT
    if section == "readme-ridge":
        return RIDGE_TEXT
    if section == "readme-logit":
        return LOGISTIC_TEXT
    if section == "readme-knn":
        return KNN_TEXT
    if section == "readme-kmeans":
        return KMEANS_TEXT

    return "Select a section."

@app.callback(
    Output("logit-pred-output", "children"),
    Input("log-year", "value"),
    Input("log-class", "value"),
    Input("log-drive", "value"),
    Input("log-engine-cyl", "value"),
    Input("log-engine-disp", "value"),
    Input("log-fuel-type", "value"),
    Input("log-fuel-type1", "value"),
    Input("log-city-mpg", "value"),
    Input("log-hwy-mpg", "value"),
    Input("log-unadj-city", "value"),
    Input("log-unadj-hwy", "value"),
    Input("log-combined-mpg", "value"),
    Input("log-annual-cost", "value"),
    Input("log-save-5yr", "value"),
    Input("log-barrels", "value"),
    Input("log-co2", "value"),
)
def update_logit_prediction(
    year,
    cls,
    drive,
    engine_cyl,
    engine_disp,
    fuel_type,
    fuel_type1,
    city_mpg,
    hwy_mpg,
    unadj_city,
    unadj_hwy,
    combined_mpg,
    annual_cost,
    save_5yr,
    barrels,
    co2,
):
    vals = [
        year,
        cls,
        drive,
        engine_cyl,
        engine_disp,
        fuel_type,
        fuel_type1,
        city_mpg,
        hwy_mpg,
        unadj_city,
        unadj_hwy,
        combined_mpg,
        annual_cost,
        save_5yr,
        barrels,
        co2,
    ]
    if any(v is None for v in vals):
        return "Enter values above to see a predicted transmission type."

    X_new = pd.DataFrame(
        {
            "year": [year],
            "class": [cls],
            "drive": [drive],
            "engine_cylinders": [engine_cyl],
            "engine_displacement": [engine_disp],
            "fuel_type": [fuel_type],
            "fuel_type_1": [fuel_type1],
            "city_mpg_ft1": [city_mpg],
            "highway_mpg_ft1": [hwy_mpg],
            "unadjusted_city_mpg_ft1": [unadj_city],
            "unadjusted_highway_mpg_ft1": [unadj_hwy],
            "combined_mpg_ft1": [combined_mpg],
            "annual_fuel_cost_ft1": [annual_cost],
            "save_or_spend_5_year": [save_5yr],
            "annual_consumption_in_barrels_ft1": [barrels],
            "tailpipe_co2_in_grams_mile_ft1": [co2],
        }
    )

    proba = trans_model.predict_proba(X_new)[0]
    classes = trans_model.named_steps["model"].classes_
    proba_pos = proba[pos_idx]
    pred_label = trans_model.predict(X_new)[0]

    return (
        f"Predicted transmission: {pred_label}  "
        f"(P({pos_class}) = {proba_pos:.2f})"
    )


@app.callback(
    Output("knn-graph", "figure"),
    Output("knn-metrics", "children"),
    Input("knn-target", "value"),
    Input("knn-k", "value"),
)
def update_knn_models(target, k):
    if k is None or k < 1:
        k = 5

    if target == "year":
        y_train = y_knn_year_train
        y_test = y_knn_year_test
        target_label = "Model Year"
    else:
        y_train = y_knn_mpg_train
        y_test = y_knn_mpg_test
        target_label = "Combined MPG"

    knn_model = Pipeline(
        steps=[
            ("preprocess", knn_preprocess),
            ("model", KNeighborsRegressor(n_neighbors=int(k))),
        ]
    )

    knn_model.fit(X_knn_train, y_train)
    y_pred = knn_model.predict(X_knn_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred,
        }
    )

    fig = px.scatter(
        results_df,
        x="Actual",
        y="Predicted",
        title=f"KNN Regression: Actual vs Predicted ({target_label}, K={int(k)})",
    )
    min_val = results_df[["Actual", "Predicted"]].min().min()
    max_val = results_df[["Actual", "Predicted"]].max().max()
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(dash="dash"),
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    metrics = html.Ul(
        [
            html.Li(f"Target: {target_label}"),
            html.Li(f"K (neighbors): {int(k)}"),
            html.Li(f"Test RMSE: {rmse:.2f}"),
            html.Li(f"Test R²: {r2:.3f}"),
        ]
    )

    return fig, metrics

@app.callback(
    Output("ridge-scatter", "figure"),
    Input("ridge-x", "value"),
    Input("ridge-y", "value"),
)
def update_ridge_scatter(x_col, y_col):
    # Guard defaults
    if x_col is None:
        x_col = "engine_displacement"
    if y_col is None:
        y_col = "combined_mpg_ft1_log"

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        opacity=0.2,
    )
    fig.update_layout(
        title=f"Relationship between {x_col} and {y_col}",
        title_x=0.5,
        xaxis_title=x_col,
        yaxis_title=y_col,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="black")
    fig.update_yaxes(showgrid=True, gridcolor="black")

    return fig


if __name__ == "__main__":
    app.run(debug=True)

"""
Predict the price per m² of homes sold in Paris and its inner suburbs
from DVF transactions enriched with Filosofi socio-economic data
"""

import argparse
import os
from pathlib import Path

import duckdb

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from skore import EstimatorReport

from features import build_features
from import_data import download_sources, get_melodi_indicator
from check import  check_code_commune_formatting, check_missing_values, check_unseen_categories
from model import create_pipeline

load_dotenv()
con = duckdb.connect(database=":memory:")

LEARNING_RATE = 0.03
NUM_LEAVES = 31
NUMERIC_FEATURES = [
    "log_surface",
    "log_distance_eiffel",
    "nombre_pieces_principales",
    "nombre_lots",
    "log_surface_terrain",
    "ratio_carrez",
    "nbre_menages",
    "tx_pauvrete",
    "nv_vie_moyen",
    "part_proprietaires",
    "part_logements_sociaux",
    "part_logements_avant_1945",
    "surface_moyenne_logements",
    "NIVVIE_MEDIAN",
    "TAUX_PAUVRETE",
]
CATEGORICAL_FEATURES = ["code_commune", "type_local"]
DEPARTEMENTS_PARIS = ["75", "92", "93", "94"]

jeton_api = os.environ["JETON_API"]

DATA_LOCATION = Path("data")
SOURCES = {
    "dvf.parquet": (
        "https://minio.lab.sspcloud.fr/"
        "projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet"
    ),
    "filosofi_carreaux_200m_2021.parquet": (
        "https://www.data.gouv.fr/api/1/datasets/r/55432374-a91d-43d0-923d-4514dc3eb951"
    ),
}


# ENVIRONMENT CONFIGURATION ---------------------------------------

parser = argparse.ArgumentParser(description="Paramètres du modèle LightGBM")
parser.add_argument("--n_trees", type=int, default=1500, help="Nombre d'arbres")
args = parser.parse_args()

# Ce qui permet de ne pas changer le nom de cette variable meme si demain l'argument change
n_trees = args.n_trees

print(f"Valeur de l'argument n_trees: {n_trees}")



# DATA IMPORT ---------------------------------------

print("Lecture des données ------------------------")

DATA_LOCATION.mkdir(exist_ok=True)

for nom_fichier, url_source in SOURCES.items():
    chemin = DATA_LOCATION / nom_fichier
    if chemin.exists():
        print(f"{nom_fichier:42s} déjà présent ({chemin.stat().st_size / 1e6:.0f} Mo)")
    else:
        print(f"Téléchargement de {nom_fichier} ...")
        download_sources(url_source, chemin)
        print(f"  -> {chemin.stat().st_size / 1e6:.0f} Mo")


# gpd c'est geopandad qui permet de faire de la lecture et la jointure des données spaciales
dvf = gpd.read_parquet(DATA_LOCATION / "dvf.parquet")


# QUALITY DIAGNOSTICS ---------------------------------------

# DuckDB cannot read the geometry column as Pandas formatted it
dvf_attributs = dvf.drop(columns="geometry")

# dvf_attributs.columns would work too: DuckDB anticipates its wider use later
column_names = con.sql("SELECT column_name FROM (DESCRIBE dvf_attributs)").to_df()[
    "column_name"
]

check_code_commune_formatting(connection=con, df=dvf_attributs)

for var in column_names:
    check_missing_values(connection=con, df=dvf_attributs, variable=var)


# SCOPE RESTRICTION ---------------------------------------

print("Restriction aux départements de la petite couronne")

dvf = dvf.loc[dvf["code_departement"].isin(DEPARTEMENTS_PARIS)]
dvf = dvf.loc[dvf["valeur_fonciere"] < 1e6]


# ADDITIONAL DATA ---------------------------------------

# Filosofi gridded income and demographic data (200m cells)

filo_cols = [
    "idcar_200m",
    "ind",
    "men",
    "men_pauv",
    "men_prop",
    "ind_snv",
    "log_soc",
    "log_av45",
    "men_surf",
    "geometry",
]
filo = gpd.read_parquet(
    DATA_LOCATION / "filosofi_carreaux_200m_2021.parquet", columns=filo_cols
)
filo = filo.rename(
    columns={
        "men": "nbre_menages",
        "men_pauv": "nbre_menages_pauvres",
        "men_prop": "nbre_menages_prop",
        "ind_snv": "somme_nv_vie",
        "log_soc": "nbre_logements_sociaux",
        "log_av45": "nbre_logements_avant_1945",
        "men_surf": "surface_totale_logements",
    }
)

print(
    f"Reprojection de {dvf.shape[0]} transactions immobilières "
    f"du CRS {dvf.crs.to_epsg()} à {filo.crs.to_epsg()}"
)

dvf = dvf.to_crs(filo.crs)
dvf = dvf.sjoin(filo, how="left", predicate="within")


# Filosofi municipality-level data (Melodi API)

niveau_vie = get_melodi_indicator(
    code="MED_SL", value_name="NIVVIE_MEDIAN", jeton_api=jeton_api
)
taux_pauvrete = get_melodi_indicator(
    code="PR_MD60", value_name="TAUX_PAUVRETE", jeton_api=jeton_api
)

dvf = dvf.merge(niveau_vie, left_on="code_commune", right_on="CODGEO", how="left").drop(
    columns="CODGEO"
)
dvf = dvf.merge(
    taux_pauvrete, left_on="code_commune", right_on="CODGEO", how="left"
).drop(columns="CODGEO")

print(dvf.head(5))


# DESCRIPTIVE STATISTICS ---------------------------------------

print("Statistiques descriptives ---------------------")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)

grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


# FEATURE ENGINEERING ---------------------------------------

print("Feature engineering")

dvf = build_features(dvf)

# Missing features are imputed in the pipeline: only rows without a target
# are dropped.
dvf_start_data = dvf.dropna(subset=["prix_m2"])

# PRICE PER M² BY MUNICIPALITY ---------------------------------------

prix_m2_commune = (
    dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()
)

print("Bottom 5 communes (prix au m2 le plus bas) :")
print(prix_m2_commune.head(5))
print("Top 5 communes (prix au m2 le plus élevé) :")
print(prix_m2_commune.tail(5))

# MODEL ---------------------------------------

pipe = create_pipeline(
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    n_trees = n_trees,
    learning_rate = LEARNING_RATE,
    num_leaves = NUM_LEAVES,
)

# TRAIN/TEST SPLIT ---------------------------------------

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")


# SPLIT VALIDATION ---------------------------------------

for string_var in CATEGORICAL_FEATURES:
    check_unseen_categories(X_train, X_test, string_var)


# TRAINING AND EVALUATION ---------------------------------------

pipe.fit(X_train, y_train)

rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
print(f"RMSE sur le test : {rmse_test:,.0f} €/m²")

residuals = X_test[["code_commune"]].copy()
residuals["y_test"] = y_test
residuals["pred"] = pipe.predict(X_test)

rmse_commune = (
    residuals.groupby("code_commune")
    .apply(
        lambda d: root_mean_squared_error(d["y_test"], d["pred"]),
        include_groups=False,
    )
    .rename("rmse")
    .reset_index()
)

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
rmse_commune = rmse_commune.merge(cog, on="code_commune").sort_values("rmse")

print("Bottom 5 communes (RMSE la plus faible) :")
print(rmse_commune.head(5))
print("Top 5 communes (RMSE la plus élevée) :")
print(rmse_commune.tail(5))


# EVALUATION WITH SKORE ---------------------------------------

print("Évaluation skore ---------------------")

# Unfitted copy: skore fits the model itself and can therefore report
# the training time.
report = EstimatorReport(
    clone(pipe),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(report)

metrics = report.metrics.summarize(data_source="both").frame()
print(metrics)

report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.savefig("prediction_error.png", dpi=150, bbox_inches="tight")
plt.close()

importance = report.inspection.permutation_importance(data_source="test").frame()
print(importance)

print("Fin du script !")


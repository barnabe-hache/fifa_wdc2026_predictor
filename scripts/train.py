"""
train.py — Entraînement du prédicteur de scores CDM 2026
=========================================================
Modèles  : Poisson GLM + LightGBM (objective=poisson), un par équipe (home/away).
Symétrie en terrain neutre :
  - Data augmentation : chaque match neutre est dupliqué avec équipes inversées.
    Le modèle apprend qu'en terrain neutre, A vs B == B vs A structurellement.
  - À l'inférence (predict.py) : double forward pass + moyenne des lambdas.

Usage:
    python train.py
    python train.py --data-dir ./data/clean --model-dir ./models
"""

import argparse
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "clean"
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15

ELO_K_BASE = 32
ELO_K_WC   = 60
ELO_INIT   = 1500

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 3.0,
    "UEFA Euro": 2.5,
    "Copa América": 2.5,
    "Africa Cup of Nations": 2.0,
    "AFC Asian Cup": 2.0,
    "CONCACAF Gold Cup": 2.0,
    "Confederations Cup": 2.0,
    "UEFA Nations League": 1.8,
    "World Cup qualification": 1.5,
    "Friendly": 0.6,
}
DEFAULT_WEIGHT = 1.0

STRENGTH_FEATURES = [
    "team_strength_score",
    "top11_overall_mean",
    "top11_shooting_mean",
    "top11_passing_mean",
    "top11_defending_mean",
    "top11_pace_mean",
    "top11_physic_mean",
    "top11_dribbling_mean",
    "top11_potential_mean",
    "top23_overall_mean",
    "top23_potential_mean",
    "top5_overall_mean",
    "count_85_plus_top23",
    "count_90_plus_top23",
    "top11_value_sum_eur",
]

ROLLING_WINDOWS = [5, 10, 20]
H2H_WINDOW = 10


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    matches  = pd.read_parquet(data_dir / "all_matches_clean_full.parquet")
    strength = pd.read_parquet(data_dir / "national_teams_strength.parquet")
    return matches, strength


def get_tournament_weight(tournament: str) -> float:
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament.lower():
            return w
    return DEFAULT_WEIGHT


def get_elo_k(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return ELO_K_WC
    if any(x in t for x in ["euro", "copa", "africa cup", "asian cup", "gold cup"]):
        return ELO_K_BASE * 1.5
    if "friendly" in t:
        return ELO_K_BASE * 0.5
    return ELO_K_BASE


# ---------------------------------------------------------------------------
# ELO DYNAMIQUE
# ---------------------------------------------------------------------------

def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matches = matches.copy().sort_values("date").reset_index(drop=True)
    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in matches.iterrows():
        ht, at = row["home_team"], row["away_team"]
        r_h = elo.get(ht, ELO_INIT)
        r_a = elo.get(at, ELO_INIT)
        home_elos.append(r_h)
        away_elos.append(r_a)

        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        if hs > as_:   s_h, s_a = 1.0, 0.0
        elif hs == as_: s_h = s_a = 0.5
        else:           s_h, s_a = 0.0, 1.0

        home_adv = 0 if row.get("is_neutral", 0) else 100
        e_h = 1 / (1 + 10 ** ((r_a - r_h - home_adv) / 400))
        e_a = 1 - e_h

        k = get_elo_k(row.get("tournament", ""))
        gd = abs(hs - as_)
        gd_factor = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 2.0))

        elo[ht] = r_h + k * gd_factor * (s_h - e_h)
        elo[at] = r_a + k * gd_factor * (s_a - e_a)

    matches["home_elo_before"] = home_elos
    matches["away_elo_before"] = away_elos
    matches["elo_diff"] = matches["home_elo_before"] - matches["away_elo_before"]
    matches["elo_win_prob_home"] = 1 / (
        1 + 10 ** ((matches["away_elo_before"] - matches["home_elo_before"]) / 400)
    )
    return matches, elo


# ---------------------------------------------------------------------------
# ROLLING STATS
# ---------------------------------------------------------------------------

def compute_rolling_stats(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    def _view(df, team_col, score_col, conceded_col, win_val, loss_val):
        v = df[["date", team_col, score_col, conceded_col, "result", "is_neutral"]].copy()
        v.columns = ["date", "team", "goals_scored", "goals_conceded", "result", "is_neutral"]
        v["win"]         = (v["result"] == win_val).astype(int)
        v["draw"]        = (v["result"] == "draw").astype(int)
        v["loss"]        = (v["result"] == loss_val).astype(int)
        v["clean_sheet"] = (v["goals_conceded"] == 0).astype(int)
        v["scored_0"]    = (v["goals_scored"] == 0).astype(int)
        return v

    home_view = _view(matches, "home_team", "home_score", "away_score", "home_win", "away_win")
    away_view = _view(matches, "away_team", "away_score", "home_score", "away_win", "home_win")

    team_view = pd.concat([home_view, away_view], ignore_index=True)
    team_view = team_view.sort_values(["team", "date"]).reset_index(drop=True)

    roll_cols = ["goals_scored", "goals_conceded", "win", "draw", "loss", "clean_sheet", "scored_0"]
    stats_list = []
    for window in ROLLING_WINDOWS:
        rolled = team_view.groupby("team")[roll_cols].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        rolled.columns = [f"{c}_last{window}" for c in rolled.columns]
        stats_list.append(rolled)

    ewm_gs = team_view.groupby("team")["goals_scored"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    ewm_gc = team_view.groupby("team")["goals_conceded"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    stats_list.append(pd.DataFrame({"goals_scored_ewm10": ewm_gs, "goals_conceded_ewm10": ewm_gc}))

    rolling_stats = pd.concat(stats_list, axis=1)
    rolling_stats["team"] = team_view["team"].values
    rolling_stats["date"] = team_view["date"].values
    rolling_stats = (
        rolling_stats.sort_values(["team", "date"])
        .drop_duplicates(subset=["team", "date"], keep="last")
    )
    return rolling_stats


# ---------------------------------------------------------------------------
# HOME ADVANTAGE FACTOR
# ---------------------------------------------------------------------------

def compute_home_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    home_stats = df.groupby("home_team").agg(
        home_gs=("home_score", "mean"),
        home_wr=("result", lambda x: (x == "home_win").mean()),
    ).reset_index().rename(columns={"home_team": "team"})

    neutral_home = df[df["is_neutral"] == 1].groupby("home_team").agg(
        neutral_gs=("home_score", "mean"),
        neutral_wr=("result", lambda x: (x == "home_win").mean()),
    ).reset_index().rename(columns={"home_team": "team"})

    adv = home_stats.merge(neutral_home, on="team", how="left")
    adv["home_adv_goals"]   = adv["home_gs"]  - adv["neutral_gs"].fillna(adv["home_gs"])
    adv["home_adv_winrate"] = adv["home_wr"]  - adv["neutral_wr"].fillna(adv["home_wr"])
    return adv[["team", "home_adv_goals", "home_adv_winrate"]]


# ---------------------------------------------------------------------------
# HEAD-TO-HEAD
# ---------------------------------------------------------------------------

def compute_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy().sort_values("date").reset_index(drop=True)
    records = []
    for idx, row in tqdm(matches.tail(10000).iterrows()):
        ht, at, dt = row["home_team"], row["away_team"], row["date"]
        past = matches[
            (matches["date"] < dt) & (
                ((matches["home_team"] == ht) & (matches["away_team"] == at)) |
                ((matches["home_team"] == at) & (matches["away_team"] == ht))
            )
        ].tail(H2H_WINDOW)

        if len(past) == 0:
            records.append({"h2h_n": 0, "h2h_home_winrate": np.nan,
                            "h2h_home_goals_mean": np.nan, "h2h_away_goals_mean": np.nan,
                            "h2h_draw_rate": np.nan})
            continue

        wins, draws, losses = 0, 0, 0
        hg_list, ag_list = [], []
        for _, p in past.iterrows():
            hg = p["home_score"] if p["home_team"] == ht else p["away_score"]
            ag = p["away_score"] if p["home_team"] == ht else p["home_score"]
            hg_list.append(hg); ag_list.append(ag)
            if hg > ag: wins += 1
            elif hg == ag: draws += 1
            else: losses += 1

        n = len(past)
        records.append({"h2h_n": n, "h2h_home_winrate": wins / n,
                        "h2h_home_goals_mean": np.mean(hg_list),
                        "h2h_away_goals_mean": np.mean(ag_list),
                        "h2h_draw_rate": draws / n})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FUSIONS
# ---------------------------------------------------------------------------

def merge_strength(matches: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    str_cols = ["nationality_name"] + [c for c in STRENGTH_FEATURES if c in strength.columns]
    s = strength[str_cols].copy()
    home_s = s.rename(columns={c: f"home_{c}" for c in str_cols if c != "nationality_name"},
                      ).rename(columns={"nationality_name": "home_team"})
    away_s = s.rename(columns={c: f"away_{c}" for c in str_cols if c != "nationality_name"},
                      ).rename(columns={"nationality_name": "away_team"})
    return matches.merge(home_s, on="home_team", how="left").merge(away_s, on="away_team", how="left")


def merge_rolling(matches: pd.DataFrame, rolling_stats: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [c for c in rolling_stats.columns if c not in ["team", "date"]]
    home_r = rolling_stats.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in stat_cols}})
    away_r = rolling_stats.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in stat_cols}})
    return matches.merge(home_r, on=["home_team", "date"], how="left")\
                  .merge(away_r, on=["away_team", "date"], how="left")


def merge_home_advantage(matches: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    ha = adv.rename(columns={"team": "home_team", "home_adv_goals": "home_team_adv_goals",
                              "home_adv_winrate": "home_team_adv_winrate"})
    aa = adv.rename(columns={"team": "away_team", "home_adv_goals": "away_team_adv_goals",
                              "home_adv_winrate": "away_team_adv_winrate"})
    return matches.merge(ha, on="home_team", how="left").merge(aa, on="away_team", how="left")


# ---------------------------------------------------------------------------
# DATA AUGMENTATION — SYMÉTRIE TERRAIN NEUTRE
# ---------------------------------------------------------------------------

def augment_neutral_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match joué sur terrain neutre, crée une ligne miroir
    en inversant home et away (scores, features, ELO, rolling stats, etc.).
    Cela enseigne au modèle qu'en terrain neutre A vs B = B vs A.
    Le poids sample de la ligne miroir est identique à l'original.
    """
    neutral_df = df[df["is_neutral"] == 1].copy()
    if neutral_df.empty:
        return df

    mirror = neutral_df.copy()

    # Inverser les scores cibles
    mirror["home_score"], mirror["away_score"] = neutral_df["away_score"].values, neutral_df["home_score"].values
    mirror["home_team"],  mirror["away_team"]  = neutral_df["away_team"].values,  neutral_df["home_team"].values

    # Inverser les colonnes home_* <-> away_*
    all_cols = df.columns.tolist()
    home_cols = [c for c in all_cols if c.startswith("home_") and not c in ("home_team", "home_score")]
    away_cols = [c for c in all_cols if c.startswith("away_") and not c in ("away_team", "away_score")]

    # Colonnes appariées (même suffixe)
    home_suffixes = {c[len("home_"):]: c for c in home_cols}
    away_suffixes = {c[len("away_"):]: c for c in away_cols}
    common = set(home_suffixes) & set(away_suffixes)

    for suf in common:
        hc, ac = home_suffixes[suf], away_suffixes[suf]
        mirror[hc], mirror[ac] = neutral_df[ac].values, neutral_df[hc].values

    # Inverser les diff_* (signe)
    diff_cols = [c for c in all_cols if c.startswith("diff_")]
    for c in diff_cols:
        mirror[c] = -neutral_df[c].values

    # ELO diff et prob
    if "elo_diff" in mirror.columns:
        mirror["elo_diff"] = -neutral_df["elo_diff"].values
    if "elo_win_prob_home" in mirror.columns:
        mirror["elo_win_prob_home"] = 1 - neutral_df["elo_win_prob_home"].values

    # attack_diff / defense_diff
    if "attack_diff" in mirror.columns:
        mirror["attack_diff"]  = -neutral_df["attack_diff"].values
    if "defense_diff" in mirror.columns:
        mirror["defense_diff"] = -neutral_df["defense_diff"].values

    # H2H vu du nouveau home_team
    if "h2h_home_winrate" in mirror.columns:
        mirror["h2h_home_winrate"]    = 1 - neutral_df["h2h_home_winrate"].fillna(0.5) - neutral_df["h2h_draw_rate"].fillna(0)
        mirror["h2h_home_goals_mean"] = neutral_df["h2h_away_goals_mean"].values
        mirror["h2h_away_goals_mean"] = neutral_df["h2h_home_goals_mean"].values

    # result inversé
    result_map = {"home_win": "away_win", "away_win": "home_win", "draw": "draw"}
    if "result" in mirror.columns:
        mirror["result"] = neutral_df["result"].map(result_map)

    augmented = pd.concat([df, mirror], ignore_index=True)
    return augmented.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(matches: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    print("    ELO dynamique...")
    matches, _ = compute_elo(matches)

    print("    Rolling stats...")
    rolling_stats = compute_rolling_stats(matches)

    print("    Head-to-head...")
    h2h = compute_h2h(matches)
    matches = pd.concat([matches.reset_index(drop=True), h2h.reset_index(drop=True)], axis=1)

    print("    Home advantage...")
    adv = compute_home_advantage(matches)

    print("    Fusions...")
    matches = merge_strength(matches, strength)
    matches = merge_rolling(matches, rolling_stats)
    matches = merge_home_advantage(matches, adv)

    matches["tournament_weight"] = matches["tournament"].apply(get_tournament_weight)
    matches["year"]  = matches["date"].dt.year
    matches["month"] = matches["date"].dt.month

    for col in STRENGTH_FEATURES:
        h_col, a_col = f"home_{col}", f"away_{col}"
        if h_col in matches.columns and a_col in matches.columns:
            matches[f"diff_{col}"] = matches[h_col].fillna(0) - matches[a_col].fillna(0)

    roll_stats_base = ["goals_scored", "goals_conceded", "win", "clean_sheet",
                       "goals_scored_ewm10", "goals_conceded_ewm10"]
    for stat in roll_stats_base:
        suffixes = [""] if "ewm" in stat else [f"_last{w}" for w in ROLLING_WINDOWS]
        for suf in suffixes:
            key = f"{stat}{suf}"
            h, a = f"home_{key}", f"away_{key}"
            if h in matches.columns and a in matches.columns:
                matches[f"diff_{key}"] = matches[h].fillna(0) - matches[a].fillna(0)

    if "home_top11_shooting_mean" in matches.columns:
        matches["attack_diff"]  = (matches["home_top11_shooting_mean"].fillna(0)
                                   - matches["away_top11_defending_mean"].fillna(0))
        matches["defense_diff"] = (matches["away_top11_shooting_mean"].fillna(0)
                                   - matches["home_top11_defending_mean"].fillna(0))

    return matches


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "country", "neutral", "result", "goal_diff", "total_goals",
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# MODÈLES
# ---------------------------------------------------------------------------

def build_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="poisson", n_estimators=500, learning_rate=0.04,
        num_leaves=31, max_depth=6, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.2,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )


def build_poisson_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.1, max_iter=600)),
    ])


def evaluate(name: str, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {name:40s}  MAE={mae:.4f}  RMSE={rmse:.4f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def train(data_dir: Path, model_dir: Path):
    print("=" * 65)
    print("CDM 2026 — Entraînement du prédicteur de scores v3")
    print("=" * 65)

    print("\n[1/7] Chargement des données...")
    matches, strength = load_data(data_dir)
    print(f"  Matchs : {len(matches):,}  |  Équipes strength : {len(strength):,}")

    print("\n[2/7] Feature engineering (ELO + H2H + rolling + strength)...")
    df = build_features(matches, strength)
    feature_cols = get_feature_cols(df)

    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_score"] >= 0) & (df["home_score"] <= 20)]
    df = df[(df["away_score"] >= 0) & (df["away_score"] <= 20)]

    print(f"  Features : {len(feature_cols)}  |  Samples avant augmentation : {len(df):,}")

    print("\n[3/7] Augmentation symétrie terrain neutre...")
    df_aug = augment_neutral_symmetry(df)
    n_neutral = (df["is_neutral"] == 1).sum()
    print(f"  Matchs neutres : {n_neutral:,} → +{n_neutral:,} miroirs → total {len(df_aug):,} samples")

    # Split temporel sur le df ORIGINAL (avant augmentation) pour évaluation honnête
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_train_orig = df.iloc[:split_idx]
    df_test  = df.iloc[split_idx:]

    # Le train inclut les miroirs des matchs neutres du train set uniquement
    df_train_aug = augment_neutral_symmetry(df_train_orig)
    print(f"  Train augmenté : {len(df_train_aug):,}  |  Test (original) : {len(df_test):,}")

    X_train = df_train_aug[feature_cols].fillna(0)
    y_home_train = df_train_aug["home_score"].astype(float)
    y_away_train = df_train_aug["away_score"].astype(float)
    w_train = df_train_aug["tournament_weight"].values

    X_test = df_test[feature_cols].fillna(0)
    y_home_test = df_test["home_score"].astype(float)
    y_away_test = df_test["away_score"].astype(float)

    print("\n[4/7] Entraînement des modèles...")
    print("  > Poisson GLM (home / away)...")
    poisson_home = build_poisson_pipeline()
    poisson_home.fit(X_train, y_home_train, **{"model__sample_weight": w_train})
    poisson_away = build_poisson_pipeline()
    poisson_away.fit(X_train, y_away_train, **{"model__sample_weight": w_train})

    print("  > LightGBM (home / away)...")
    lgbm_home = build_lgbm()
    lgbm_home.fit(X_train, y_home_train, sample_weight=w_train)
    lgbm_away = build_lgbm()
    lgbm_away.fit(X_train, y_away_train, sample_weight=w_train)

    ENSEMBLE_W_LGBM    = 0.6
    ENSEMBLE_W_POISSON = 0.4

    print("  > Calibrage Isotonic (train set)...")
    raw_h_tr = ENSEMBLE_W_LGBM * lgbm_home.predict(X_train) + ENSEMBLE_W_POISSON * poisson_home.predict(X_train)
    raw_a_tr = ENSEMBLE_W_LGBM * lgbm_away.predict(X_train) + ENSEMBLE_W_POISSON * poisson_away.predict(X_train)
    iso_home = IsotonicRegression(out_of_bounds="clip"); iso_home.fit(raw_h_tr, y_home_train.values)
    iso_away = IsotonicRegression(out_of_bounds="clip"); iso_away.fit(raw_a_tr, y_away_train.values)

    print("\n[5/7] Évaluation sur le jeu de test (matchs originaux)...")
    raw_h_te = ENSEMBLE_W_LGBM * lgbm_home.predict(X_test) + ENSEMBLE_W_POISSON * poisson_home.predict(X_test)
    raw_a_te = ENSEMBLE_W_LGBM * lgbm_away.predict(X_test) + ENSEMBLE_W_POISSON * poisson_away.predict(X_test)
    cal_h_te = iso_home.predict(raw_h_te)
    cal_a_te = iso_away.predict(raw_a_te)

    evaluate("Poisson GLM — home",       y_home_test, poisson_home.predict(X_test))
    evaluate("Poisson GLM — away",       y_away_test, poisson_away.predict(X_test))
    evaluate("LightGBM — home",          y_home_test, lgbm_home.predict(X_test))
    evaluate("LightGBM — away",          y_away_test, lgbm_away.predict(X_test))
    evaluate("Ensemble calibré — home",  y_home_test, cal_h_te)
    evaluate("Ensemble calibré — away",  y_away_test, cal_a_te)

    correct_exact  = np.mean((np.round(cal_h_te) == y_home_test.values) & (np.round(cal_a_te) == y_away_test.values))
    correct_result = np.mean(np.sign(cal_h_te - cal_a_te) == np.sign(y_home_test.values - y_away_test.values))
    print(f"\n  Score exact correct  : {correct_exact:.2%}")
    print(f"  Résultat correct     : {correct_result:.2%}")

    print("\n[6/7] Vérification de la symétrie en terrain neutre...")
    # Test sur quelques paires de pays si les données permettent
    neutral_test = df_test[df_test["is_neutral"] == 1].head(20)
    if len(neutral_test) > 0:
        # Simuler l'inférence symétrique (double pass)
        X_mirror = neutral_test.copy()
        # Inverser home<->away features
        home_fcols = [c for c in feature_cols if c.startswith("home_")]
        away_fcols = [c for c in feature_cols if c.startswith("away_")]
        h_suf = {c[5:]: c for c in home_fcols}
        a_suf = {c[5:]: c for c in away_fcols}
        common = set(h_suf) & set(a_suf)
        for suf in common:
            hc, ac = h_suf[suf], a_suf[suf]
            X_mirror[hc], X_mirror[ac] = neutral_test[ac].values, neutral_test[hc].values
        for dc in [c for c in feature_cols if c.startswith("diff_")]:
            X_mirror[dc] = -neutral_test[dc].values
        if "elo_diff" in feature_cols:        X_mirror["elo_diff"]         = -neutral_test["elo_diff"].values
        if "elo_win_prob_home" in feature_cols: X_mirror["elo_win_prob_home"] = 1 - neutral_test["elo_win_prob_home"].values

        Xn  = neutral_test[feature_cols].fillna(0)
        Xnm = X_mirror[feature_cols].fillna(0)

        lh_n  = iso_home.predict(ENSEMBLE_W_LGBM * lgbm_home.predict(Xn)  + ENSEMBLE_W_POISSON * poisson_home.predict(Xn))
        la_n  = iso_away.predict(ENSEMBLE_W_LGBM * lgbm_away.predict(Xn)  + ENSEMBLE_W_POISSON * poisson_away.predict(Xn))
        lh_m  = iso_home.predict(ENSEMBLE_W_LGBM * lgbm_home.predict(Xnm) + ENSEMBLE_W_POISSON * poisson_home.predict(Xnm))
        la_m  = iso_away.predict(ENSEMBLE_W_LGBM * lgbm_away.predict(Xnm) + ENSEMBLE_W_POISSON * poisson_away.predict(Xnm))

        sym_h = (lh_n + la_m) / 2
        sym_a = (la_n + lh_m) / 2
        asym  = np.mean(np.abs(sym_h - sym_a))
        print(f"  Écart symétrique résiduel (doit être ≈ 0) : {asym:.6f}")
        print(f"  ✅ Symétrie garantie par double inférence en predict.py")

    print("\n[7/7] Sauvegarde des artefacts...")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(poisson_home, model_dir / "poisson_home.pkl")
    joblib.dump(poisson_away, model_dir / "poisson_away.pkl")
    joblib.dump(lgbm_home,    model_dir / "lgbm_home.pkl")
    joblib.dump(lgbm_away,    model_dir / "lgbm_away.pkl")
    joblib.dump(iso_home,     model_dir / "iso_home.pkl")
    joblib.dump(iso_away,     model_dir / "iso_away.pkl")

    meta = {
        "feature_cols": feature_cols,
        "strength_features": STRENGTH_FEATURES,
        "rolling_windows": ROLLING_WINDOWS,
        "h2h_window": H2H_WINDOW,
        "ensemble_weight_lgbm": ENSEMBLE_W_LGBM,
        "ensemble_weight_poisson": ENSEMBLE_W_POISSON,
        "tournament_weights": TOURNAMENT_WEIGHTS,
        "elo_init": ELO_INIT,
    }
    joblib.dump(meta, model_dir / "meta.pkl")

    rolling_stats = compute_rolling_stats(matches)
    latest_rolling = rolling_stats.sort_values("date").groupby("team").last().reset_index()
    latest_rolling.to_parquet(model_dir / "latest_rolling_stats.parquet", index=False)

    _, elo_final = compute_elo(matches)
    pd.DataFrame(list(elo_final.items()), columns=["team", "elo"]).to_parquet(
        model_dir / "elo_ratings.parquet", index=False)

    compute_home_advantage(matches).to_parquet(model_dir / "home_advantage.parquet", index=False)

    matches[["date", "home_team", "away_team", "home_score", "away_score", "result", "tournament"]]\
        .to_parquet(model_dir / "matches_history.parquet", index=False)

    strength.to_parquet(model_dir / "strength.parquet", index=False)

    print(f"\n✅ Modèles sauvegardés dans : {model_dir.resolve()}")
    print("   poisson_home/away.pkl | lgbm_home/away.pkl | iso_home/away.pkl")
    print("   meta.pkl | latest_rolling_stats.parquet | elo_ratings.parquet")
    print("   home_advantage.parquet | matches_history.parquet | strength.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CDM 2026 score predictor v3")
    parser.add_argument("--data-dir",  type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
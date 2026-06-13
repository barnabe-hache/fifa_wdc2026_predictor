"""
predict.py — Prédicteur de scores CDM 2026 v3
==============================================
Symétrie terrain neutre : double inférence (A vs B + B vs A) + moyenne des lambdas.
  predict_score("France", "Brazil", neutral=True)
    == predict_score("Brazil", "France", neutral=True)   ✓ garanti

Usage CLI :
    python predict.py "France" "Brazil"
    python predict.py "France" "Brazil" --no-neutral

Usage Python :
    from predict import predict_score
    r = predict_score("France", "Brazil")
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR    = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MAX_GOALS          = 8
DEFAULT_TOURNAMENT = "FIFA World Cup"
ELO_INIT           = 1500
H2H_WINDOW         = 10
ROLLING_WINDOWS    = [5, 10, 20]

STRENGTH_FEATURES = [
    "team_strength_score", "top11_overall_mean", "top11_shooting_mean",
    "top11_passing_mean", "top11_defending_mean", "top11_pace_mean",
    "top11_physic_mean", "top11_dribbling_mean", "top11_potential_mean",
    "top23_overall_mean", "top23_potential_mean", "top5_overall_mean",
    "count_85_plus_top23", "count_90_plus_top23", "top11_value_sum_eur",
]

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 3.0, "UEFA Euro": 2.5, "Copa América": 2.5,
    "Africa Cup of Nations": 2.0, "AFC Asian Cup": 2.0, "CONCACAF Gold Cup": 2.0,
    "Confederations Cup": 2.0, "UEFA Nations League": 1.8,
    "World Cup qualification": 1.5, "Friendly": 0.6,
}


# ---------------------------------------------------------------------------
# PREDICTOR
# ---------------------------------------------------------------------------

class Predictor:

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._load()

    def _load(self):
        md = self.model_dir
        self.poisson_home = joblib.load(md / "poisson_home.pkl")
        self.poisson_away = joblib.load(md / "poisson_away.pkl")
        self.lgbm_home    = joblib.load(md / "lgbm_home.pkl")
        self.lgbm_away    = joblib.load(md / "lgbm_away.pkl")
        self.iso_home     = joblib.load(md / "iso_home.pkl")
        self.iso_away     = joblib.load(md / "iso_away.pkl")
        self.meta         = joblib.load(md / "meta.pkl")
        self.rolling      = pd.read_parquet(md / "latest_rolling_stats.parquet")
        self.strength     = pd.read_parquet(md / "strength.parquet")
        self.elo_df       = pd.read_parquet(md / "elo_ratings.parquet")
        self.adv_df       = pd.read_parquet(md / "home_advantage.parquet")
        self.history      = pd.read_parquet(md / "matches_history.parquet")
        self.history["date"] = pd.to_datetime(self.history["date"])
        self.feature_cols = self.meta["feature_cols"]
        self.w_lgbm       = self.meta["ensemble_weight_lgbm"]
        self.w_poisson    = self.meta["ensemble_weight_poisson"]

    # ── Lookups ───────────────────────────────────────────────────────────

    def _elo(self, team: str) -> float:
        r = self.elo_df[self.elo_df["team"] == team]
        return float(r["elo"].iloc[0]) if not r.empty else ELO_INIT

    def _get_strength(self, team: str) -> dict:
        r = self.strength[self.strength["nationality_name"] == team]
        return r.iloc[0].to_dict() if not r.empty else {}

    def _get_rolling(self, team: str) -> dict:
        r = self.rolling[self.rolling["team"] == team]
        return r.iloc[0].to_dict() if not r.empty else {}

    def _get_adv(self, team: str) -> dict:
        r = self.adv_df[self.adv_df["team"] == team]
        return r.iloc[0].to_dict() if not r.empty else {}

    def _tournament_weight(self, t: str) -> float:
        for key, w in TOURNAMENT_WEIGHTS.items():
            if key.lower() in t.lower():
                return w
        return 1.0

    # ── H2H ──────────────────────────────────────────────────────────────

    def _h2h(self, team_a: str, team_b: str) -> dict:
        """
        Calcule le H2H du point de vue de team_a (wins = victoires de team_a).
        Toujours appelé avec (team_a, team_b) dans l'ordre canonique alphabétique
        quand neutral=True pour que le résultat soit stable quel que soit l'appelant.
        """
        past = self.history[
            ((self.history["home_team"] == team_a) & (self.history["away_team"] == team_b)) |
            ((self.history["home_team"] == team_b) & (self.history["away_team"] == team_a))
        ].tail(H2H_WINDOW)

        if len(past) == 0:
            return {
                "h2h_n": 0, "h2h_home_winrate": np.nan,
                "h2h_home_goals_mean": np.nan, "h2h_away_goals_mean": np.nan,
                "h2h_draw_rate": np.nan,
                "h2h_wins": 0, "h2h_draws": 0, "h2h_losses": 0,
                "h2h_goals_scored": [], "h2h_goals_conceded": [],
            }

        wins, draws, losses = 0, 0, 0
        hg_list, ag_list = [], []
        for _, p in past.iterrows():
            hg = p["home_score"] if p["home_team"] == team_a else p["away_score"]
            ag = p["away_score"] if p["home_team"] == team_a else p["home_score"]
            hg_list.append(float(hg)); ag_list.append(float(ag))
            if hg > ag: wins += 1
            elif hg == ag: draws += 1
            else: losses += 1

        n = len(past)
        return {
            "h2h_n": n,
            "h2h_home_winrate": wins / n,
            "h2h_home_goals_mean": float(np.mean(hg_list)),
            "h2h_away_goals_mean": float(np.mean(ag_list)),
            "h2h_draw_rate": draws / n,
            "h2h_wins": wins, "h2h_draws": draws, "h2h_losses": losses,
            "h2h_goals_scored": hg_list, "h2h_goals_conceded": ag_list,
        }

    # ── Construction feature row ──────────────────────────────────────────

    def _build_row(self, home_team: str, away_team: str,
                   neutral: bool, tournament: str,
                   elo_h: float, elo_a: float, h2h: dict) -> pd.DataFrame:
        row: dict = {}

        row["is_neutral"]        = int(neutral)
        row["tournament_weight"] = self._tournament_weight(tournament)
        row["year"]              = pd.Timestamp.now().year
        row["month"]             = pd.Timestamp.now().month

        row["home_elo_before"]   = elo_h
        row["away_elo_before"]   = elo_a
        row["elo_diff"]          = elo_h - elo_a
        row["elo_win_prob_home"] = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))

        h_str = self._get_strength(home_team)
        a_str = self._get_strength(away_team)
        for col in STRENGTH_FEATURES:
            row[f"home_{col}"] = h_str.get(col, np.nan)
            row[f"away_{col}"] = a_str.get(col, np.nan)
        for col in STRENGTH_FEATURES:
            row[f"diff_{col}"] = (row.get(f"home_{col}") or 0) - (row.get(f"away_{col}") or 0)

        hs = h_str.get("top11_shooting_mean", 0) or 0
        hd = h_str.get("top11_defending_mean", 0) or 0
        as_ = a_str.get("top11_shooting_mean", 0) or 0
        ad = a_str.get("top11_defending_mean", 0) or 0
        row["attack_diff"]  = hs - ad
        row["defense_diff"] = as_ - hd

        h_roll = self._get_rolling(home_team)
        a_roll = self._get_rolling(away_team)
        roll_stats = ["goals_scored", "goals_conceded", "win", "draw", "loss", "clean_sheet", "scored_0"]
        for stat in roll_stats:
            for w in ROLLING_WINDOWS:
                key = f"{stat}_last{w}"
                row[f"home_{key}"] = h_roll.get(key, 0) or 0
                row[f"away_{key}"] = a_roll.get(key, 0) or 0
        row["home_goals_scored_ewm10"]   = h_roll.get("goals_scored_ewm10", 0) or 0
        row["away_goals_scored_ewm10"]   = a_roll.get("goals_scored_ewm10", 0) or 0
        row["home_goals_conceded_ewm10"] = h_roll.get("goals_conceded_ewm10", 0) or 0
        row["away_goals_conceded_ewm10"] = a_roll.get("goals_conceded_ewm10", 0) or 0

        for stat in ["goals_scored", "goals_conceded", "win", "clean_sheet"]:
            for w in ROLLING_WINDOWS:
                key = f"{stat}_last{w}"
                row[f"diff_{key}"] = row.get(f"home_{key}", 0) - row.get(f"away_{key}", 0)
        row["diff_goals_scored_ewm10"]   = row["home_goals_scored_ewm10"] - row["away_goals_scored_ewm10"]
        row["diff_goals_conceded_ewm10"] = row["home_goals_conceded_ewm10"] - row["away_goals_conceded_ewm10"]

        h_adv = self._get_adv(home_team)
        a_adv = self._get_adv(away_team)
        row["home_team_adv_goals"]   = h_adv.get("home_adv_goals", 0) or 0
        row["home_team_adv_winrate"] = h_adv.get("home_adv_winrate", 0) or 0
        row["away_team_adv_goals"]   = a_adv.get("home_adv_goals", 0) or 0
        row["away_team_adv_winrate"] = a_adv.get("home_adv_winrate", 0) or 0

        row["h2h_n"]               = h2h.get("h2h_n", 0) or 0
        row["h2h_home_winrate"]    = h2h.get("h2h_home_winrate") or 0
        row["h2h_home_goals_mean"] = h2h.get("h2h_home_goals_mean") or 0
        row["h2h_away_goals_mean"] = h2h.get("h2h_away_goals_mean") or 0
        row["h2h_draw_rate"]       = h2h.get("h2h_draw_rate") or 0

        df = pd.DataFrame([row])
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols].fillna(0)

    # ── Inférence brute ───────────────────────────────────────────────────

    def _raw_lambda(self, X: pd.DataFrame) -> tuple[float, float]:
        """Retourne (lambda_home, lambda_away) calibrés pour une feature row."""
        p_h = max(float(self.poisson_home.predict(X)[0]), 0.05)
        p_a = max(float(self.poisson_away.predict(X)[0]), 0.05)
        l_h = max(float(self.lgbm_home.predict(X)[0]),   0.05)
        l_a = max(float(self.lgbm_away.predict(X)[0]),   0.05)
        raw_h = self.w_lgbm * l_h + self.w_poisson * p_h
        raw_a = self.w_lgbm * l_a + self.w_poisson * p_a
        cal_h = max(float(self.iso_home.predict([raw_h])[0]), 0.05)
        cal_a = max(float(self.iso_away.predict([raw_a])[0]), 0.05)
        return cal_h, cal_a, p_h, p_a, l_h, l_a, raw_h, raw_a

    def _mirror_row(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Crée la feature row miroir (home ↔ away inversés) pour la double inférence.
        """
        Xm = X.copy()
        fc = self.feature_cols

        home_fcols = {c[5:]: c for c in fc if c.startswith("home_")}
        away_fcols = {c[5:]: c for c in fc if c.startswith("away_")}
        common = set(home_fcols) & set(away_fcols)
        for suf in common:
            hc, ac = home_fcols[suf], away_fcols[suf]
            Xm[hc], Xm[ac] = X[ac].values, X[hc].values

        for dc in [c for c in fc if c.startswith("diff_")]:
            Xm[dc] = -X[dc].values

        if "elo_diff" in fc:          Xm["elo_diff"]          = -X["elo_diff"].values
        if "elo_win_prob_home" in fc: Xm["elo_win_prob_home"] = 1 - X["elo_win_prob_home"].values
        if "attack_diff" in fc:       Xm["attack_diff"]       = -X["attack_diff"].values
        if "defense_diff" in fc:      Xm["defense_diff"]      = -X["defense_diff"].values

        return Xm

    # ── Dérivés probabilistes ─────────────────────────────────────────────

    @staticmethod
    def _score_matrix(lh: float, la: float) -> np.ndarray:
        ph = np.array([poisson.pmf(k, lh) for k in range(MAX_GOALS + 1)])
        pa = np.array([poisson.pmf(k, la) for k in range(MAX_GOALS + 1)])
        return np.outer(ph, pa)

    @staticmethod
    def _result_probs(m: np.ndarray) -> dict:
        return {
            "home_win": float(np.sum(np.tril(m, -1))),
            "draw":     float(np.trace(m)),
            "away_win": float(np.sum(np.triu(m, 1))),
        }

    @staticmethod
    def _top_scores(m: np.ndarray, n: int = 5) -> list:
        flat = m.flatten(); top = np.argsort(flat)[::-1][:n]; sz = m.shape[0]
        return [{"home": int(i // sz), "away": int(i % sz), "proba": round(float(flat[i]), 4)}
                for i in top]

    @staticmethod
    def _xg_analysis(lh: float, la: float) -> dict:
        def over(thr):
            return round(1 - sum(poisson.pmf(i, lh) * poisson.pmf(j, la)
                                 for i in range(thr + 2) for j in range(thr + 2)
                                 if i + j <= thr), 4)
        return {
            "xg_home": round(lh, 3), "xg_away": round(la, 3),
            "xg_diff": round(lh - la, 3), "xg_total": round(lh + la, 3),
            "btts_prob":     round((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la)), 4),
            "over_0_5_prob": over(0), "over_1_5_prob": over(1),
            "over_2_5_prob": over(2), "over_3_5_prob": over(3), "over_4_5_prob": over(4),
        }

    @staticmethod
    def _clean_sheet_probs(lh: float, la: float) -> dict:
        return {
            "clean_sheet_home_prob": round(float(poisson.pmf(0, la)), 4),
            "clean_sheet_away_prob": round(float(poisson.pmf(0, lh)), 4),
        }

    @staticmethod
    def _dominance_index(lh, la, elo_h, elo_a, rp) -> float:
        xg_s  = float(np.tanh((lh - la) / 1.5))
        elo_s = float(np.tanh((elo_h - elo_a) / 300))
        win_s = 2 * rp["home_win"] - 1
        return round(float(np.clip(0.4 * xg_s + 0.35 * elo_s + 0.25 * win_s, -1, 1)), 4)

    @staticmethod
    def _upset(rp, elo_h, elo_a) -> dict:
        if elo_h >= elo_a:
            return {"favorite": "home", "upset_prob": round(rp["away_win"], 4)}
        return {"favorite": "away", "upset_prob": round(rp["home_win"], 4)}

    @staticmethod
    def _confidence(lh, la, elo_h, elo_a, h2h_n, rp) -> float:
        probs   = list(rp.values())
        clarity = float(np.clip((max(probs) - 1/3) / 0.67, 0, 1))
        elo_sig = float(np.clip(abs(elo_h - elo_a) / 500, 0, 1))
        h2h_b   = float(np.clip(h2h_n / H2H_WINDOW, 0, 1)) * 0.5
        return round(float(np.clip(0.5 * clarity + 0.35 * elo_sig + 0.15 * h2h_b, 0, 1)), 4)

    def _team_info(self, team: str) -> dict:
        s   = self._get_strength(team)
        r   = self._get_rolling(team)
        adv = self._get_adv(team)
        info: dict = {"name": team, "elo": round(self._elo(team), 1)}
        for col in STRENGTH_FEATURES:
            v = s.get(col)
            info[col] = round(float(v), 3) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None
        for w in ROLLING_WINDOWS:
            for stat in ["goals_scored", "goals_conceded", "win", "draw", "loss", "clean_sheet"]:
                key = f"{stat}_last{w}"
                v = r.get(key)
                info[key] = round(float(v), 3) if v is not None else None
        info["goals_scored_ewm10"]   = round(float(r["goals_scored_ewm10"]), 3) if r.get("goals_scored_ewm10") is not None else None
        info["goals_conceded_ewm10"] = round(float(r["goals_conceded_ewm10"]), 3) if r.get("goals_conceded_ewm10") is not None else None
        info["home_adv_goals"]   = round(float(adv.get("home_adv_goals", 0)), 3)
        info["home_adv_winrate"] = round(float(adv.get("home_adv_winrate", 0)), 3)
        return info

    # ── Predict principal ─────────────────────────────────────────────────

    def predict_score(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = DEFAULT_TOURNAMENT,
    ) -> dict:
        """
        Prédit le score et retourne un dict exhaustif.
        En terrain neutre : double inférence (A vs B + B vs A) + moyenne des λ.
        Garantit predict_score(A,B,neutral=True) == predict_score(B,A,neutral=True).
        """
        elo_h = self._elo(home_team)
        elo_a = self._elo(away_team)

        # H2H : toujours du point de vue de home_team
        h2h_full = self._h2h(home_team, away_team)
        h2h_feat = {k: h2h_full[k] for k in
                    ["h2h_n", "h2h_home_winrate", "h2h_home_goals_mean",
                     "h2h_away_goals_mean", "h2h_draw_rate"]}

        # Feature row forward (home_team joue home)
        X_fwd = self._build_row(home_team, away_team, neutral, tournament, elo_h, elo_a, h2h_feat)
        cal_h_fwd, cal_a_fwd, p_h_f, p_a_f, l_h_f, l_a_f, raw_h_f, raw_a_f = self._raw_lambda(X_fwd)

        if neutral:
            # ── Double inférence ─────────────────────────────────────────
            # H2H inversé (du point de vue de away_team)
            h2h_inv = self._h2h(away_team, home_team)
            h2h_feat_inv = {k: h2h_inv[k] for k in
                            ["h2h_n", "h2h_home_winrate", "h2h_home_goals_mean",
                             "h2h_away_goals_mean", "h2h_draw_rate"]}
            # Feature row inverse (away_team joue home)
            X_inv = self._build_row(away_team, home_team, neutral, tournament, elo_a, elo_h, h2h_feat_inv)
            cal_h_inv, cal_a_inv, _, _, _, _, _, _ = self._raw_lambda(X_inv)

            # λ symétriques : λ_A = moy(λ_home(A,B), λ_away(B,A))
            lambda_home = (cal_h_fwd + cal_a_inv) / 2
            lambda_away = (cal_a_fwd + cal_h_inv) / 2

            # Pour le reporting des sous-modèles, on garde le forward pass
            p_h, p_a, l_h, l_a = p_h_f, p_a_f, l_h_f, l_a_f
            raw_h, raw_a = raw_h_f, raw_a_f
        else:
            lambda_home = cal_h_fwd
            lambda_away = cal_a_fwd
            p_h, p_a, l_h, l_a = p_h_f, p_a_f, l_h_f, l_a_f
            raw_h, raw_a = raw_h_f, raw_a_f

        # Dérivés
        matrix       = self._score_matrix(lambda_home, lambda_away)
        result_probs = self._result_probs(matrix)
        best_idx     = np.unravel_index(np.argmax(matrix), matrix.shape)
        predicted    = max(result_probs, key=result_probs.get)
        xg           = self._xg_analysis(lambda_home, lambda_away)
        cs           = self._clean_sheet_probs(lambda_home, lambda_away)
        dominance    = self._dominance_index(lambda_home, lambda_away, elo_h, elo_a, result_probs)
        upset        = self._upset(result_probs, elo_h, elo_a)
        confidence   = self._confidence(lambda_home, lambda_away, elo_h, elo_a,
                                        h2h_full["h2h_n"], result_probs)

        h2h_out = {
            "h2h_n":              h2h_full["h2h_n"],
            "h2h_home_winrate":   round(h2h_full.get("h2h_home_winrate") or 0, 4),
            "h2h_away_winrate":   round(1 - (h2h_full.get("h2h_home_winrate") or 0)
                                          - (h2h_full.get("h2h_draw_rate") or 0), 4),
            "h2h_draw_rate":      round(h2h_full.get("h2h_draw_rate") or 0, 4),
            "h2h_wins":           h2h_full.get("h2h_wins", 0),
            "h2h_draws":          h2h_full.get("h2h_draws", 0),
            "h2h_losses":         h2h_full.get("h2h_losses", 0),
            "h2h_avg_goals_home": round(h2h_full.get("h2h_home_goals_mean") or 0, 3),
            "h2h_avg_goals_away": round(h2h_full.get("h2h_away_goals_mean") or 0, 3),
            "h2h_last_results": [
                "W" if hg > ag else ("D" if hg == ag else "L")
                for hg, ag in zip(h2h_full.get("h2h_goals_scored", []),
                                  h2h_full.get("h2h_goals_conceded", []))
            ],
        }

        h_known = home_team in self.strength["nationality_name"].values
        a_known = away_team in self.strength["nationality_name"].values

        return {
            "prediction":          {"home": int(best_idx[0]), "away": int(best_idx[1])},
            "predicted_result":    predicted,
            "confidence_score":    confidence,
            "lambda_home":         round(lambda_home, 4),
            "lambda_away":         round(lambda_away, 4),
            "lambda_home_raw":     round(raw_h, 4),
            "lambda_away_raw":     round(raw_a, 4),
            "poisson_lambda_home": round(p_h, 4),
            "poisson_lambda_away": round(p_a, 4),
            "lgbm_lambda_home":    round(l_h, 4),
            "lgbm_lambda_away":    round(l_a, 4),
            "result_probs":        {k: round(v, 4) for k, v in result_probs.items()},
            "top5_scores":         self._top_scores(matrix),
            "score_matrix":        matrix,
            "xg_analysis":         xg,
            "clean_sheet_probs":   cs,
            "elo_home":            round(elo_h, 1),
            "elo_away":            round(elo_a, 1),
            "elo_diff":            round(elo_h - elo_a, 1),
            "elo_win_prob_home":   round(1 / (1 + 10 ** ((elo_a - elo_h) / 400)), 4),
            "dominance_index":     dominance,
            "upset":               upset,
            "h2h":                 h2h_out,
            "home_team_info":      self._team_info(home_team),
            "away_team_info":      self._team_info(away_team),
            "match_context": {
                "home_team":  home_team, "away_team": away_team,
                "tournament": tournament, "neutral": neutral,
                "home_known_in_strength": h_known,
                "away_known_in_strength": a_known,
                "symmetric_inference":    neutral,
            },
        }


# ---------------------------------------------------------------------------
# INTERFACE FONCTIONNELLE
# ---------------------------------------------------------------------------

_predictor: Predictor | None = None


def predict_score(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament: str = DEFAULT_TOURNAMENT,
    model_dir: Path = MODEL_DIR,
) -> dict:
    global _predictor
    if _predictor is None or _predictor.model_dir != model_dir:
        _predictor = Predictor(model_dir)
    return _predictor.predict_score(home_team, away_team, neutral, tournament)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: dict):
    ctx = r["match_context"]
    pred, rp = r["prediction"], r["result_probs"]
    xg, cs, h2h = r["xg_analysis"], r["clean_sheet_probs"], r["h2h"]
    W = 60
    sym = " [symétrique ✓]" if ctx.get("symmetric_inference") else ""
    print("\n" + "═" * W)
    print(f"  {ctx['home_team']:>20}  vs  {ctx['away_team']:<20}")
    print(f"  {ctx['tournament']}  |  Neutre : {ctx['neutral']}{sym}")
    print("═" * W)
    bar = "▓" * int(r["confidence_score"] * 20) + "░" * (20 - int(r["confidence_score"] * 20))
    print(f"\n  🏆 Score prédit   : {pred['home']} – {pred['away']}")
    print(f"  📊 Résultat       : {r['predicted_result'].replace('_',' ').upper()}")
    print(f"  🎯 Confiance      : [{bar}] {r['confidence_score']:.0%}")
    dom = r["dominance_index"]
    print(f"  ⚖  Dominance      : {dom:+.3f}  ({'→ home' if dom > 0.1 else ('← away' if dom < -0.1 else '≈ équilibré')})")
    print(f"\n  ELO  {ctx['home_team']}: {r['elo_home']:.0f}  {ctx['away_team']}: {r['elo_away']:.0f}  (diff {r['elo_diff']:+.0f})")
    print(f"  λ calibré : {r['lambda_home']:.3f} / {r['lambda_away']:.3f}  |  xG total : {xg['xg_total']}")
    print(f"\n  Probas résultat :")
    for lbl, k in [(ctx['home_team'], "home_win"), ("Nul", "draw"), (ctx['away_team'], "away_win")]:
        v = rp[k]; bbar = "█" * int(v * 40)
        print(f"    {lbl:<20} {v:.1%}  {bbar}")
    print(f"\n  BTTS: {xg['btts_prob']:.1%}  |  Over 1.5: {xg['over_1_5_prob']:.1%}  |  Over 2.5: {xg['over_2_5_prob']:.1%}")
    print(f"  CS home: {cs['clean_sheet_home_prob']:.1%}  |  CS away: {cs['clean_sheet_away_prob']:.1%}")
    print(f"\n  Top 5 scores : " + "  ".join(f"{s['home']}-{s['away']}({s['proba']:.1%})" for s in r["top5_scores"]))
    if h2h["h2h_n"] > 0:
        print(f"\n  H2H : {h2h['h2h_wins']}V {h2h['h2h_draws']}N {h2h['h2h_losses']}D")
        print(f"  Derniers : {' '.join(h2h['h2h_last_results'][-8:])}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--no-neutral", dest="neutral", action="store_false", default=True)
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT)
    parser.add_argument("--model-dir",  type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    _print_result(predict_score(args.home_team, args.away_team,
                                neutral=args.neutral, tournament=args.tournament,
                                model_dir=args.model_dir))
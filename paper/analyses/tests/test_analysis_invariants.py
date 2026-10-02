"""Invariants the S11/S12 machinery has to satisfy before it runs on the snapshot.

Each test fails if the underlying feature is wrong, not merely absent: the cell
collapse has to reproduce a record-level fit, the quota partition has to preserve
the quantity it claims to preserve, a failed mixed-model fit must not reach a
result file as a number, and a results path must never be written twice.
"""

import ast
import builtins
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glmm  # noqa: E402
import mexico_confirmed_cases  # noqa: E402
import paths  # noqa: E402
import s11_site_definitions as s11  # noqa: E402
import s12_composition_ladder as s12  # noqa: E402
import validate_glmm  # noqa: E402


def synthetic_records(n_records=40_000, n_sites=12):
    """Records with the three binary covariates, site-varying composition and mortality."""
    rng = np.random.default_rng()
    site = rng.integers(0, n_sites, n_records)
    elderly_rate = np.linspace(0.05, 0.45, n_sites)[site]
    elderly = rng.random(n_records) < elderly_rate
    male = rng.random(n_records) < 0.5
    comorbidity = rng.random(n_records) < (0.15 + 0.35 * elderly)
    logit = -4.0 + 2.6 * elderly + 0.35 * male + 0.8 * comorbidity + 0.4 * (site / n_sites)
    died = rng.random(n_records) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "site": site, "site_residence": site,
        "municipality": site * 100 + rng.integers(0, 4, n_records),
        "elderly": elderly.astype(int), "male": male.astype(int),
        "has_comorbidity": comorbidity.astype(int), "died": died.astype(int),
    })


def test_cell_collapsed_fit_equals_record_level_fit():
    df = synthetic_records()
    cells = s11.cells_by(df, "site")
    _, collapsed = s11.fit_record_model(cells)

    design = np.column_stack([np.ones(len(df))] +
                             [df[c].to_numpy(dtype=float) for c in s11.COVARIATES])
    record_level = sm.GLM(df["died"].to_numpy(dtype=float), design,
                          family=sm.families.Binomial()).fit()

    assert collapsed["intercept"] == pytest.approx(record_level.params[0], abs=1e-8)
    for i, covariate in enumerate(s11.COVARIATES):
        assert collapsed[f"beta_{covariate}"] == pytest.approx(record_level.params[i + 1],
                                                               abs=1e-8)


def test_quota_partition_preserves_size_and_elderly_count(tmp_path, monkeypatch):
    monkeypatch.setattr(s11, "OUTPUT_DIR", tmp_path)
    df = synthetic_records()
    rng = np.random.default_rng()

    preserving = s11.partition_scheme(df, rng, replications=20, preserve_composition=True)
    assert preserving["realized_quota_deviations"]["size_deviations"] == 0
    assert preserving["realized_quota_deviations"]["elderly_deviations"] == 0

    unrestricted = s11.partition_scheme(df, rng, replications=20, preserve_composition=False)
    assert unrestricted["realized_quota_deviations"]["size_deviations"] == 0

    observed_sd = float(df.groupby("site")["elderly"].mean().std(ddof=1))
    assert preserving["mean_elderly_share_sd"] == pytest.approx(observed_sd, rel=1e-9)
    assert unrestricted["mean_elderly_share_sd"] < 0.2 * observed_sd


def test_partition_draws_are_written_for_every_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(s11, "OUTPUT_DIR", tmp_path)
    s11.partition_scheme(synthetic_records(), np.random.default_rng(), replications=15,
                         preserve_composition=True)
    draws = pd.read_csv(tmp_path / "s11_partition_composition_preserving_draws.csv")
    assert len(draws) == 15
    assert draws["slope_difference"].notna().all()


def test_degenerate_exposure_is_counted_as_a_failure_not_a_slope():
    """Sites that share one elderly share carry no ecological slope; the draw fails."""
    df = synthetic_records(n_records=6_000, n_sites=6)
    df["site"] = np.arange(len(df)) % 6  # equal-sized sites
    df["site_residence"] = df["site"]
    df["elderly"] = (df.groupby("site").cumcount() % 5 == 0).astype(int)
    assert df.groupby("site")["elderly"].mean().nunique() == 1
    cells = s11.cells_by(df, "site")
    groups = np.array(sorted(cells["group"].unique()))
    differences, _, _, failures = s11.bootstrap_sites(cells, groups, np.random.default_rng(),
                                                      draws=25)
    assert np.isnan(differences).all()
    assert failures["no_exposure_variation"] + failures["fit"] == 25
    assert s11.failure_summary(differences, failures, 25)["unstable"] is True


def test_archive_existing_keeps_every_superseded_payload(tmp_path):
    target = tmp_path / "result.json"
    for value in range(5):
        paths.write_result({"timestamp": "2026-09-26 12:00:00", "value": value}, target)

    assert json.loads(target.read_text())["value"] == 4
    archived = sorted((tmp_path / "superseded").glob("result_*.json"))
    assert len(archived) == 4
    assert {json.loads(p.read_text())["value"] for p in archived} == {0, 1, 2, 3}


def test_write_table_archives_rather_than_overwrites(tmp_path):
    target = tmp_path / "table.csv"
    paths.write_table(pd.DataFrame({"a": [1]}), target)
    paths.write_table(pd.DataFrame({"a": [2]}), target)
    assert pd.read_csv(target)["a"].tolist() == [2]
    archived = list((tmp_path / "superseded").glob("table_*.csv"))
    assert len(archived) == 1
    assert pd.read_csv(archived[0])["a"].tolist() == [1]


def grouped_binomial(n_groups=120, cells_per_group=20, sigma=0.4):
    rng = np.random.default_rng()
    groups = np.repeat(np.arange(n_groups), cells_per_group)
    x = rng.normal(size=len(groups))
    u = rng.normal(0, sigma, n_groups)
    n = rng.integers(20, 200, len(groups)).astype(float)
    probability = 1 / (1 + np.exp(-(-2.0 + 0.75 * x + u[groups])))
    deaths = rng.binomial(n.astype(int), probability).astype(float)
    design = np.column_stack([np.ones(len(groups)), x])
    return design, deaths, n, groups


def test_glmm_recovers_the_fixed_effect_and_the_variance_component():
    errors, sigmas = [], []
    for _ in range(8):
        design, deaths, n, groups = grouped_binomial()
        fit = glmm.fit_random_intercept(design, deaths, n, groups)
        errors.append((fit["beta"][1] - 0.75) / fit["se"][1])
        sigmas.append(fit["sigma"])
    assert abs(np.mean(errors)) < 2.0
    assert np.mean(sigmas) == pytest.approx(0.4, rel=0.15)


def test_glmm_separates_a_null_variance_component_from_a_real_one():
    null = [glmm.fit_random_intercept(*grouped_binomial(sigma=0.0)) for _ in range(4)]
    real = [glmm.fit_random_intercept(*grouped_binomial(sigma=0.5)) for _ in range(4)]
    assert max(f["sigma"] for f in null) < 0.08
    assert min(f["sigma"] for f in real) > 0.35
    assert np.mean([f["sigma"] for f in real]) == pytest.approx(0.5, rel=0.15)
    assert max(f["boundary_lrt"] for f in null) < 20
    assert min(f["boundary_lrt"] for f in real) > 500
    assert not any(f["singular"] for f in real)


def test_glmm_approaches_the_plain_binomial_fit_when_the_variance_vanishes():
    design, deaths, n, groups = grouped_binomial(sigma=0.0)
    fit = glmm.fit_random_intercept(design, deaths, n, groups)
    plain = sm.GLM(np.column_stack([deaths, n - deaths]), design,
                   family=sm.families.Binomial()).fit()
    assert fit["beta"][1] == pytest.approx(plain.params[1], abs=0.01)
    assert fit["se"][1] == pytest.approx(plain.bse[1], rel=0.05)


def test_glmm_raises_rather_than_returning_a_number_when_the_optimizer_fails(monkeypatch):
    design, deaths, n, groups = grouped_binomial(n_groups=8, cells_per_group=5)

    def failed(*args, **kwargs):
        return optimize.OptimizeResult(x=np.zeros(design.shape[1] + 1), success=False,
                                       message="forced failure", fun=np.nan, nit=0)

    monkeypatch.setattr(glmm.optimize, "minimize", failed)
    with pytest.raises(glmm.ConvergenceError):
        glmm.fit_random_intercept(design, deaths, n, groups)


def contextual_frame(n_records=9_000, n_sites=8):
    """A frame with the columns `contextual_model` reads, and its spline knots."""
    rng = np.random.default_rng()
    site = rng.integers(0, n_sites, n_records)
    age = rng.integers(20, 90, n_records)
    elderly = (age >= s12.mexico_confirmed_cases.AGE_THRESHOLD).astype(int)
    frame = pd.DataFrame({"site": site, "age": age, "elderly": elderly,
                          "male": rng.integers(0, 2, n_records),
                          "age_band": (age // 10 * 10).astype(str),
                          "comorbidity_unknown": rng.random(n_records) < 0.02})
    for column in s12.COMORBIDITIES:
        frame[column] = (rng.random(n_records) < 0.1).astype(int)
    state_effect = rng.normal(0, 0.5, n_sites)[site]  # states differ, so sigma is not zero
    logit = -4.0 + 2.5 * elderly + 0.3 * frame["male"] + state_effect
    frame["died"] = (rng.random(n_records) < 1 / (1 + np.exp(-logit))).astype(int)
    knots = np.percentile(frame["age"], s12.SPLINE_PERCENTILES).round(1)
    return frame, knots


def test_h6_reports_no_odds_ratio_when_glmer_does_not_complete(monkeypatch):
    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda *args: {"completed": False, "stderr_tail": "forced"})
    result = s12.contextual_model(*contextual_frame())
    assert result["converged"] is False
    assert result["evaluable"] is False
    assert result["elderly_share_per_sd"] is None


def test_h6_is_not_evaluable_when_the_variance_is_at_the_boundary(monkeypatch):
    frame, knots = contextual_frame()
    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda cells, terms, index: r_payload_static(len(terms) + 1,
                                                                    singular=True))
    result = s12.contextual_model(frame, knots)
    assert result["gates"]["variance_component_off_the_boundary"] is False
    assert result["evaluable"] is False
    assert result["elderly_share_per_sd"]["or"] > 0  # reported, not used


def r_payload_static(n_terms, **overrides):
    estimate = [0.1] * n_terms
    payload = {"completed": True, "converged": True, "singular": False,
               "estimate": estimate, "se": [0.01] * n_terms,
               "random_intercept_sd": 0.3, "lme4_version": "1.1.31",
               "profile_ci": [0.05, 0.15]}
    payload.update(overrides)
    return payload


def test_h6_requires_the_two_implementations_to_agree(monkeypatch):
    """The gate the amendment adds: a fit nobody can reproduce is not evaluable."""
    frame, knots = contextual_frame()

    def disagreeing(cells, terms, index):
        return r_payload_static(len(terms) + 1)

    monkeypatch.setattr(s12, "fit_contextual_in_r", disagreeing)
    result = s12.contextual_model(frame, knots)
    assert result["reproduction"]["largest_absolute_coefficient_difference"] > 1e-4
    assert result["gates"]["reproduced_independently"] is False
    assert result["evaluable"] is False


def test_h6_reports_the_profile_interval_glmer_returned(monkeypatch):
    frame, knots = contextual_frame()
    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda cells, terms, index: r_payload_static(len(terms) + 1))
    result = s12.contextual_model(frame, knots)
    share = result["elderly_share_per_sd"]
    assert share["interval_used"] == "profile likelihood"
    assert share["profile_ci"] == [pytest.approx(np.exp(0.05)), pytest.approx(np.exp(0.15))]


def test_sentinel_codes_carry_no_municipality_or_sector():
    rng = np.random.default_rng()
    n = 400
    raw = pd.DataFrame({
        "ENTIDAD_UM": rng.integers(1, 33, n), "ENTIDAD_RES": rng.integers(1, 33, n),
        "MUNICIPIO_RES": rng.choice([1, 5, 570, 997, 998, 999], n),
        "SECTOR": rng.choice([4, 6, 9, 12, 99], n),
        "FECHA_DEF": "9999-99-99", "FECHA_SINTOMAS": "2021-01-15",
        "EDAD": rng.integers(0, 100, n), "SEXO": rng.integers(1, 3, n),
    })
    for column in mexico_confirmed_cases.COMORBIDITY_COLS:
        raw[column] = rng.choice([1, 2, 98], n)

    frame = mexico_confirmed_cases.records(raw)
    sentinel = raw["MUNICIPIO_RES"].isin([997, 998, 999])
    assert frame.loc[sentinel, "municipality"].isna().all()
    assert frame.loc[~sentinel, "municipality"].notna().all()
    assert frame.loc[raw["SECTOR"] == 99, "sector"].isna().all()
    assert frame.loc[raw["SECTOR"] != 99, "sector"].notna().all()

    valid = frame[frame["municipality"].notna()]
    assert (valid["municipality"] // 1000 == valid["site_residence"]).all()
    assert valid["municipality"].nunique() == len(
        raw.loc[~sentinel, ["ENTIDAD_RES", "MUNICIPIO_RES"]].drop_duplicates())


def test_fitting_the_record_model_once_is_the_same_as_refitting_per_partition():
    """The partition reassigns records, which leaves the pooled covariate cells alone."""
    df = synthetic_records()
    _, pooled = s11.fit_record_model(s11.cells_by(df, "site"))
    regrouped = df.assign(site=np.random.default_rng().permutation(df["site"].to_numpy()))
    _, refitted = s11.fit_record_model(s11.cells_by(regrouped, "site"))
    for key, value in pooled.items():
        assert refitted[key] == pytest.approx(value, abs=1e-12)


def test_paired_municipality_contrast_matches_a_brute_force_regrouping():
    df = synthetic_records(n_records=30_000, n_sites=8)
    df["municipality"] = df["site"] * 1000 + np.arange(len(df)) % 3
    point = s11.paired_municipality_state(df, np.random.default_rng(), draws=2, threshold=50)

    known = df[df["municipality"].notna()]
    sizes = known.groupby("municipality").size()
    matched = known[known["municipality"].isin(sizes[sizes >= 50].index)]
    predicted, _ = s11.fit_record_model(s11.cells_by(matched, "municipality"))
    municipality = s11.discrepancy(
        s11.site_table(s11.cells_by(matched, "municipality"), predicted))["slope_difference"]
    state = s11.discrepancy(
        s11.site_table(s11.cells_by(matched, "site_residence"), predicted))["slope_difference"]

    assert point["municipality"] == pytest.approx(municipality, abs=1e-9)
    assert point["state_of_residence"] == pytest.approx(state, abs=1e-9)
    assert point["paired_difference"] == pytest.approx(municipality - state, abs=1e-9)


def test_clustered_municipality_draws_keep_duplicated_states_apart():
    df = synthetic_records(n_records=12_000, n_sites=5)
    df["municipality"] = df["site"] * 1000 + np.arange(len(df)) % 2
    draws = s11.MINIMUM_VALID_DRAWS + 20
    result = s11.clustered_municipality_interval(df, np.random.default_rng(), draws=draws)
    assert result["failed_draws"] < 20
    assert result["valid_draws"] == draws - result["failed_draws"]
    assert np.isfinite(result["slope_difference_ci"]).all()
    assert result["failed_draws"] == sum(result["failure_reasons"].values())


def test_every_valid_municipality_reaches_the_no_threshold_analysis():
    df = synthetic_records(n_records=40_000, n_sites=6)
    df["municipality"] = df["site"] * 1000 + np.arange(len(df)) % 2
    df.loc[df.index[:100], "municipality"] = np.nan
    streams = s11.substreams(1, s11.ANALYSES)
    out = s11.analyse_municipalities(df, streams, draws=5)
    assert out["records_without_a_municipality_code"] == 100
    assert out["all_valid_municipalities"]["n_sites"] == df["municipality"].nunique()
    assert out["all_valid_municipalities"]["n_records"] == int(df["municipality"].notna().sum())


def test_var_weights_projection_reproduces_a_grouped_binomial_fit():
    """With an integer response the weighted projection is the grouped-binomial fit."""
    df = synthetic_records()
    cells = s11.cells_by(df, "site")
    predicted, _ = s11.fit_record_model(cells)
    table = s11.site_table(cells, predicted)
    result = s11.discrepancy(table)

    exog = sm.add_constant(table["prop_elderly"])
    grouped = sm.GLM(np.column_stack([table["deaths"], table["n"] - table["deaths"]]),
                     exog, family=sm.families.Binomial()).fit()
    weighted = sm.GLM(table["deaths"] / table["n"], exog, family=sm.families.Binomial(),
                      var_weights=table["n"]).fit()
    assert weighted.params.iloc[1] == pytest.approx(grouped.params.iloc[1], rel=1e-8)
    assert result["binomial_log_odds_slope"] == pytest.approx(grouped.params.iloc[1], rel=1e-8)


def test_s12_bootstrap_draw_equals_an_explicit_refit_on_duplicated_states():
    frame, _ = contextual_frame(n_records=6_000, n_sites=6)
    frame["has_comorbidity"] = frame[s12.COMORBIDITIES].max(axis=1)
    model = s12.BootstrapModel(frame, "elderly + male", ["elderly", "male"], False)
    weights = np.array([2, 0, 1, 1, 3, 1], dtype=np.float32)[:len(model.states)]
    value, reason = model.draw(weights)
    assert reason is None

    duplicated = pd.concat([frame[frame["site"] == state].assign(site=f"{state}_{copy}")
                            for index, state in enumerate(model.states)
                            for copy in range(int(weights[index]))], ignore_index=True)
    cells, _ = s12.fit_and_predict(s12.build_cells(duplicated, ["elderly", "male"]),
                                   "elderly + male")
    assert value == pytest.approx(s12.slopes(s12.state_table(cells))["slope_difference"],
                                  rel=1e-6)


def test_every_s12_model_sees_the_same_state_multiplicities(tmp_path, monkeypatch):
    monkeypatch.setattr(s12, "BOOTSTRAP_SHARDS", tmp_path)
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "1111111111aa")
    frame, knots = contextual_frame(n_records=6_000, n_sites=6)
    frame["has_comorbidity"] = frame[s12.COMORBIDITIES].max(axis=1)
    frame["onset_month"] = "2021-01"
    formulas = s12.model_specifications(knots)[:3]
    covariate_sets = [s12.covariates_for(f, o) for _, f, o in formulas]

    seen = []
    real = s12.BootstrapModel.draw

    def record(self, weights):
        seen.append(np.asarray(weights).copy())
        return real(self, weights)

    original = s12.BootstrapModel.draw
    s12.BootstrapModel.draw = record
    try:
        s12.bootstrap_ladder(frame, formulas, covariate_sets, seed=7, draws=4)
    finally:
        s12.BootstrapModel.draw = original

    per_model = [seen[i * 4:(i + 1) * 4] for i in range(len(formulas))]
    for draw in range(4):
        for other in per_model[1:]:
            assert np.array_equal(per_model[0][draw], other[draw])


def test_s12_counts_a_failed_draw_instead_of_aborting(tmp_path, monkeypatch):
    monkeypatch.setattr(s12, "BOOTSTRAP_SHARDS", tmp_path)
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "2222222222bb")
    frame, _ = contextual_frame(n_records=4_000, n_sites=5)
    frame["has_comorbidity"] = frame[s12.COMORBIDITIES].max(axis=1)
    formulas = [("model_1", "elderly", False)]
    covariate_sets = [["elderly"]]

    calls = {"n": 0}
    real = s12.sm.GLM

    class Exploding(real):
        def fit(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise np.linalg.LinAlgError("forced failure")
            return real.fit(self, *args, **kwargs)

    monkeypatch.setattr(s12.sm, "GLM", Exploding)
    values, failures = s12.bootstrap_ladder(frame, formulas, covariate_sets, seed=3, draws=6)

    failed = int(np.isnan(values["model_1"]).sum())
    assert 0 < failed < 6
    assert sum(failures["model_1"].values()) == failed
    assert "fit: LinAlgError" in failures["model_1"]


def test_the_quadrature_objective_does_not_depend_on_the_path_to_it():
    design, deaths, n, groups = grouped_binomial(n_groups=30, cells_per_group=10)
    model = glmm.RandomIntercept(design, deaths, n, groups)
    params = np.array([-2.0, 0.75, np.log(0.4)])

    cold = model.loglik(params, warm_start=False)
    model.loglik(params + np.array([1.5, -1.0, 1.0]))
    warm = model.loglik(params)
    model.loglik(params - np.array([2.0, 2.0, -2.0]))
    again = model.loglik(params)
    assert warm == pytest.approx(cold, rel=1e-12)
    assert again == pytest.approx(cold, rel=1e-12)


def test_the_fit_is_stable_as_the_quadrature_is_refined():
    design, deaths, n, groups = grouped_binomial(n_groups=30, cells_per_group=10)
    fits = {nodes: glmm.fit_random_intercept(design, deaths, n, groups, nodes=nodes)
            for nodes in (7, 15, 25)}
    reference = fits[25]
    for nodes in (7, 15):
        assert fits[nodes]["beta"][1] == pytest.approx(reference["beta"][1], abs=1e-4)
        assert fits[nodes]["sigma"] == pytest.approx(reference["sigma"], rel=1e-3)


def test_the_profile_interval_agrees_with_the_marginal_hessian_when_groups_are_many():
    design, deaths, n, groups = grouped_binomial(n_groups=120, cells_per_group=20)
    fit = glmm.fit_random_intercept(design, deaths, n, groups)
    profile = glmm.profile_interval(fit, 1)
    wald = glmm.wald(fit["beta"][1], fit["se"][1])
    assert np.log(wald["ci"][0]) == pytest.approx(profile[0], abs=0.02 * fit["se"][1] * 10)
    assert np.log(wald["ci"][1]) == pytest.approx(profile[1], abs=0.02 * fit["se"][1] * 10)
    assert fit["hessian"]["positive_definite"]
    assert fit["hessian"]["max_relative_difference_between_steps"] < 1e-3
    assert abs(glmm.gradient_check(fit)) / abs(fit["log_likelihood"]) < 1e-6


def test_a_dirty_analysis_tree_stops_a_reported_run(tmp_path, monkeypatch):
    run = lambda *args: subprocess.run(args, cwd=tmp_path, check=True,
                                       capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "test")
    (tmp_path / "paper" / "analyses").mkdir(parents=True)
    (tmp_path / "paper" / "analyses" / "s0.py").write_text("x = 1\n")
    run("git", "add", "paper/analyses/s0.py")
    run("git", "commit", "-q", "-m", "first", "--no-gpg-sign")

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    paths.require_clean_tree()

    (tmp_path / "paper" / "analyses" / "s0.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        paths.require_clean_tree()


def test_the_unknown_rate_covariate_is_state_constant_unrounded_and_clustered():
    frame, knots = contextual_frame(n_records=12_000, n_sites=6)
    frame["has_comorbidity"] = frame[s12.COMORBIDITIES].max(axis=1)
    frame["onset_month"] = "2021-01"
    formulas = s12.model_specifications(knots)
    entry = s12.unknown_comorbidity_sensitivity(frame, formulas)

    rates = frame.groupby("site")["comorbidity_unknown"].mean()
    assert set(entry["per_state_unknown_rate"].values()) == set(rates.astype(float))
    assert any(value not in (0.0, round(value, 3)) for value in rates)  # unrounded
    assert entry["unknown_rate_coefficient"]["clusters"] == frame["site"].nunique()
    assert "unknown_rate" not in formulas[4][1]  # absent from the registered primary model


def test_invalid_onset_dates_leave_the_month_missing_and_the_cohorts_matched():
    rng = np.random.default_rng()
    dates = ["2021-01-15", "9999-99-99", "2019-06-01", "2022-06-01"] * 40
    n = len(dates)
    raw = pd.DataFrame({
        "ENTIDAD_UM": rng.integers(1, 33, n), "ENTIDAD_RES": rng.integers(1, 33, n),
        "MUNICIPIO_RES": rng.integers(1, 100, n), "SECTOR": 4,
        "FECHA_DEF": "9999-99-99", "FECHA_SINTOMAS": dates,
        "EDAD": rng.integers(0, 100, n), "SEXO": rng.integers(1, 3, n),
    })
    for column in mexico_confirmed_cases.COMORBIDITY_COLS:
        raw[column] = 2

    frame = mexico_confirmed_cases.records(raw)
    inside = pd.Series(dates).isin(["2021-01-15"])
    assert frame["onset_month"].notna().tolist() == inside.tolist()
    assert not frame["onset_month"].astype("string").eq("NaT").any()
    assert len(frame.dropna(subset=["onset_month"])) == int(inside.sum())


def test_codes_outside_the_catalogue_carry_no_site():
    rng = np.random.default_rng()
    n = 200
    raw = pd.DataFrame({
        "ENTIDAD_UM": rng.integers(1, 33, n),
        "ENTIDAD_RES": rng.choice([1, 15, 32, 97, 99], n),
        "MUNICIPIO_RES": rng.choice([1, 570, 571, 996, 999], n),
        "SECTOR": 4, "FECHA_DEF": "9999-99-99", "FECHA_SINTOMAS": "2021-01-15",
        "EDAD": rng.integers(0, 100, n), "SEXO": rng.integers(1, 3, n),
    })
    for column in mexico_confirmed_cases.COMORBIDITY_COLS:
        raw[column] = 2

    frame = mexico_confirmed_cases.records(raw)
    assert frame.loc[raw["ENTIDAD_RES"].isin([97, 99]), "site_residence"].isna().all()
    assert frame.loc[raw["ENTIDAD_RES"].isin([1, 15, 32]), "site_residence"].notna().all()
    outside = raw["MUNICIPIO_RES"].isin([571, 996, 999]) | raw["ENTIDAD_RES"].isin([97, 99])
    assert frame.loc[outside, "municipality"].isna().all()
    assert frame.loc[~outside, "municipality"].notna().all()


def test_the_fit_agrees_across_its_prespecified_variance_starts():
    design, deaths, n, groups = grouped_binomial(n_groups=40, cells_per_group=20)
    fit = glmm.fit_random_intercept(design, deaths, n, groups, nodes=11)
    assert len(fit["variance_starts"]["converged"]) >= 2
    assert fit["variance_starts"]["largest_log_likelihood_spread"] < 1e-4
    assert fit["variance_starts"]["largest_sigma_spread"] < 1e-3


def test_a_profile_that_cannot_bracket_the_cutoff_reports_no_interval(monkeypatch):
    design, deaths, n, groups = grouped_binomial(n_groups=30, cells_per_group=10)
    fit = glmm.fit_random_intercept(design, deaths, n, groups, nodes=11)
    monkeypatch.setattr(glmm.optimize, "brentq",
                        lambda *args, **kwargs: float("nan"))
    assert glmm.profile_interval(fit, 1) is None


def test_too_few_valid_draws_suppress_the_interval():
    values = np.full(2000, np.nan)
    values[:10] = np.linspace(0.1, 0.2, 10)
    assert s11.interval_or_none(values) is None
    summary = s11.failure_summary(values, {"fit: LinAlgError": 1990}, 2000)
    assert summary["valid_draws"] == 10
    assert summary["unstable"] is True
    assert summary["interval_is_conditional_on_valid_draws"] is True


def test_the_preflight_refuses_a_stale_or_partial_validation(tmp_path):
    complete = {
        "estimator_fingerprint": validate_glmm.fingerprint(),
        "units_expected": validate_glmm.unit_names(),
        "units_completed": validate_glmm.unit_names(),
        "s9_against_lme4": {"agrees": True}, "reduced_design": {"agrees": True},
        "h6_shape": {"agrees": True},
        "quadrature_refinement": {"shift_to_the_finest_in_beta": 1e-7},
    }
    good = tmp_path / "good.json"
    good.write_text(json.dumps(complete))
    assert validate_glmm.preflight(good)["units"] == len(complete["units_completed"])

    for label, damage in [
        ("stale", {"estimator_fingerprint": "0" * 16}),
        ("partial", {"units_completed": complete["units_completed"][:-3]}),
        ("raised", {"units_that_raised": {"h6_shape": "boom"}}),
        ("disagrees", {"h6_shape": {"agrees": False}}),
        ("unstable quadrature", {"quadrature_refinement": {"shift_to_the_finest_in_beta": 0.5}}),
    ]:
        broken = tmp_path / f"{label}.json"
        broken.write_text(json.dumps({**complete, **damage}))
        with pytest.raises(validate_glmm.ValidationIncomplete):
            validate_glmm.preflight(broken)

    with pytest.raises(validate_glmm.ValidationIncomplete):
        validate_glmm.preflight(tmp_path / "absent.json")


def test_an_unstable_bootstrap_is_not_evaluable_rather_than_false():
    steady = {"slope_difference_ci": [0.2, 0.4], "valid_draws": 2000, "unstable": False}
    shaky = {"slope_difference_ci": [0.2, 0.4], "valid_draws": 2000, "unstable": True}
    starved = {"slope_difference_ci": None, "valid_draws": 40, "unstable": True}

    held = s12.disposition("c", steady, direction=True)
    assert (held["evaluable"], held["holds"]) == (True, True)
    missed = s12.disposition("c", steady, direction=False)
    assert (missed["evaluable"], missed["holds"]) == (True, False)
    for bootstrap in (shaky, starved):
        entry = s12.disposition("c", bootstrap, direction=True)
        assert entry["evaluable"] is False
        assert entry["holds"] is False

    s11_held = s11.disposition("c", steady, 0.3, lambda bounds: bounds[0] > 0)
    assert (s11_held["evaluable"], s11_held["holds"]) == (True, True)
    s11_shaky = s11.disposition("c", shaky, 0.3, lambda bounds: bounds[0] > 0)
    assert (s11_shaky["evaluable"], s11_shaky["holds"]) == (False, False)


def test_s12_reports_no_interval_below_the_minimum_valid_draws():
    values = np.full(2000, np.nan)
    values[:s12.MINIMUM_VALID_DRAWS - 1] = 0.3
    assert s12.interval_or_none(values) is None
    values[:s12.MINIMUM_VALID_DRAWS] = 0.3
    assert s12.interval_or_none(values) == [pytest.approx(0.3), pytest.approx(0.3)]


def test_s12_counts_its_own_convergence_failure_as_a_failed_draw():
    assert RuntimeError in s12.FIT_FAILURES  # fit_and_predict raises it on nonconvergence


def test_likelihood_differences_agree_between_implementations():
    """An additive offset cancels in a difference, so differences must match exactly."""
    design, deaths, n, groups = grouped_binomial(n_groups=20, cells_per_group=15)
    model = glmm.RandomIntercept(design, deaths, n, groups, nodes=11)
    first = np.array([-2.0, 0.75, np.log(0.4)])
    second = np.array([-1.9, 0.60, np.log(0.5)])
    mine = model.loglik(first, warm_start=False) - model.loglik(second, warm_start=False)
    constant = validate_glmm.binomial_constant(deaths, n)
    shifted = ((model.loglik(first, warm_start=False) + constant)
               - (model.loglik(second, warm_start=False) + constant))
    assert shifted == pytest.approx(mine, rel=1e-12)


def names_a_function_uses_but_never_binds(path):
    """Every name a module's top-level functions load without binding, importing or sharing."""
    tree = ast.parse(Path(path).read_text())
    imported = {alias.asname or alias.name.split(".")[0]
                for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    module_level = {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
    module_level |= {node.target.id for node in tree.body
                     if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    defined = {node.name for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

    findings = {}
    for function in [node for node in tree.body if isinstance(node, ast.FunctionDef)]:
        bound = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                arguments = node.args
                bound |= {argument.arg for argument in [*arguments.posonlyargs,
                                                        *arguments.args,
                                                        *arguments.kwonlyargs]}
                bound |= {argument.arg
                          for argument in (arguments.vararg, arguments.kwarg) if argument}
                if not isinstance(node, ast.Lambda):
                    bound.add(node.name)
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                bound.add(node.optional_vars.id)
            elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound |= set(node.names)
        used = {node.id for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        unknown = used - bound - imported - module_level - defined - set(dir(builtins))
        if unknown:
            findings[function.name] = sorted(unknown)
    return findings


def test_no_analysis_function_uses_a_name_it_does_not_have():
    """Twice now, moving code out of a long function left it referring to a lost local.

    Both failures surfaced only when a run reached the line, four hours in: `load` lost
    `onset` when `records` was extracted, and `main` lost `share` when the H6 summary
    became a function. The analyses read a 1.9 GB snapshot, so no test calls them end to
    end; this reads them instead.
    """
    offenders = {path.name: found
                 for path in sorted(Path(s12.__file__).parent.glob("*.py"))
                 if (found := names_a_function_uses_but_never_binds(path))}
    assert not offenders, offenders


def test_the_undefined_name_check_sees_a_name_that_is_not_there(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("CONSTANT = 2\n\n"
                      "def good(x):\n"
                      "    [y for y in range(x)]\n"
                      "    with open('f') as handle:\n"
                      "        return handle, CONSTANT, good\n\n"
                      "def bad(x):\n"
                      "    return x + missing\n")
    assert names_a_function_uses_but_never_binds(module) == {"bad": ["missing"]}


def test_a_finished_bootstrap_model_is_not_redrawn(tmp_path, monkeypatch):
    """Six models at 2,000 draws is hours; a failure in the last must not cost the rest."""
    monkeypatch.setattr(s12, "BOOTSTRAP_SHARDS", tmp_path)
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "aaaaaaaaaaaa")
    values = np.array([0.3, np.nan, 0.5])
    reasons = {"fit: LinAlgError": 1}
    assert s12.load_bootstrap_shard("model_5", 7, 3) is None

    s12.save_bootstrap_shard("model_5", 7, 3, values, reasons)
    back = s12.load_bootstrap_shard("model_5", 7, 3)
    assert np.isnan(back["values"][1])                      # a failed draw stays failed
    assert back["values"][[0, 2]].tolist() == [0.3, 0.5]
    assert back["reasons"] == reasons
    assert s12.load_bootstrap_shard("model_5", 8, 3) is None   # a different seed is a different run
    assert s12.load_bootstrap_shard("model_5", 7, 2000) is None

    # Draws produced by code that has since changed are not draws of the current model.
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "bbbbbbbbbbbb")
    assert s12.load_bootstrap_shard("model_5", 7, 3) is None


def test_the_h6_summary_renders_from_what_contextual_model_returns(monkeypatch):
    """A print written against a shape its producer stopped returning cost four hours."""
    frame, knots = contextual_frame(n_records=3_000, n_sites=5)
    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda cells, terms, index: r_payload_static(len(terms) + 1))
    line = s12.describe_contextual(s12.contextual_model(frame, knots))
    assert "elderly share per SD" in line and "evaluable" in line

    failed = {"elderly_share_per_sd": None, "convergence_message": "glmer did not finish"}
    assert "glmer did not finish" in s12.describe_contextual(failed)
    assert "no reason recorded" in s12.describe_contextual({"elderly_share_per_sd": None})


def test_a_finished_stage_is_read_back_rather_than_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(s12, "STAGE_SHARDS", tmp_path)
    monkeypatch.setenv("ANALYSIS_COMMIT", "abc123def456")
    calls = []

    def compute():
        calls.append(1)
        return {"slope": np.float64(0.42), "draws": np.array([1.0, 2.0])}

    first = s12.cached_stage("ladder_primary", compute)
    second = s12.cached_stage("ladder_primary", compute)
    assert len(calls) == 1                       # the second call did not recompute
    assert second["slope"] == 0.42               # numpy scalars survive the round trip
    assert second["draws"] == [1.0, 2.0]
    assert first["slope"] == second["slope"]

    monkeypatch.setenv("ANALYSIS_COMMIT", "999999999999")
    s12.cached_stage("ladder_primary", compute)
    assert len(calls) == 2                       # different code, so not reused


def snapshot_shaped_records(n_records=20_000, n_sites=5, months=6):
    """Every column `main` reads, so the whole analysis can run on a frame in memory."""
    frame, _ = contextual_frame(n_records, n_sites)
    rng = np.random.default_rng()
    frame["has_comorbidity"] = frame[s12.COMORBIDITIES].max(axis=1)
    frame["comorbidity_unknown"] = frame["comorbidity_unknown"].astype(int)
    onset = (pd.Timestamp(s12.SNAPSHOT_DATE)
             - pd.to_timedelta(rng.integers(1, months * 28, n_records), unit="D"))
    frame["onset"] = pd.Series(onset).where(rng.random(n_records) > 0.03)
    frame["onset_month"] = (frame["onset"].dt.to_period("M").astype("string")
                            .where(frame["onset"].notna()))
    return frame


def counting(function, name, log):
    def wrapper(*arguments, **keywords):
        log.append(name)
        return function(*arguments, **keywords)
    return wrapper


def test_a_checkpoint_needs_a_commit_to_belong_to(monkeypatch):
    """Without one, every run shares a namespace and reads back the last run's answer."""
    monkeypatch.delenv(paths.SUPPLIED_COMMIT, raising=False)
    monkeypatch.setattr(s12, "_git", lambda *arguments: subprocess.CompletedProcess(
        arguments, returncode=128, stdout="", stderr="not a git repository"))
    with pytest.raises(RuntimeError, match="no commit is available"):
        s12._run_commit()


def test_a_checkpoint_whose_fingerprint_disagrees_is_not_reused(tmp_path, monkeypatch):
    """The filename carries a commit prefix, which is a name; the record is the identity."""
    monkeypatch.setattr(s12, "STAGE_SHARDS", tmp_path)
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "dddddddddddd")

    for field, replacement in (("data_sha256", "0" * 64),
                               ("cache_schema", s12.CACHE_SCHEMA + 1),
                               ("commit", "eeeeeeeeeeee")):
        assert s12.cached_stage(field, lambda: {"slope": 1.0})["slope"] == 1.0
        path, = tmp_path.glob(f"{field}_*.json")
        record = json.loads(path.read_text())
        assert record["fingerprint"]["data_sha256"] == paths.EXPECTED_SNAPSHOT_SHA256
        record["fingerprint"][field] = replacement
        path.write_text(json.dumps(record))
        assert s12.cached_stage(field, lambda: {"slope": 2.0})["slope"] == 2.0


def test_a_glmer_that_never_finished_is_not_cached_as_the_h6_answer(tmp_path, monkeypatch):
    """An R timeout must not stand as the registered verdict on every later resume.

    The distinction the cache has to make: a fit that completed and then failed a gate is
    an outcome H6 registers, and a fit the environment never produced is not.
    """
    monkeypatch.setattr(s12, "STAGE_SHARDS", tmp_path)
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "cccccccccccc")
    frame, knots = contextual_frame(n_records=2_000, n_sites=4)
    cacheable = {"cacheable": lambda value: value["glmer_completed"]}

    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda *arguments: {"completed": False, "stderr_tail": "timed out"})
    failed = s12.cached_stage("h6_contextual",
                              lambda: s12.contextual_model(frame, knots), **cacheable)
    assert failed["glmer_completed"] is False
    assert not list(tmp_path.glob("h6_contextual_*.json"))

    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda cells, terms, index: r_payload_static(len(terms) + 1))
    fitted = s12.cached_stage("h6_contextual",
                              lambda: s12.contextual_model(frame, knots), **cacheable)
    assert fitted["glmer_completed"] is True
    assert fitted["evaluable"] is False          # a gate failed, and that is a result
    assert len(list(tmp_path.glob("h6_contextual_*.json"))) == 1


def test_the_whole_ladder_runs_and_assembles_its_output(tmp_path, monkeypatch):
    """Eighteen fits reach an output dict nothing had ever executed.

    Two runs died assembling it, hours in: once on a key its producer had stopped
    returning, once on a local lost when the H6 summary became a function. Neither is
    reachable from a unit test of the pieces. The snapshot is 1.9 GB and glmer is not
    installed here, so the loader and the R fit are stubbed and everything between is
    the registered code, at a size that runs in under a minute.
    """
    monkeypatch.setenv(paths.SUPPLIED_COMMIT, "0123456789ab")
    monkeypatch.setattr(s12, "COMORBIDITIES", s12.COMORBIDITIES[:3])
    monkeypatch.setattr(s12, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(s12, "STAGE_SHARDS", tmp_path / "stages")
    monkeypatch.setattr(s12, "BOOTSTRAP_SHARDS", tmp_path / "bootstrap")
    monkeypatch.setattr(s12, "BOOTSTRAP_DRAWS", 6)
    monkeypatch.setattr(s12, "MINIMUM_VALID_DRAWS", 2)
    monkeypatch.setattr(validate_glmm, "preflight",
                        lambda: {"units": 56, "validated_fingerprint": "stub"})
    monkeypatch.setattr(mexico_confirmed_cases, "load", snapshot_shaped_records)
    monkeypatch.setattr(s12, "fit_contextual_in_r",
                        lambda cells, terms, index: r_payload_static(len(terms) + 1))
    calls = []
    for name in ("run_ladder", "contextual_model", "unknown_comorbidity_sensitivity",
                 "save_bootstrap_shard"):
        monkeypatch.setattr(s12, name, counting(getattr(s12, name), name, calls))

    s12.main()
    computed = [calls.count(name) for name in ("run_ladder", "contextual_model",
                                               "unknown_comorbidity_sensitivity",
                                               "save_bootstrap_shard")]
    assert computed == [3, 1, 1, 6]

    written = json.loads((tmp_path / "s12_composition_ladder.json").read_text())
    assert set(written["primary_ladder"]) == {name for name, _, _
                                              in s12.model_specifications(range(5))}
    for hypothesis in ("h4_composition_contrast", "h5_temporal_contrast",
                       "h6_contextual_association"):
        assert {"criterion", "evaluable", "holds"} <= set(written[hypothesis])
    assert written["h6_contextual_association"]["per_sd_or"] > 0
    assert written["primary_ladder"]["model_5_age_spline"]["bootstrap"]["valid_draws"] >= 2
    assert (tmp_path / "s12_bootstrap_draws.csv").exists()

    # The resume path, which is the one a crash makes you take. The stubbed loader draws a
    # new frame on every call, so a stage that was recomputed cannot return the same slope.
    s12.main()
    assert [calls.count(name) for name in ("run_ladder", "contextual_model",
                                           "unknown_comorbidity_sensitivity",
                                           "save_bootstrap_shard")] == computed
    resumed = json.loads((tmp_path / "s12_composition_ladder.json").read_text())
    for model, entry in written["primary_ladder"].items():
        assert resumed["primary_ladder"][model]["slope_difference"] == entry["slope_difference"]
    assert (resumed["sensitivity_unknown_comorbidity_rate"]["slope_difference"]
            == written["sensitivity_unknown_comorbidity_rate"]["slope_difference"])
    assert len(list((tmp_path / "superseded").glob("s12_composition_ladder_*.json"))) == 1


def test_the_comparison_tells_a_flat_surface_from_an_optimizer_that_stopped(tmp_path):
    """One number cannot distinguish the two, and only one of them is a defect.

    `elderly_share_z` has to be among the names, because the comparison reports the
    difference on the coefficient H6 is about by name.
    """
    design, deaths, n, groups = grouped_binomial(n_groups=30, cells_per_group=10)
    fit = glmm.fit_random_intercept(design, deaths, n, groups, errors=False)
    names = ["intercept", "elderly_share_z"]

    settled = {"completed": True, "estimate": list(fit["beta"]),
               "random_intercept_sd": fit["sigma"]}
    agreed = s12.compare_implementations(settled, fit, names)
    assert agreed["largest_absolute_coefficient_difference"] == pytest.approx(0, abs=1e-12)
    assert agreed["our_optimizer_stopped_short"] is False
    assert agreed["our_estimate_scores_higher_by"] == pytest.approx(0, abs=1e-6)

    # The same comparison, against a reproduction that stopped before the optimum.
    nudged = np.asarray(fit["beta"], dtype=float) + np.array([0.0, 0.05])
    parameters = np.concatenate([nudged, [np.log(fit["sigma"])]])
    short = {**fit, "beta": nudged, "params": parameters,
             "log_likelihood": float(fit["model"].loglik(parameters, warm_start=False))}
    caught = s12.compare_implementations(settled, short, names)
    assert caught["our_optimizer_stopped_short"] is True
    assert caught["our_estimate_scores_higher_by"] < 0
    assert caught["largest_difference_term"] == "elderly_share_z"
    # the difference is lme4 minus ours, and it is ours that was nudged upward
    assert caught["registered_coefficient_lme4_minus_glmm_py"] == pytest.approx(-0.05,
                                                                                abs=1e-9)


def test_a_float_of_difference_is_not_an_optimizer_that_stopped_short(tmp_path):
    """The real H6 fit trips a bare inequality: lme4 scores 4e-06 higher on -751691.

    One implementation always stops a float ahead of the other, so a comparison with no
    tolerance reports a shortfall on every run and means nothing by meaning always.
    """
    design, deaths, n, groups = grouped_binomial(n_groups=30, cells_per_group=10)
    fit = glmm.fit_random_intercept(design, deaths, n, groups, errors=False)
    names = ["intercept", "elderly_share_z"]

    settled = {"completed": True, "estimate": list(fit["beta"]),
               "random_intercept_sd": fit["sigma"]}
    nudged = np.asarray(fit["beta"], dtype=float) + np.array([0.0, 1e-7])
    parameters = np.concatenate([nudged, [np.log(fit["sigma"])]])
    hair = {**fit, "beta": nudged, "params": parameters,
            "log_likelihood": float(fit["model"].loglik(parameters, warm_start=False))}
    close = s12.compare_implementations(settled, hair, names)
    # Which of the two scores higher is not stable at this scale: the gap here runs to
    # 1e-11 and has been seen with either sign, which is the whole reason a bare
    # inequality decides nothing. Only the magnitude is meaningful.
    assert abs(close["our_estimate_scores_higher_by"]) < s12.OPTIMIZER_SHORTFALL
    assert close["our_optimizer_stopped_short"] is False


def test_the_comparison_reports_nothing_when_a_fit_is_missing():
    assert s12.compare_implementations({"completed": False}, None, [])["comparable"] is False
    assert s12.compare_implementations(
        {"completed": True}, None, [])["largest_absolute_coefficient_difference"] is None


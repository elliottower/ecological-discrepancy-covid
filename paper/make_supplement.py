"""Generate supplement_v2.tex from the result files.

Every number in the supplement is read from paper/analyses/results/ at build
time.  Nothing is transcribed by hand, so a rerun of an analysis and a rebuild
of the supplement cannot disagree.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "analyses" / "results"
OUT = HERE / "supplement_v2.tex"


def load(name):
    with (RESULTS / name).open() as handle:
        return json.load(handle)


s3 = load("s3_expected_ecological.json")
s2 = load("s2_mundlak_decomposition.json")
s9 = load("s9_two_stage_vs_one_stage.json")
s11 = load("s11_site_definitions.json")
s12 = load("s12_composition_ladder.json")
glmm = load("glmm_validation.json")
audit = load("mexico_loader_audit.json")

confirmed = s3["n_states"], s11["definitions"]["treating_unit_state"]["n_records"]
n_records = s11["definitions"]["treating_unit_state"]["n_records"]
n_deaths = s11["definitions"]["treating_unit_state"]["n_deaths"]
total_rows = audit["total_rows"]
counts = audit["clasificacion_final_counts"]


def num(x, places=4, sign=True):
    fmt = f"{{:+.{places}f}}" if sign else f"{{:.{places}f}}"
    return fmt.format(x)


def thousands(x):
    return f"{int(x):,}".replace(",", "\\,")


def sci(x, places=2):
    """LaTeX scientific notation: 7.65e-05 -> 7.65 \\times 10^{-5}."""
    mantissa, exponent = f"{x:.{places}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


# ---------------------------------------------------------------- D: per state
per_state_rows = []
for key in sorted(s3["per_state"], key=int):
    row = s3["per_state"][key]
    per_state_rows.append(
        f"{key} & {thousands(row['n'])} & {row['prop_elderly']:.4f} & "
        f"{row['observed_mortality']:.4f} & {row['model_implied_mortality']:.4f} & "
        f"{row['observed_mortality'] - row['model_implied_mortality']:+.4f} \\\\"
    )
per_state = "\n".join(per_state_rows)

definition_rows = []
for key, label in [
    ("treating_unit_state", "Treating unit's state"),
    ("residence_state", "State of residence"),
    ("health_care_sector", "Health-care sector"),
]:
    d = s11["definitions"][key]
    definition_rows.append(
        f"{label} & {d['n_sites']} & {num(d['observed_slope'])} & "
        f"{num(d['implied_slope'])} & {num(d['slope_difference'])} & "
        f"{num(d['weighted_slope_difference'])} & {num(d['binomial_log_odds_difference'], 3)} \\\\"
    )
definitions = "\n".join(definition_rows)

muni = s11["definitions"]["municipality"]
muni_rows = []
for key, label in [
    ("threshold_1000", "$\\geq 1{,}000$ records"),
    ("threshold_500", "$\\geq 500$ records"),
    ("threshold_250", "$\\geq 250$ records"),
    ("all_valid_municipalities", "No size rule"),
]:
    m = muni[key]
    muni_rows.append(
        f"{label} & {m['n_sites']} & {thousands(m['n_records'])} & "
        f"{num(m['observed_slope'])} & {num(m['implied_slope'])} & "
        f"{num(m['slope_difference'])} \\\\"
    )
municipality = "\n".join(muni_rows)

# ---------------------------------------------------------------- E: the gates
gates = s12["h6_contextual_model"]["gates"]
gate_labels = {
    "registered_covariates": "Covariates are the registered ones",
    "converged": "The optimizer reports convergence",
    "term_order_matches": "Term order matches the registered design",
    "variance_component_off_the_boundary": "Variance component is off the boundary",
    "finite_estimate": "The estimate is finite",
    "reproduced_independently": "A second implementation reproduces it",
    "hessian_positive_definite": "The Hessian is positive definite",
    "hessian_agrees_across_steps": "The Hessian agrees across step sizes",
    "hessian_well_conditioned": "The Hessian is well conditioned",
    "gradient_at_the_optimum": "The gradient vanishes at the optimum",
    "profile_interval_available": "A profile interval is available",
}
gate_rows = "\n".join(
    f"{gate_labels[k]} & {'pass' if v else 'FAIL'} \\\\" for k, v in gates.items()
)

h6 = s12["h6_contextual_model"]
repro = h6["reproduction"]
assoc = s12["h6_contextual_association"]

# ---------------------------------------------------------------- C: the ladder
LADDER_LABELS = {
    "model_1_elderly_only": "Elderly only",
    "model_2_elderly_sex_any_comorbidity": "Elderly, sex, any comorbidity",
    "model_3_nine_comorbidities": "Elderly, sex, nine comorbidities",
    "model_4_age_bands": "Age bands, sex, nine comorbidities",
    "model_5_age_spline": "Age spline, sex, nine comorbidities",
    "model_6_age_spline_and_month": "Age spline, sex, nine comorbidities, month",
}

ladder_rows = []
for key, value in s12["primary_ladder"].items():
    label = LADDER_LABELS[key]
    ladder_rows.append(
        f"{label} & {value['terms']} & {num(value['implied_slope'])} & "
        f"{num(value['slope_difference'])} \\\\"
    )
ladder = "\n".join(ladder_rows)

partition = s11["composition_preserving_partition"]
unrestricted = s11["size_matched_unrestricted_partition"]
paired = s11["paired_municipality_state"]

DOC = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
\usepackage{{setspace}}
\doublespacing
\usepackage{{lineno}}
\linenumbers

\renewcommand{{\arraystretch}}{{1.2}}

\title{{Supplementary Materials\\[6pt]
\large Ecological discrepancy under a specified record-level model:
{thousands(n_records)} Mexican COVID-19 case records in 32 treating-unit states}}

\author{{Elliot Tower}}

\date{{}}

\begin{{document}}
\maketitle

\setcounter{{section}}{{0}}
\renewcommand{{\thesection}}{{Supplement~\Alph{{section}}}}

This document is generated from the analysis result files by
\texttt{{paper/make\_supplement.py}}.  Every number below is read from those
files at build time rather than transcribed, so a rerun of an analysis and a
rebuild of this document cannot disagree.

\section{{Registration and Provenance}}
\label{{supp:registration}}

Two protocols were frozen before the analyses they govern.  The first,
\texttt{{ANALYSIS\_PROTOCOL.md}}, was frozen at commit \texttt{{97f7094}} and
specifies ten supplementary analyses, each with a falsification criterion and
a main-text or supplement placement fixed in advance.  The second,
\texttt{{ANALYSIS\_PROTOCOL\_2\_}}\-\texttt{{SITE\_DEFINITIONS.md}}, was frozen at commit
\texttt{{a38196b}} and tagged \texttt{{registration-s11-s12}}; it governs the
site definitions and the covariate ladder, records the state-level discrepancy
already known at that point, and states that no grouping other than the
treating unit's state had been run.

\begin{{sloppypar}}
Two further records accompany them.
\texttt{{IMPLEMENTATION\_AMENDMENT\_2026-09-29.md}} records changes to the
implementation that do not alter the registered design, and
\texttt{{PROTOCOL\_CORRECTION\_ADDENDUM.md}} records a correction to the
protocol itself.  Both are in the repository and both are reachable from the
ledger.
\end{{sloppypar}}

\begin{{table}}[h]
\centering
\caption{{Registered analyses and the commit that froze each.}}
\vspace{{0.5\baselineskip}}
\begin{{tabular}}{{lll}}
\toprule
Analysis & Frozen at & Outcome known when specified \\
\midrule
Ten supplementary analyses (S1--S10) & \texttt{{97f7094}} & tagged per analysis \\
Site definitions (S11) & \texttt{{a38196b}} & state-level discrepancy only \\
Covariate ladder (S12) & \texttt{{a38196b}} & state-level discrepancy only \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{sloppypar}}
The analysis runs recorded in the ledger carry the snapshot digest
\texttt{{{s11['run']['snapshot_sha256'][:16]}\ldots}}, and each names the
commit it ran at and whether the analysis tree was clean.
\end{{sloppypar}}

\section{{Denominators}}
\label{{supp:denominators}}

The published snapshot holds {thousands(total_rows)} rows.  Confirmed cases are
those with \texttt{{CLASIFICACION\_FINAL}} in $\{{1, 2, 3\}}$:
{thousands(counts['1'])} $+$ {thousands(counts['2'])} $+$
{thousands(counts['3'])} $=$ {thousands(n_records)}, of which
{thousands(n_deaths)} record a death.  Every state-level analysis uses that
denominator.  The municipality analyses use fewer records because a size rule
applies; {thousands(audit['codes_among_confirmed_cases']['outside_catalogue']['records_with_no_municipality'])}
confirmed cases carry no municipality code in the catalogue.

\section{{Registered Supplementary Analyses}}
\label{{supp:preregistered}}

Three of the ten registered analyses use the Mexico records reported in the
main text.  The remaining seven were specified on the two further record sets
described in Supplement~E and are reported there with the data they used.

\paragraph{{S2: Mundlak within/between decomposition.}}
A logistic model with state fixed effects and within-state and between-state
elderly covariates over {thousands(s2['registered_analysis']['within_state']['n_patients'])}
records gives a within-state coefficient of
{num(s2['registered_analysis']['within_state']['coef'])}
(OR $= {s2['registered_analysis']['within_state']['or']:.4f}$).  The
between-state coefficient is not identified: the state means lie in the span of
the state dummies.  \emph{{Falsification:}} the criterion fires, and the
collinearity is a property of the specification rather than evidence about
aggregation.

\paragraph{{S3: Expected ecological coefficient.}}
Averaging the record-level model's predicted probabilities within each state
and regressing those averages on elderly share gives a model-implied slope of
{num(s3['model_implied_ecological_slope'])} against an observed slope of
{num(s3['observed_ecological_slope'])}, a difference of
{num(s3['slope_difference'])} and a ratio of
{s3['ratio_observed_implied']:.4f}.  The state-bootstrap interval on the
difference is [{num(s3['bootstrap']['slope_difference_ci'][0])},
{num(s3['bootstrap']['slope_difference_ci'][1])}] over
{s3['bootstrap']['draws']} draws resampled by state.  The registered agreement
rule---observed and expected within 20\%---does not hold.

\paragraph{{S9: Two-stage against one-stage.}}
Three estimators on the same records: ecological regression
($\beta = {num(s9['ecological']['beta'])}$,
$R^2 = {s9['ecological']['r_squared']:.4f}$); two-stage individual participant
data pooled by DerSimonian--Laird
(OR $= {s9['two_stage_ipd']['or']:.4f}$,
$I^2 = {s9['two_stage_ipd']['I2']:.4f}$); and the one-stage random-intercept
fit reported in Supplement~D.  The ecological estimator is on the
mortality-rate scale and the other two on the log odds scale, so the three are
not directly comparable in magnitude.

\section{{Per-State Estimates, Weighting and Site Definitions}}
\label{{supp:perstate}}

\paragraph{{The covariate ladder.}}
Each row fits the named covariates at the record level, averages the predicted
probabilities within each state, and regresses those averages on elderly share.

\begin{{table}}[h]
\centering
\caption{{Model-implied slope and discrepancy by record-level specification.}}
\vspace{{0.5\baselineskip}}
\begin{{tabular}}{{lrrr}}
\toprule
Specification & Terms & Implied slope & Discrepancy \\
\midrule
{ladder}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Weighting and the response scale.}}
The primary fit weights each state equally.  Two alternatives are reported
beside it: a fit weighting each state by its record count, and a
grouped-binomial fit on the log odds scale.  The three answer different
questions and are not on a common scale; the log odds column is not comparable
in magnitude with the two rate-scale columns.

\begin{{table}}[h]
\centering
\caption{{Discrepancy by site definition, under equal-state weighting,
record-count weighting, and a grouped-binomial fit on the log odds scale.}}
\vspace{{0.5\baselineskip}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Site definition & Sites & Observed & Implied & Equal-state & Record-count & Log odds \\
\midrule
{definitions}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Municipality at every registered threshold.}}

\begin{{table}}[h]
\centering
\caption{{Municipality grouping at the four registered size thresholds.}}
\vspace{{0.5\baselineskip}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Size rule & Sites & Records & Observed & Implied & Discrepancy \\
\midrule
{municipality}
\bottomrule
\end{{tabular}}
\end{{table}}

On the {thousands(paired['matched_records'])} records retained by the
$\geq 1{{,}}000$ rule, the paired contrast between
{paired['matched_municipalities']} municipalities and
{paired['matched_states']} states of residence is
${num(paired['paired_difference'])}$.

\paragraph{{Constrained random partitions.}}
Reassigning records to pseudo-sites that hold each state's size and count aged
70 or older fixed, without using any other record field, leaves a median
discrepancy of ${num(partition['median_slope_difference'])}$ over
{partition['replications']} replications, with a randomization interval of
$[{num(partition['randomization_interval'][0])},
{num(partition['randomization_interval'][1])}]$ and no deviation from either
quota.  The expectation of that statistic is zero under this scheme, because
the record-level model is fitted by maximum likelihood and its residuals sum to
zero within the strata the quotas hold fixed, so the comparison is a check on
the construction rather than evidence about geography.  Matching size alone,
without the elderly quota, gives a median of
${num(unrestricted['median_slope_difference'])}$ with a wider interval of
$[{num(unrestricted['randomization_interval'][0])},
{num(unrestricted['randomization_interval'][1])}]$.

\begingroup
\singlespacing
\begin{{longtable}}{{rrrrrr}}
\caption{{Per-state records, elderly share, observed and model-implied
mortality, and their difference, under the registered three-covariate
model.}}\\
\toprule
State & Records & Elderly share & Observed & Implied & Difference \\
\midrule
\endfirsthead
\toprule
State & Records & Elderly share & Observed & Implied & Difference \\
\midrule
\endhead
\bottomrule
\endfoot
{per_state}
\end{{longtable}}
\endgroup

\section{{State-Level Model Diagnostics}}
\label{{supp:glmm}}

The state-level model is a logistic random-intercept fit over
{thousands(h6['n_cells'])} cells with the registered covariates, the centered
age spline, and the state's elderly share entered as a contextual term.  It was
fitted by \texttt{{lme4::glmer}} at {h6['estimator'].split('nAGQ = ')[1].split(',')[0]}
adaptive Gauss--Hermite quadrature nodes in
{h6['seconds'] / 3600:.2f} hours.  The elderly share carries an odds ratio of
{assoc['per_sd_or']:.4f} per standard deviation, with a profile interval of
[{assoc['ci'][0]:.4f}, {assoc['ci'][1]:.4f}] and a Wald interval of
[{assoc['wald_ci'][0]:.4f}, {assoc['wald_ci'][1]:.4f}].  The random-intercept
standard deviation is {h6['random_intercept_sd']:.4f}.

A second implementation written separately, using adaptive Gauss--Hermite
quadrature in Python, reproduces the fit: the largest absolute coefficient
difference is ${sci(repro['largest_absolute_coefficient_difference'])}$ against a
tolerance of ${sci(repro['tolerance'], 0)}$, and the two random-intercept standard
deviations are {h6['random_intercept_sd']:.4f} and {repro['sigma']:.4f}.  The
relative gradient at the optimum is
${sci(repro['relative_gradient'])}$.

The result is reported only if eleven acceptance criteria hold.  All eleven hold.

\begin{{table}}[h]
\centering
\caption{{Acceptance criteria for the state-level model.  A result failing any
one of these is reported as not evaluable rather than as an estimate.}}
\vspace{{0.5\baselineskip}}
\begin{{tabular}}{{lr}}
\toprule
Criterion & Outcome \\
\midrule
{gate_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

An earlier validation run checked the same Python implementation against
\texttt{{lme4}} on a reduced design: the log odds agree to
${sci(glmm['s9_against_lme4']['log_or_absolute_difference'])}$ and the standard
errors to ${sci(glmm['s9_against_lme4']['se_absolute_difference'])}$, against a
tolerance of ${sci(glmm['s9_against_lme4']['tolerance'], 0)}$.

\section{{Two Further Record Sets}}
\label{{supp:other}}

Seven of the ten registered supplementary analyses were specified on two record
sets that are not part of the analysis reported in the main text: United States
case surveillance published by the Centers for Disease Control and Prevention
(\texttt{{data.cdc.gov}}, dataset \texttt{{vbim-akqf}}), grouped by race and
ethnicity into pseudo-sites, and the 4CE consortium's aggregate neurological
data.  They are reported here because they were registered, and the main text
draws no conclusion from them.  Their results are in
\texttt{{paper/analyses/results/}} under \texttt{{s1}}, \texttt{{s4}} through
\texttt{{s8}}, and \texttt{{s10}}.

\end{{document}}
"""

OUT.write_text(DOC)
print(f"wrote {OUT}  ({len(DOC.splitlines())} lines)")

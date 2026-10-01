"""Kaplan-Meier (Greenwood) and Cox PH (Efron / Breslow): golden values from R ``survival``.

VALIDATION METHOD. Expected numbers were produced by R 4.5.1 with survival 3.8.3
(``arch -arm64 Rscript``) on the SAME fixed data embedded here (``survival::aml``
for KM / log-rank / Cox, and a 36-subject synthetic exposure-response set with
integer times, hence tied event times, for the two-covariate Cox) and hard-coded.
Stated tolerance vs R: 1e-5 relative for every KM and Cox quantity.

R script (essential calls; output follows):

    show <- function(tag, v) cat(sprintf("%s: %s\\n", tag, paste(sprintf("%.12g", v), collapse = ", ")))
    km  <- survfit(Surv(time, status) ~ 1, data = aml)               # conf.type "log" (R default)
    km2 <- survfit(Surv(time, status) ~ 1, data = aml, conf.type = "plain")
    km3 <- survfit(Surv(time, status) ~ 1, data = aml, conf.type = "log-log")
    show("KM surv", km$surv); show("KM stderr_logS", km$std.err)     # std.err = Greenwood on the log scale
    show("KM lower_log", km$lower); show("KM upper_log", km$upper)
    summary(km)$table[c("median", "0.95LCL", "0.95UCL")]
    sd1 <- survdiff(Surv(time, status) ~ x, data = aml)             # log-rank
    cx <- coxph(Surv(time, status) ~ m, data = aml, ties = "efron") # m = 1 if Maintained; also ties = "breslow"
    cx <- coxph(Surv(tm, ev) ~ auc + wt, ties = tt)                 # C2 data below; tt in efron, breslow
    cu <- coxph(Surv(tm, ev) ~ auc, ties = tt)

R output (selected):

    KM time:   5, 8, 9, 12, 13, 16, 18, 23, 27, 28, 30, 31, 33, 34, 43, 45, 48, 161
    KM surv:   0.913043478261, 0.826086956522, 0.782608695652, 0.739130434783, 0.695652173913, 0.695652173913,
               0.645962732919, 0.546583850932, 0.496894409938, 0.496894409938, 0.441683919945, 0.386473429952,
               0.331262939959, 0.276052449965, 0.220841959972, 0.165631469979, 0.0828157349896, 0.0828157349896
    KM stderr_logS: 0.0643489452088, 0.095672974647, 0.109896745566, 0.123876020852, 0.137919321092, 0.137919321092,
               0.156576641377, 0.196219924093, 0.218158583837, 0.218158583837, 0.247955755309, 0.281672148868,
               0.321167749404, 0.369434779888, 0.431835682396, 0.519437570767, 0.87739124108, 0.87739124108
    KM median(lcl,ucl): 27, 18, 45     ;  by group: Maintained 31 (18, NA), Nonmaintained 23 (8, NA)
    LOGRANK chisq 3.39638869898, p 0.0653393220405, obs 7, 11, exp 10.6893359923, 7.3106640077
    COXAML efron   coef -0.915532575015 se 0.511934275172 p 0.073714860639  loglik -42.7248392628 -41.0326155965
    COXAML breslow coef -0.904219723686 se 0.512247907304 p 0.0775302512086 loglik -42.8981238972 -41.2501143501
    COX2 efron   coef 0.00963897104614, -0.0266848688141  se 0.00526428584137, 0.01538420901
                 loglik -71.7882449581 -69.1104035259   lrt 5.35568286433
    COX2 breslow coef 0.00966715599182, -0.0266615582224  se 0.00527551227004, 0.0153987356142
                 loglik -71.9995927522 -69.3322116403
    COX1 efron   coef 0.00807928799738 se 0.00539243819718 hr_ci 1.00811201352, 0.997513400066, 1.01882323759
    COX1 breslow coef 0.00807600182157 se 0.00540344139334
"""
import json
import math

import numpy as np
import pytest

from app.compute.er_survival import cox_ph, kaplan_meier, logrank_test

REL = 1e-5   # stated tolerance vs R survival

AML_T = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161, 5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
AML_E = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
AML_M = [1] * 11 + [0] * 12                 # 1 = Maintained

C2_T = [5, 26, 14, 42, 21, 16, 29, 12, 13, 18, 16, 4, 16, 2, 23, 20, 22, 16, 11, 45, 15, 7, 7, 19, 1, 6, 21, 12, 3,
        8, 19, 43, 8, 38, 13, 20]
C2_E = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
C2_AUC = [121.3, 116.3, 160.5, 64.3, 93.8, 61.7, 179.0, 80.4, 153.0, 82.5, 100.0, 135.2, 71.9, 114.6, 70.9, 61.9,
          76.6, 139.8, 82.5, 93.6, 50.5, 118.7, 118.0, 78.6, 152.4, 112.5, 115.2, 69.5, 86.4, 131.0, 138.6, 85.2,
          110.8, 64.5, 47.4, 92.5]
C2_WT = [65, 56, 90, 66, 108, 68, 67, 73, 86, 61, 77, 73, 64, 56, 88, 55, 80, 99, 75, 79, 63, 65, 81, 47, 74, 57,
         74, 79, 80, 64, 102, 89, 68, 57, 75, 80]

# Heavy-ties fixture: n = 200, 109 events on only 8 distinct integer times (tie groups of ~14), three
# covariates (x1, x2 continuous to 3 dp; x3 binary). R output for ``coxph(Surv(t, e) ~ x1 + x2 + x3, ties = tt)``:
#   HEAVY efron   coef 0.401084679804, -0.302583290797, 0.525730813565  se 0.112304892695, 0.0840171142633, 0.195746333213
#                 loglik(null,final) -503.696467203, -491.177912624
#   HEAVY breslow coef 0.36385732203, -0.276139195733, 0.490996376005   se 0.111324069546, 0.083673534548, 0.195715903061
#                 loglik(null,final) -511.983758771, -501.325453663
HV_T = [int(c) for c in "48816614166241215831155811215124281177223322332128254418125265188285112381138241122481811881138188511788283222888813886343131542221411854512283155125462112174213242411855324511828322215478258122235632"]
HV_E = [int(c) for c in "10011001110111111011101101000001100101111101110011111100110001000001101001110101011000001011110100100100100111000100101111111010110010011001100101010110111010011000110100111010001110101110110010111100"]
HV_X3 = [int(c) for c in "11010101000001101011000111100010100000010101010110111110010000000001100001011100100010001011100110101110101101001010011110101011101111100111001101010001101100110111011010100110110000000011010001001111"]
HV_X1 = [0.082, -0.464, 0.051, 0.686, -1.757, 1.684, -0.458, -0.596, -1.047, 0.932, 0.675, 1.244, 0.893, 0.263,
         0.329, 0.935, -0.878, -0.046, 0.382, -0.453, 0.722, -0.352, 0.673, 0.141, 0.463, -1.518, -0.86, 1.345,
         0.178, -0.081, 0.964, 0.751, -0.047, -0.643, 1.961, 0.691, -1.572, 0.839, 0.768, 0.814, -0.404, 1.471,
         -0.748, 1.211, 0.293, 1.697, -0.389, 0.696, 0.845, -0.324, 0.011, -0.415, 0.478, 0.689, -0.292, 0.346,
         -0.582, -0.521, -1.923, -1.174, -0.674, 0.108, 1.52, 0.269, 0.091, 0.348, -1.401, 0.048, -0.868, -0.574,
         2.036, 1.847, 0.977, 0.543, 0.504, -0.965, -1.255, 0.335, -0.447, -0.777, -0.08, -0.072, 0.189, -0.74,
         0.017, -1.239, 1.227, 1.519, -1.104, 0.427, -0.221, 0.114, 0.063, 1.193, 0.553, -0.869, 0.08, -0.018,
         -0.798, 0.204, -0.023, 1.359, -0.188, 0.441, -2.248, -1.678, 0.327, 0.129, 0.909, -1.152, -0.285, -2.492,
         -0.837, -0.433, -0.097, -0.555, 0.954, -1.448, -1.283, 0.46, -0.379, 1.337, 0.689, 0.333, -1.714, -0.618,
         0.268, -0.644, 0.282, 0.299, -0.831, -1.114, 0.967, 0.048, 0.055, 0.961, 1.733, -2.809, 1.455, 0.998,
         -0.296, 1.18, -0.083, 1.597, 1.139, 0.412, -0.623, 0.087, -0.876, -0.702, 0.18, -0.128, -0.667, 1.649,
         -0.553, -0.029, -0.003, -2.191, 2.616, -0.576, 0.738, -2.15, -0.168, -1.639, -0.508, 1.266, 0.781, 0.62,
         0.501, -1.755, -1.14, 0.546, 1.36, 0.455, 0.569, 0.545, 0.699, 0.481, -1.105, 0.973, -0.902, -1.469, 0.129,
         -1.334, -1.082, 0.102, -0.822, -0.89, 0.456, 1.11, 0.731, 0.898, -1.483, 0.654, -0.013, -0.717, -0.302,
         -0.383, -1.049, -1.95]
HV_X2 = [0.395, 0.117, 1.345, 1.345, -0.342, 1.356, 1.401, -0.874, -0.577, 0.549, -1.156, -0.711, 0.929, -0.352,
         0.552, 0.166, 0.971, 0.261, 0.248, -0.413, -0.808, -1.653, 2.869, 0.357, 1.169, -1.34, 0.212, 0.785, -0.811,
         1.994, 1.782, 0.065, 1.103, 0.993, 1.179, 0.748, -0.326, 0.263, -1.204, -0.181, 0.345, 0.194, 0.881, 0.464,
         -0.747, -0.74, -1.042, 0.174, 0.04, -2.817, 0.546, 1.468, -0.551, 0.672, -0.859, 1.406, -0.972, -0.335,
         0.863, 0.584, -0.518, 1.269, 1.096, 1.145, 1.564, 1.738, -1.308, -0.818, 0.269, -0.471, -2.538, 1.53,
         -0.972, -0.133, -0.532, -0.639, 0.248, -1.709, -0.449, -1.769, -0.359, 0.242, 1.026, -0.464, 2.487, 0.559,
         1.604, 1.195, -0.589, -0.349, -0.98, 0.066, -3.024, -0.129, 0.134, -1.657, -1.049, 0.399, 0.366, 0.372,
         -2.281, 2.265, -0.29, 3.144, -0.718, 0.035, -0.351, 0.674, 1.99, 0.142, 0.927, -0.079, -1.91, 0.505, -0.716,
         0.33, -0.699, 0.909, -0.935, 2.238, -0.464, 1.273, -1.101, 0.288, -1.254, 0.41, 1.549, 0.935, -0.525,
         -1.096, -0.61, -1.39, 1.329, 0.359, 0.605, -1.159, -0.456, -1.811, 0.759, 0.656, -2.303, 1.619, 1.919,
         -1.328, -0.548, 0.875, -0.959, -0.346, -0.473, -2.037, -0.519, -1.28, -0.583, 0.495, 0.095, 0.068, -0.749,
         -0.456, 2.808, -1.046, -2.938, -1.745, -1.29, -2.282, -0.744, -0.505, 0.145, 0.657, 2.571, 0.107, -0.43,
         -1.865, -1.141, -1.087, -0.188, -1.85, 0.374, 1.478, 0.164, 1.621, -0.552, 0.065, 0.114, 0.277, 0.6, -1.176,
         -1.724, -1.336, 1.165, 0.849, 1.235, -0.62, -0.355, 0.692, -1.916, -0.627, 0.237, 0.121, -1.08, -0.788]

KM_TIME = [5, 8, 9, 12, 13, 16, 18, 23, 27, 28, 30, 31, 33, 34, 43, 45, 48, 161]
KM_NRISK = [23, 21, 19, 18, 17, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 2, 1]
KM_NEVENT = [2, 2, 1, 1, 1, 0, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
KM_NCENS = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
KM_SURV = [0.913043478261, 0.826086956522, 0.782608695652, 0.739130434783, 0.695652173913, 0.695652173913,
           0.645962732919, 0.546583850932, 0.496894409938, 0.496894409938, 0.441683919945, 0.386473429952,
           0.331262939959, 0.276052449965, 0.220841959972, 0.165631469979, 0.0828157349896, 0.0828157349896]
KM_SELOG = [0.0643489452088, 0.095672974647, 0.109896745566, 0.123876020852, 0.137919321092, 0.137919321092,
            0.156576641377, 0.196219924093, 0.218158583837, 0.218158583837, 0.247955755309, 0.281672148868,
            0.321167749404, 0.369434779888, 0.431835682396, 0.519437570767, 0.87739124108, 0.87739124108]
KM_LO_LOG = [0.804854797783, 0.684839468072, 0.6309579106, 0.579799217968, 0.530878341628, 0.530878341628,
             0.475257724706, 0.372078087249, 0.32401654569, 0.32401654569, 0.271675995392, 0.222515297747,
             0.176520182581, 0.133822195629, 0.0947332405633, 0.0598407178101, 0.0148346082548, 0.0148346082548]
KM_HI_LOG = [1, 0.996466605023, 0.970708759208, 0.942246527232, 0.9115684501, 0.9115684501, 0.877982262317,
             0.802933352802, 0.762010637765, 0.762010637765, 0.718078477475, 0.671242442974, 0.621657726531,
             0.569449296312, 0.514826379784, 0.458446770886, 0.462327406565, 0.462327406565]
KM_LO_PLAIN = [0.79788896017, 0.671182777938, 0.614039741744, 0.55967507956, 0.507605633313, 0.507605633313,
               0.447726732276, 0.336376455802, 0.284430823732, 0.284430823732, 0.227032447132, 0.173114099622,
               0.122740464819, 0.0761687058063, 0.0339252152411, 0, 0, 0]
KM_HI_PLAIN = [1, 0.980991135106, 0.951177649561, 0.918585790005, 0.883698714513, 0.883698714513, 0.844198733563,
               0.756791246061, 0.709357996144, 0.709357996144, 0.656335392757, 0.599832760282, 0.539785415098,
               0.475936194125, 0.407758704704, 0.334257379863, 0.225230247028, 0.225230247028]
KM_LO_LL = [0.694947537532, 0.600610411237, 0.554211506669, 0.509209438754, 0.465641749762, 0.465641749762,
            0.413952950717, 0.319250413971, 0.275565628012, 0.275565628012, 0.227381168463, 0.182836618649,
            0.141825693482, 0.104441343234, 0.0709966489829, 0.0421142812616, 0.00695596679854, 0.00695596679854]
KM_HI_LL = [0.977515669901, 0.930903558387, 0.903207489882, 0.873375781698, 0.841721291392, 0.841721291392,
            0.805307764482, 0.726449173821, 0.684213788448, 0.684213788448, 0.63709268746, 0.587476914418,
            0.535273789989, 0.480284660417, 0.422167443381, 0.360361360273, 0.286760367981, 0.286760367981]


def _close(got, want, rel=REL):
    assert got == pytest.approx(want, rel=rel, abs=1e-12), (got, want)


def _col(table, key):
    return [row[key] for row in table]


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def test_km_table_counts_match_r():
    km = kaplan_meier(AML_T, AML_E)
    assert km["status"] == "ok" and km["n"] == 23 and km["n_events"] == 18
    assert _col(km["table"], "time") == KM_TIME
    assert _col(km["table"], "n_risk") == KM_NRISK
    assert _col(km["table"], "n_event") == KM_NEVENT
    assert _col(km["table"], "n_censor") == KM_NCENS


def test_km_survival_and_greenwood_se_match_r():
    km = kaplan_meier(AML_T, AML_E)
    for got, want in zip(_col(km["table"], "surv"), KM_SURV):
        _close(got, want)
    # Greenwood: se(S) = S * se(log S); R reports se(log S) as `std.err`.
    for row, s, se_log in zip(km["table"], KM_SURV, KM_SELOG):
        _close(row["se"], s * se_log)
        _close(row["se_log"], se_log)


def test_km_log_ci_is_the_r_default_and_matches():
    km = kaplan_meier(AML_T, AML_E)
    assert km["conf_type"] == "log"
    for got, want in zip(_col(km["table"], "lo"), KM_LO_LOG):
        _close(got, want)
    for got, want in zip(_col(km["table"], "hi"), KM_HI_LOG):
        _close(got, want)


def test_km_plain_and_loglog_ci_match_r():
    plain = kaplan_meier(AML_T, AML_E, conf_type="plain")
    for got, want in zip(_col(plain["table"], "lo"), KM_LO_PLAIN):
        _close(got, want)
    for got, want in zip(_col(plain["table"], "hi"), KM_HI_PLAIN):
        _close(got, want)
    ll = kaplan_meier(AML_T, AML_E, conf_type="log-log")
    for got, want in zip(_col(ll["table"], "lo"), KM_LO_LL):
        _close(got, want)
    for got, want in zip(_col(ll["table"], "hi"), KM_HI_LL):
        _close(got, want)


def test_km_median_and_its_ci_match_r():
    m = kaplan_meier(AML_T, AML_E)["median"]
    assert (m["estimate"], m["lo"], m["hi"]) == (27, 18, 45)
    mt = kaplan_meier([t for t, g in zip(AML_T, AML_M) if g], [e for e, g in zip(AML_E, AML_M) if g])["median"]
    assert (mt["estimate"], mt["lo"], mt["hi"]) == (31, 18, None)
    mn = kaplan_meier([t for t, g in zip(AML_T, AML_M) if not g], [e for e, g in zip(AML_E, AML_M) if not g])["median"]
    assert (mn["estimate"], mn["lo"], mn["hi"]) == (23, 8, None)


@pytest.mark.parametrize("t,e,want", [
    ([1, 2, 3, 4], [1, 1, 1, 1], (2.5, 1, None)),            # S == 0.5 on a step -> midpoint rule (R)
    ([1, 2, 2.5, 4], [1, 1, 0, 1], (3, 1, None)),            # flat at 0.5 until t=4 -> (2 + 4) / 2
    ([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 0, 1], (3.5, 2, None)),
    ([1, 2, 3, 4, 5], [1, 1, 0, 1, 0], (4, 2, None)),
    ([2, 4, 4, 6, 8, 8, 9], [1, 1, 1, 0, 1, 0, 1], (8, 4, None)),
    ([1, 2], [1, 0], (1, 1, None)),                           # S ends exactly at 0.5
    ([1, 2, 3, 4], [1, 0, 0, 0], (None, 1, None)),            # median not reached
    ([2, 4, 6], [0, 0, 0], (None, None, None)),               # all censored
])
def test_km_median_conventions_match_r_edge_cases(t, e, want):
    m = kaplan_meier(t, e)["median"]
    assert (m["estimate"], m["lo"], m["hi"]) == want


def test_logrank_matches_r_survdiff():
    r = logrank_test(AML_T, AML_E, AML_M)
    assert r["status"] == "ok" and r["df"] == 1
    _close(r["chisq"], 3.39638869898)
    _close(r["p"], 0.0653393220405)
    by = {g["group"]: g for g in r["groups"]}
    _close(by["0"]["observed"], 11); _close(by["1"]["observed"], 7)
    _close(by["0"]["expected"], 7.3106640077); _close(by["1"]["expected"], 10.6893359923)


def test_km_invariants_monotone_survival_and_deterministic():
    rng = np.random.default_rng(11)
    t = np.round(rng.exponential(10, 200), 1) + 0.1
    e = (rng.random(200) < 0.7).astype(int)
    a = kaplan_meier(t.tolist(), e.tolist())
    b = kaplan_meier(t.tolist(), e.tolist())
    s = _col(a["table"], "surv")
    assert all(x >= y for x, y in zip(s, s[1:])) and a == b


def test_km_refuses_bad_input():
    assert kaplan_meier([1, 2], [1])["status"] == "invalid_input"
    assert kaplan_meier([1, -2, 3], [1, 1, 0])["status"] == "invalid_input"
    assert kaplan_meier([1, 2, 3], [1, 2, 0])["status"] == "invalid_input"
    assert kaplan_meier([], [])["status"] == "invalid_input"


# ── Cox proportional hazards ─────────────────────────────────────────────────

def test_cox_aml_efron_matches_r():
    r = cox_ph(AML_T, AML_E, {"m": AML_M})
    assert r["status"] == "ok" and r["ties"] == "efron" and r["converged"]
    c = r["covariates"][0]
    _close(c["coef"], -0.915532575015); _close(c["se"], 0.511934275172)
    _close(c["z"], -1.78837913267); _close(c["p"], 0.073714860639)
    _close(c["hr"], 0.400303377733); _close(c["hr_lo"], 0.146767538182); _close(c["hr_hi"], 1.09181359999)
    _close(r["loglik_null"], -42.7248392628); _close(r["loglik"], -41.0326155965)
    _close(r["lr_chisq"], 3.3844473326)


def test_cox_aml_breslow_matches_r():
    r = cox_ph(AML_T, AML_E, {"m": AML_M}, ties="breslow")
    c = r["covariates"][0]
    assert r["ties"] == "breslow"
    _close(c["coef"], -0.904219723686); _close(c["se"], 0.512247907304)
    _close(c["p"], 0.0775302512086)
    _close(c["hr"], 0.404857662735); _close(c["hr_lo"], 0.148346107212); _close(c["hr_hi"], 1.10491424518)
    _close(r["loglik_null"], -42.8981238972); _close(r["loglik"], -41.2501143501)
    _close(r["lr_chisq"], 3.29601909421)


def test_cox_two_covariates_with_tied_times_efron_and_breslow_match_r():
    ef = cox_ph(C2_T, C2_E, {"auc": C2_AUC, "wt": C2_WT})
    for cv, coef, se, p in zip(ef["covariates"], (0.00963897104614, -0.0266848688141),
                               (0.00526428584137, 0.01538420901), (0.0670987378368, 0.0828183636624)):
        _close(cv["coef"], coef); _close(cv["se"], se); _close(cv["p"], p)
    _close(ef["loglik_null"], -71.7882449581); _close(ef["loglik"], -69.1104035259)
    _close(ef["lr_chisq"], 5.35568286433)
    br = cox_ph(C2_T, C2_E, {"auc": C2_AUC, "wt": C2_WT}, ties="breslow")
    for cv, coef, se, p in zip(br["covariates"], (0.00966715599182, -0.0266615582224),
                               (0.00527551227004, 0.0153987356142), (0.0668831761854, 0.0833782778191)):
        _close(cv["coef"], coef); _close(cv["se"], se); _close(cv["p"], p)
    _close(br["loglik"], -69.3322116403)
    assert ef["n_tied_event_times"] > 0, "fixture must actually contain tied event times"


def test_cox_single_exposure_hr_per_unit_and_per_sd():
    r = cox_ph(C2_T, C2_E, {"auc": C2_AUC})
    c = r["covariates"][0]
    _close(c["coef"], 0.00807928799738); _close(c["se"], 0.00539243819718)
    _close(c["hr"], 1.00811201352); _close(c["hr_lo"], 0.997513400066); _close(c["hr_hi"], 1.01882323759)
    _close(r["loglik"], -70.7392599413)
    sd = 33.205516849                      # R: sd(auc)
    _close(c["sd"], sd)
    z = 1.959963984540054
    _close(c["hr_per_sd"], math.exp(0.00807928799738 * sd))
    _close(c["hr_per_sd_lo"], math.exp((0.00807928799738 - z * 0.00539243819718) * sd))
    _close(c["hr_per_sd_hi"], math.exp((0.00807928799738 + z * 0.00539243819718) * sd))


def test_cox_recovers_known_hazard_ratio_within_sampling_error():
    rng = np.random.default_rng(5)
    n, beta = 3000, 0.5
    x = rng.normal(0.0, 1.0, n)
    t_ev = rng.exponential(1.0 / (0.1 * np.exp(beta * x)))
    t_ce = rng.exponential(12.0, n)
    t = np.minimum(t_ev, t_ce)
    e = (t_ev <= t_ce).astype(int)
    r = cox_ph(t.tolist(), e.tolist(), {"x": x.tolist()})
    c = r["covariates"][0]
    assert abs(c["coef"] - beta) < 3.0 * c["se"], (c["coef"], c["se"])


def test_cox_monotone_likelihood_is_refused():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]          # earliest events carry the highest exposure
    r = cox_ph(t, [1] * 12, {"x": x})
    assert r["status"] in ("monotone_likelihood", "non_convergence")
    assert "covariates" not in r


def test_cox_refusals_zero_variance_few_events_invalid():
    t = list(range(1, 21))
    assert cox_ph(t, [1] * 20, {"x": [3.0] * 20})["status"] == "zero_variance_covariate"
    assert cox_ph(t, [1, 1, 1] + [0] * 17, {"x": list(range(20))})["status"] == "insufficient_events"
    assert cox_ph(t, [0] * 20, {"x": list(range(20))})["status"] == "insufficient_events"
    assert cox_ph([1, 2], [1], {"x": [1, 2]})["status"] == "invalid_input"
    assert cox_ph([1, -2, 3], [1, 1, 1], {"x": [1, 2, 3]})["status"] == "invalid_input"
    assert cox_ph(t, [1] * 20, {"x": list(range(20))}, ties="exact")["status"] == "invalid_input"
    assert cox_ph(t, [1] * 20, {})["status"] == "invalid_input"


def test_cox_events_per_variable_warning_and_json_safe():
    r = cox_ph(C2_T, C2_E, {"auc": C2_AUC, "wt": C2_WT})
    s = json.dumps(r)
    assert "NaN" not in s and "Infinity" not in s
    assert len(r["cov"]) == 2
    rng = np.random.default_rng(2)
    ev = np.zeros(24, dtype=int)
    ev[rng.choice(24, 7, replace=False)] = 1
    few = cox_ph(list(range(1, 25)), ev.tolist(), {"x": rng.normal(0, 1, 24).tolist()})
    assert few["status"] == "ok" and any("events per" in w.lower() for w in few["warnings"])


def test_km_and_cox_handle_a_few_thousand_subjects_with_heavy_ties():
    rng = np.random.default_rng(17)
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    t_ev = np.ceil(rng.exponential(1.0 / (0.1 * np.exp(0.3 * x))))        # integer times -> many ties
    t_ce = np.ceil(rng.exponential(15.0, n))
    t, e = np.minimum(t_ev, t_ce), (t_ev <= t_ce).astype(int)
    assert kaplan_meier(t.tolist(), e.tolist())["status"] == "ok"
    r = cox_ph(t.tolist(), e.tolist(), {"x": x.tolist()})
    c = r["covariates"][0]
    assert r["status"] == "ok" and r["n_tied_event_times"] > 10
    assert abs(c["coef"] - 0.3) < 4.0 * c["se"]


def test_cox_heavy_ties_three_covariates_matches_r_for_efron_and_breslow():
    """Large tie groups (~14 events per time) are where Efron and Breslow differ most; both must match R."""
    cov = {"x1": HV_X1, "x2": HV_X2, "x3": HV_X3}
    ef = cox_ph(HV_T, HV_E, cov)
    for cv, coef, se in zip(ef["covariates"], (0.401084679804, -0.302583290797, 0.525730813565),
                            (0.112304892695, 0.0840171142633, 0.195746333213)):
        _close(cv["coef"], coef); _close(cv["se"], se)
    _close(ef["loglik_null"], -503.696467203); _close(ef["loglik"], -491.177912624)
    br = cox_ph(HV_T, HV_E, cov, ties="breslow")
    for cv, coef, se in zip(br["covariates"], (0.36385732203, -0.276139195733, 0.490996376005),
                            (0.111324069546, 0.083673534548, 0.195715903061)):
        _close(cv["coef"], coef); _close(cv["se"], se)
    _close(br["loglik_null"], -511.983758771); _close(br["loglik"], -501.325453663)
    assert ef["n_events"] == 109 and ef["n_tied_event_times"] == 8

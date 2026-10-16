# ============================================================================
#  Supplementary material for the manuscript
#      "<ARTICLE TITLE>"  (<AUTHORS>),
#      submitted to Fuzzy Sets and Systems.
#  <-- replace the title/authors line above before submission -->
#
#  Interactive computation of the interval negation N(X, Y).
#
#  Given a 4x4 row-stochastic matrix A and two intervals X = [x1, x2] and
#  Y = [x3, x4], this application computes the negation N(X, Y) using exact
#  rational arithmetic, and reports every intermediate quantity of its
#  construction so that each step can be inspected and reproduced.
#
#  Requirements : Python >= 3.9, numpy, pandas, streamlit
#  Usage        : streamlit run interval_negation.py
#                 (then open the local URL that Streamlit prints)
# ============================================================================
import numpy as np
import pandas as pd
import streamlit as st
from fractions import Fraction
 
 
def F(x):
    """Return x as an exact rational, rounded to the nearest fraction whose
    denominator does not exceed 1_000_000 (removes floating-point noise)."""
    return Fraction(x).limit_denominator(1_000_000)
 
 
def compute_negation(A, X, Y, lambda_min=None):
    """Compute the interval negation N(X, Y) for the row-stochastic matrix A
    and the intervals X = [x1, x2], Y = [x3, x4], using exact rational
    arithmetic.  Returns a dictionary holding N together with every
    intermediate quantity.  If the construction reaches control == 1 and
    ``lambda_min`` is not supplied, the function returns early with
    res['needs_lambda_min'] = True and res['lambda_value'] = lambda."""
    res = {}
 
    alpha_1 = F(A[0, 1] / (A[0, 0] + A[0, 1]))
    alpha_2 = F(A[0, 3] / (A[0, 2] + A[0, 3]))
    alpha   = F(A[0, 2] + A[0, 3])
    K1 = F((1 - alpha_1) * X[0, 0] + alpha_1 * X[0, 1])
    K2 = F((1 - alpha_2) * Y[0, 0] + alpha_2 * Y[0, 1])
    l  = F(max(X[0, 1], Y[0, 1]) - min(X[0, 0], Y[0, 0]))
    l1 = F(X[0, 1] - X[0, 0]); l2 = F(Y[0, 1] - Y[0, 0])
    res.update(alpha_1=alpha_1, alpha_2=alpha_2, alpha=alpha,
               K1=K1, K2=K2, l=l, l1=l1, l2=l2)
 
    # Degenerate case: both intervals reduce to a single point.
    if l1 + l2 == 0:
        z = F(X[0, 0])
        res['case'] = 'degenerate (l1+l2=0)'; res['A1XY'] = z
        res['N'] = ([1 - z, 1 - z], [1 - z, 1 - z])
        res['Z'] = [1 - z, 1 - z]; res['T'] = [1 - z, 1 - z]
        return res
 
    q1 = F(l1 / l); q2 = F(l2 / l); res.update(q1=q1, q2=q2)
 
    # Three cases according to the order of K1 and K2.
    if K1 < K2:
        case = 'K1<K2'; A1XY = F((1 - alpha) * K1 + alpha * K2); d = F(K2 - K1)
        u = F((Y[0, 1] - X[0, 0]) / l); sign_B = +1; pA_first = True
    elif K2 < K1:
        case = 'K2<K1'; A1XY = F(alpha * K2 + (1 - alpha) * K1); d = F(K1 - K2)
        u = F((X[0, 1] - Y[0, 0]) / l); sign_B = -1; pA_first = False
    else:
        case = 'K1==K2'; A1XY = F(alpha * K2 + (1 - alpha) * K1); d = F(K1 - K2)
        u = F(max((X[0, 1] - Y[0, 0]) / l, (Y[0, 1] - X[0, 0]) / l))
        sign_B = +1; pA_first = (u == Y[0, 1] - X[0, 0])
 
    m = F(d / l); res.update(case=case, A1XY=A1XY, d=d, m=m, u=u)
 
    t1 = F((X[0, 0] - A1XY) / l); t2 = F((X[0, 1] - A1XY) / l)
    t3 = F((Y[0, 0] - A1XY) / l); t4 = F((Y[0, 1] - A1XY) / l)
 
    def clamp_unit(v):
        """Keep v if it lies in [0, 1]; otherwise discard it (return 2)."""
        return 2 if (v < 0 or 1 < v) else v
 
    lm1 = clamp_unit(F(-A1XY / t1)) if t1 != 0 else 2
    lm2 = clamp_unit(F((1 - A1XY) / t2)) if t2 != 0 else 2
    lm3 = clamp_unit(F(-A1XY / t3)) if t3 != 0 else 2
    lm4 = clamp_unit(F((1 - A1XY) / t4)) if t4 != 0 else 2
    lmax = F(min(lm1, lm2, lm3, lm4)); lamda = F(l / lmax)
    u1 = A1XY + lmax * t1; u2 = A1XY + lmax * t2
    u3 = A1XY + lmax * t3; u4 = A1XY + lmax * t4
    res.update(t1=t1, t2=t2, t3=t3, t4=t4, lm1=lm1, lm2=lm2, lm3=lm3, lm4=lm4,
               lmax=lmax, lamda=lamda, u1=u1, u2=u2, u3=u3, u4=u4)
 
    B = np.dot(A, np.array([[-alpha], [-alpha], [1 - alpha], [1 - alpha]]))
    C = np.dot(A, np.array([[-alpha_1], [1 - alpha_1], [0], [0]]))
    D = np.dot(A, np.array([[0], [0], [-alpha_2], [1 - alpha_2]]))
    BCD = sign_B * m * B + q1 * C + q2 * D
    cl = F(next(x for x in BCD.ravel() if abs(x) > 1e-9))
    bcd = [F(BCD[i, 0]) for i in range(4)]
    res.update(cl=cl, bcd=bcd)
 
    # Selection of p: two candidates; keep the first one lying in [0, 1].
    pA = u - (1 - alpha_1) * q2 - alpha_2 * q1
    pB = u - alpha_1 * q2 - (1 - alpha_2) * q1
    p_formula_A = "p = u - (1 - alpha_1) q2 - alpha_2 q1"
    p_formula_B = "p = u - alpha_1 q2 - (1 - alpha_2) q1"
    order         = [pA, pB] if pA_first else [pB, pA]
    order_formula = [p_formula_A, p_formula_B] if pA_first else [p_formula_B, p_formula_A]
    p = order[0]; p_formula = order_formula[0]
    if p < 0 or 1 < p:
        p = order[1]; p_formula = order_formula[1]
    pB_used = (p_formula == p_formula_B)      # True -> case M1 <= M2 (branch hm1..hm4)
    res['p'] = p; res['p_formula'] = p_formula
 
    def hdiv(num, den):
        """Safe division: returns 2 (out-of-range marker) when den == 0."""
        return F(Fraction(num) / Fraction(den)) if den != 0 else 2
 
    # Candidate h_max values for the case M1 <= M2 (branch hm1..hm4).
    hm1 = hdiv(1 - A1XY, alpha * p + alpha_1 * q2)
    hm2 = hdiv(A1XY,     -alpha * p + (1 - alpha_1) * q2)
    hm3 = hdiv(1 - A1XY, -(1 - alpha) * p + alpha_2 * q1)
    hm4 = hdiv(A1XY,     (1 - alpha) * p + (1 - alpha_2) * q1)
    # Candidate h_max values for the case M2 <= M1 (branch hmi..hmiv).
    hmi   = hdiv(1 - A1XY, -alpha * p + alpha_1 * q2)
    hmii  = hdiv(A1XY,    alpha * p + (1 - alpha_1) * q2)
    hmiii = hdiv(1 - A1XY, (1 - alpha) * p + alpha_2 * q1)
    hmiv  = hdiv(A1XY,    -(1 - alpha) * p + (1 - alpha_2) * q1)
 
    # Discard the candidates that fall outside [0, 1].
    if not 0 <= hm1 <= 1: hm1 = 2
    if not 0 <= hm2 <= 1: hm2 = 2
    if not 0 <= hm3 <= 1: hm3 = 2
    if not 0 <= hm4 <= 1: hm4 = 2
    if not 0 <= hmi <= 1: hmi = 2
    if not 0 <= hmii <= 1: hmii = 2
    if not 0 <= hmiii <= 1: hmiii = 2
    if not 0 <= hmiv <= 1: hmiv = 2
 
    hmax_c1 = F(min(hm1, hm2, hm3, hm4))      # candidate for the case M1 <= M2
    hmax_c2 = F(min(hmi, hmii, hmiii, hmiv))  # candidate for the case M2 <= M1
    hmax1 = hmax_c1 if 0 <= hmax_c1 <= 1 else -1
    hmax2 = hmax_c2 if 0 <= hmax_c2 <= 1 else -1
 
    # The branch is chosen by HOW p was selected (p == pB  <=>  M1 <= M2),
    # not by the numerical value of u - p (which the two branches can share).
    # If the branch dictated by the theory leaves [0, 1] (its h_max is -1),
    # the other branch is used instead.
    case_M1_le_M2 = pB_used
    if case_M1_le_M2:
        use_branch_1 = (hmax1 != -1)
    else:
        use_branch_1 = (hmax2 == -1)
 
    if hmax1 == -1 and hmax2 == -1:
        res['hmax_warning'] = "Neither hmax1 nor hmax2 lies in [0, 1]; check the data."
 
    if use_branch_1:
        hmax = hmax1
        M1 = F(1 - A1XY - alpha * p * hmax); M2 = F(p * hmax + M1)
    else:
        hmax = hmax2
        M2 = F(1 - A1XY - (1 - alpha) * p * hmax); M1 = F(M2 + p * hmax)
 
    res.update(hm1=hm1, hm2=hm2, hm3=hm3, hm4=hm4,
               hmi=hmi, hmii=hmii, hmiii=hmiii, hmiv=hmiv,
               hmax1=hmax1, hmax2=hmax2, hmax=hmax, M1=M1, M2=M2,
               case_M1_le_M2=case_M1_le_M2)
 
    v1 = F(M1 - alpha_1 * q2 * hmax); v2 = F(v1 + q2 * hmax)
    v3 = F(M2 - alpha_2 * q1 * hmax); v4 = F(v3 + q1 * hmax)
    s1 = F((v1 - 1 + A1XY) / hmax); s2 = F((v2 - 1 + A1XY) / hmax)
    s3 = F((v3 - 1 + A1XY) / hmax); s4 = F((v4 - 1 + A1XY) / hmax)
    res.update(v1=v1, v2=v2, v3=v3, v4=v4, s1=s1, s2=s2, s3=s3, s4=s4)
 
    eta = np.dot(A, np.array([s1, s2, s3, s4]))
    ch = F(next(x for x in eta if x != 0))
    res['ch'] = ch; res['eta'] = [F(eta[i]) for i in range(4)]
 
    control = 0
    if cl * ch < 0:
        mu = lamda
    else:
        gamma = F(-min(t1, t2, t3, t4) / (max(t1, t2, t3, t4) - min(t1, t2, t3, t4)))
        delta = F(-min(s1, s2, s3, s4) / (max(s1, s2, s3, s4) - min(s1, s2, s3, s4)))
        Delta = F((max(gamma, 1 - gamma) + max(delta, 1 - delta)) / 2)
        Theta = F((min(gamma, 1 - gamma) + min(delta, 1 - delta)) / 2)
        if Delta != Theta:
            g  = F(Fraction(Theta) / Fraction(Delta))
            g1 = F(Fraction(Delta) / Fraction(Theta))
        else:
            if max(m, p) > 0:
                g = F(max(m, p)); g1 = F(1 / max(m, p))
            else:
                g = g1 = None
                res['warning'] = 'check: Delta == Theta and max(m, p) <= 0'
        Rr = F(Fraction(1, 2) * (g1 - g) / (1 - g))
        r_DT = Fraction(Delta) / Fraction(Theta)   # Delta / Theta
        r_TD = Fraction(Theta) / Fraction(Delta)   # Theta / Delta
        lamda_prime = F(lamda / Rr)
        if lamda_prime <= Fraction(1, 2):
            control = 1 if lamda < F(Rr * Rr - r_DT) else 0
            mu_prime = F(Rr * g - g * lamda_prime)
        else:
            control = 1 if lamda < F(Rr * Rr * r_TD - r_DT) else 0
            mu_prime = F(Rr - g1 * lamda_prime)
        mu = F(mu_prime * Rr)
        res.update(gamma=gamma, delta=delta, Delta=Delta, Theta=Theta,
                   g=g, g1=g1, R_=Rr, lamda_prime=lamda_prime, mu_prime=mu_prime,
                   control=control)
 
        if control == 1:
            if lambda_min is None:
                res['needs_lambda_min'] = True
                res['lambda_value'] = lamda
                return res
            Ss = F((r_DT + lambda_min) / Rr)
            lamda_prime = F(lamda / Ss)
            if lamda_prime < Fraction(1, 2):
                mu_prime = F(Rr * g - g * lamda_prime)
            else:
                mu_prime = F(Rr - g1 * lamda_prime)
            mu = F(mu_prime * Ss)
            res.update(Ss=Ss, lamda_prime=lamda_prime, mu_prime=mu_prime,
                       lambda_min=lambda_min)
 
    z1 = 1 - A1XY + mu * hmax * s1; z2 = 1 - A1XY + mu * hmax * s2
    z3 = 1 - A1XY + mu * hmax * s3; z4 = 1 - A1XY + mu * hmax * s4
    res.update(mu=mu, N=([z1, z2], [z3, z4]), Z=[z1, z2], T=[z3, z4], control=control)
    res['A1ZT'] = 1 - A1XY; res['h1'] = q2 * hmax; res['h2'] = q1 * hmax
    res['U'] = ([u1, u2], [u3, u4]); res['V'] = ([v1, v2], [v3, v4])
    res['min_lower'] = min(u1, u3); res['max_upper'] = max(u2, u4)
    return res
 
 
# ----------------------------------------------------------------------------
#  Formatting helpers
# ----------------------------------------------------------------------------
def tex(fr):
    """Format a Fraction as LaTeX: 'a' or '\\frac{a}{b}' ('?' for None)."""
    if fr is None:
        return "?"
    fr = Fraction(fr)
    if fr.denominator == 1:
        return str(fr.numerator)
    sign = "-" if fr < 0 else ""
    return f"{sign}\\frac{{{abs(fr.numerator)}}}{{{fr.denominator}}}"
 
 
def col_vec(values):
    """Build a LaTeX column vector from a list of values."""
    return r"\begin{pmatrix}" + r"\\[6pt]".join(tex(v) for v in values) + r"\end{pmatrix}"
 
 
def frac_str(fr):
    """Format a Fraction as plain text, e.g. '3/5' ('?' for None)."""
    if fr is None:
        return "?"
    return str(Fraction(fr))
 
 
def parse_frac(text):
    """Parse user input ('1/6', '0.5', '1') into an exact Fraction."""
    return Fraction(str(text).strip())
 
 
# ----------------------------------------------------------------------------
#  Predefined example matrices (entered as text fractions)
# ----------------------------------------------------------------------------
EXAMPLES = {
    "Example 1": [["1/3", "1/3", "1/6", "1/6"],
                  ["1/3", "1/6", "1/3", "1/6"],
                  ["1/3", "1/6", "1/6", "1/3"],
                  ["1/2", "1/6", "1/6", "1/6"]],
    "Example 2": [["1/2", "1/6", "1/6", "1/6"],
                  ["1/6", "1/2", "1/6", "1/6"],
                  ["1/6", "1/6", "1/2", "1/6"],
                  ["1/6", "1/6", "1/6", "1/2"]],
    "Example 3": [["1/6", "1/3", "1/3", "1/6"],
                  ["1/6", "1/3", "1/6", "1/3"],
                  ["1/6", "1/6", "1/3", "1/3"],
                  ["1/3", "1/6", "1/3", "1/6"]],
    "Example 4": [["1/2", "1/6", "1/6", "1/6"],
                  ["1/3", "1/6", "1/3", "1/6"],
                  ["1/6", "1/3", "1/6", "1/3"],
                  ["1/6", "1/6", "1/6", "1/2"]],
}
COLS = ["c1", "c2", "c3", "c4"]
 
 
# ----------------------------------------------------------------------------
#  User interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Interval negation N[X,Y]", layout="centered")
st.title("Interval negation  N[X, Y]")
 
if "matrix" not in st.session_state:
    st.session_state.matrix = pd.DataFrame(EXAMPLES["Example 2"], columns=COLS)
 
st.subheader("Matrix A (4x4)")
st.caption("Enter each entry as a fraction (e.g. `1/6`) or a decimal (e.g. `0.5`).")
 
left, right = st.columns([2, 1])
with right:
    preset = st.selectbox("Load example", list(EXAMPLES.keys()), index=1)
    if st.button("Load into the table"):
        st.session_state.matrix = pd.DataFrame(EXAMPLES[preset], columns=COLS)
 
matrix_df = st.data_editor(
    st.session_state.matrix,
    hide_index=True,
    column_config={c: st.column_config.TextColumn(width="small") for c in COLS},
    key="matrix_editor",
)
 
st.subheader("Intervals")
cX1, cX2, cY1, cY2 = st.columns(4)
with cX1: x1 = st.text_input("x1  (X lower)", "1/4")
with cX2: x2 = st.text_input("x2  (X upper)", "3/4")
with cY1: x3 = st.text_input("x3  (Y lower)", "1/4")
with cY2: x4 = st.text_input("x4  (Y upper)", "1/2")
 
compute_clicked = st.button("Compute", type="primary")
 
# On "Compute": read and store the inputs, reset the auxiliary parameter.
if compute_clicked:
    try:
        A_fr = [[parse_frac(matrix_df.iloc[i, j]) for j in range(4)] for i in range(4)]
        X_fr = [parse_frac(x1), parse_frac(x2)]
        Y_fr = [parse_frac(x3), parse_frac(x4)]
    except Exception:
        st.error("Some entry is not a valid number. Use fractions such as `1/6` or decimals such as `0.5`.")
        st.stop()
    st.session_state.data = {"A_fr": A_fr, "X_fr": X_fr, "Y_fr": Y_fr}
    st.session_state.lambda_min = None
 
# Computation and output.
if "data" in st.session_state:
    A_fr = st.session_state.data["A_fr"]
    X_fr = st.session_state.data["X_fr"]
    Y_fr = st.session_state.data["Y_fr"]
    A = np.array([[float(A_fr[i][j]) for j in range(4)] for i in range(4)])
    X = np.array([[float(X_fr[0]), float(X_fr[1])]])
    Y = np.array([[float(Y_fr[0]), float(Y_fr[1])]])
 
    if X_fr[0] > X_fr[1] or Y_fr[0] > Y_fr[1]:
        st.warning("Check the intervals: a lower endpoint is greater than its upper endpoint.")
    for i in range(4):
        if sum(A_fr[i]) != 1:
            st.info(f"Note: row {i + 1} of A sums to {frac_str(sum(A_fr[i]))} (not 1).")
 
    lm_current = st.session_state.get("lambda_min")
    try:
        res = compute_negation(A, X, Y, lambda_min=lm_current)
    except (ZeroDivisionError, StopIteration):
        st.error("The computation cannot be completed with these data (division by zero at some step).")
        st.stop()
 
    # Echo of the input (always shown).
    st.divider()
    A_tex = "\\\\".join(" & ".join(tex(A_fr[i][j]) for j in range(4)) for i in range(4))
    st.latex(r"A=\begin{pmatrix}" + A_tex + r"\end{pmatrix}\qquad "
             r"X=\left[" + tex(X_fr[0]) + "," + tex(X_fr[1]) + r"\right]\quad "
             r"Y=\left[" + tex(Y_fr[0]) + "," + tex(Y_fr[1]) + r"\right]")
 
    # If the auxiliary parameter lambda_min is required, ask for it and stop.
    if res.get("needs_lambda_min") and lm_current is None:
        lam = res['lambda_value']
        st.warning(f"The value of λ is {frac_str(lam)}. "
                   f"Enter an auxiliary value with 0 < λ_min ≤ λ:")
        val = st.text_input("λ_min  (0 < λ_min ≤ λ)", key="input_lm")
        if st.button("Continue with this value"):
            try:
                lm = parse_frac(val)
            except Exception:
                st.error("That value is not valid. Use a fraction (e.g. 1/100) or a decimal (e.g. 0.01).")
                st.stop()
            if not (0 < lm <= lam):
                st.error(f"The value must satisfy 0 < λ_min ≤ {frac_str(lam)}.")
                st.stop()
            st.session_state.lambda_min = lm
            res = compute_negation(A, X, Y, lambda_min=lm)   # resume the computation
        else:
            st.stop()
 
    if st.session_state.get("lambda_min") is not None:
        st.info(f"Auxiliary value used: λ_min = {frac_str(st.session_state.lambda_min)}")
 
    # ===== Main result =====
    st.subheader("Result")
    st.caption(f"Case: **{res['case']}**")
    N = res['N']
    st.latex(
        r"N[X,Y]=N\left(\left[" + tex(X_fr[0]) + "," + tex(X_fr[1]) + r"\right],"
        r"\left[" + tex(Y_fr[0]) + "," + tex(Y_fr[1]) + r"\right]\right)=\left(\left["
        + tex(N[0][0]) + "," + tex(N[0][1]) + r"\right],"
        r"\left[" + tex(N[1][0]) + "," + tex(N[1][1]) + r"\right]\right)"
    )
 
    if res['case'].startswith('degenerate'):
        st.info("Degenerate case: both intervals are single points (l1 + l2 = 0).")
        st.stop()
 
    # ===== Step-by-step details (collapsed by default) =====
    st.subheader("Computation details")
 
    def row_of(pairs):
        """Display a row of 'label = value' pairs in equal-width columns."""
        cols = st.columns(len(pairs))
        for c, (label, value) in zip(cols, pairs):
            c.markdown(f"**{label}** = {frac_str(value)}")
 
    with st.expander("Input interval quantities"):
        row_of([("α₁", res['alpha_1']), ("α₂", res['alpha_2']), ("α", res['alpha'])])
        row_of([("K1", res['K1']), ("K2", res['K2']), ("A₁(X,Y)", res['A1XY'])])
        row_of([("l1", res['l1']), ("l2", res['l2']), ("d", res['d'])])
        row_of([("q1", res['q1']), ("q2", res['q2']), ("l", res['l'])])
 
    with st.expander("Case parameters  (m, u, p)"):
        row_of([("m", res['m']), ("u", res['u']), ("p", res['p'])])
        st.markdown(f"**p formula used:** `{res['p_formula']}`")
 
    with st.expander("Computation of λ  (t, lmax, λ, U)"):
        row_of([("t1", res['t1']), ("t2", res['t2']), ("t3", res['t3']), ("t4", res['t4'])])
        row_of([("lmax", res['lmax']), ("λ", res['lamda'])])
        st.markdown(f"**U** = ([{frac_str(res['U'][0][0])}, {frac_str(res['U'][0][1])}], "
                    f"[{frac_str(res['U'][1][0])}, {frac_str(res['U'][1][1])}])")
 
    with st.expander("Vectors B, C, D  (BCD, cl)"):
        b = res['bcd']
        st.latex(r"A(X,Y)=" + col_vec([res['A1XY']] * 4) + r"+l\cdot" + col_vec(b))
        row_of([("cl", res['cl'])])
 
    with st.expander("Computation of h  (hmax, M, V, A(Z,T), ch)"):
        row_of([("hmax", res['hmax'])])
        row_of([("h1", res['h1']), ("h2", res['h2'])])
        row_of([("M1", res['M1']), ("M2", res['M2'])])
        e = res['eta']
        V_vals = [res['V'][0][0], res['V'][0][1], res['V'][1][0], res['V'][1][1]]
        s_vec = [res['s1'], res['s2'], res['s3'], res['s4']]
        st.latex(r"V=" + col_vec(V_vals) + r"=" + col_vec([res['A1ZT']] * 4) + r"+h_{max}\cdot" + col_vec(s_vec))
        st.latex(r"A(Z,T)=" + col_vec([res['A1ZT']] * 4) + r"+h\cdot" + col_vec(e))
        row_of([("ch", res['ch'])])
 
    if 'Delta' in res:
        with st.expander("Computation of μ  (γ, δ, Δ, Θ, g, R, λ', μ', control)"):
            row_of([("γ", res['gamma']), ("δ", res['delta'])])
            row_of([("Δ", res['Delta']), ("Θ", res['Theta'])])
            row_of([("Θ/Δ", res['g']), ("Δ/Θ", res['g1'])])
            row_of([("R", res['R_']), ("control", res.get('control'))])
            row_of([("λ'", res['lamda_prime']), ("μ'", res['mu_prime'])])
            if 'Ss' in res:
                row_of([("S", res['Ss']), ("λ_min", res['lambda_min'])])
 
    with st.expander("μ and result"):
        row_of([("μ", res['mu']), ("h", res['mu'] * res['hmax'])])
        st.latex(
            r"N[X,Y]=\left(\left[" + tex(N[0][0]) + "," + tex(N[0][1]) + r"\right],"
            r"\left[" + tex(N[1][0]) + "," + tex(N[1][1]) + r"\right]\right)"
        )
 
    with st.expander("Additional quantities"):
        row_of([("lm1", res['lm1']), ("lm2", res['lm2']), ("lm3", res['lm3']), ("lm4", res['lm4'])])
        row_of([("hm1", res['hm1']), ("hm2", res['hm2']), ("hm3", res['hm3']), ("hm4", res['hm4'])])
        row_of([("hmi", res['hmi']), ("hmii", res['hmii']), ("hmiii", res['hmiii']), ("hmiv", res['hmiv'])])
        row_of([("hmax1", res['hmax1']), ("hmax2", res['hmax2'])])
 
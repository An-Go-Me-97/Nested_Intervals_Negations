/* ============================================================================
 *  Supplementary material for the manuscript
 *      "<ARTICLE TITLE>"  (<AUTHORS>),
 *      submitted to Fuzzy Sets and Systems.
 *  <-- replace the title/authors line above before submission -->
 *
 *  Interactive computation of the interval negation N(X, Y).
 *
 *  Given a 4x4 row-stochastic matrix A and two intervals X = [x1, x2] and
 *  Y = [x3, x4], this application computes the negation N(X, Y) using exact
 *  rational arithmetic, and reports every intermediate quantity of its
 *  construction so that each step can be inspected and reproduced.
 *
 *  This is a JavaScript port of the original Python/Streamlit program: the
 *  numerical results are identical.  It runs entirely in the browser; simply
 *  open the accompanying .html file in any modern web browser (no server or
 *  installation required).  The only external dependency is KaTeX
 *  (katex.min.js / katex.min.css), used for rendering the mathematics.
 *
 *  Files:
 *    Nested_Intervals_Negation.html  -- page markup
 *    Nested_Intervals_Negation.css   -- styles
 *    Nested_Intervals_Negation.js    -- all the logic (this file)
 * ========================================================================= */

/* === Exact rational arithmetic (frac.js) === */

"use strict";
// Exact rational arithmetic (BigInt), replicating Python's fractions.Fraction
// and Fraction.limit_denominator, including the float/Fraction interplay.

function bgcd(a, b) { a = a < 0n ? -a : a; b = b < 0n ? -b : b;
  while (b) { [a, b] = [b, a % b]; } return a; }
function bfloordiv(n, d) { // floor division for BigInt, d > 0
  let q = n / d, r = n % d; if (r !== 0n && (r < 0n)) q -= 1n; return q; }
function babs(a) { return a < 0n ? -a : a; }

class Frac {
  constructor(n, d = 1n) {
    n = BigInt(n); d = BigInt(d);
    if (d === 0n) throw new Error("zero denominator");
    if (d < 0n) { n = -n; d = -d; }
    const g = bgcd(n, d) || 1n;
    this.n = n / g; this.d = d / g;
  }
  static fromDouble(x) {
    if (!Number.isFinite(x)) throw new Error("non-finite");
    if (Number.isInteger(x)) return new Frac(BigInt(x), 1n);
    const buf = new DataView(new ArrayBuffer(8));
    buf.setFloat64(0, x);
    const bits = buf.getBigUint64(0);
    const sign = (bits >> 63n) & 1n;
    let exp = Number((bits >> 52n) & 0x7ffn);
    let mant = bits & 0xfffffffffffffn;
    if (exp === 0) { exp = -1074; } else { mant |= 0x10000000000000n; exp -= 1075; }
    let num = mant, den = 1n;
    if (exp >= 0) num <<= BigInt(exp); else den <<= BigInt(-exp);
    if (sign) num = -num;
    return new Frac(num, den);
  }
  limitDenominator(maxDen = 1000000n) {
    maxDen = BigInt(maxDen);
    if (maxDen < 1n) throw new Error("max_denominator<1");
    if (this.d <= maxDen) return new Frac(this.n, this.d);
    let p0 = 0n, q0 = 1n, p1 = 1n, q1 = 0n;
    let n = this.n, d = this.d;
    while (true) {
      const a = bfloordiv(n, d);
      const q2 = q0 + a * q1;
      if (q2 > maxDen) break;
      [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
      [n, d] = [d, n - a * d];
    }
    const k = bfloordiv(maxDen - q0, q1);
    const b1 = new Frac(p0 + k * p1, q0 + k * q1);
    const b2 = new Frac(p1, q1);
    return (b2.sub(this).abs().cmp(b1.sub(this).abs()) <= 0) ? b2 : b1;
  }
  add(o) { o = F.cast(o); return new Frac(this.n * o.d + o.n * this.d, this.d * o.d); }
  sub(o) { o = F.cast(o); return new Frac(this.n * o.d - o.n * this.d, this.d * o.d); }
  mul(o) { o = F.cast(o); return new Frac(this.n * o.n, this.d * o.d); }
  div(o) { o = F.cast(o); return new Frac(this.n * o.d, this.d * o.n); }
  neg() { return new Frac(-this.n, this.d); }
  abs() { return new Frac(babs(this.n), this.d); }
  cmp(o) { o = F.cast(o); const L = this.n * o.d, R = o.n * this.d; return L < R ? -1 : (L > R ? 1 : 0); }
  eq(o) { return this.cmp(o) === 0; }
  lt(o){return this.cmp(o)<0;} le(o){return this.cmp(o)<=0;} gt(o){return this.cmp(o)>0;} ge(o){return this.cmp(o)>=0;}
  toNumber() { return Number(this.n) / Number(this.d); }
  toString() { return this.d === 1n ? this.n.toString() : this.n.toString() + "/" + this.d.toString(); }
}

// Return x as an exact rational, rounded to the nearest fraction whose
// denominator does not exceed 1_000_000 (removes floating-point noise).
// Equivalent to Python's  Fraction(x).limit_denominator(1_000_000).
function F(x) {
  const fr = (x instanceof Frac) ? x : Frac.fromDouble(x);
  return fr.limitDenominator(1000000n);
}
F.cast = (o) => (o instanceof Frac) ? o : (typeof o === "bigint" ? new Frac(o, 1n)
                 : (Number.isInteger(o) ? new Frac(BigInt(o), 1n) : Frac.fromDouble(o)));



/* === Core routine: compute_negation (core.js) === */

"use strict";
// ---------------------------------------------------------------------------
//  Core routine: interval negation N(X, Y) in exact rational arithmetic.
//  This is a faithful JavaScript port of the original Python implementation.
//  Floating-point (IEEE-754 double) is used exactly where the Python code
//  used numpy floats, and exact rational arithmetic (the Frac class) is used
//  exactly where the Python code used fractions.Fraction, so the results are
//  bit-for-bit identical to the Python program.
// ---------------------------------------------------------------------------

const ONE = new Frac(1n), ZERO = new Frac(0n), TWO = new Frac(2n), HALF = new Frac(1n, 2n);
const fmax = Math.max, fmin = Math.min, fabs = Math.abs;
// min / max of a list of Frac values
function fminFrac(...a) { return a.reduce((m, x) => x.lt(m) ? x : m); }
function fmaxFrac(...a) { return a.reduce((m, x) => x.gt(m) ? x : m); }

// Compute the interval negation N(X, Y) for the row-stochastic matrix A and
// the intervals X = [x1, x2], Y = [x3, x4], using exact rational arithmetic.
// Returns an object holding N together with every intermediate quantity.  If
// the construction reaches control == 1 and `lambda_min` is not supplied, the
// function returns early with res.needs_lambda_min = true and
// res.lambda_value = lambda.
//
//   A          : 4x4 array of Numbers (floats), the row-stochastic matrix.
//   X, Y       : [[Number, Number]], the input intervals as floats.
//   lambda_min : Frac | null, the auxiliary value (used only when control==1).
//   chTol      : when true, the selection of `ch` uses the tolerance
//                abs(x) > 1e-9 (Atanassov program); when false it uses
//                x !== 0 (nested-intervals program).
function computeNegation(A, X, Y, lambda_min = null, chTol = false) {
  const res = {};

  const alpha_1 = F(A[0][1] / (A[0][0] + A[0][1]));
  const alpha_2 = F(A[0][3] / (A[0][2] + A[0][3]));
  const alpha   = F(A[0][2] + A[0][3]);
  const oma1 = ONE.sub(alpha_1), oma2 = ONE.sub(alpha_2), oma = ONE.sub(alpha);
  const K1 = F(oma1.toNumber() * X[0][0] + alpha_1.toNumber() * X[0][1]);
  const K2 = F(oma2.toNumber() * Y[0][0] + alpha_2.toNumber() * Y[0][1]);
  const l  = F(fmax(X[0][1], Y[0][1]) - fmin(X[0][0], Y[0][0]));
  const l1 = F(X[0][1] - X[0][0]), l2 = F(Y[0][1] - Y[0][0]);
  Object.assign(res, { alpha_1, alpha_2, alpha, K1, K2, l, l1, l2 });

  // Degenerate case: both intervals reduce to a single point.
  if (l1.add(l2).eq(ZERO)) {
    const z = F(X[0][0]);
    res.case = 'degenerate (l1+l2=0)'; res.A1XY = z;
    res.N = [[ONE.sub(z), ONE.sub(z)], [ONE.sub(z), ONE.sub(z)]];
    res.Z = [ONE.sub(z), ONE.sub(z)]; res.T = [ONE.sub(z), ONE.sub(z)];
    return res;
  }

  const q1 = F(l1.div(l)), q2 = F(l2.div(l));
  Object.assign(res, { q1, q2 });

  // Three cases according to the order of K1 and K2.
  let caseName, A1XY, d, u, sign_B, pA_first;
  if (K1.lt(K2)) {
    caseName = 'K1<K2'; A1XY = F(oma.mul(K1).add(alpha.mul(K2))); d = F(K2.sub(K1));
    u = F((Y[0][1] - X[0][0]) / l.toNumber()); sign_B = 1; pA_first = true;
  } else if (K2.lt(K1)) {
    caseName = 'K2<K1'; A1XY = F(alpha.mul(K2).add(oma.mul(K1))); d = F(K1.sub(K2));
    u = F((X[0][1] - Y[0][0]) / l.toNumber()); sign_B = -1; pA_first = false;
  } else {
    caseName = 'K1==K2'; A1XY = F(alpha.mul(K2).add(oma.mul(K1))); d = F(K1.sub(K2));
    u = F(fmax((X[0][1] - Y[0][0]) / l.toNumber(), (Y[0][1] - X[0][0]) / l.toNumber()));
    sign_B = 1; pA_first = u.eq(Frac.fromDouble(Y[0][1] - X[0][0]));
  }

  const m = F(d.div(l));
  Object.assign(res, { case: caseName, A1XY, d, m, u });

  const t1 = F((X[0][0] - A1XY.toNumber()) / l.toNumber());
  const t2 = F((X[0][1] - A1XY.toNumber()) / l.toNumber());
  const t3 = F((Y[0][0] - A1XY.toNumber()) / l.toNumber());
  const t4 = F((Y[0][1] - A1XY.toNumber()) / l.toNumber());

  // Keep v if it lies in [0, 1]; otherwise discard it (return 2).
  const clamp_unit = (v) => (v.lt(ZERO) || v.gt(ONE)) ? TWO : v;
  const lm1 = t1.eq(ZERO) ? TWO : clamp_unit(F(A1XY.neg().div(t1)));
  const lm2 = t2.eq(ZERO) ? TWO : clamp_unit(F(ONE.sub(A1XY).div(t2)));
  const lm3 = t3.eq(ZERO) ? TWO : clamp_unit(F(A1XY.neg().div(t3)));
  const lm4 = t4.eq(ZERO) ? TWO : clamp_unit(F(ONE.sub(A1XY).div(t4)));
  const lmax = F(fminFrac(lm1, lm2, lm3, lm4)); const lamda = F(l.div(lmax));
  const u1 = A1XY.add(lmax.mul(t1)), u2 = A1XY.add(lmax.mul(t2));
  const u3 = A1XY.add(lmax.mul(t3)), u4 = A1XY.add(lmax.mul(t4));
  Object.assign(res, { t1, t2, t3, t4, lm1, lm2, lm3, lm4, lmax, lamda, u1, u2, u3, u4 });

  // B, C, D are matrix-vector products A . v, computed in float (as numpy did).
  const mv = (vec) => [0,1,2,3].map(i =>
    A[i][0]*vec[0] + A[i][1]*vec[1] + A[i][2]*vec[2] + A[i][3]*vec[3]);
  const B = mv([alpha.neg().toNumber(), alpha.neg().toNumber(), oma.toNumber(), oma.toNumber()]);
  const C = mv([alpha_1.neg().toNumber(), oma1.toNumber(), 0, 0]);
  const D = mv([0, 0, alpha_2.neg().toNumber(), oma2.toNumber()]);
  const mN = m.toNumber(), q1N = q1.toNumber(), q2N = q2.toNumber();
  const BCD = [0,1,2,3].map(i => sign_B*mN*B[i] + q1N*C[i] + q2N*D[i]);
  const cl = F(BCD.find(x => fabs(x) > 1e-9));
  const bcd = BCD.map(x => F(x));
  Object.assign(res, { cl, bcd });

  // Selection of p: two candidates; keep the first one lying in [0, 1].
  const pA = u.sub(oma1.mul(q2)).sub(alpha_2.mul(q1));
  const pB = u.sub(alpha_1.mul(q2)).sub(oma2.mul(q1));
  const p_formula_A = "p = u - (1 - alpha_1) q2 - alpha_2 q1";
  const p_formula_B = "p = u - alpha_1 q2 - (1 - alpha_2) q1";
  const order        = pA_first ? [pA, pB] : [pB, pA];
  const order_formula = pA_first ? [p_formula_A, p_formula_B] : [p_formula_B, p_formula_A];
  let p = order[0], p_formula = order_formula[0];
  if (p.lt(ZERO) || p.gt(ONE)) { p = order[1]; p_formula = order_formula[1]; }
  const pB_used = (p_formula === p_formula_B);   // true -> case M1 <= M2 (branch hm1..hm4)
  res.p = p; res.p_formula = p_formula;

  // Safe division: returns 2 (out-of-range marker) when den == 0.
  const hdiv = (nu, de) => de.eq(ZERO) ? TWO : F(nu.div(de));

  // Candidate h_max values for the case M1 <= M2 (branch hm1..hm4).
  let hm1 = hdiv(ONE.sub(A1XY), alpha.mul(p).add(alpha_1.mul(q2)));
  let hm2 = hdiv(A1XY,          alpha.neg().mul(p).add(oma1.mul(q2)));
  let hm3 = hdiv(ONE.sub(A1XY), oma.neg().mul(p).add(alpha_2.mul(q1)));
  let hm4 = hdiv(A1XY,          oma.mul(p).add(oma2.mul(q1)));
  // Candidate h_max values for the case M2 <= M1 (branch hmi..hmiv).
  let hmi   = hdiv(ONE.sub(A1XY), alpha.neg().mul(p).add(alpha_1.mul(q2)));
  let hmii  = hdiv(A1XY,          alpha.mul(p).add(oma1.mul(q2)));
  let hmiii = hdiv(ONE.sub(A1XY), oma.mul(p).add(alpha_2.mul(q1)));
  let hmiv  = hdiv(A1XY,          oma.neg().mul(p).add(oma2.mul(q1)));

  // Discard the candidates that fall outside [0, 1].
  const disc = (v) => (v.ge(ZERO) && v.le(ONE)) ? v : TWO;
  hm1 = disc(hm1); hm2 = disc(hm2); hm3 = disc(hm3); hm4 = disc(hm4);
  hmi = disc(hmi); hmii = disc(hmii); hmiii = disc(hmiii); hmiv = disc(hmiv);

  const hmax_c1 = F(fminFrac(hm1, hm2, hm3, hm4));      // candidate for the case M1 <= M2
  const hmax_c2 = F(fminFrac(hmi, hmii, hmiii, hmiv));  // candidate for the case M2 <= M1
  const hmax1 = (hmax_c1.ge(ZERO) && hmax_c1.le(ONE)) ? hmax_c1 : -1;
  const hmax2 = (hmax_c2.ge(ZERO) && hmax_c2.le(ONE)) ? hmax_c2 : -1;

  // The branch is chosen by HOW p was selected (p == pB  <=>  M1 <= M2),
  // not by the numerical value of u - p (which the two branches can share).
  // If the branch dictated by the theory leaves [0, 1] (its h_max is -1),
  // the other branch is used instead.
  const case_M1_le_M2 = pB_used;
  let use_branch_1;
  if (case_M1_le_M2) use_branch_1 = (hmax1 !== -1);
  else use_branch_1 = (hmax2 === -1);

  if (hmax1 === -1 && hmax2 === -1)
    res.hmax_warning = "Neither hmax1 nor hmax2 lies in [0, 1]; check the data.";

  let hmax, M1, M2;
  if (use_branch_1) {
    hmax = hmax1;
    M1 = F(ONE.sub(A1XY).sub(alpha.mul(p).mul(hmax))); M2 = F(p.mul(hmax).add(M1));
  } else {
    hmax = hmax2;
    M2 = F(ONE.sub(A1XY).sub(oma.mul(p).mul(hmax))); M1 = F(M2.add(p.mul(hmax)));
  }
  Object.assign(res, { hm1, hm2, hm3, hm4, hmi, hmii, hmiii, hmiv,
    hmax1, hmax2, hmax, M1, M2, case_M1_le_M2 });

  const v1 = F(M1.sub(alpha_1.mul(q2).mul(hmax))), v2 = F(v1.add(q2.mul(hmax)));
  const v3 = F(M2.sub(alpha_2.mul(q1).mul(hmax))), v4 = F(v3.add(q1.mul(hmax)));
  const s1 = F(v1.sub(ONE).add(A1XY).div(hmax)), s2 = F(v2.sub(ONE).add(A1XY).div(hmax));
  const s3 = F(v3.sub(ONE).add(A1XY).div(hmax)), s4 = F(v4.sub(ONE).add(A1XY).div(hmax));
  Object.assign(res, { v1, v2, v3, v4, s1, s2, s3, s4 });

  const sN = [s1.toNumber(), s2.toNumber(), s3.toNumber(), s4.toNumber()];
  const eta = [0,1,2,3].map(i => A[i][0]*sN[0] + A[i][1]*sN[1] + A[i][2]*sN[2] + A[i][3]*sN[3]);
  const ch = chTol ? F(eta.find(x => fabs(x) > 1e-9)) : F(eta.find(x => x !== 0));
  res.ch = ch; res.eta = eta.map(x => F(x));

  let control = 0, mu;
  if (cl.mul(ch).lt(ZERO)) {
    mu = lamda;
  } else {
    const tmin = fminFrac(t1, t2, t3, t4), tmax = fmaxFrac(t1, t2, t3, t4);
    const smin = fminFrac(s1, s2, s3, s4), smax = fmaxFrac(s1, s2, s3, s4);
    const gamma = F(tmin.neg().div(tmax.sub(tmin)));
    const delta = F(smin.neg().div(smax.sub(smin)));
    const Delta = F(fmaxFrac(gamma, ONE.sub(gamma)).add(fmaxFrac(delta, ONE.sub(delta))).div(TWO));
    const Theta = F(fminFrac(gamma, ONE.sub(gamma)).add(fminFrac(delta, ONE.sub(delta))).div(TWO));
    let g, g1;
    if (!Delta.eq(Theta)) {
      g = F(Theta.div(Delta)); g1 = F(Delta.div(Theta));
    } else {
      const mp = fmaxFrac(m, p);
      if (mp.gt(ZERO)) { g = F(mp); g1 = F(ONE.div(mp)); }
      else { g = null; g1 = null; res.warning = 'check: Delta == Theta and max(m, p) <= 0'; }
    }
    const Rr = F(HALF.mul(g1.sub(g)).div(ONE.sub(g)));
    const r_DT = Delta.div(Theta);   // Delta / Theta
    const r_TD = Theta.div(Delta);   // Theta / Delta
    let lamda_prime = F(lamda.div(Rr)), mu_prime;
    if (lamda_prime.le(HALF)) {
      control = lamda.lt(F(Rr.mul(Rr).sub(r_DT))) ? 1 : 0;
      mu_prime = F(Rr.mul(g).sub(g.mul(lamda_prime)));
    } else {
      control = lamda.lt(F(Rr.mul(Rr).mul(r_TD).sub(r_DT))) ? 1 : 0;
      mu_prime = F(Rr.sub(g1.mul(lamda_prime)));
    }
    mu = F(mu_prime.mul(Rr));
    Object.assign(res, { gamma, delta, Delta, Theta, g, g1, R_: Rr,
      lamda_prime, mu_prime, control });

    if (control === 1) {
      if (lambda_min === null) {
        res.needs_lambda_min = true; res.lambda_value = lamda;
        return res;
      }
      const Ss = F(r_DT.add(lambda_min).div(Rr));
      lamda_prime = F(lamda.div(Ss));
      mu_prime = lamda_prime.lt(HALF) ? F(Rr.mul(g).sub(g.mul(lamda_prime)))
                                      : F(Rr.sub(g1.mul(lamda_prime)));
      mu = F(mu_prime.mul(Ss));
      Object.assign(res, { Ss, lamda_prime, mu_prime, lambda_min });
    }
  }

  const z1 = ONE.sub(A1XY).add(mu.mul(hmax).mul(s1)), z2 = ONE.sub(A1XY).add(mu.mul(hmax).mul(s2));
  const z3 = ONE.sub(A1XY).add(mu.mul(hmax).mul(s3)), z4 = ONE.sub(A1XY).add(mu.mul(hmax).mul(s4));
  Object.assign(res, { mu, N: [[z1, z2], [z3, z4]], Z: [z1, z2], T: [z3, z4], control });
  res.A1ZT = ONE.sub(A1XY); res.h1 = q2.mul(hmax); res.h2 = q1.mul(hmax);
  res.U = [[u1, u2], [u3, u4]]; res.V = [[v1, v2], [v3, v4]];
  res.min_lower = fminFrac(u1, u3); res.max_upper = fmaxFrac(u2, u4);
  return res;
}



/* === Browser UI helpers (ui_common.js) === */

/* ===========================================================================
 *  Browser UI helpers (shared by both programs).
 *  These replace the Streamlit front-end of the original Python programs.
 *  All mathematics is rendered with KaTeX; Greek and fraktur symbols are kept
 *  as symbols (exactly as in the original interface).
 * ======================================================================== */

// Parse user input ('1/6', '0.5', '1') into an exact Fraction (Frac).
// Mirrors Python's Fraction(str(text).strip()) for the supported forms.
function parseFrac(text) {
  let s = String(text).trim();
  if (s.length === 0) throw new Error("empty");
  if (s.includes("/")) {                      // "a/b"
    const parts = s.split("/");
    return new Frac(BigInt(parts[0].trim()), BigInt(parts[1].trim()));
  }
  if (s.includes(".")) {                       // exact decimal "0.5" -> 1/2
    let sign = 1n;
    if (s[0] === "-") { sign = -1n; s = s.slice(1); }
    else if (s[0] === "+") { s = s.slice(1); }
    const dot = s.indexOf(".");
    let intp = s.slice(0, dot) || "0";
    let frac = s.slice(dot + 1);
    const num = BigInt((intp + frac) || "0") * sign;
    const den = 10n ** BigInt(frac.length);
    return new Frac(num, den);
  }
  return new Frac(BigInt(s), 1n);              // integer
}

// Format a Fraction as LaTeX: 'a' or '\frac{a}{b}' ('?' for null/undefined).
function tex(fr) {
  if (fr === null || fr === undefined) return "?";
  if (typeof fr === "number") return String(fr);   // out-of-range sentinel (e.g. 2, -1)
  if (fr.d === 1n) return fr.n.toString();
  const sign = fr.n < 0n ? "-" : "";
  const absn = fr.n < 0n ? -fr.n : fr.n;
  return sign + "\\frac{" + absn.toString() + "}{" + fr.d.toString() + "}";
}

// Format a Fraction as plain text, e.g. '3/5' ('?' for null/undefined).
function frac_str(fr) {
  if (fr === null || fr === undefined) return "?";
  if (fr instanceof Frac) return fr.toString();
  return String(fr);
}

// Build a LaTeX column vector from a list of values.
function col_vec(values) {
  return "\\begin{pmatrix}" + values.map(tex).join("\\\\[6pt]") + "\\end{pmatrix}";
}

// Format a matrix (array of rows of fractions) as a LaTeX pmatrix.
function mat_tex(M) {
  return "\\begin{pmatrix}"
    + M.map(row => row.map(tex).join(" & ")).join("\\\\[4pt]")
    + "\\end{pmatrix}";
}

// Render a display-mode LaTeX string into an HTML block.
function latexBlock(s) {
  return '<div class="math">' + katex.renderToString(s, { displayMode: true, throwOnError: false }) + '</div>';
}

// A row of "label = value" pairs in equal-width columns (like Streamlit row_of).
function row_of(pairs) {
  let html = '<div class="row">';
  for (const [label, value] of pairs) {
    html += '<div class="cell"><b>' + label + '</b> = ' + frac_str(value) + '</div>';
  }
  return html + '</div>';
}

// Simple message boxes.
function box(kind, html) { return '<div class="box ' + kind + '">' + html + '</div>'; }

// Read the 4x4 matrix of text inputs into an array of Frac rows.
function readMatrix() {
  const A = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      row.push(parseFrac(document.getElementById("a" + i + j).value));
    }
    A.push(row);
  }
  return A;
}

// Load an example matrix into the input grid.
function loadExample(name) {
  const M = EXAMPLES[name];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      document.getElementById("a" + i + j).value = M[i][j];
}


/* === Program-specific UI (nested intervals) === */

/* ---------------------------------------------------------------------------
 *  Predefined example matrices (entered as text fractions).
 * ------------------------------------------------------------------------ */
const EXAMPLES = {
  "Example 1": [["1/3","1/3","1/6","1/6"],["1/3","1/6","1/3","1/6"],["1/3","1/6","1/6","1/3"],["1/2","1/6","1/6","1/6"]],
  "Example 2": [["1/2","1/6","1/6","1/6"],["1/6","1/2","1/6","1/6"],["1/6","1/6","1/2","1/6"],["1/6","1/6","1/6","1/2"]],
  "Example 3": [["1/6","1/3","1/3","1/6"],["1/6","1/3","1/6","1/3"],["1/6","1/6","1/3","1/3"],["1/3","1/6","1/3","1/6"]],
  "Example 4": [["1/2","1/6","1/6","1/6"],["1/3","1/6","1/3","1/6"],["1/6","1/3","1/6","1/3"],["1/6","1/6","1/6","1/2"]],
};

// Auxiliary-parameter state (the two-step lambda_min flow).
const state = { lambda_min: null, inputs: null };

// Called by the "Compute" button: read and store the inputs, reset lambda_min.
function onCompute() {
  try {
    const A_fr = readMatrix();
    const X_fr = [parseFrac(document.getElementById("x1").value),
                  parseFrac(document.getElementById("x2").value)];
    const Y_fr = [parseFrac(document.getElementById("x3").value),
                  parseFrac(document.getElementById("x4").value)];
    state.inputs = { A_fr, X_fr, Y_fr };
  } catch (e) {
    render(box("error", "Some entry is not a valid number. Use fractions such as <code>1/6</code> or decimals such as <code>0.5</code>."));
    return;
  }
  state.lambda_min = null;
  runAndRender();
}

// Perform the computation and render the output.
function runAndRender() {
  const { A_fr, X_fr, Y_fr } = state.inputs;
  // Numeric (float) versions, exactly as the Python program built them.
  const A = A_fr.map(row => row.map(f => f.toNumber()));
  const X = [[X_fr[0].toNumber(), X_fr[1].toNumber()]];
  const Y = [[Y_fr[0].toNumber(), Y_fr[1].toNumber()]];

  let out = "";

  // Validation notices.
  if (X_fr[0].gt(X_fr[1]) || Y_fr[0].gt(Y_fr[1]))
    out += box("warn", "Check the intervals: a lower endpoint is greater than its upper endpoint.");
  for (let i = 0; i < 4; i++) {
    const s = A_fr[i].reduce((a, b) => a.add(b), new Frac(0n));
    if (!s.eq(new Frac(1n)))
      out += box("info", "Note: row " + (i + 1) + " of A sums to " + frac_str(s) + " (not 1).");
  }

  let res;
  try {
    res = computeNegation(A, X, Y, state.lambda_min, /*chTol=*/false);
  } catch (e) {
    out += box("error", "The computation cannot be completed with these data (division by zero at some step).");
    render(out); return;
  }

  // Echo of the input (always shown).
  const A_tex = A_fr.map(r => r.map(tex).join(" & ")).join("\\\\");
  out += latexBlock("A=\\begin{pmatrix}" + A_tex + "\\end{pmatrix}\\qquad "
    + "X=\\left[" + tex(X_fr[0]) + "," + tex(X_fr[1]) + "\\right]\\quad "
    + "Y=\\left[" + tex(Y_fr[0]) + "," + tex(Y_fr[1]) + "\\right]");

  // If the auxiliary parameter lambda_min is required, ask for it and stop.
  if (res.needs_lambda_min && state.lambda_min === null) {
    const lam = res.lambda_value;
    out += box("warn", "The value of &lambda; is " + frac_str(lam)
      + ". Enter an auxiliary value with 0 &lt; &lambda;_min &le; &lambda;:");
    out += '<div class="lmrow"><input id="lm_input" type="text" placeholder="e.g. 1/100">'
      + '<button id="lm_btn">Continue with this value</button></div>'
      + '<div id="lm_err"></div>';
    render(out);
    document.getElementById("lm_btn").onclick = function () {
      let lm;
      try { lm = parseFrac(document.getElementById("lm_input").value); }
      catch (e) { document.getElementById("lm_err").innerHTML = box("error", "That value is not valid. Use a fraction (e.g. 1/100) or a decimal (e.g. 0.01)."); return; }
      if (!(lm.gt(new Frac(0n)) && lm.le(lam))) {
        document.getElementById("lm_err").innerHTML = box("error", "The value must satisfy 0 &lt; &lambda;_min &le; " + frac_str(lam) + ".");
        return;
      }
      state.lambda_min = lm;
      runAndRender();                           // resume the computation
    };
    return;
  }

  if (state.lambda_min !== null)
    out += box("info", "Auxiliary value used: &lambda;_min = " + frac_str(state.lambda_min));

  // ===== Main result =====
  out += '<h2>Result</h2>';
  out += '<p class="caption">Case: <b>' + res.case + '</b></p>';
  const N = res.N;
  out += latexBlock("N[X,Y]=N\\left(\\left[" + tex(X_fr[0]) + "," + tex(X_fr[1]) + "\\right],"
    + "\\left[" + tex(Y_fr[0]) + "," + tex(Y_fr[1]) + "\\right]\\right)=\\left(\\left["
    + tex(N[0][0]) + "," + tex(N[0][1]) + "\\right],"
    + "\\left[" + tex(N[1][0]) + "," + tex(N[1][1]) + "\\right]\\right)");

  if (res.case.startsWith("degenerate")) {
    out += box("info", "Degenerate case: both intervals are single points (l1 + l2 = 0).");
    render(out); return;
  }

  // ===== Step-by-step details (collapsed by default) =====
  out += '<h2>Computation details</h2>';

  out += expander("Input interval quantities",
    row_of([["&alpha;&#8321;", res.alpha_1], ["&alpha;&#8322;", res.alpha_2], ["&alpha;", res.alpha]]) +
    row_of([["K1", res.K1], ["K2", res.K2], ["A&#8321;(X,Y)", res.A1XY]]) +
    row_of([["l1", res.l1], ["l2", res.l2], ["d", res.d]]) +
    row_of([["q1", res.q1], ["q2", res.q2], ["l", res.l]]));

  out += expander("Case parameters  (m, u, p)",
    row_of([["m", res.m], ["u", res.u], ["p", res.p]]) +
    '<p><b>p formula used:</b> <code>' + res.p_formula + '</code></p>');

  out += expander("Computation of &lambda;  (t, lmax, &lambda;, U)",
    row_of([["t1", res.t1], ["t2", res.t2], ["t3", res.t3], ["t4", res.t4]]) +
    row_of([["lmax", res.lmax], ["&lambda;", res.lamda]]) +
    '<p><b>U</b> = ([' + frac_str(res.U[0][0]) + ', ' + frac_str(res.U[0][1]) + '], ['
      + frac_str(res.U[1][0]) + ', ' + frac_str(res.U[1][1]) + '])</p>');

  out += expander("Vectors B, C, D  (BCD, cl)",
    latexBlock("A(X,Y)=" + col_vec([res.A1XY, res.A1XY, res.A1XY, res.A1XY]) + "+l\\cdot" + col_vec(res.bcd)) +
    row_of([["cl", res.cl]]));

  const V_vals = [res.V[0][0], res.V[0][1], res.V[1][0], res.V[1][1]];
  const s_vec = [res.s1, res.s2, res.s3, res.s4];
  out += expander("Computation of h  (hmax, M, V, A(Z,T), ch)",
    row_of([["hmax", res.hmax]]) +
    row_of([["h1", res.h1], ["h2", res.h2]]) +
    row_of([["M1", res.M1], ["M2", res.M2]]) +
    latexBlock("V=" + col_vec(V_vals) + "=" + col_vec([res.A1ZT, res.A1ZT, res.A1ZT, res.A1ZT]) + "+h_{max}\\cdot" + col_vec(s_vec)) +
    latexBlock("A(Z,T)=" + col_vec([res.A1ZT, res.A1ZT, res.A1ZT, res.A1ZT]) + "+h\\cdot" + col_vec(res.eta)) +
    row_of([["ch", res.ch]]));

  if ("Delta" in res) {
    let body =
      row_of([["&gamma;", res.gamma], ["&delta;", res.delta]]) +
      row_of([["&Delta;", res.Delta], ["&Theta;", res.Theta]]) +
      row_of([["&Theta;/&Delta;", res.g], ["&Delta;/&Theta;", res.g1]]) +
      row_of([["R", res.R_], ["control", res.control]]) +
      row_of([["&lambda;'", res.lamda_prime], ["&mu;'", res.mu_prime]]);
    if ("Ss" in res) body += row_of([["S", res.Ss], ["&lambda;_min", res.lambda_min]]);
    out += expander("Computation of &mu;  (&gamma;, &delta;, &Delta;, &Theta;, g, R, &lambda;', &mu;', control)", body);
  }

  out += expander("&mu; and result",
    row_of([["&mu;", res.mu], ["h", res.mu.mul(res.hmax)]]) +
    latexBlock("N[X,Y]=\\left(\\left[" + tex(N[0][0]) + "," + tex(N[0][1]) + "\\right],"
      + "\\left[" + tex(N[1][0]) + "," + tex(N[1][1]) + "\\right]\\right)"));

  out += expander("Additional quantities",
    row_of([["lm1", res.lm1], ["lm2", res.lm2], ["lm3", res.lm3], ["lm4", res.lm4]]) +
    row_of([["hm1", res.hm1], ["hm2", res.hm2], ["hm3", res.hm3], ["hm4", res.hm4]]) +
    row_of([["hmi", res.hmi], ["hmii", res.hmii], ["hmiii", res.hmiii], ["hmiv", res.hmiv]]) +
    row_of([["hmax1", res.hmax1], ["hmax2", res.hmax2]]));

  render(out);
}

function expander(title, bodyHtml) {
  return '<details><summary>' + title + '</summary><div class="exp">' + bodyHtml + '</div></details>';
}
function render(html) { document.getElementById("output").innerHTML = html; }

// Wire up the controls once the page has loaded.
window.addEventListener("DOMContentLoaded", function () {
  loadExample("Example 2");                    // default matrix (as in the original)
  document.getElementById("compute").onclick = onCompute;
  document.getElementById("load").onclick = function () {
    loadExample(document.getElementById("example").value);
  };
});


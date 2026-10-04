// Just enough dense linear algebra to train a linear readout: ridge regression.
// Matrices are row-major Float64Arrays.

// In-place Cholesky factorisation of a symmetric positive-definite n×n matrix.
// Only the lower triangle is read and written. Returns false if not SPD.
export function cholesky(A, n) {
  for (let j = 0; j < n; j++) {
    const rj = j * n;
    let d = A[rj + j];
    for (let k = 0; k < j; k++) d -= A[rj + k] * A[rj + k];
    if (!(d > 0)) return false;
    const djj = Math.sqrt(d);
    A[rj + j] = djj;
    const inv = 1 / djj;
    for (let i = j + 1; i < n; i++) {
      const ri = i * n;
      let s = A[ri + j];
      for (let k = 0; k < j; k++) s -= A[ri + k] * A[rj + k];
      A[ri + j] = s * inv;
    }
  }
  return true;
}

// Solve L Lᵀ X = B for X, where L is the lower Cholesky factor (n×n) and
// B is n×m (row-major). B is overwritten with the solution.
export function choleskySolve(L, n, B, m) {
  // Forward: L Y = B
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let c = 0; c < m; c++) {
      let s = B[i * m + c];
      for (let k = 0; k < i; k++) s -= L[ri + k] * B[k * m + c];
      B[i * m + c] = s / L[ri + i];
    }
  }
  // Backward: Lᵀ X = Y
  for (let i = n - 1; i >= 0; i--) {
    for (let c = 0; c < m; c++) {
      let s = B[i * m + c];
      for (let k = i + 1; k < n; k++) s -= L[k * n + i] * B[k * m + c];
      B[i * m + c] = s / L[i * n + i];
    }
  }
  return B;
}

// Solve (A + λI) X = B for symmetric PSD A (n×n, full or lower triangle filled).
// Adds extra jitter if the factorisation fails for numerical reasons.
export function solveRegularized(A, n, B, m, lambda) {
  let jitter = lambda;
  for (let attempt = 0; attempt < 8; attempt++) {
    const M = new Float64Array(A);
    for (let i = 0; i < n; i++) M[i * n + i] += jitter;
    if (cholesky(M, n)) return choleskySolve(M, n, new Float64Array(B), m);
    let trace = 0;
    for (let i = 0; i < n; i++) trace += A[i * n + i];
    jitter = Math.max(jitter * 10, 1e-10 * (trace / n || 1));
  }
  throw new Error('Ridge system could not be factorised');
}

// Streams training pairs (feature f, target y) and accumulates FᵀF and FᵀY,
// so the design matrix never has to be stored. Only the lower triangle of FᵀF
// is accumulated, which halves the work.
export class RidgeAccumulator {
  constructor(nFeat, nOut) {
    this.nFeat = nFeat;
    this.nOut = nOut;
    this.FtF = new Float64Array(nFeat * nFeat);
    this.FtY = new Float64Array(nFeat * nOut);
    this.count = 0;
  }

  add(f, y) {
    const n = this.nFeat;
    const m = this.nOut;
    const FtF = this.FtF;
    const FtY = this.FtY;
    for (let i = 0; i < n; i++) {
      const fi = f[i];
      if (fi === 0) continue;
      const ri = i * n;
      for (let j = 0; j <= i; j++) FtF[ri + j] += fi * f[j];
      const ro = i * m;
      for (let c = 0; c < m; c++) FtY[ro + c] += fi * y[c];
    }
    this.count++;
  }

  // Returns Wout as an nOut×nFeat row-major matrix, so y = Wout · f.
  solve(lambda) {
    const n = this.nFeat;
    const m = this.nOut;
    const W = solveRegularized(this.FtF, n, this.FtY, m, lambda * Math.max(1, this.count));
    return transpose(W, n, m);
  }
}

export function transpose(A, rows, cols) {
  const T = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) T[c * rows + r] = A[r * cols + c];
  return T;
}

// Ridge regression from an explicit design matrix X (S×F) and targets Y (S×C).
// Uses the dual form (S×S system) when there are fewer samples than features,
// which is the usual case for few-shot classification. Returns W (C×F).
export function ridgeSolve(X, S, F, Y, C, lambda) {
  const lam = lambda * S;
  if (S < F) {
    // W = Xᵀ (X Xᵀ + λI)⁻¹ Y
    const K = new Float64Array(S * S);
    for (let i = 0; i < S; i++) {
      const ri = i * F;
      for (let j = 0; j <= i; j++) {
        const rj = j * F;
        let s = 0;
        for (let k = 0; k < F; k++) s += X[ri + k] * X[rj + k];
        K[i * S + j] = s;
        K[j * S + i] = s;
      }
    }
    const alpha = solveRegularized(K, S, Y, C, lam); // S×C
    const W = new Float64Array(C * F);
    for (let s = 0; s < S; s++) {
      const rs = s * F;
      for (let c = 0; c < C; c++) {
        const a = alpha[s * C + c];
        if (a === 0) continue;
        const rc = c * F;
        for (let k = 0; k < F; k++) W[rc + k] += a * X[rs + k];
      }
    }
    return W;
  }
  const acc = new RidgeAccumulator(F, C);
  for (let s = 0; s < S; s++) acc.add(X.subarray(s * F, (s + 1) * F), Y.subarray(s * C, (s + 1) * C));
  const W = solveRegularized(acc.FtF, F, acc.FtY, C, lam);
  return transpose(W, F, C);
}

// y = W · f for W (rows×cols).
export function matVec(W, rows, cols, f, out = new Float64Array(rows)) {
  for (let r = 0; r < rows; r++) {
    const rr = r * cols;
    let s = 0;
    for (let k = 0; k < cols; k++) s += W[rr + k] * f[k];
    out[r] = s;
  }
  return out;
}

// Top-k principal directions of row vectors in X (S×F) by power iteration
// with deflation. Returns { mean, components: Float64Array[k] }.
export function pca(X, S, F, k = 2, iters = 60) {
  const mean = new Float64Array(F);
  for (let s = 0; s < S; s++) for (let j = 0; j < F; j++) mean[j] += X[s * F + j];
  for (let j = 0; j < F; j++) mean[j] /= S || 1;
  const comps = [];
  const v = new Float64Array(F);
  const tmp = new Float64Array(S);
  for (let c = 0; c < k; c++) {
    for (let j = 0; j < F; j++) v[j] = Math.sin(j * 1.7 + c * 3.1 + 0.3);
    for (let it = 0; it < iters; it++) {
      for (let s = 0; s < S; s++) {
        let d = 0;
        for (let j = 0; j < F; j++) d += (X[s * F + j] - mean[j]) * v[j];
        tmp[s] = d;
      }
      v.fill(0);
      for (let s = 0; s < S; s++) for (let j = 0; j < F; j++) v[j] += tmp[s] * (X[s * F + j] - mean[j]);
      for (const p of comps) {
        let d = 0;
        for (let j = 0; j < F; j++) d += v[j] * p[j];
        for (let j = 0; j < F; j++) v[j] -= d * p[j];
      }
      let norm = 0;
      for (let j = 0; j < F; j++) norm += v[j] * v[j];
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < F; j++) v[j] /= norm;
    }
    comps.push(new Float64Array(v));
  }
  return { mean, components: comps };
}

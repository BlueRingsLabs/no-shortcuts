# Module 02: Linear Algebra

*Part I: Foundations · about 45 hours*

Every model in this course is, at the bottom, a pile of matrix multiplications. Let's understand them.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is the geometric meaning of the rank of a matrix?
- Why should you almost never call `np.linalg.inv` to solve a linear system?
- What do the singular values of a data matrix tell you?

## When you finish it, you can

- Manipulate vectors, norms, dot products and angles, and use them as similarity measures.
- Treat matrices as linear maps; reason about rank, null space, inverses and conditioning.
- Derive least squares as an orthogonal projection and solve it stably (QR, lstsq) instead of inverting matrices.
- Compute and interpret eigendecompositions and the SVD, and use the SVD for low-rank approximation.

## Lessons

02.1. [Vectors, dot products and norms](02.1-vectors-norms.md)

02.2. [Matrices as functions: multiplication, rank, inverses](02.2-matrices-as-maps.md)

02.3. [Least squares and projections](02.3-least-squares.md)

02.4. [Eigendecomposition and the SVD](02.4-eigen-svd.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/02-linear-algebra/labs/lab_02_1_vectors.py
```

- [`lab_02_1_vectors.py`](labs/lab_02_1_vectors.py)
- [`lab_02_2_matrices.py`](labs/lab_02_2_matrices.py)
- [`lab_02_3_least_squares.py`](labs/lab_02_3_least_squares.py)
- [`lab_02_4_eigen_svd.py`](labs/lab_02_4_eigen_svd.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)

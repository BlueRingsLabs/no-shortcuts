# Module 03: Calculus and Optimization

*Part I: Foundations · about 40 hours*

Training a model is minimizing a function. This module is about how you minimize functions, and why it sometimes doesn't work.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is the gradient of f(w) = ||Xw - y||^2 with respect to w?
- Why does gradient descent zig-zag in a narrow valley?

## When you finish it, you can

- Compute derivatives, partial derivatives and gradients of the functions used in ML, including matrix expressions.
- Apply the chain rule on computational graphs, which is all backpropagation is.
- Use Jacobians, Hessians and Taylor expansions to reason about local behaviour and curvature.
- Implement gradient descent, reason about step sizes and convexity, and verify gradients numerically.
- Solve constrained problems with Lagrange multipliers and read KKT conditions.

## Lessons

03.1. [Derivatives, gradients and the chain rule](03.1-derivatives-gradients.md)

03.2. [Jacobians, Hessians and Taylor: the local picture](03.2-jacobians-hessians.md)

03.3. [Gradient descent and convexity](03.3-gradient-descent-convexity.md)

03.4. [Constrained optimization: Lagrange multipliers and KKT](03.4-constrained-optimization.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/03-calculus-optimization/labs/lab_03_1_gradients.py
```

- [`lab_03_1_gradients.py`](labs/lab_03_1_gradients.py)
- [`lab_03_2_jacobians_hessians.py`](labs/lab_03_2_jacobians_hessians.py)
- [`lab_03_3_gradient_descent.py`](labs/lab_03_3_gradient_descent.py)
- [`lab_03_4_constrained.py`](labs/lab_03_4_constrained.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)

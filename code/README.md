# Companion code

Every miniature from the addendum, with the test that checks it against the real library.

| Chapter | Miniature | Check |
|---|---|---|
| C. NumPy | `mini_numpy.py` | `python test_mini_numpy.py` |
| D. pandas | `mini_pandas.py` | `python test_mini_pandas.py` |
| E. scikit-learn | `sk/mini_sklearn.py` | `cd sk && python test_mini_sklearn.py` |
| F. PyTorch | `torch_ch/mini_torch.py` | `cd torch_ch && python test_mini_torch.py` |
| G. sentence-transformers | `st/mini_st.py` | `cd st && python test_mini_st.py` (downloads `all-MiniLM-L6-v2`, 90.9 MB) |

The other files are the scripts that produced the outputs quoted in the chapters:

- chapter B: `ch1_probe.py`, `ch1_traceback.py`;
- chapter C: `np_demos.py`, `np_probe.py`;
- chapter D: `pd_probe.py`;
- chapter E: `sk/naive.py`, `sk/step2.py`, `sk/step3.py` and `sk/run_checks.py` (the estimator of each stage, graded by `check_estimator`), `sk/init_bad.py`, `sk/leakage.py`, `sk/grid.py`;
- chapter F: `torch_ch/train_demo.py`, `torch_ch/pitfalls.py`;
- chapter G: `st/fifteen.py`, `st/demos.py`, `st/unicode_diff.py`.

Install the versions in `../requirements.txt` first, as the main README explains.

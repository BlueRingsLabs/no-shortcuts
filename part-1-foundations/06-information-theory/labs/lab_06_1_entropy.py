# %% [markdown]
# # Lab 06.1: Bits, losses and divergences
#
# 1. Entropy as optimal code length: build a Huffman code and compare its average length with H(p).
# 2. Cross-entropy loss = negative log-likelihood = KL + constant.
# 3. KL is not symmetric; forward vs reverse KL fitting a Gaussian to a bimodal mixture.
# 4. The Gaussian KL closed form vs Monte Carlo.
# 5. Perplexity, and why it depends on the tokenizer.

# %%
import heapq
from collections import Counter

import numpy as np
from scipy import stats
from scipy.optimize import minimize

rng = np.random.default_rng(0)


def H(p, base=2):
    p = np.asarray(p, float)
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(base)


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return np.sum(p[m] * np.log(p[m] / q[m]))

# %% [markdown]
# ## 1. Huffman coding approaches the entropy

# %%
def huffman_lengths(probs):
    heap = [(p, i, (i,)) for i, p in enumerate(probs)]
    heapq.heapify(heap)
    lengths = np.zeros(len(probs), int)
    counter = len(probs)
    while len(heap) > 1:
        p1, _, s1 = heapq.heappop(heap)
        p2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1              # every merge adds one bit to the codes below it
        heapq.heappush(heap, (p1 + p2, counter, s1 + s2))
        counter += 1
    return lengths


dyadic = [0.5, 0.25, 0.125, 0.125]
L = huffman_lengths(dyadic)
assert np.isclose(np.dot(dyadic, L), H(dyadic)) and np.isclose(H(dyadic), 1.75)

text = ("it is a truth universally acknowledged that a single man in possession of a good fortune "
        "must be in want of a wife ") * 20
counts = Counter(text)
p_chars = np.array(list(counts.values())) / len(text)
L = huffman_lengths(p_chars)
avg = np.dot(p_chars, L)
print(f"characters: H = {H(p_chars):.3f} bits, Huffman average {avg:.3f} bits, ASCII 8 bits")
assert H(p_chars) <= avg < H(p_chars) + 1, "Huffman is within 1 bit of the entropy"

print(f"H(fair die) = {H([1/6] * 6):.3f}, H(0.9 coin) = {H([0.9, 0.1]):.3f} bits")

# %% [markdown]
# ## 2. Cross-entropy loss, NLL and KL

# %%
n, K = 5000, 4
p_true = np.array([0.1, 0.2, 0.3, 0.4])
y = rng.choice(K, size=n, p=p_true)
q_model = np.array([0.15, 0.25, 0.25, 0.35])
nll = -np.mean(np.log(q_model[y]))
p_hat = np.bincount(y, minlength=K) / n
assert np.isclose(nll, -np.sum(p_hat * np.log(q_model)))                 # NLL = H(p_hat, q)
assert np.isclose(nll, kl(p_hat, q_model) + H(p_hat, base=np.e))          # = KL + entropy
# the best model is the empirical distribution itself
assert -np.sum(p_hat * np.log(p_hat)) < nll
print(f"NLL {nll:.4f} nats = KL {kl(p_hat, q_model):.4f} + H(data) {H(p_hat, np.e):.4f}")

assert np.isclose(-np.log(0.2), 1.6094, atol=1e-4) and np.isclose(-np.log(0.7), 0.3567, atol=1e-4)

# %% [markdown]
# ## 3. Asymmetry, and forward vs reverse KL
#
# p = mixture of N(-3, 1) and N(3, 1). Fit a single Gaussian q by minimizing each direction of KL on a grid.

# %%
p2, q2 = [0.5, 0.5], [0.9, 0.1]
print(f"KL(p||q) = {kl(p2, q2):.3f}   KL(q||p) = {kl(q2, p2):.3f}")
assert np.isclose(kl(p2, q2), 0.511, atol=1e-3) and np.isclose(kl(q2, p2), 0.368, atol=1e-3)

x = np.linspace(-10, 10, 4001)
dx = x[1] - x[0]
p_mix = 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)


def gauss(params):
    mu, log_s = params
    return stats.norm.pdf(x, mu, np.exp(log_s)) + 1e-300


def forward_kl(params):
    return np.sum(p_mix * np.log((p_mix + 1e-300) / gauss(params))) * dx


def reverse_kl(params):
    q = gauss(params)
    return np.sum(q * np.log(q / (p_mix + 1e-300))) * dx


fwd = minimize(forward_kl, x0=[0.5, 0.0], method="Nelder-Mead").x
rev = minimize(reverse_kl, x0=[0.5, 0.0], method="Nelder-Mead").x
print(f"forward KL fit: mu={fwd[0]:+.2f}, sd={np.exp(fwd[1]):.2f}   (covers both modes)")
print(f"reverse KL fit: mu={rev[0]:+.2f}, sd={np.exp(rev[1]):.2f}   (locks onto one mode)")
assert abs(fwd[0]) < 0.1 and np.exp(fwd[1]) > 2.8                  # mean 0, sd = sqrt(1 + 9) ~ 3.16
assert abs(abs(rev[0]) - 3) < 0.1 and abs(np.exp(rev[1]) - 1) < 0.1

# %% [markdown]
# ## 4. Gaussian KL closed form vs Monte Carlo

# %%
mu, s = 1.3, 0.6
closed = 0.5 * (mu**2 + s**2 - np.log(s**2) - 1)
z = rng.normal(mu, s, 1_000_000)
mc = np.mean(stats.norm.logpdf(z, mu, s) - stats.norm.logpdf(z, 0, 1))
print(f"KL(N({mu},{s}^2) || N(0,1)): closed form {closed:.4f}, Monte Carlo {mc:.4f}")
assert abs(closed - mc) < 0.005

# %% [markdown]
# ## 5. Perplexity depends on what a "token" is
#
# Same text, same (unigram) modelling approach, two tokenizations: characters and words. The total number of nats to
# encode the text is what's comparable; per-token perplexities are not.

# %%
def unigram_stats(tokens):
    c = Counter(tokens)
    probs = np.array([c[t] / len(tokens) for t in tokens])
    nats = -np.sum(np.log(probs))
    return np.exp(nats / len(tokens)), nats


ppl_char, nats_char = unigram_stats(list(text))
ppl_word, nats_word = unigram_stats(text.split(" "))
print(f"char-level: perplexity {ppl_char:.1f}, total {nats_char:.0f} nats   "
      f"word-level: perplexity {ppl_word:.1f}, total {nats_word:.0f} nats")
assert ppl_word > ppl_char and nats_word < nats_char
print("higher per-token perplexity, fewer total nats: per-token numbers aren't comparable across tokenizers")

# %%
print("\nAll checks passed.")

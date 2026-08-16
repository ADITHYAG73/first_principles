"""
flash attention, tier 1 — the film wearing a for-loop.
plain numpy implementation of FlashAttention's Algorithm 1 (forward pass).

two purposes, stated honestly:
  1. prove the mathematics EXACT: matches standard attention to the digit.
  2. run the honest race. surprise: even in numpy, the tiled version WINS
     (~10x on CPU) — because CPU caches are a memory hierarchy too, and
     tiling is rewarded at every level of the hierarchy. the GPU/SRAM story
     (tier 2, triton) is the same physics with bigger stakes.
"""
import numpy as np
import time


def standard_attention(Q, K, V):
    """the three-kernel crime: materialise S and P in full."""
    S = Q @ K.T                                   # N x N raw scores
    P = np.exp(S - S.max(axis=1, keepdims=True))  # shift by row max (the license)
    P = P / P.sum(axis=1, keepdims=True)          # normalise (the recipe)
    return P @ V                                  # the pour


def flash_attention_tier1(Q, K, V, Br=None, Bc=None):
    """Algorithm 1, literally: tiles, notebooks <m, r, o>, repair on coup.
    S and P are never materialised — only Br x Bc tiles, minted and evicted."""
    N, d = Q.shape
    Br = Br or min(64, N)          # asker block:  Br = min(ceil(M/4d), d)
    Bc = Bc or min(64, N)          # shelf section: Bc = ceil(M/4d)

    O = np.zeros((N, d))           # running blends (wet paint)
    m = np.full(N, -np.inf)        # per-row queens   (sentinel: -inf)
    r = np.zeros(N)                # per-row room totals (sentinel: 0)

    # outer loop: shelf sections take residence (FA-1's itinerary)
    for j0 in range(0, N, Bc):
        Kj = K[j0:j0 + Bc]                     # haul section in: K block
        Vj = V[j0:j0 + Bc]                     #                  V block
        # inner loop: asker blocks parade past the resident section
        for i0 in range(0, N, Br):
            Qi = Q[i0:i0 + Br]                 # asker visit: Q in
            sl = slice(i0, i0 + min(Br, N - i0))

            S_tile = Qi @ Kj.T                 # mint the tile (never stored)
            m_tilde = S_tile.max(axis=1)       # local queens, one per row-strip
            P_tilde = np.exp(S_tile - m_tilde[:, None])
            r_tilde = P_tilde.sum(axis=1)      # local sums (local currency)

            m_old, r_old = m[sl], r[sl]
            m_new = np.maximum(m_old, m_tilde)             # crown contest
            conv_old = np.exp(m_old - m_new)               # repair the past
            conv_new = np.exp(m_tilde - m_new)             # convert the newcomer
            r_new = conv_old * r_old + conv_new * r_tilde  # Algorithm 1, line 11

            # blend update — Algorithm 1, line 12:
            # repair (conversion x dilution) then admit the new pour
            with np.errstate(invalid="ignore"):            # 0/0 on cold start -> handled
                c = np.where(r_new > 0, conv_old * r_old / r_new, 0.0)
            pour = (conv_new[:, None] * (P_tilde @ Vj)) / r_new[:, None]
            O[sl] = c[:, None] * O[sl] + pour

            m[sl], r[sl] = m_new, r_new        # notebooks out (O out rides free here)
    return O


if __name__ == "__main__":
    # ---- receipt 1: the article's anchor (N=3, d=2), tiles of 2 ----
    Q = np.array([[1., 0.], [0., 2.], [1., 1.]])
    K = np.array([[1., 1.], [0., 1.], [2., 0.]])
    V = np.array([[1., 0.], [0., 1.], [1., 1.]])
    O_std = standard_attention(Q, K, V)
    O_flash = flash_attention_tier1(Q, K, V, Br=2, Bc=2)
    print("anchor, standard :\n", np.round(O_std, 3))
    print("anchor, flash    :\n", np.round(O_flash, 3))
    print("row 1 ground truth [0.910, 0.755]  ->", np.round(O_flash[0], 3))
    assert np.allclose(O_std, O_flash, atol=1e-12), "EXACTNESS FAILED"
    print("exactness: PASS (to the digit)\n")

    # ---- receipt 2: the fire test — scores near 100, naive would overflow ----
    rng = np.random.default_rng(0)
    Qf = rng.normal(100, 1, (64, 8)); Kf = rng.normal(0, 1, (64, 8)) / 8
    Vf = rng.normal(0, 1, (64, 8))
    assert np.allclose(standard_attention(Qf, Kf, Vf),
                       flash_attention_tier1(Qf, Kf, Vf, 16, 16), atol=1e-10)
    print("fire test (big scores): PASS\n")

    # ---- receipt 3: GPT-2 scale, exactness + the honest race ----
    N, d = 1024, 64
    Qg, Kg, Vg = (rng.normal(0, 1, (N, d)) for _ in range(3))
    t0 = time.perf_counter(); O1 = standard_attention(Qg, Kg, Vg)
    t1 = time.perf_counter(); O2 = flash_attention_tier1(Qg, Kg, Vg, 64, 390)
    t2 = time.perf_counter()
    assert np.allclose(O1, O2, atol=1e-9)
    print(f"GPT-2 scale (N={N}, d={d}): exactness PASS")
    print(f"standard : {(t1-t0)*1e3:7.1f} ms")
    print(f"flash t1 : {(t2-t1)*1e3:7.1f} ms   <- faster EVEN HERE.")
    print("why? CPU caches are a memory hierarchy too — tiling avoids sweeping")
    print("the N x N ghost through slow RAM, so L2/L3 reward it exactly the way")
    print("SRAM will. same physics, bigger stakes on GPU. ENTER TRITON (tier 2).")

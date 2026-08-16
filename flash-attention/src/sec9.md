# Receipts, confessions, and the road ahead

We have come to the end of the derivations. How about we put these ideas to the test and see them live?? That would be pretty cool!!

Let's implement our algorithm in plain python — the $\langle m, r, o \rangle$ loops, literally the film wearing a for-loop. And watch two things happen: it matches standard attention **to the digit** (our film table is literally the test fixture)... And — honest surprise, I expected it to lose — it already runs ~10x faster even in numpy. Why? Because CPU caches are a memory hierarchy too, and tiling is rewarded at every level of the hierarchy: the tiled version never sweeps the N x N ghost through slow RAM. The GPU story is the same physics with far bigger stakes — SRAM control that numpy simply cannot express.

<code: tier-1 numpy implementation — to be added>

So you will ask: it matches, but where is the speedup you promised? We need infrastructure that gives us *control of the memory hierarchy*. ENTER TRITON — a python-embedded language where you decide what lives in SRAM. (fun fact from this article's own story: Triton's creator, Phil Tillet, is the same person who first flipped the flash attention loops — the flip we re-derived in section 6.)

<link: public kaggle notebook — triton kernel + benchmark, so you can run the speedup yourself — to be added>

<widget: the measured benchmark — my own version of the paper's Fig. 2 — to be added>

## Honesty time — the confession list

This article walked past several things ON PURPOSE, and you deserve to know what and why:

1. **the backward pass, fully** — we derived recomputation (the soul), and I quoted the gradient formulas with shapes checked, but the complete derivation (the softmax jacobian and all four gradient flows) lives in my notes. It deserves its own article, and it will get one.

2. **the paper's formal proofs** — we *manufactured* theorem 2 with a pencil, but the paper also proves a lower bound: no exact attention algorithm can beat this asymptotically across all SRAM sizes. We counted; they proved. Worth reading in the original.

3. **block-sparse flash attention** — the paper's own approximate extension (its section 3.3). We stayed exact-only, deliberately.

4. **causal masking and dropout** — algorithm 2's extra lines. Mechanical once you own the forward pass, so I leave them as an exercise to the reader (chuckles).

5. **flash attention 2, 3 and 4** — remember the loop flip we discovered ourselves? That is literally FA-2's opening move. Next stop on this staircase.

6. **MQA, GQA and the economics of inference** — that is the KV-cache article's territory. And its coming.

## The staircase

This article stands on the first one — matmuls, couriers, and the roofline <link: free-norm article> — and borrowed its whole vocabulary. The next flight is already visible: **KV cache**, then **mixture of experts**. Same method every time: smallest countable case, pencil first, no borrowed formulas.

This was a great journey personally — highly rewarding in terms of learning. See you at the next one!!

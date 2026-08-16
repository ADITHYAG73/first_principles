# Receipts, confessions, and the road ahead

We have come to the end of the derivations. How about we put these ideas to the test and see them live?? That would be pretty cool!!

Let's implement our algorithm in plain python — the $\langle m, r, o \rangle$ loops, literally the film wearing a for-loop. And watch two things happen: it matches standard attention **to the digit** (our film table is literally the test fixture)... And — honest surprise, I expected it to lose — it already runs ~10x faster even in numpy. Why? Because CPU caches are a memory hierarchy too, and tiling is rewarded at every level of the hierarchy: the tiled version never sweeps the N x N ghost through slow RAM. The GPU story is the same physics with far bigger stakes — SRAM control that numpy simply cannot express.

<code: tier-1 numpy implementation — to be added>

So you will ask: it matches, but where is the saving you promised? numpy cannot show you, because numpy cannot *see* the bench. We need infrastructure that gives us *control of the memory hierarchy*. ENTER TRITON — a python-embedded language where you decide what lives in SRAM. (fun fact from this article's own story: Triton's creator, Phil Tillet, is the same person who first flipped the flash attention loops — the flip we re-derived in section 6.)

<link: public kaggle notebook — triton kernel + benchmark, so you can run the speedup yourself — to be added>

<widget: the measured benchmark — my own version of the paper's Fig. 2 — to be added>

## Reading our own receipt

Everything above was measured on a Kaggle T4, $d = 64$, fp16 in and fp32 on the accumulator. Three receipts, in rising order of brutality.

**Receipt one — it is still exactly attention.** Largest disagreement with the materialise-everything version, across the whole sweep: $3.7 \times 10^{-4}$. That is fp16 rounding, not a different answer. We reordered the arithmetic and changed nothing about the result — which was the whole promise in section 1.

**Receipt two — the courier bill, and this one is the article.** At $N = 32{,}768$ the naive version needs **10,261 MiB** to hold its ghost. Ours needs **25 MiB**. That is a **410×** difference, and look at the shape of it: naive climbs quadratically, ours crawls up linearly — 10 MiB at $N=1024$, 25 MiB at $N=32{,}768$. This is the $N^2$ versus $N^2d^2/M$ story of section 7, showing up on a real machine, in megabytes you can read off a meter.

**Receipt three — the wall, exactly where we said it was.** At $N = 65{,}536$ the naive version does not get slower. It *dies* — the GPU refuses to allocate $S$ at all. Our kernel answers the same question in 882 ms using 49 MiB. Section 4's wall 1 is not a metaphor; it is a hardware error message.

### And now the confession, because the clock is the one receipt I did not get

Wall-clock, ours versus torch: **0.25× at $N=1024$, climbing to 0.88× at $N=32{,}768$.** Read that honestly — our kernel is still *slower*, by about 14% at the long end. The line climbs beautifully and never quite crosses.

I could have hidden this by benchmarking against a deliberately clumsy baseline. I would rather tell you what the gap actually is, because the gap is *interesting*.

Two things are in it, and I measured which is which. **The first is an empty GPU.** With one head, our grid is $N/B_r$ programs on 40 multiprocessors — at $N=1024$ that is 16 programs on a 40-lane road. Run 12 heads instead, as a real transformer does, and the ratio at $N=1024$ jumps from 0.25× to 0.46×. But do the same at $N = 16{,}384$ and it moves by $-0.008$ — nothing. So the empty grid is a *small-$N$* story only, and it is gone by the time the sequences get long.

**The second is simply that I am one person and cuBLAS is not.** The thing we are racing is not a naive baseline in the compute sense: `Q @ K.T` is a hand-tuned assembly GEMM and `torch.softmax` is a fused kernel, both polished for a decade. When I let the autotuner pipeline our loads — `num_stages = 4`, so the courier fetches the next shelf section while the bench is still working on the current one — our kernel got **2.4× faster** without a single line of the algorithm changing. The algorithm was never the slow part. The *engineering* was, and that engineering is a career, not a section 9.

So: the memory receipts are ours, measured, and they are the argument. The clock belongs to the library engineers, and on production kernels the paper's own Fig. 2 shows what the same idea does when those engineers get hold of it — 41.7 ms down to 7.3 ms. What you derived with a pencil is real. What you have not derived is a decade of kernel tuning, and you should not pretend otherwise.

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

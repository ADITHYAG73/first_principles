# The scaffold — tiles, rent, and who commutes

In the last section we found the theme: REPAIR IS THE PRICE OF REFUSAL TO WAIT. We illustrated it on a single row, one number at a time. In reality, hardware is **batch-o-philic** — it always yearns for batches. If you have read my previous article on GEMM vs free-norm <link: article 1>, you are familiar with blockwise matmul: $Q$ and $K$ are giants, so what GPUs actually compute is

$$S_{\text{tile}} = Q_{\text{tile}}\, K_{\text{tile}}^\top$$

A whole *tile* of scores minted at once — several askers' scores against one section of keys.

Does our notebook survive group arrivals? Absolutely yes — because all said and done, **softmax is and will always be a row-wise operation**, and a tile is nothing but several rows stacked together. Each asker-row of the tile is one asker's *installment* of scores, and each such row-strip does its own private bookkeeping the moment its minted: a **local queen** $\tilde{m}_i$ (the strip's max) and a **local sum** $\tilde{r}_i$ (the strip's mini-denominator, shifted by its local queen). The strip then presents itself to that asker's notebook as one composite newcomer, and the deposit converts by the *same* law we already own — currency conversion, second costume:

$$m_{\text{new}} = \max(m_{\text{old}},\, \tilde{m}) \qquad\qquad r_{\text{new}} = e^{\,m_{\text{old}} - m_{\text{new}}}\, r_{\text{old}} \;+\; e^{\,\tilde{m} - m_{\text{new}}}\, \tilde{r}$$

Elegance check: a "block" of one element has $\tilde{m} = s$ and $\tilde{r} = e^0 = 1$, and the formula collapses to exactly last section's law. **one rule for all** — the game we played was the general case in disguise. And the repair scalar $c$ for the running blends? Unchanged, applied per row. Tiles arrive as rectangles; queens rule rows.

> **notebox — where is P?** careful: what the notebook holds are *provisional weights under the current regime* — what P would be **if the row ended here**. The finished P is never minted anywhere, forward pass or otherwise. It exists only implicitly, inside $\langle m, r \rangle$. Remember this; it becomes the hero of the backward-pass story.

## The rent arithmetic

Tiles solved the *arrival* question. Now the *size* question — how big can a tile be? Not a style choice: the bench decides.

During one visit, the bench must simultaneously hold four block tenants:

$$Q_{\text{tile}}: B_r \times d \qquad K_{\text{tile}}: B_c \times d \qquad V_{\text{tile}}: B_c \times d \qquad O_{\text{tile}}: B_r \times d$$

Start with the square toy — one knob, $B_r = B_c = B$. Total rent $4Bd$, bench capacity $M$:

$$4Bd = M \qquad \Rightarrow \qquad B = \left\lceil \frac{M}{4d} \right\rceil$$

(why ceiling and not floor? Don't read generosity into it — these are sizing *heuristics*, not physical caps. $M$ itself is an approximate budget, real kernels round block sizes to friendly powers of two, and the last ragged block is handled separately. The ceiling just guarantees a nonzero block.)

Plug in the A100-class numbers: $M = 100{,}000$, $d = 64$ → $B \approx 390$ rows per block. Do the division yourself — its your first rent calculation.

**and now the audit that breaks the square plan.** we forgot a fifth tenant: the minted score tile itself, $B_r \times B_c$. At $B = 390$: $390 \times 390 \approx 152{,}000$ numbers — **bigger than the entire bench**, before the four slabs even sit down. Bankrupt.

So something must shrink — and here is the crucial point: nothing gets *truncated*. We never cut a minted tile; we choose the block shapes **before minting**. And the side that gets capped is the **asker side**:

$$B_c = \left\lceil \frac{M}{4d} \right\rceil \approx 390 \qquad\qquad B_r = \min\!\left(\left\lceil \frac{M}{4d} \right\rceil,\; d\right) = 64$$

Only 64 askers shop at a time, while the shelf of keys stays long at 390. Why cap the askers and not the shelf? Two reasons, and the second is the punchline. First: the shelf *wants* to be long — a long shelf means fewer sections, fewer rounds, and (as the schedule below shows) fewer re-hauls. Second: at $B_r = d$, look at the tile's rent —

$$B_r \times B_c = 64 \times 390 = 390 \times 64 = B_c \times d$$

**the tile's rent exactly equals one slab's rent** — the same as a K-block. The quadratic tenant, tamed to slab class. Five tenants, one rent tier, everything fits. That's not "smaller" — that's the *principled* size.

<widget: the bench — drag $B_r$ and watch the tile explode past the budget, then snap it to $B_r = d$ and watch it tamed>

## The schedule — who commutes?

Now imagine the machine running. Toy world: $N = 4$ tokens, $d = 2$, blocks of 2 — so two shelf sections (K,V rows 1–2, then rows 3–4) and two asker blocks (Q rows 1–2 with their O rows and notebooks, then rows 3–4).

The paper ships **one** itinerary <cite: Algorithm 1 — outer loop over K,V blocks, inner over Q blocks>: shelf sections take residence one at a time, and the asker blocks parade past each resident section.


And here is the anatomy that decides everything. Watch one asker visit: the block hauls in its Q rows (the questionnaires), hauls in its O rows, works, and — the crucial part — **hauls its O rows back OUT**. Why does O ride both ways? Because O is *wet paint*: half-finished blends, mid-repair, that must survive eviction (the next asker block needs the bench) and return next round for more repair. An asker visit costs three trucks: Q in, O in, O out. A shelf section, by contrast, is read-only merchandise — K in, V in, nothing to write back. **two trucks.**

Tally the toy world's bill and the O-traffic dominates the entire schedule — the wet paint's round-trips are the single biggest line item.

Now the question the paper didn't ask: *what if the roles swapped?* let an asker block take residence, and parade the shelf sections past it:


Count again — and the bill drops. Because now **the wet paint never travels**: a resident asker's blend stays on the bench through every parade, dries, and ships out exactly once, finished. The O round-trips vanish. The verdict, one sentence: **the light traveler should commute.** the shelf carries two trucks a visit; the askers carry three; FA-1 sat the light party and commuted the heavy one.

And if you just felt clever deriving that — you re-derived history. This exact flip was first implemented by Phil Tillet in Triton and then formalised as **FlashAttention-2** <cite: FA-2 paper; Tillet provenance>. One honesty note though: the traffic saving you just counted is real but constant-factor; FA-2's *headline* motive was parallelism — with askers outer, different blocks run on different compute units sharing nothing. That story is next-article territory.

<widget: the itinerary comic — flip through the schedule page by page, then verify the timetables urself>

So the machine is now fully assembled: tiles minted, notebooks running per row, blends repairing in place — and the full $S$ never exists anywhere. Next section: count what this machine actually hauls, and collect the prize.

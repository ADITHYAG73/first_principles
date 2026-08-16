# The theorem — and where it stops working

Recall from the crime scene <link: sec3>: standard attention's movement bill was

$$T_{\text{std}} = 4Nd + 4N^2$$

In the last section we built the tiled machine and its schedule. Moment of truth — did all these trials and tribulations actually reduce the bill? Let's measure.

**the shelf's haul:** every K,V row visits the bench exactly once → $2Nd$. Cheap, irreducible-class.

**the askers' commute** (the inner loop, in programming terms <cite: confirmed against Algorithm 1's structure>): one full parade of Q costs $Nd$. And how many times must the parade run? Once per shelf section — $N/B_c$ rounds. So

$$\text{commute} = \frac{N}{B_c} \times Nd = \frac{N^2 d}{B_c}$$

Now recall the rent: $B_c = M/4d$. Substitute — do it yourself, its one line:

$$\text{commute} = \frac{N^2 d}{M/4d} = \frac{4\,N^2 d^2}{M}$$

Hola!! We got the cost down. Who is the saviour? Of course — its $M$. We could not kill the quadratic term, but we **divided it by the bench**. A bigger bench now means cheaper postage — a lever that simply did not exist before. And this expression — $\Theta(N^2 d^2 / M)$ — is the paper's Theorem 2 <cite: exact statement at citation pass>. **you did not read that theorem. You just manufactured it** from rent arithmetic and a timetable.

(what about O? It commutes alongside Q — your own visit anatomy from last section, O in and O out — which multiplies the *constant*, not the shape. Same $N^2d^2/M$ species; the paper's $\Theta$ absorbs both. We derive with Q for cleanliness and bill honestly in the footnote.)

## The two ledgers meet

At GPT-2 scale ($N = 1024$, $d = 64$, $M = 100{,}000$), set the bills side by side — and be honest about the ladder:

- commute term alone vs the crime term: **~26× less traffic**
- add the shelf haul and the honest constants: **~15×**
- the paper's *measured* number on real silicon: **9×** <cite: Fig. 2>

26 → 15 → 9: each step toward reality adds overhead — and the ordering itself is the lesson. Our pencil count brackets the measurement from above, exactly as a first-principles count should.

<widget: two ledgers — slide N and watch the gap>

## The breakeven — where the bench stops saving you

So is flash attention always a win? Form the ratio of flash's commute against standard's crime term:

$$\frac{4N^2d^2/M}{4N^2} = \frac{d^2}{M}$$

The $N$'s cancel — read that again. The ratio is **independent of sequence length**. Whether flash wins is a pure property of head size versus bench size, the same at every context length. Flash wins when $d^2/M < 1$, I.e.

$$d < \sqrt{M} \approx 316$$

That's the wall. Now walk the real world along the curve:

- **GPT-2 heads**, $d = 64$: ratio 0.04 — flash hauls 4% of the crime it replaced
- **Llama-class heads**, $d = 128$: ratio 0.16 — still a thumping win
- **the Gemma family**, $d = 256$: ratio 0.66 — flirting with the wall <cite: HF config, Google architecture blog>
- **Gemma 4's global layers**, $d = 512$: ratio 2.6 — **crossed the wall in march 2026, and promptly lost flash attention entirely.** every FA kernel caps at head_dim 256; those layers fall back to slower attention <cite: Dao-AILab issue, HF discussions>. The wall we derived with a pencil is roughly where the real ecosystem stops building kernels.
- and the one-fat-head fantasy, $d = 3072$: ratio ≈ 94 — flash would haul ninety-four crimes' worth. Nobody builds this. Now you know why.

<widget: the breakeven curve — hover any d>

Which lands the closing insight, and its my favourite in this article: **multi-head attention is flash attention's secret ally.** the 2017 design split $d_{\text{model}}$ into many small heads (for expressiveness reasons); the 2022 trick wins precisely and only in the small-$d$ regime. Neither team planned for the other. Deep structure compounds — no prophets required.

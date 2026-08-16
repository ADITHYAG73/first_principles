# Anatomy of standard attention — the crime scene

By now you are familiar: the journey to $O$ passes through $S$ and $P$. Now let's watch how that journey actually runs on hardware — and count what it costs.

Your inputs $Q, K, V$ all reside in **HBM** (High Bandwidth Memory) — the GPU's large main memory. Large, but *slow relative to where the work happens*. The work happens on the **SRAM workbench** — tiny (a few hundred KB per compute unit) but blazing fast. And between the two runs the **courier** — the memory machinery hauling numbers back and forth. One law governs this whole article: *compute on the bench is nearly free; courier trips are the expensive currency.* <cite: paper §2.1 — A100: HBM 40 GB at 1.5 TB/s, SRAM ~19 TB/s>

(readers of my first article <link: free-norm> will recognise this cast — warehouse, bench, courier. If you are new: that article derives why this economy works the way it does; here we just use it.)

Standard attention runs as **three separate kernels** — three separate programs, and nothing survives on the bench between programs. Watch the courier:

1. Haul $Q$ and $K$ in from HBM to compute $S$ — **in: $2Nd$**
2. Write $S$ to HBM — **out: $N^2$**
3. Load $S$ back from HBM to compute $P$ — **in: $N^2$**
4. Write $P$ to HBM — **out: $N^2$**
5. Haul $P$ and $V$ in from HBM — **in: $N^2 + Nd$**
6. Write $O$ to HBM — **out: $Nd$**

(why wasn't $V$ fetched back in step 1? Because kernel 3 is a *different program* — the bench was wiped clean twice in between. That wipe IS the crime, as you are about to see.)

Total movement bill:

$$T_{\text{std}} = 4Nd + 4N^2$$

<cite: this is the paper's Theorem — standard attention requires ❤22❤ HBM accesses; pin exact section at citation pass>

Now — how much of this is IRREDUCIBLE? The inputs must come in ($Q + K + V = 3Nd$) and the answer must go out ($O = Nd$). So $4Nd$ is the fundamental cost of the operation, non-negotiable.

Everything else — the entire $4N^2$ — is **ghost traffic**: four trips that exist for no mathematical reason. Look at them again: $S$ was written out (step 2) only to be *immediately read back* (step 3). $P$ likewise (steps 4–5). The questionnaire matrix takes two full round-trips through slow memory purely because the pipeline is chopped into three programs that cannot hand objects to each other across the bench. The courier is shuttling paperwork between departments of the same office.

And at real scale the ghost dominates utterly. GPT-2 scale: $N = 1024$, $d = 64$. The ratio of ghost to legitimate is $N^2/Nd = N/d = 16$. **out of every 17 numbers the courier hauls, 16 are ghost traffic — roughly 94% of the entire bill.** the biggest thing the machine builds, it builds only to throw across the slow wire. Twice.

<widget: two-ledgers — the hatched region is the ghost>

So the sec1 paradox half-resolves: now we know *where* the 40.3 GB lives. The questions that remain: can this traffic be reduced at all? And if so — what is the tradeoff we must accept? The next two sections find out.

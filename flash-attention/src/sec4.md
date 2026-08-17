# The two walls

The dream is obvious, so let's state it: **fuse the three kernels into one.** compute $S$, launder it, pour $O$ — all in a single visit to the bench, so $S$ and $P$ live and die on the bench and never touch HBM. Traffic collapses to the irreducible $4Nd$. Why hasn't everyone always done this? Two walls.

## Wall 1 — the bench is too small

Say the bench holds $M = 100{,}000$ numbers (an A100-class budget — roughly one compute unit's SRAM in half precision; treat it as an order-of-magnitude figure, not a datasheet number). Now look at the size of $S$. At GPT-2 scale, $N = 1024$, so $S$ has $N^2 \approx 1.05$ **million** entries — roughly *ten times the entire bench*, before $Q$, $K$, $V$, $P$, $O$ ask for their seats. The biggest object in the pipeline cannot visit the workshop whole. So clearly you can't fit $S$ in — forget computing softmax on it and multiplying by $V$.

Fine — so we chunk. Store only a piece of $S$ at a time. And here wall 2 rises.

## Wall 2 — the denominator wants the whole row

Think about what softmax needs. Its denominator is $\sum_k e^{s_k}$ — it touches **every score in the row** — and even the max-shift (our fireproofing license from earlier) needs the row's max, which you cannot know until you have seen the whole row.

And here is the brutal part: the chunking that wall 1 forces runs *along the rows*. When the machine tiles, each asker-row's scores come into existence in **installments** — a row of $S$ is that token's questionnaire, 1024 scores long, and it gets minted a few hundred scores at a time as different chunks of $K$ visit the bench. At no moment do all 1024 coexist anywhere. It is like hearing a sentence one phrase at a time, with no replays allowed.

(careful with what gets chopped: a token's *own* vectors — its $q$, $k$, $v$ rows, each $d$ wide — always travel whole; no one splits a token in half. What arrives in installments is the row of **scores**. $d$-wide things are representations; $N$-wide things are score-lists — and it is the score-lists that get chopped.)

## The interlock

So here is the contradiction, stated plainly:

**wall 1 forces chopping. Wall 2 forbids it.** the bench cannot hold a row's scores whole; the softmax cannot finish on a fragment. Size forbids wholeness; the denominator demands it.

Every attempt to fuse the kernels smashes into this interlock — which is exactly why the three-kernel crime survived for years. The next section dissolves the contradiction, and it does so in a way I genuinely find beautiful: the machine will **commit early, and repair itself.**

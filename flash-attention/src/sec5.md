# Online softmax — the game

In the last section we stopped at a contradiction: we could not store whole rows, but softmax needed whole rows. So what do we do? As always when we hit a roadblock — go back to fundamentals, find the single atomic problem, and solve it in its simplest form. The hardware flexing its muscles at scale is just show. We own the brain.

Let's play a game.

$$s = [\,101,\; 100,\; 102\,]$$

(Look familiar? It should — this is row 1 of our anchor, $[1, 0, 2]$, shifted up by 100. Shift invariance says the two rows have identical softmax. Keep that in your back pocket.)

You will be given this vector one element at a time, and your goal is — of course — to compute its softmax (what did you guys expect?). But here is the thing: this is a game, and games have rules:

1. **each number arrives once.** whatever you want to do with it, do it while its in your hands — you will never see her again!!
2. **you may keep a small notebook** — a few running scalars of your choosing, updated as you go. But never the raw elements themselves; no stockpiling the vector in disguise. (derived quantities of the number *currently in your hands* are fair game — you just can't warehouse it.)
3. **no time travel.** you will not know the later elements in advance. THIS MOMENT IS ALL You HAVE → MAKE IT COUNT.

And the stakes, so you know this is not a puzzle-page filler: **if a winning notebook exists, the whole-row requirement dissolves, chunking becomes legal, and the quadratic ghost dies.** the entire article hangs on this game being winnable.

One hint before you begin: on a per-row basis the bottleneck is never COMPUTE — you have unlimited freedom to compute, or *recompute* rather (well, that's a giveaway in itself.. But its fine). So: 101 arrives, and you will never see it again — what do you do? Then 100 arrives — what changes? Then 102 — and why is 102 special? Or is it special at all?

<details><summary><b>open the box below to find the answers to all</b></summary>

When you receive 101, the first thing most of you thought: exponentiate it and store the value. Fair enough, good start — but the machine refuses. You can't compute $e^{101}$, let alone store it (sec 2's fire, remember). Ok, the rescue we learned: subtract the max, 102, first. **but hang on — you have not seen 102 yet.** at this moment all you have is 101, and you must make do with it. Is the game unplayable? Definitely not. Watch.

## The notebook

Track two scalars: $m$, the running max, and $r$, the running exponential sum — the denominator under construction. Sentinels: $m = -\infty$, $r = 0$ (chosen so the general rule needs no special first-step case: $\max(-\infty, s) = s$ crowns any first arrival automatically).

**101 knocks.** $m = \max(-\infty, 101) = 101$. And $r = e^{101 - m} = e^0 = 1$. No fire — we subtracted the best max we have *so far*. A provisional license.

**100 knocks.** $\max(101, 100)$ — the max survives. The newcomer just deposits in the reigning currency: $r = 1 + e^{100-101} = 1 + e^{-1}$. So far so good.

**102 knocks.** what is the max now? Oopsy — **we have a coup.** the old queen falls; 102 > 101, long live the new queen. And now look at the problem: our running sum $r$ was computed with respect to the OLD queen. The regime just changed, and every term in $r$ is measured in dead currency. Are we doomed? You bet not. Pay close attention — let's reverse-engineer.

What we have: $\;r_{\text{have}} = e^{101-101} + e^{100-101} = 1 + e^{-1}$

What we want: $\;r_{\text{want}} = e^{101-102} + e^{100-102} = e^{-1} + e^{-2}$

We can't admit 102 yet — first we repair the existing sum. (though she is the new queen, a queen's arithmetic is only valid if her subjects are in order; they better fix themselves before facing her wrath.) so here is the question:

$$r_{\text{want}} = \text{factor} \times r_{\text{have}} \qquad \text{— what is that factor?}$$

**FACTOR IS ALL YOU NEED!!**

Look at the two sums term by term: every exponent dropped by exactly 1. So multiply the whole of $r_{\text{have}}$ by $e^{-1}$ — one wholesale multiplication fixes every term at once. And the holy-moly moment: $e^{-1} = e^{101 - 102} = e^{\,m_{\text{old}} - m_{\text{new}}}$ — **the factor is built entirely from the two numbers the notebook legally holds.** no stored elements, no revisiting, no time travel. Repair on demand. Simple as that.

Sum repaired: $r = e^{-1} + e^{-2}$. NOW the queen is admitted — her own term, $e^{102-102} = e^0 = 1$, joins the room:

$$r = 1 + e^{-1} + e^{-2} = 1.5032$$

And that's it. You have conquered the game. Notice the rhythm that emerged, because it is the law of everything that follows: **two strokes per arrival — repair the past, then admit the newcomer.** always in that order.

</details>

## The citizenship laws

What did the game teach us? Two laws:

1. Every number in the notebook is measured against the reigning queen — the max is the *currency*, not a gatekeeper. A coup re-denominates everything.
2. You cannot admit a newcomer into $r$ while the past sits in dead currency. **coup first demands repair; admission comes second.**

In mathematical form — the update run at every arrival $s$:

$$m_{\text{new}} = \max(m_{\text{old}},\, s) \qquad\qquad r_{\text{new}} = e^{\,m_{\text{old}} - m_{\text{new}}}\; r_{\text{old}} \;+\; e^{\,s - m_{\text{new}}}$$

Verify for yourself: when there is NO coup, $m_{\text{old}} = m_{\text{new}}$ and the factor is $e^0 = 1$ — the repair *gracefully does nothing*, and the law collapses to plain admission. One law, no special cases.

> **and now open the FlashAttention paper to Algorithm 1, line 11.** that's this equation — their $\ell$ is our $r$, their $e^{m - m^{\text{new}}}$ is our repair factor. **you did not read that line. You just derived it.** this is the entire thesis of this article, cashing out for the first time.

## The pour joins the game

But — are we done? What was our final goal anyway? You bet we still have $O$ to think about. Our old friend $O = PV$. The ultimate pour.

Recall the game's rule 1 — and realise it applies to $V$'s rows too. One rule for all: each score arrives *holding hands with its ingredient* $v$, and the ingredient is present now and gone after. You cannot stockpile the $v$'s and pour at the end — stockpiling is the crime we are escaping. So the pour must happen AS scores arrive, with weights that are not final yet. **we refuse to wait.** the notebook grows a third resident: $o$, the running blend, poured provisionally and — you guessed it — *repaired* when regimes change.

So what is the blend's repair factor? Here is the derivation, and it is three lines:

<details><summary><b>open the box: the blend's repair factor</b></summary>

Take any one citizen $j$ already inside the blend. Its provisional weight (old regime) and its correct weight (new regime):

$$w_j^{\text{stale}} = \frac{e^{\,s_j - m_{\text{old}}}}{r_{\text{old}}} \qquad\qquad w_j^{\text{correct}} = \frac{e^{\,s_j - m_{\text{new}}}}{r_{\text{new}}}$$

Form the ratio, flip the bottom fraction, regroup:

$$\frac{w_j^{\text{correct}}}{w_j^{\text{stale}}} = \frac{e^{\,s_j - m_{\text{new}}}}{e^{\,s_j - m_{\text{old}}}} \cdot \frac{r_{\text{old}}}{r_{\text{new}}} = e^{\,m_{\text{old}} - m_{\text{new}}} \cdot \frac{r_{\text{old}}}{r_{\text{new}}}$$

And look — **$s_j$ cancelled.** the correction is the SAME for every citizen, regardless of their score. That uniformity is the license: one scalar, multiplied into the whole blend vector, repairs every term simultaneously:

$$c = e^{\,m_{\text{old}} - m_{\text{new}}} \cdot \frac{r_{\text{old}}}{r_{\text{new}}} \qquad\qquad o_{\text{new}} = c \cdot o_{\text{old}} + \frac{e^{\,s - m_{\text{new}}}}{r_{\text{new}}}\, v$$

</details>

> **notebox — the two gears of $c$, and a trap.** the scalar $c$ has two factors doing two different jobs. The **conversion gear** $e^{\,m_{\text{old}}-m_{\text{new}}}$ fires only at a coup (idles at $e^0 = 1$ otherwise). The **dilution gear** $r_{\text{old}}/r_{\text{new}}$ fires at *every* admission — because every newcomer's deposit grows the room, so every incumbent's share must shrink. The trap every reader falls into (I did): "no coup → no repair." wrong — no coup means no *conversion*, but the past still *dilutes*, because the room still grew. Dilution is not punishment; it is justice — each incumbent keeps exactly the share the full-row softmax will eventually grant it.

So the full notebook is the triplet $\langle m, r, o \rangle$ — max, running sum, running blend. Let's run the whole film, with ingredients $v_1 = [1,0]$, $v_2 = [0,1]$, $v_3 = [1,1]$ riding alongside our scores:

| arrival | $m$ (queen) | $r$ | $o$ (running blend) |
|---|---|---|---|
| — (cold start) | $-\infty$ | 0 | $[0,\,0]$ |
| $s = 101$ | 101 | 1 | $[1,\,0]$ |
| $s = 100$ | 101 | $1.368$ | $[0.731,\,0.269]$ |
| $s = 102$ — **coup** | 102 | $1.503$ | $[0.910,\,0.755]$ |

And the moment of truth. Compute the same row the ordinary, full-row way — softmax of $[101,100,102]$ gives weights $[0.245,\, 0.090,\, 0.665]$, pour them onto the three $v$'s — and you land on

$$o_{\text{ground truth}} = [\,0.910,\; 0.755\,]$$

**to the digit.** the notebook never saw the whole row at once, committed early, repaired itself — and arrived exactly where the patient full-row computation arrives. (and one more delight: those weights $[0.245, 0.090, 0.665]$ — go look at row 1 of $P$ back in section 2. Same numbers. Of course: same gaps.)

<widget: the film player — step through this exact table, two strokes per arrival, coup and all. toggle naive mode to watch the jar catch fire at $e^{101}$>

> **and Algorithm 1, line 12:** the blend update you just derived — the $c \cdot o + \text{new pour}$ — is that line, in the paper written per-row in matrix clothing ($\mathrm{diag}(\ell)^{-1}$ and friends). Both of the algorithm's famously cryptic lines are now things you invented at a bench with three numbers.

**REPAIR IS THE PRICE OF REFUSAL TO WAIT.**

# The backward pass — resurrection

Thus far, everything we discussed was the forward pass. But what makes neural networks what they are? Correct — the **backward pass**, the thing that actually trains these models. And the backward pass has a demand that threatens everything we just built.

## A quick primer on backprop

Consider a product: $c = a \times b$, with $a = 2$, $b = 3$.

I nudge $a$ a little. How much does $c$ change — how big is $c$'s move relative to $a$'s nudge? Take your time.

<details markdown="1"><summary><b>check the box to reveal</b></summary>

Nudge $a$ by $\Delta$: $c' = (a + \Delta)b = ab + \Delta b$. So $c$ moved by $\Delta b$, and the move-per-nudge is exactly $b$. And by symmetry, nudge $b$ and the sensitivity is $a$.

</details>

So the derivative with respect to one input **is the other input's value**. Which means: to compute gradients at backward time — which happens *later*, after the forward pass is long finished — the machine must have **kept its inputs**. Every framework calls this "saved for backward," and it is exactly why training eats so much more memory than inference.

## The ghost at the door

Now scale it up. $O = PV$ is nothing but a grid of these little products, and the same law scales wholesale. The gradient recipe for a matmul — the golden rule, quoted with shapes checked (full derivation lives in my companion notes — it deserves, and will get, its own article):

$$C = AB \;\;\Longrightarrow\;\; dA = dC\, B^\top \qquad\qquad dB = A^\top dC$$

(the transposes are positioned so the shapes close — that is the whole mnemonic)

Apply it: $dV = P^\top\, dO$.

But hold on a second. **do you have P with you?**

What is P's shape? $N \times N$. So — clear no-go, you can't store it; storing it resurrects the exact quadratic crime this whole article exists to kill. But can you compute $dV$ *without* P? No way — the recipe demands it. So you want to compute something, you have not stored its ingredients, and you are *not permitted* to store them.

All you CAN do is compute. Recompute. Recompute... N times — nobody is denying you that. **memory is the premium; compute is not.** so what do you do?

Correct: **you recompute.** simple as that. What is P anyway? Softmax of the score tiles. Do you have the raw scores stored? No — but you DO have $Q$ and $K$, pristine in HBM. Re-mint the tile: one cheap matmul, $S_{\text{tile}} = Q_{\text{tile}} K_{\text{tile}}^\top$. And the shift and the denominator? You kept them: the final queen $m_i$ and the final room total $r_i$ — **two scalars per row, $2N$ numbers total** ($N$ maxes plus $N$ sums). At GPT-2 scale that's 2,048 numbers kept, versus the 1.05 **million** we refused to store. Peanuts.

So at backward time, any weight resurrects by its own definition:

$$P_{ij} = \frac{e^{\,S_{ij} - m_i}}{r_i}$$

Not an approximation — the *definition*, evaluated late. The film was never stored; it is **re-shot on demand, from actors who never left the building.**

And here is the receipt, from our own clinic row. The film ended with $m = 102$, $r = 1.5032$. Pretend you are the backward pass arriving now — re-mint the middle score ($s_2 = 100$) and resurrect its weight:

$$P_2 = \frac{e^{\,100 - 102}}{1.5032} = \frac{0.1353}{1.5032} = 0.090$$

Check it against the film's ground-truth weights $[0.245,\, 0.090,\, 0.665]$ — **to the digit.** one exponential, one division, and a burned frame walks again.

## The trade, named — and the paradox closed

So name what the machine is doing: **paying FLOPs to avoid bytes.** deliberately redundant arithmetic — every backward tile re-minted — purchased because arithmetic is the cheap currency and HBM traffic is the dear one. And now go back to section 1's receipt, the paradox that opened this article: flash attention does 75.2 GFLOPs where standard does 66.6 — *more work* — and finishes six times faster. **the extra GFLOPs are the recomputation. The missing gigabytes are the never-stored S and P.** the paradox was never a paradox; it was a currency exchange, and flash attention simply knows the exchange rate.

Training is memory-intensive, and memory is the premium. So:

**RECOMPUTATION IS THE PRICE OF REFUSING TO STORE.**

Hold it next to its twin from the forward pass — *repair is the price of refusal to wait* — and you are holding the entire soul of flash attention: the same trade, struck twice. Refuse to hold the big thing; pay in the cheap currency; keep the mathematics exact to the digit. Two laws, one machine.

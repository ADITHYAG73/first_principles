# What is attention?

$$O = \mathrm{softmax}(QK^\top)\,V$$

I know some of you may be thinking: we have been seeing this equation forever, what do you offer new? My answer — nothing new, technically speaking. But what I promise is that by the end of this article you would have internalised it in an intuitive fashion, derived every piece of it with your own hands. That's the dimension I really care about. It's a promise, and I believe I can deliver.

Let's go to the smallest case we can comfortably hold in our heads. (I was tempted to go square, but am deliberately staying rectangular — later sections live in rectangular block sizes $B_r$, $B_c$, and I don't want the two side-lengths masquerading as each other. Square cases we reserve for simplifying upper-bound calculations, and we will say so when we do.)

Consider three matrices, each of shape 3×2:

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \qquad K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 2 & 0 \end{bmatrix} \qquad V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

**terminology freeze** — pin this table before anything else, we will use it for the entire article:

| symbol | meaning | shape |
|---|---|---|
| $N$ | context length — how many tokens (here 3). the lever the *user* pulls | — |
| $d$ | head dimension — how long each token's vector is (here 2). an *architect's* constant, small and fixed | — |
| $Q, K, V$ | queries, keys, values. **row $i$ of each belongs to token $i$** — three outfits the same token wears | $N \times d$ |
| $S = QK^\top$ | raw scores | $N \times N$ |
| $P = \mathrm{softmax}(S)$ | proportions (row-wise softmax) | $N \times N$ |
| $O$ | outputs | $N \times d$ |

> **notebox — the two levers.** $d$-wide things are *representations* (a token's vector). $N$-wide things are *score-lists* (a token judged against everyone). Keep them separate in your head at all times; half the confusion in this subject is these two masquerading as each other.

Now, I ask you to compute $O$. The journey has three stops.

## Stop 1: the raw scores

$$S = QK^\top = \begin{bmatrix} 1 & 0 & 2 \\ 2 & 2 & 0 \\ 2 & 1 & 2 \end{bmatrix}$$

Note $S$ is **square**, unlike its inputs — and think about what it means. Its rows are the tokens of $N$... And its columns? You are right — also the tokens of $N$. Entry $S_{ij}$ is the dot product of token $i$'s query row with token $j$'s key row: token $i$ *asking*, token $j$ *being asked*. $S$ is the full questionnaire — every token interrogating every token.

And pay close attention: $S_{12} = 0$ but $S_{21} = 2$. They **need not** be the same. The importance of token 2 *to* token 1 is a different question from the importance of token 1 *to* token 2 — relevance in language is directional ("sat" urgently needs "cat"; "cat" barely needs "sat").

> **notebox — why two matrices?** if each token had just ONE vector $x_i$, the score would be $x_i \cdot x_j$ — and dot products don't care about order, so the matrix would be *forced* symmetric. Giving each token separate query and key vectors is precisely what buys the *freedom* to be asymmetric. (can $S_{12} = S_{21}$ still happen? Sure — specific vectors can align that way, and if you set $Q = K$ the whole matrix turns symmetric. The split doesn't forbid symmetry; it stops *forcing* it.)

## Stop 2: from scores to proportions — why softmax?

Here is the plan for stop 3, stated upfront so stop 2 has a purpose: each token's output will be a **blend** of the value rows — some amount of $v_1$, some of $v_2$, some of $v_3$, mixed according to relevance. And there is the problem: raw scores cannot be blend amounts. A score of 2 is not "2 of something"; scores can even be negative, and a recipe cannot call for −0.3 cups of milk. We need to convert each row of raw scores into legal *proportions*. Normalization is not a math ritual — it is what turns opinions into recipe amounts.

So let's design the converter. Take a row vector and ask: how much share does each element deserve?

**candidate 1 — divide by the sum.** try $s_{\text{row}} = [1, 2, 3]$: sum is 6, shares $[1/6,\, 2/6,\, 3/6]$. Works! Now try $s_{\text{row}} = [1, -1, 0]$:

$$\text{sum} = 0 \qquad s_{\text{norm}} = [\,1/0,\; -1/0,\; 0/0\,]$$

Oops. The machine just beeps. Division by zero — and even when the sum isn't zero, negative "shares" sneak through. Dead.

**candidate 2 — take modulus, then divide by the sum.** for $[1, -1, 0]$: modulus sum is 2, shares $[1/2,\, 1/2,\, 0]$. Positive, sums to one... But do you see the problem? Try $[1, 1, 0]$:

$$[1, -1, 0] \;\longrightarrow\; [1/2,\, 1/2,\, 0] \qquad [1, 1, 0] \;\longrightarrow\; [1/2,\, 1/2,\, 0]$$

Two vectors living in different parts of the space — one *dislikes* its second element, the other *likes* it — collapse to the identical shares. The modulus erased the sign, and the sign carried the opinion. Semantic confusion.

So let's write down what we actually demand. **three properties**, and notice each dead candidate died against a different one:

1. **every share ≥ 0** — a recipe holds no negative amounts (candidate 1 died here, plus the fire)
2. **shares sum to 1** — proportions of a whole (think percentages)
3. **faithfulness** — the converter must preserve what the scores *said*: higher score, bigger share; dislike stays small; different opinions land on different shares (candidate 2 died here — it met properties 1 and 2 perfectly and still lobotomised the meaning)

> **notebox — the weights ambush.** the word "weights" is doing double duty in ML. *model weights* — learned parameters — are signed floats, negative all the time. *attention weights* — these shares we are building — are computed fresh from each input and must be legal proportions. Same english word, different mathematical citizens. When this article says weights, it means the shares.

The survivor: the **exponential — followed by the very same divide-by-the-sum we started with**, which works now because $e^x$ makes everything positive first. $e^x > 0$ always, and $e^x$ is strictly increasing, so order is preserved and different opinions stay different. That pair — exponentiate, then normalize — is softmax:

$$\mathrm{softmax}(s)_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

Check it on the killer case $[1, -1, 0]$:

$$e^1 + e^{-1} + e^0 \approx 2.718 + 0.368 + 1 = 4.086 \qquad \Rightarrow \qquad [\,0.665,\; 0.090,\; 0.245\,]$$

All shares positive, sum to 1, and the disliked element gets a *small but honest* share. And on $[1, 1, 0]$: $[\,0.422,\; 0.422,\; 0.155\,]$ — different vector, different shares, no collapse. Faithfulness lives.

> **notebox — a name for later.** numbers that are all ≥ 0 and sum to 1, used to mix vectors, make what mathematicians call a **convex combination** — the blend can never escape the region spanned by its ingredients. Remember the phrase; it pays rent in stop 3.

## Stop 2.5: don't set the machine on fire

Now consider $s_{\text{row}} = [101, 100, 102]$. Apply softmax directly: $e^{101} \approx 10^{43.9}$. The machine beeps again — float16 overflows at ~65,504, and even float32 dies near $e^{89}$. These are perfectly reasonable scores and the formula is *uncomputable*.

The rescue: subtract the row's max from every element first. Max is 102, so

$$[101, 100, 102] \;\longrightarrow\; [-1, -2, 0]$$

Every shifted score ≤ 0, every exponential ≤ 1. Fireproof forever. But hang on — am I allowed to just mutilate the input because its numbers are inconvenient? Here is the beautiful part:

**s₁ = [101, 100, 102] and s₂ = [−1, −2, 0] have the exact same softmax.**

DON'T BELIEVE ME? Three layers of conviction, escalating:

<details><summary><b>open the box: proof, witness, miracle</b></summary>

**the proof (three lines, fully general).** shift every score by any constant $c$:

$$\mathrm{softmax}(s + c)_j = \frac{e^{s_j + c}}{\sum_k e^{s_k + c}} = \frac{e^c \cdot e^{s_j}}{e^c \cdot \sum_k e^{s_k}} = \frac{e^{s_j}}{\sum_k e^{s_k}} = \mathrm{softmax}(s)_j$$

The $e^c$ cancels top and bottom — for ANY shift, ANY vector, computable or not.

**the witness (both sides computable — check me).** take $[1, 2, 3]$ vs its shifted twin $[-2, -1, 0]$:

$$[1,2,3]: \frac{[e^1, e^2, e^3]}{e^1+e^2+e^3} = [\,0.090,\; 0.245,\; 0.665\,] \qquad [-2,-1,0]: \frac{[e^{-2}, e^{-1}, e^0]}{e^{-2}+e^{-1}+1} = [\,0.090,\; 0.245,\; 0.665\,]$$

Identical, to the digit. The proof isn't lying.

**the miracle.** now $[101, 100, 102]$: the left side CANNOT be computed on any hardware you own — and yet you know its answer exactly, because the shifted twin $[-1, -2, 0]$ gives $[\,0.245,\; 0.090,\; 0.665\,]$, and the proof guarantees the twins agree. An impossible computation, answered. That is what a theorem buys you that arithmetic can't.

</details>

**SOFTMAX IS SHIFT INVARIANT.** the overflow was the *motive* for subtracting the max; shift invariance is the *license* that makes it legal. Motive plus license — remember this pair, it returns with a vengeance later in this article.

## Stop 3: the pour

Apply the (max-shifted) softmax to every row of our $S$:

$$P = \begin{bmatrix} 0.245 & 0.090 & 0.665 \\ 0.468 & 0.468 & 0.063 \\ 0.422 & 0.155 & 0.422 \end{bmatrix} \qquad \text{(every row ≥ 0, every row sums to 1 — verify!)}$$

And pour: row $i$ of $O$ is token $i$'s blend of the value rows, mixed by its own proportions:

$$o_i = P_{i1}\, v_1 + P_{i2}\, v_2 + P_{i3}\, v_3 \qquad\qquad O = PV = \begin{bmatrix} 0.910 & 0.755 \\ 0.532 & 0.532 \\ 0.845 & 0.578 \end{bmatrix}$$

Look at what happened geometrically: each output row sits *inside* the region spanned by $v_1, v_2, v_3$ — the convex combination promise cashing in. Token 1's blend leans hard toward $v_3$ (share 0.665, its top score); token 2 sits nearly on the fence between $v_1$ and $v_2$ (0.468 each). The questionnaire decided where each token stands.

> **notebox — what softmax does and does NOT constrain.** the mandate (≥ 0, sums to 1) lives on $P$ and only on $P$. Rows of $O$ inherit no such law — their entries are whatever the ingredients dictate, negative and unnormalised as real learned values are. *softmax disciplines the shares, never the goods.*

So what has a token actually DONE by replacing itself with such a blend? Before attention, token 2 was only itself. After, its representation is built from material that lived in *other tokens' rows* — pulled over in proportion to measured relevance. One token asked, everyone answered, and the relevant answers flowed in.

There is an everyday word for what these tokens just did.

**ATTENTION IS HOW TOKENS COMMUNICATE.**

Each token asks every token how relevant it is ($S$, the questionnaires), converts the answers into faithful shares ($P$, the laundering), and rebuilds itself as a blend of what the relevant ones were carrying ($O$, the pour). That one sentence is this entire article's anchor — because communication, it turns out, is exactly the thing that gets expensive. Every token talking to every token is $N^2$ conversations. Beautiful for expressiveness. Brutal, as the next section counts, for the courier.

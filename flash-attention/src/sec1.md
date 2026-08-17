# Flash Attention, From First Principles

In this article we are going to see, in detail, flash attention. *Attention Is All You Need* <link: Vaswani et al., 2017> was a watershed moment in the world of generative modeling, especially in NLP. Hard to believe it was nearly a decade ago — and here we are, in a world where LLMs are chipping away at long-unsolved Erdős problems <verify-citation>, designing drugs <verify-citation>, and (for better and worse) even orchestrating cyber operations <verify-citation>. Agents are an integral part of our lives now, and in this article we are going to see one of the atomic concepts that powered this revolution. Sit tight.. You will enjoy this ride, I can assure you that.

What if I told you — you can do MORE computation, in LESS time, and still deliver the **exact same output**? Not an approximation. The same attention, computed in a different order. We are generous on computation but frugal on time; we would rather compute — or *recompute* — than wait. In this article we will see exactly how and why flash attention is faster while being exactly attention itself.

Don't take my word for it. Here is the receipt, straight from the paper's own measurements <cite: FlashAttention paper, Fig. 2 — GPT-2 medium, N = 1024, A100>:

| | GFLOPs (work done) | GB moved | time |
|---|---|---|---|
| standard attention | 66.6 | 40.3 | 41.7 ms |
| flash attention | **75.2** | **4.4** | **7.3 ms** |

Read that again. Flash attention does MORE arithmetic — and finishes almost six times faster, moving nine times less data. How on earth?

Here is my promise: by the end of this article, you will not have *read* the answer — you will have **derived the machine that produces the first two of those numbers**, with a pencil, starting from a 3×2 matrix: why the work goes *up*, why the data movement collapses, and how far each one moves. The work and the data movement are both yours to count. The third number, the clock, you will *not* derive — and the reason why is one of the more interesting things in here. You will earn that one by measuring it. Let's go.

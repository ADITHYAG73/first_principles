# Towards First Principles

Technical explainers and research notes on AI by Adithya Giridharan, written with Claude as tutor and editor.

Each piece is built the long way round. Start at the smallest case you can hold in your head, count it upward with a
pencil, and only then check whether the machine agrees. Every number is measured, and every figure is one you can
poke at.

**Read it here: https://adithyag73.github.io/first_principles/**

## Articles

| Date | Article | Kind |
|---|---|---|
| 8 Nov 2026 | [I gave Opus 5.5 the prompt and nothing else](https://adithyag73.github.io/first_principles/weight-space-programming/): a blind replication of *Sparks of Weight-Space Programming*, where Claude Opus 5.5 hand-edits Qwen3.5-0.8B's weights and is compared with GPT-6 Astra | research note |
| 11 Oct 2026 | [We trained a small model to query a graph with RL. It got better, and it stopped listening.](https://adithyag73.github.io/first_principles/bird-graph-rl/) | research note |
| 11 Oct 2026 | [RL post-training a graph-query agent on Fireworks: a 6-point gain, and what it is made of](https://adithyag73.github.io/first_principles/bird-graph-rl-fireworks/) | research note |
| 22 Sep 2026 | [A Forking-Tokens Experiment with Qwen3.5-9B](https://adithyag73.github.io/first_principles/forking-tokens/) | research note |
| 17 Aug 2026 | [FlashAttention, From First Principles](https://adithyag73.github.io/first_principles/flash-attention/) | explainer, 7 interactive figures |
| 28 Jul 2026 | [Towards Free Normalization, From First Principles](https://adithyag73.github.io/first_principles/free-normalization/) | explainer, 6 interactive figures |

## What's in the repo

- **One folder per article.** Each has an `index.html` (the published page) plus whatever stands behind it: source
  markdown, build scripts, interactive widgets, figures, code, logs and data.
- **`index.html`** at the root is the site's home page, listing every article.
- **`methodology/SKILL.md`** is the working method behind every article: the tutoring protocol, the verification rules,
  the technical-writing standard and the publishing steps.
- **`weight-space-programming/SKILL.md`** lets a coding agent such as Claude Code or Codex replicate that experiment.

The site is served by GitHub Pages from `main`. Changes go through pull requests.

## Found an error?

Open an issue. Every number in these pieces should trace back to a measurement, a derivation or a cited source, and if
one doesn't, I want to know.

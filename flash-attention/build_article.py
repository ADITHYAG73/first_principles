#!/usr/bin/env python3
"""build_article.py — assemble flash-attention/index.html from the nine final mds.
usage: python3 build_article.py <mds_dir> <out_html>"""
import sys, re, markdown, pathlib

MDS = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])

CITES = {
  "Vaswani": ("https://arxiv.org/abs/1706.03762", "Vaswani et al., 2017"),
  "free-norm": ("../free-normalization/", "the free-normalization article"),
  "erdos": ("https://www.quantamagazine.org/why-the-legendary-erdos-problems-are-falling-to-ai-20260803/", "Quanta, Aug 2026"),
  "drugs": ("https://www.nobelprize.org/prizes/chemistry/2024/press-release/", "Nobel Chemistry 2024 — AlphaFold"),
  "cyber": ("https://www.anthropic.com/news/disrupting-AI-espionage", "Anthropic, 2025"),
  "fa1": ("https://arxiv.org/abs/2205.14135", "Dao et al., 2022"),
  "fa2": ("https://arxiv.org/abs/2307.08691", "Dao, 2023 — FlashAttention-2"),
  "triton": ("https://github.com/triton-lang/triton", "Triton"),
  "gemma": ("https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/", "Google — Gemma architecture"),
  "gemma-hf": ("https://huggingface.co/docs/transformers/en/model_doc/gemma2", "HF Gemma config"),
  "dao-issue": ("https://github.com/Dao-AILab/flash-attention/issues", "Dao-AILab/flash-attention issues"),
  "kaggle": ("https://www.kaggle.com/code/adithyagiri/flashattention", "the public kaggle notebook"),
}
def cite(key, text=None):
    url, label = CITES[key]
    return f'<a class="cite" href="{url}" target="_blank" rel="noopener">{text or label}</a>'

# placeholder -> replacement (applied on raw md before conversion)
def patch(md, name):
    r = {
      "<link: Vaswani et al., 2017>": cite("Vaswani"),
      "<verify-citation>": "",  # handled by inline cites below
      "llms are chipping away at long-unsolved Erdős problems": f'llms are chipping away at long-unsolved Erdős problems ({cite("erdos")})',
      "designing drugs": f'designing drugs ({cite("drugs")})',
      "even orchestrating cyber operations": f'even orchestrating cyber operations ({cite("cyber")})',
      "<cite: FlashAttention paper, Fig. 2 — GPT-2 medium, N = 1024, A100>": f'({cite("fa1","FlashAttention paper, Fig. 2 — GPT-2 medium, N=1024, A100")})',
      "<cite: paper §2.1 — A100: HBM 40 GB at 1.5 TB/s, SRAM ~19 TB/s>": f'({cite("fa1","paper §2.1: A100 — HBM 40 GB @ 1.5 TB/s, SRAM ~19 TB/s")})',
      "<link: free-norm>": cite("free-norm"),
      "<link: article 1>": cite("free-norm"),
      "<link: free-norm article>": cite("free-norm"),
      # theorem slot: the md carries a corrupted math token (❥22❥), not $\Theta(...)$,
    # so it is matched by regex below rather than by an exact-string key.
      "<cite: Algorithm 1 — outer loop over K,V blocks, inner over Q blocks>": f'({cite("fa1","Algorithm 1 — outer loop over K,V blocks, inner over Q blocks")})',
      "<cite: FA-2 paper; Tillet provenance>": f'({cite("fa2")}; the flip was first implemented by Phil Tillet in {cite("triton")})',
      "<link: sec3>": '<a href="#sec3">the crime scene</a>',
      "<cite: confirmed against Algorithm 1's structure>": f'(confirmed against Algorithm 1 \u2014 {cite("fa1")})',
      "<cite: exact statement at citation pass>": f'({cite("fa1","Theorem 2")})',
      "<cite: Fig. 2>": f'({cite("fa1","Fig. 2")})',
      "<cite: HF config, Google architecture blog>": f'({cite("gemma-hf")}; {cite("gemma")})',
      "<cite: Dao-AILab issue, HF discussions>": f'({cite("dao-issue")})',
    }
    for k, v in r.items(): md = md.replace(k, v)
    # widget slots -> iframes
    W = {
      "two-ledgers": "widgets/two_ledgers_widget.html",
      "two ledgers": "widgets/two_ledgers_widget.html",
      "the film player": "widgets/online_softmax_film_player.html",
      "the bench": "widgets/the_bench_rent_widget.html",
      "the itinerary comic": "widgets/the_itinerary_comic.html",
      "the breakeven curve": "widgets/the_breakeven_widget.html",
      "the measured benchmark": "widgets/widget6-benchmark.html",
    }
    def wsub(m_):
        body = m_.group(1).strip()
        for key, src in W.items():
            if body.lower().startswith(key):
                return f'\n<iframe class="widget" src="{src}" loading="lazy" title="{body}"></iframe>\n'
        return f'<!-- unmatched widget slot: {body} -->'
    # Theorem 2, FlashAttention paper §3.2 (Analysis: IO Complexity).
    md = re.sub(r"<cite:\s*this is the paper's Theorem[^>]*>",
        lambda _m: f'(this is the paper\u2019s Theorem 2, \u00a73.2: standard attention requires \u0398(Nd + N\u00b2) HBM accesses, '
                   f'while FlashAttention requires \u0398(N\u00b2d\u00b2M\u207b\u00b9) \u2014 {cite("fa1")})', md)
    md = re.sub(r"<widget:\s*([^>]+)>", wsub, md)
    md = re.sub(r"<code:\s*([^>]+)>",
        lambda m_: f'<p class="codelink">→ <a href="code/flash_attention_numpy.py">flash_attention_numpy.py</a> — run it: exactness PASS at the anchor and GPT-2 scale, and ~10× faster even on CPU ({m_.group(1).strip()})</p>', md)
    md = md.replace("<link: public kaggle notebook — triton kernel + benchmark, so you can run the speedup yourself — to be added>",
        f'{cite("kaggle")} (triton kernel + benchmark \u2014 run the speedup urself; notebook file also in <a href="code/triton_flash_attention_kaggle_v3.ipynb">code/</a>)')
    return md

# math protection: convert $$..$$ and $..$ to \[..\]/\(..\) AFTER md conversion is risky;
# instead shield math from markdown, restore after.
def shield(md):
    store = []
    def keep(m_):
        store.append(m_.group(0)); return f"@@M{len(store)-1}@@"
    md = re.sub(r"\$\$.*?\$\$", keep, md, flags=re.S)
    md = re.sub(r"\$[^\$\n]+\$", keep, md)
    return md, store
def unshield(html, store):
    for i, s in enumerate(store): html = html.replace(f"@@M{i}@@", s)
    return html

TITLES = {  # sidebar labels
  "sec1": "the receipt", "sec2": "what is attention?", "sec3": "the crime scene",
  "sec4": "the two walls", "sec5": "online softmax — the game", "sec6": "the scaffold",
  "sec7": "the theorem & the wall", "sec8": "the backward pass", "sec9": "receipts & confessions",
}

sections, nav = [], []
for i in range(1, 10):
    name = f"sec{i}"
    md = (MDS / f"{name}.md").read_text()
    md = patch(md, name)
    md, store = shield(md)
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = unshield(html, store)
    sections.append(f'<section id="{name}">{html}</section>')
    nav.append(f'<a href="#{name}">{i}. {TITLES[name]}</a>')

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flash attention, from first principles</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
 :root{{--paper:#FDFCF9;--ink:#1A1A1E;--ink-soft:#5A5A63;--hairline:#E4E1DA;
  --serif:Charter,Georgia,'Times New Roman',serif;--mono:'SF Mono',ui-monospace,Menlo,Consolas,monospace}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);line-height:1.65}}
 .wrap{{display:flex;max-width:1080px;margin:0 auto}}
 nav{{position:sticky;top:0;align-self:flex-start;width:230px;padding:28px 18px;font-family:var(--mono);
  font-size:12px;border-right:1px solid var(--hairline);height:100vh;overflow-y:auto}}
 nav a{{display:block;color:var(--ink-soft);text-decoration:none;padding:5px 0}}
 nav a:hover{{color:var(--ink)}}
 main{{flex:1;padding:34px 40px;max-width:780px;min-width:0}}
 h1{{font-size:26px;line-height:1.25}} h2{{font-size:19px;margin-top:2em}} h3{{font-size:16px}}
 table{{border-collapse:collapse;font-size:14px;margin:1em 0;display:block;overflow-x:auto}}
 th,td{{border:0.5px solid var(--hairline);padding:6px 10px;text-align:left}}
 th{{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--ink-soft);font-weight:400}}
 blockquote{{border-left:3px solid var(--hairline);margin:1.2em 0;padding:2px 16px;background:#fff;border-radius:0 4px 4px 0}}
 code{{font-family:var(--mono);font-size:.9em;background:#f1efe9;padding:1px 5px;border-radius:3px}}
 details{{border:1px solid var(--hairline);border-radius:5px;padding:10px 14px;margin:1em 0;background:#fff}}
 summary{{cursor:pointer;font-family:var(--mono);font-size:13px}}
 iframe.widget{{width:100%;height:640px;border:none;margin:1.2em 0;border-radius:6px;background:#f3f1ec}}
 a{{color:#1E5AA8}} a.cite{{font-size:.88em}}
 .katex-display{{overflow-x:auto;overflow-y:hidden;padding:4px 0}}
 section{{margin-bottom:3.2em;border-bottom:1px solid var(--hairline);padding-bottom:1.4em}}
 @media(max-width:860px){{nav{{display:none}}main{{padding:22px 18px}}}}
</style></head><body>
<div class="wrap">
<nav><strong style="color:var(--ink)">flash attention</strong><br><br>{''.join(nav)}
<br><a href="../">← all articles</a></nav>
<main>
{''.join(sections)}
<footer style="font-family:var(--mono);font-size:11px;color:var(--ink-soft);padding:20px 0">
first_principles · derived at a bench, verified by machine · <a href="../">index</a></footer>
</main></div></body></html>"""
OUT.write_text(page)
print(f"built {OUT} ({len(page):,} bytes), {len(sections)} sections")

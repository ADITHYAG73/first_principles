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
  "fa-repo": ("https://github.com/Dao-AILab/flash-attention", "Dao-AILab/flash-attention"),
  "fa-headdim": ("https://github.com/Dao-AILab/flash-attention/issues/2427", "flash-attention issue #2427 — head_dim=512 for Gemma 4 global layers"),
  "gemma4-hf": ("https://huggingface.co/google/gemma-4-31B-it/discussions/30", "HF: FlashAttention supports head dimension at most 256"),
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
      "<cite: repo support matrix>": f'({cite("fa-repo","the repo\u2019s own support matrix")})',
      "<cite: FA-2 abstract>": f'({cite("fa2","FlashAttention-2, abstract")})',
      "<cite: Dao-AILab issue, HF discussions>": f'({cite("fa-headdim")}; {cite("gemma4-hf")})',
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

sections = []
for i in range(1, 10):
    name = f"sec{i}"
    md = (MDS / f"{name}.md").read_text()
    md = patch(md, name)
    md, store = shield(md)
    # "toc" gives every heading an id, which the client-side contents list needs
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "md_in_html"])
    html = unshield(html, store)
    sections.append(f'<section id="{name}" data-title="{i}. {TITLES[name]}">{html}</section>')

# per-widget heights: a single fixed height leaves dead space under short widgets
# and makes tall ones scroll inside themselves, trapping the reader's wheel.
# these are fallbacks only — each widget reports its true height via postMessage.
WIDGET_H = {
  "two_ledgers_widget": 560, "online_softmax_film_player": 720,
  "the_bench_rent_widget": 700, "the_itinerary_comic": 760,
  "the_breakeven_widget": 620, "widget6-benchmark": 470,
}
body = "".join(sections)
for stem, h in WIDGET_H.items():
    body = body.replace(f'src="widgets/{stem}.html" loading="lazy"',
                        f'src="widgets/{stem}.html" style="height:{h}px" loading="lazy"')

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flash Attention, From First Principles</title>
<meta name="description" content="Why attention runs out of memory before it runs out of maths — online softmax derived by hand, then the tiled kernel that never writes the N×N matrix down.">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  :root {{
    --paper: #FDFCF9; --ink: #1A1A1E; --ink-soft: #5A5A63; --machine: #8A8A93;
    --matmul: #2B5BA8; --norm: #C4552D; --hairline: #E4E1DA;
    --serif: Charter, Georgia, 'Times New Roman', serif;
    --mono: 'SF Mono', ui-monospace, Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html {{ background: #F3F1EC; }}
  body {{ margin: 0; font-family: var(--serif); color: var(--ink); line-height: 1.62; font-size: 17px; }}
  .sheet {{ max-width: 760px; margin: 0 auto; background: var(--paper); border-left: 1px solid var(--hairline);
    border-right: 1px solid var(--hairline); padding: 48px 40px 80px; }}
  @media (max-width: 820px) {{ .sheet {{ padding: 32px 18px 60px; border: none; }} body {{ font-size: 16px; }} }}
  h1 {{ font-size: 34px; line-height: 1.2; margin: 0 0 4px; letter-spacing: -.01em; }}
  section h1 {{ font-size: 26px; margin: 0 0 12px; }}
  section:first-of-type h1 {{ font-size: 34px; }}
  h2 {{ font-size: 24px; margin: 48px 0 12px; letter-spacing: -.005em; }}
  h3 {{ font-size: 18.5px; margin: 32px 0 8px; }}
  p {{ margin: 0 0 16px; }}
  hr {{ border: none; border-top: 1px solid var(--hairline); margin: 40px 0; }}
  a {{ color: var(--matmul); text-decoration: none; border-bottom: 1px solid rgba(43,91,168,.3); }}
  a:hover {{ border-bottom-color: var(--matmul); }}
  a.cite {{ font-size: .88em; }}
  code {{ font-family: var(--mono); font-size: .85em; background: #F2F0EA; padding: 1px 5px; border-radius: 3px; }}
  blockquote {{ margin: 0 0 18px; padding: 10px 16px; border-left: 3px solid var(--norm); background: #fff;
    border-radius: 0 4px 4px 0; color: var(--ink-soft); }}
  blockquote p:last-child {{ margin-bottom: 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 15px; margin: 0 0 16px; display: block; overflow-x: auto; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--hairline); vertical-align: top; }}
  th {{ font-family: var(--mono); font-size: 12.5px; font-weight: 500; color: var(--ink-soft); }}
  details {{ border: 1px solid var(--hairline); border-left: 3px solid var(--norm); border-radius: 4px;
    background: #fff; padding: 10px 16px; margin: 0 0 18px; }}
  details summary {{ cursor: pointer; font-size: 13.5px; color: var(--norm); font-style: italic; list-style: none; }}
  details summary::before {{ content: '\\25B8\\00a0'; }}
  details[open] summary::before {{ content: '\\25BE\\00a0'; }}
  details[open] summary {{ margin-bottom: 8px; }}
  .katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0 10px;
    scrollbar-width: thin; scrollbar-color: var(--machine) transparent; }}
  .katex-display::-webkit-scrollbar {{ height: 7px; }}
  .katex-display::-webkit-scrollbar-track {{ background: #F2F0EA; border-radius: 4px; }}
  .katex-display::-webkit-scrollbar-thumb {{ background: var(--machine); border-radius: 4px; }}
  iframe.widget {{ width: 100%; border: none; display: block; margin: 26px -20px; width: calc(100% + 40px); }}
  @media (max-width: 820px) {{ iframe.widget {{ margin: 22px -8px; width: calc(100% + 16px); }} }}
  section {{ margin-bottom: 8px; }}
  section + section {{ border-top: 1px solid var(--hairline); padding-top: 34px; margin-top: 34px; }}
  .codelink {{ font-family: var(--mono); font-size: 13.5px; }}
  .colophon {{ font-style: italic; color: var(--ink-soft); font-size: 15.5px; }}
  footer {{ font-size: 13.5px; color: var(--ink-soft); line-height: 1.7; }}
  .layout {{ display: flex; justify-content: center; align-items: flex-start; }}
  nav.toc {{ display: none; }}
  @media (min-width: 1160px) {{
    nav.toc {{ display: block; position: sticky; top: 32px; width: 210px; flex: 0 0 210px;
      margin-right: 28px; padding: 24px 0 24px 16px; max-height: calc(100vh - 64px);
      overflow-y: auto; font-size: 12.5px; line-height: 1.5; }}
    nav.toc .toc-title {{ font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em;
      text-transform: uppercase; color: var(--ink-soft); margin-bottom: 10px; }}
    nav.toc a {{ display: block; color: var(--ink-soft); border: none; padding: 3px 0 3px 10px;
      border-left: 2px solid transparent; }}
    nav.toc a.h3 {{ padding-left: 22px; font-size: 11.5px; }}
    nav.toc a:hover {{ color: var(--ink); }}
    nav.toc a.active {{ color: var(--matmul); border-left-color: var(--matmul); }}
    nav.toc .home {{ margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--hairline);
      font-family: var(--mono); font-size: 11px; }}
  }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents"><div class="toc-title">Contents</div><div id="tocitems"></div>
<div class="home"><a href="../">← all articles</a></div></nav>
<div class="sheet">
{body}
<hr />
<p class="colophon">Written by Adithya Giridharan, with Claude as tutor and editor. Every derivation in this
article was worked by hand before it was written, and every number was measured before it was claimed.</p>
<hr />
<footer>
<p>FlashAttention is the work of Dao et al., reported in
<a href="https://arxiv.org/abs/2205.14135">the original paper</a>. This article is the staircase, not the
building. Found an error? Good — that means you were counting. Reach me on
<a href="https://x.com/AG_1698">X</a> or <a href="https://github.com/ADITHYAG73">GitHub</a>.</p>
</footer>
</div>
</div>
<script>
// contents list, built from the sections and their h2s, with an active marker that
// follows the reader. same behaviour as article one.
(function() {{
  var box = document.getElementById('tocitems');
  if (!box) return;
  var links = {{}}, targets = [];
  Array.prototype.forEach.call(document.querySelectorAll('.sheet section[id]'), function(sec) {{
    var a = document.createElement('a');
    a.href = '#' + sec.id;
    a.textContent = sec.getAttribute('data-title') || sec.id;
    box.appendChild(a); links[sec.id] = a; targets.push(sec);
    Array.prototype.forEach.call(sec.querySelectorAll('h2[id]'), function(h) {{
      // headings inside a reveal box are structure for that box, not the article:
      // linking to them would scroll the reader to something still collapsed.
      if (h.closest('details')) return;
      var b = document.createElement('a');
      b.href = '#' + h.id; b.textContent = h.textContent; b.className = 'h3';
      box.appendChild(b); links[h.id] = b; targets.push(h);
    }});
  }});
  var current = null;
  function setActive(id) {{
    if (current === id) return;
    if (current && links[current]) links[current].classList.remove('active');
    if (links[id]) links[id].classList.add('active');
    current = id;
  }}
  var obs = new IntersectionObserver(function(entries) {{
    entries.forEach(function(e) {{ if (e.isIntersecting) setActive(e.target.id); }});
  }}, {{ rootMargin: '0px 0px -70% 0px' }});
  targets.forEach(function(t) {{ obs.observe(t); }});
}})();

// widgets report their own height, so no iframe is taller than its contents.
// this removes the dead space under short widgets and stops tall ones from
// scrolling internally and swallowing the reader's wheel.
window.addEventListener('message', function(e) {{
  if (!e.data || e.data.type !== 'widget-height' || !isFinite(e.data.height)) return;
  var frames = document.querySelectorAll('iframe.widget');
  for (var i = 0; i < frames.length; i++) {{
    if (frames[i].contentWindow === e.source) {{
      frames[i].style.height = Math.ceil(e.data.height) + 'px';
      return;
    }}
  }}
}});
</script>
</body>
</html>"""
OUT.write_text(page)
print(f"built {OUT} ({len(page):,} bytes), {len(sections)} sections")

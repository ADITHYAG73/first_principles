#!/usr/bin/env python3
"""build.py — turn an article.md into index.html in the forking-tokens page shell.
usage: python3 build.py <article_dir> <date> <inline_math: yes|no>
Reads <article_dir>/article.md and <article_dir>/aiuse.html; writes <article_dir>/index.html.
"""
import html, pathlib, re, sys
import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = pathlib.Path(sys.argv[1]); date = sys.argv[2]; inline_math = sys.argv[3] == "yes"
md = (d / "article.md").read_text()
aiuse = (d / "aiuse.html").read_text().strip()

# head and tail of the shell come from the forking-tokens article, so the family stays one family
shell = (ROOT / "forking-tokens" / "index.html").read_text()
style = shell[shell.index("<style>"): shell.index("</style>") + len("</style>")]
script = shell[shell.rindex("<script>"): shell.rindex("</script>") + len("</script>")]

lines = md.splitlines()
title = lines[0].lstrip("# ").strip()
body = "\n".join(lines[1:])
# drop the draft status line (an italic paragraph right under the title)
body = re.sub(r"\A\s*\*(Final draft|Draft)[^*]*\*\s*", "", body, flags=re.S)
desc = re.sub(r"\s+", " ", re.sub(r"[*_`#>\[\]()!]", "", body.strip().split("\n\n")[0]))[:240]

# protect maths from the markdown converter
store = []
def keep(m):
    store.append(m.group(0)); return f"@@MATH{len(store)-1}@@"
body = re.sub(r"\$\$.+?\$\$", keep, body, flags=re.S)
if inline_math:
    # inline maths is rewritten to \( \) so that dollar amounts in the prose are never parsed as maths
    def keep_inline(m):
        store.append("\\(" + m.group(1) + "\\)"); return f"@@MATH{len(store)-1}@@"
    body = re.sub(r"(?<![\w$])\$(?=[A-Za-z\\])([^$\n]+?)\$(?![\w$])", keep_inline, body)

# images become captioned figures; the alt text is the caption
body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'\n<figure><img src="{m.group(2)}" alt="{html.escape(m.group(1))}"><figcaption>{html.escape(m.group(1))}</figcaption></figure>\n', body)

# one <section> per h2, so the contents sidebar can find them
parts = re.split(r"^## (.+)$", body, flags=re.M)
out = []
intro = parts[0].strip()
def conv(t):
    h = markdown.markdown(t, extensions=["tables", "fenced_code"])
    for i, m in enumerate(store):
        h = h.replace(f"@@MATH{i}@@", html.escape(m, quote=False))
    return h
out.append(f'<section id="sec0" data-title="top">\n<h1>{html.escape(title)}</h1>\n<p class="byline">Adithya Giridharan · {date}</p>\n{conv(intro)}\n</section>')
for k in range(1, len(parts), 2):
    head, text = parts[k].strip(), parts[k + 1]
    label = re.sub(r"^\d+\.\s*", "", head).lower()
    out.append(f'<section id="sec{k//2+1}" data-title="{html.escape(label)}">\n<h2>{html.escape(head)}</h2>\n{conv(text)}\n</section>')
out.append(f'<section id="aiuse" data-title="how this was made">\n<p class="aiuse">{aiuse}</p>\n</section>')

katex = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true}""" + (",{left:'\\\\(',right:'\\\\)',display:false}" if inline_math else "") + """],ignoredTags:['script','noscript','style','textarea','pre','code']});"></script>"""

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:type" content="article">
<meta name="twitter:card" content="summary">
<meta name="author" content="Adithya Giridharan">
{katex}
{style}
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents"><div class="toc-title">Contents</div><div id="tocitems"></div>
<div class="home"><a href="../">← all articles</a></div></nav>
<div class="sheet">
{chr(10).join(out)}
</div>
</div>
{script}
</body>
</html>
"""
(d / "index.html").write_text(page)
print("wrote", d / "index.html", len(page), "bytes;", len(out) - 2, "sections;", len(store), "maths spans")

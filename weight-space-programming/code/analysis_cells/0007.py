# Hypothesis: the cut-offs are repetition loops (same lines over and over), not long-but-productive reasoning.
# Characterise each cut-off: where the loop starts, the loop's period, and whether the answer was already found.
import re, collections
def loop_profile(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    seen, first_rep = {}, None
    for i, l in enumerate(lines):
        if len(l) > 20 and l in seen and first_rep is None and lines[i:i+5] and all(x in seen for x in lines[i:i+5]):
            first_rep = i
        seen.setdefault(l, i)
    tail = lines[-200:]
    c = collections.Counter(tail)
    top, cnt = c.most_common(1)[0] if c else ("", 0)
    return dict(n_lines=len(lines), loop_start_frac=round(first_rep / len(lines), 3) if first_rep else None,
                tail_unique=len(c), top_line_count=cnt, top_line=top[:120])
cut = [r for r in P1 if r["category"] == "cut_off"]
prof = [loop_profile(r["text"]) for r in cut]
starts = sorted(p["loop_start_frac"] for p in prof if p["loop_start_frac"] is not None)
print("cut-offs", len(cut), "| loop detected", len(starts), "| loop start (fraction of answer) quartiles",
      [starts[int(q * (len(starts) - 1))] for q in (0, .25, .5, .75, 1)] if starts else None)
print("unique lines in last 200 lines: quartiles", sorted(p["tail_unique"] for p in prof)[::len(prof)//4])
gold_in = sum(1 for r in cut if r["gold"] and re.search(r"(?<![\d.])" + re.escape(r["gold"]) + r"(?![\d.])", r["text"][:len(r["text"])//2]))
print("cut-offs whose gold answer already appears in the first half of the thinking:", gold_in, "/", len(cut))
for r, p in list(zip(cut, prof))[:3]:
    t = r["text"]; i = int(len(t) * (p["loop_start_frac"] or 0.5))
    print("\n---", r["id"], "tokens", r["n_tokens"], "loop_start_frac", p["loop_start_frac"], "tail_unique", p["tail_unique"])
    print("[at loop start]", repr(t[i:i+300]))
    print("[last 300 chars]", repr(t[-300:]))

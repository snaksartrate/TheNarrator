"""Terminal layout: 3 columns x 2 rows (AdamW / SGD), vertical fallback when narrow."""
import shutil
import sys
import textwrap

GROUPS = [("adamw", "ADAMW"), ("sgd", "SGD + MOMENTUM")]
TITLES = {"char": "Character", "word": "Word", "bpe": "BPE"}
MIN_COL = 30


def _box():
    chars = "┌─┐│└┘"
    try:
        chars.encode(sys.stdout.encoding or "ascii")
        return chars
    except (UnicodeEncodeError, LookupError):
        return "+-+|++"


def safe(text):
    """Never let a console encoding crash the printout."""
    enc = sys.stdout.encoding or "ascii"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _panel(r, w, box):
    tl, h, tr, v, bl, br = box
    inner = w - 2
    title = f" {TITLES[r['tokenizer']]} "
    if r["status"] in ("ok", "empty"):
        body = r["continuation"] or "(empty continuation)"
    else:
        body = f"[{r['status'].upper()}] {r['error']}"
    lines = []
    for para in body.split("\n"):
        lines += textwrap.wrap(para, inner - 2) or [""]
    m = r["metrics"]
    stat = f"{r['generation_seconds']:.1f}s | {m['word_count']} words" if m and r["generation_seconds"] is not None else "-"
    out = [tl + title + h * (inner - len(title)) + tr]
    out += [v + " " + ln.ljust(inner - 2) + " " + v for ln in lines]
    out += [v + " " + stat[:inner - 2].ljust(inner - 2) + " " + v, bl + h * inner + br]
    return out


def _side_by_side(panels, gap=2):
    height = max(len(p) for p in panels)
    wid = [len(p[0]) for p in panels]
    padded = [p[:-1] + [" " * wid[i]] * (height - len(p)) + p[-1:] for i, p in enumerate(panels)]
    return [(" " * gap).join(rows) for rows in zip(*padded)]


def render(run, width=None):
    width = width or shutil.get_terminal_size((100, 20)).columns
    box = _box()
    out = ["PROMPT"] + textwrap.wrap(run["prompt"], width, replace_whitespace=False) + [""]
    for opt, label in GROUPS:
        rs = [r for r in run["results"] if r["optimizer"] == opt]
        if not rs:
            continue
        out.append(label)
        n = len(rs)
        if n * MIN_COL + 2 * (n - 1) <= width:
            colw = (width - 2 * (n - 1)) // n
            out += _side_by_side([_panel(r, colw, box) for r in rs])
        else:
            for r in rs:
                out += _panel(r, max(MIN_COL, min(width, 100)), box)
        out.append("")
    out.append(f"RUN {run['run_id']:05d}  (status: {run['status']})")
    out += textwrap.wrap(f"Saved to: {run['run_dir']}", max(width, MIN_COL), break_long_words=True)
    return safe("\n".join(out))
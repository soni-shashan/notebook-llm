"""Interactive CLI. Run `notebook-llm` (terminal) or `notebook_llm.run()` (notebook cell)."""
import argparse
import os
import re

from . import disk, library, ui
from .env import is_kaggle
from .integrations import TOOLS
from .manager import NotebookLLM
from .ui import ask, c, confirm, pick, table, title


# ---------- rendering helpers ---------------------------------------
def fit_cell(est) -> str:
    return c(est.label, ui.VERDICT_STYLE[est.verdict])


def speed_cell(est) -> str:
    return f"~{est.tok_s:.0f}" if est.tok_s else "-"


def show_candidates(cands, llm):
    ui.info(f"Hardware: {llm.hw.summary()}")
    ui.info("NEEDS = weights + KV cache (auto context) + overhead. tok/s is an estimate (+-30%).")
    rows = []
    for i, cd in enumerate(cands, 1):
        size = f"{cd.size_gb:.1f} GB" + ("" if cd.exact else "~")
        moe = " MoE" if cd.active_b and cd.params_b and cd.active_b < cd.params_b * 0.8 else ""
        rows.append((i, cd.ref + c(moe, "gray"), size, f"{cd.est.needed_gb:.1f} GB", cd.est.context,
                     fit_cell(cd.est), speed_cell(cd.est)))
    table(["#", "MODEL", "SIZE", "NEEDS", "CTX", "FIT", "TOK/S"], rows, right_cols=(0, 2, 3, 4, 6))


def show_estimate_detail(cd):
    e = cd.est
    print(f"\n  {c(cd.ref, 'bold')}  ({'exact' if cd.exact else 'estimated'} size {cd.size_gb:.1f} GB)")
    print(f"  Weights {e.weights_gb:.1f} GB + KV cache {e.kv_gb:.1f} GB (ctx {e.context}) + overhead {e.overhead_gb:.1f} GB"
          f" = {c(f'{e.needed_gb:.1f} GB', 'bold')}")
    print(f"  Verdict: {fit_cell(e)}   Speed: {c(speed_cell(e) + ' tok/s', 'bold')}   "
          f"On GPU: {e.gpu_fraction * 100:.0f}% of weights")
    if e.note:
        ui.warn(e.note)


def header(llm: NotebookLLM):
    llm.refresh_hardware()
    title("notebook_llm")
    print(f"  {llm.hw.summary()}")
    if llm.hw.gpu.available:
        print(f"  VRAM in use: {llm.hw.gpu.used_vram_gb} / {llm.hw.gpu.total_vram_gb} GB")
    elif is_kaggle():
        ui.warn("No GPU. In Kaggle: Settings > Accelerator > GPU T4 x2, then restart.")
    print(f"  Disk   : {disk.free_gb(llm.models_dir):.0f} GB free  ({llm.models_dir})")
    up = llm.server.is_running()
    print(f"  Server : {c('running', 'green') if up else c('stopped', 'red')}")
    if up:
        try:
            for m in llm.loaded():
                pct = 100 * m.get("size_vram", 0) / m["size"] if m.get("size") else 0
                print(f"  Loaded : {c(m['name'], 'bold')} ({m.get('size', 0) / 1024 ** 3:.1f} GB, {pct:.0f}% on GPU)")
        except Exception:
            pass
    print(f"  Model  : {llm.settings.model or '(none selected)'}")
    print(f"  Tunnel : {c(llm.url, 'bold', 'green') if llm.url else 'closed'}")


# ---------- flows ----------------------------------------------------
def ensure_server(llm, model=None) -> bool:
    if llm.server.is_running():
        return True
    print("  Installing/starting Ollama...")
    try:
        llm.start_server(model)
        return True
    except Exception as e:
        ui.err(str(e))
        return False


def ensure_disk(llm, cd) -> bool:
    """Check free space before a download. Returns True if OK to proceed."""
    chk = llm.space_for(cd.size_gb)
    if chk.ok:
        ui.info(f"Disk check: {chk.free_gb:.0f} GB free, need ~{chk.needed_gb:.0f} GB. OK.")
        return True
    ui.err(f"Not enough disk space in {chk.path}: {chk.free_gb:.1f} GB free, need ~{chk.needed_gb:.1f} GB.")
    print("  Ways to fix it: delete models you don't need, pick a smaller size or quantization "
          "(type qN in the list), or change the models folder in Settings.")
    if llm.installed() and confirm("Open 'Installed models' to free space now?"):
        flow_installed(llm)
        chk = llm.space_for(cd.size_gb)
        if chk.ok:
            ui.ok(f"Now {chk.free_gb:.0f} GB free. Continuing.")
            return True
    return confirm("Try the download anyway?", False)


def after_load(llm):
    if not llm.url and confirm("Open a public tunnel now?"):
        flow_open_tunnel(llm)


def model_actions(llm, cd):
    show_estimate_detail(cd)
    if cd.est.verdict == "no":
        ui.warn("This model will not run on your hardware.")
        if not confirm("Pull anyway?", False):
            return
    print("\n  1) Pull + load now (make it the active model)\n  2) Pull only\n  Enter) back")
    ch = ask("Choose")
    if ch not in ("1", "2"):
        return
    if not ensure_server(llm, cd.ref):
        return
    try:
        if not llm.client.has_model(cd.ref):
            if not ensure_disk(llm, cd):
                return
            print(f"  Pulling {cd.ref} ({cd.size_gb:.1f} GB)...")
            llm.pull(cd.ref, cd.size_gb, force=True)  # already checked above
        if ch == "1":
            print("  Loading into VRAM...")
            llm.load(cd.ref)
            ui.ok(f"{cd.ref} is loaded and will stay in memory.")
            after_load(llm)
    except Exception as e:
        ui.err(str(e))
        if "space" in str(e).lower():
            ui.warn("The disk filled up during the download. Delete a model or pick a smaller one.")


def _result_rows(res, offset=0):
    rows = []
    for i, m in enumerate(res, 1 + offset):
        sizes = ", ".join(m.sizes[:6]) + (" ..." if len(m.sizes) > 6 else "")
        if m.cloud:
            sizes = (sizes + " " if sizes else "") + c("cloud", "yellow")
        desc = m.description.split(". ")[0]
        rows.append((i, c(m.name, "bold"), m.pulls, sizes or "-", ",".join(m.capabilities[:3]),
                     (desc[:46] + "...") if len(desc) > 46 else desc))
    return rows


def flow_search(llm):
    print("  Search ollama.com/search. Examples:  coder   |   /vision   |   qwen /tools   |   /newest")
    print("  Filters: /tools /vision /thinking /embedding /cloud /newest /popular.  Or paste a model like qwen3:8b")
    q = ask("Search (blank = all models)")
    if q in ("q", "b"):
        return
    if q.startswith("http") and "/library/" in q:
        q = q.split("/library/", 1)[1].strip("/")
    if ":" in q and " " not in q:  # direct model reference
        return model_actions(llm, library.make_candidate(q, llm.hw, llm.settings.context, llm.settings.kv_cache))
    words = [w for w in q.split() if not w.startswith("/")]
    flags = {w[1:].lower() for w in q.split() if w.startswith("/")}
    cap = next((f for f in ("tools", "vision", "thinking", "embedding", "cloud") if f in flags), None)
    sort = "newest" if "newest" in flags else "popular" if "popular" in flags else None
    query = " ".join(words)

    shown, page, hidden = [], 1, 0
    while True:
        print(f"  Fetching page {page} from ollama.com...")
        try:
            res = library.search(query, page, sort, cap)
        except library.LibraryError as e:
            ui.err(str(e))
            if not shown:
                return
            res = []
        names = {m.name for m in shown}
        res = [m for m in res if m.name not in names]
        if cap != "cloud":
            hidden += sum(1 for m in res if not m.local)
            res = [m for m in res if m.local]
        if not res and not shown:
            ui.warn("No local models found (cloud-only models are hidden; add /cloud to show them).")
            return
        title(f"ollama.com: '{query or 'all models'}'{' /' + cap if cap else ''}{' /' + sort if sort else ''}")
        table(["#", "NAME", "PULLS", "SIZES", "CAPS", "DESCRIPTION"], _result_rows(res, len(shown)), right_cols=(0,))
        shown += res
        if hidden:
            ui.info(f"({hidden} cloud-only models hidden: they run on Ollama's servers, not your GPU)")
        v = ask(f"Open which model (1-{len(shown)}), m = more, Enter = back")
        if v.lower() == "m":
            page += 1
            continue
        if v.isdigit() and 1 <= int(v) <= len(shown):
            return flow_pick_tag(llm, shown[int(v) - 1])
        return


def _select_from_candidates(llm, cands, extra_hint=""):
    """Number = select, qN = quant picker for row N. Returns after handling."""
    v = ask(f"Choose a number, qN to compare quantizations of row N{extra_hint}. Enter = back")
    if not v or v in ("b", "q"):
        return "back"
    m = re.fullmatch(r"q\s*(\d+)", v.lower())
    if m and 1 <= int(m.group(1)) <= len(cands):
        flow_quants(llm, cands[int(m.group(1)) - 1].ref)
        return "again"
    if v.isdigit() and 1 <= int(v) <= len(cands):
        model_actions(llm, cands[int(v) - 1])
        return "back"
    return v  # custom tag


def flow_pick_tag(llm, m):
    while True:
        print(f"\n  Checking sizes for {m.name} (registry lookup)...")
        cands = library.candidates_for(m, llm.hw, llm.settings.context, llm.settings.kv_cache)
        title(m.name)
        show_candidates(cands, llm)
        r = _select_from_candidates(llm, cands, " or type a custom tag")
        if r == "back":
            return
        if r == "again":
            continue
        model_actions(llm, library.make_candidate(f"{m.name}:{r}", llm.hw, llm.settings.context, llm.settings.kv_cache))
        return


def flow_recommended(llm):
    while True:
        cands = [x for x in llm.recommended() if x.est.verdict != "no"] or llm.recommended()
        title("Recommended for your hardware")
        show_candidates(cands, llm)
        r = _select_from_candidates(llm, cands)
        if r == "again":
            continue
        if r not in ("back",):
            continue
        return


def flow_quants(llm, ref=None):
    """Compare every quantization (q4/q5/q8/fp16...) of one model size: fit, speed, quality."""
    if not ref:
        ref = ask("Model and size, e.g. qwen2.5-coder:7b (or just a name like qwen2.5-coder)")
        if not ref or ref in ("q", "b"):
            return
    name, _, tag = ref.partition(":")
    if not library.size_of(tag):
        print(f"  Looking up sizes for {name}...")
        try:
            lm = next((x for x in library.search(name) if x.name == name), None)
        except library.LibraryError as e:
            ui.err(str(e))
            return
        sizes = [s for s in (lm.sizes if lm else []) if "cloud" not in s]
        if not sizes:
            ui.warn(f"No sized tags found for '{name}'. Use the form name:size, e.g. qwen2.5-coder:7b")
            return
        for i, s in enumerate(sizes, 1):
            print(f"  {i}) {name}:{s}")
        i = pick("Which size", len(sizes))
        if i is None:
            return
        tag = sizes[i]
    size_tag = library.size_of(tag)
    print(f"  Fetching quantizations of {name}:{size_tag} (registry lookups)...")
    try:
        rows = library.quant_candidates(name, size_tag, llm.hw, llm.settings.context, llm.settings.kv_cache)
    except library.LibraryError as e:
        ui.err(str(e))
        return
    best = library.suggest_quant(rows)
    title(f"Quantizations of {name}:{size_tag}")
    ui.info(f"Hardware: {llm.hw.summary()}")
    out = []
    for i, r in enumerate(rows, 1):
        cd = r.cand
        mark = c("<- suggested", "green") if best is r else ""
        out.append((i, r.tag, r.variant, r.quant or "-", library.quality_of(r.quant),
                    f"{cd.size_gb:.1f} GB" + ("" if cd.exact else "~"), f"{cd.est.needed_gb:.1f} GB",
                    fit_cell(cd.est), speed_cell(cd.est), mark))
    table(["#", "TAG", "VARIANT", "QUANT", "QUALITY", "SIZE", "NEEDS", "FIT", "TOK/S", ""], out, right_cols=(0, 5, 6, 8))
    if best:
        ui.info(f"Suggested = best quality that fits your GPU and stays at 15+ tok/s: {name}:{best.tag}")
    else:
        ui.warn("Nothing fits comfortably. Try a smaller size (low-bit quants lose quality).")
    ui.info("Lower quant = smaller and faster but less accurate. q4_K_M is the usual sweet spot; q8 is near-lossless.")
    i = pick("Select which", len(rows))
    if i is not None:
        model_actions(llm, rows[i].cand)


def flow_installed(llm):
    if not ensure_server(llm):
        return
    inst = llm.installed()
    if not inst:
        ui.warn("No models installed yet. Use 'Search' or 'Recommended'.")
        return
    loaded = {m["name"] for m in llm.loaded()}
    rows = []
    for i, m in enumerate(inst, 1):
        cd = llm.candidate(m["name"])
        cd.size_gb = m.get("size", 0) / 1024 ** 3 or cd.size_gb
        mark = c("loaded", "green") if m["name"] in loaded else ""
        active = c("*", "cyan") if m["name"] == llm.settings.model else ""
        rows.append((i, m["name"], f"{cd.size_gb:.1f} GB", fit_cell(cd.est), speed_cell(cd.est), mark, active))
    title("Installed models")
    table(["#", "MODEL", "SIZE", "FIT", "TOK/S", "STATE", ""], rows, right_cols=(0, 2, 4))
    ui.info(f"Disk free: {disk.free_gb(llm.models_dir):.0f} GB in {llm.models_dir}")
    i = pick("Manage which", len(inst))
    if i is None:
        return
    name = inst[i]["name"]
    print(f"\n  {c(name, 'bold')}\n  1) Load / make active\n  2) Unload from VRAM\n  3) Benchmark (real tok/s)\n  4) Delete\n  Enter) back")
    ch = ask("Choose")
    try:
        if ch == "1":
            llm.load(name)
            ui.ok("Loaded.")
            after_load(llm)
        elif ch == "2":
            llm.unload(name)
            ui.ok("Unloaded.")
        elif ch == "3":
            flow_benchmark(llm, name)
        elif ch == "4" and confirm(f"Delete {name}?", False):
            llm.delete(name)
            ui.ok(f"Deleted. Disk free now: {disk.free_gb(llm.models_dir):.0f} GB")
    except Exception as e:
        ui.err(str(e))


def flow_benchmark(llm, name=None):
    name = name or llm.settings.model
    if not name or not ensure_server(llm):
        ui.warn("Select/load a model first.")
        return
    print(f"  Benchmarking {name} (first run includes load time)...")
    try:
        r = llm.benchmark(name)
    except Exception as e:
        ui.err(str(e))
        return
    est = llm.candidate(name).est
    gen = c(f"{r['tok_s']:.1f} tok/s", "bold", "green")
    guess = f"   (estimate was ~{est.tok_s:.0f})" if est.tok_s else ""
    print(f"  Generation : {gen}{guess}")
    print(f"  Prompt eval: {r['prompt_tok_s']:.0f} tok/s   Load time: {r['load_s']:.1f}s")
    if "gpu_pct" in r:
        print(f"  Placement  : {r['gpu_pct']:.0f}% on GPU ({r['loaded_gb']:.1f} GB in memory)")
        if r["gpu_pct"] < 99:
            ui.warn("Model is partly on CPU. Try a smaller model/quant or lower context in Settings.")


def open_tunnel_quiet(llm):
    if not ensure_server(llm):
        return None
    try:
        return llm.open_tunnel()
    except Exception as e:
        ui.err(str(e))
        return None


def flow_open_tunnel(llm):
    url = open_tunnel_quiet(llm)
    if not url:
        return
    print("\n  " + c(url, "bold", "green"))
    ui.warn("Anyone with this link can use your GPU. It changes every time you reopen the tunnel.")
    ui.info("Next: menu 'Connect your tools' prints settings for Cline, Aider, Cursor, Claude Code and more.")


def flow_connect(llm):
    """Print ready-to-paste settings for coding tools."""
    if not llm.settings.model:
        ui.warn("Load a model first (Quick start / Recommended / Search).")
        return
    if not llm.url:
        if not confirm("These tools need the public tunnel link. Open it now?"):
            return
        if not open_tunnel_quiet(llm):
            return
    while True:
        title(f"Connect your tools  ({llm.settings.model})")
        print(f"  Link: {c(llm.url, 'bold', 'green')}\n")
        for i, (_, label, _) in enumerate(TOOLS, 1):
            print(f"  {c(i, 'cyan')}) {label}")
        print(f"  {c('a', 'cyan')}) Show all     {c('s', 'cyan')}) Save all to a file")
        v = ask("Choose (Enter = back)").lower()
        if not v or v in ("b", "q"):
            return
        if v == "a":
            print("\n" + llm.connect())
        elif v == "s":
            base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
            path = os.path.join(base, "notebook_llm_connect.txt")
            with open(path, "w") as f:
                f.write(llm.connect())
            ui.ok(f"Saved to {path}")
        elif v.isdigit() and 1 <= int(v) <= len(TOOLS):
            print("\n" + llm.connect(TOOLS[int(v) - 1][0]))
        print()


def flow_tunnel(llm):
    if not llm.url:
        return flow_open_tunnel(llm)
    print(f"\n  Tunnel: {c(llm.url, 'bold', 'green')}\n  1) Connect your tools (Cline, Aider, Cursor...)\n  2) Close tunnel\n  3) New tunnel (new link)\n  Enter) back")
    ch = ask("Choose")
    if ch == "1":
        flow_connect(llm)
    elif ch == "2":
        llm.close_tunnel()
        ui.ok("Tunnel closed.")
    elif ch == "3":
        llm.close_tunnel()
        flow_open_tunnel(llm)


def flow_server(llm):
    up = llm.server.is_running()
    print(f"\n  Server is {c('running', 'green') if up else c('stopped', 'red')}\n  1) {'Restart' if up else 'Start'}\n  2) Stop\n  3) Show log tail + GPU report\n  Enter) back")
    ch = ask("Choose")
    try:
        if ch == "1":
            llm.restart_server() if up else llm.start_server()
            ui.ok("Server running.")
        elif ch == "2":
            llm.stop_server()
            ui.ok("Stopped.")
        elif ch == "3":
            print(llm.server.gpu_report() or "(no GPU lines in log)")
            print(c("--- log ---", "gray"))
            print(llm.server.log_tail(15))
    except Exception as e:
        ui.err(str(e))


def flow_settings(llm):
    s = llm.settings
    print(f"\n  1) Context length : {s.context or 'auto'}\n  2) KV cache type  : {s.kv_cache}   (f16 | q8_0 | q4_0)\n"
          f"  3) Flash attention: {s.flash_attention}\n  4) Keep-alive     : {s.keep_alive}   (-1 = forever, or e.g. 30m)\n"
          f"  5) Models folder  : {llm.models_dir}  ({disk.free_gb(llm.models_dir):.0f} GB free)\n  Enter) back")
    ch = ask("Change which")
    if ch == "1":
        v = ask("Context tokens (number, or 'auto')", "auto")
        s.context = None if v == "auto" else int(v) if v.isdigit() else s.context
    elif ch == "2":
        v = ask("KV cache", s.kv_cache)
        s.kv_cache = v if v in ("f16", "q8_0", "q4_0") else s.kv_cache
    elif ch == "3":
        s.flash_attention = confirm("Enable flash attention?", s.flash_attention)
    elif ch == "4":
        s.keep_alive = ask("Keep-alive", s.keep_alive)
    elif ch == "5":
        ui.warn("Models already downloaded stay in the old folder and will not be visible from the new one.")
        for i, d in enumerate(disk.candidate_dirs(), 1):
            print(f"  {i}) {d}  ({disk.free_gb(d):.0f} GB free)")
        v = ask("Folder path or number", llm.models_dir)
        dirs = disk.candidate_dirs()
        s.models_dir = dirs[int(v) - 1] if v.isdigit() and 1 <= int(v) <= len(dirs) else v
    else:
        return
    s.save()
    ui.ok("Saved.")
    if llm.server.is_running() and confirm("Restart server to apply now?"):
        llm.restart_server()
        if s.model and llm.client.has_model(s.model):
            llm.load(s.model)


def flow_quickstart(llm):
    c0 = llm.best_fit()
    show_estimate_detail(c0)
    print("\n  Quick start = install, start server, pull the best-fitting model, load it, open tunnel.")
    m = ask("Model to use ('auto' = shown above)", "auto")
    try:
        url = llm.quickstart(m)
    except Exception as e:
        ui.err(str(e))
        return
    ui.ok(f"Ready: {llm.settings.model}")
    print("\n  " + c(url or "(no tunnel)", "bold", "green"))
    if url and confirm("Show setup for your coding tools now?"):
        flow_connect(llm)


MENU = [
    ("Quick start (auto: install, model, tunnel)", flow_quickstart),
    ("Search Ollama library", flow_search),
    ("Recommended models for my GPU", flow_recommended),
    ("Quant picker (compare q4 / q5 / q8 / fp16)", flow_quants),
    ("Installed models (load / benchmark / delete)", flow_installed),
    ("Connect your tools (Cline, Aider, Cursor, Claude Code...)", flow_connect),
    ("Tunnel (open / get link / close)", flow_tunnel),
    ("Server (start / stop / logs)", flow_server),
    ("Settings (context, KV cache, keep-alive, folder)", flow_settings),
    ("Benchmark active model", flow_benchmark),
]


def run(argv=None) -> None:
    """Interactive menu. Call this from a notebook cell: `import notebook_llm; notebook_llm.run()`."""
    llm = NotebookLLM()
    while True:
        header(llm)
        print()
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {c(i, 'cyan')}) {label}")
        print(f"  {c('0', 'cyan')}) Quit (server and tunnel keep running)")
        try:
            ch = ask("Choose")
            if ch in ("0", "q", "quit", "exit"):
                return
            if ch.isdigit() and 1 <= int(ch) <= len(MENU):
                MENU[int(ch) - 1][1](llm)
        except KeyboardInterrupt:
            print()
            return


# ---------- non-interactive commands (for `!notebook-llm ...`) ----------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="notebook-llm", description="Run with no arguments for the interactive menu.")
    p.add_argument("--no-color", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("gpus", help="show hardware + recommended models")
    s = sub.add_parser("search", help="search the Ollama library")
    s.add_argument("query", nargs="?", default="")
    k = sub.add_parser("check", help="will MODEL fit? estimated tok/s")
    k.add_argument("model")
    k.add_argument("--ctx", type=int)
    qn = sub.add_parser("quants", help="compare quantizations, e.g. qwen2.5-coder:7b")
    qn.add_argument("model")
    cn = sub.add_parser("connect", help="print settings for a tool (continue, cline, aider, openwebui, claude-code, cursor, antigravity, python)")
    cn.add_argument("tool", nargs="?")
    q = sub.add_parser("quickstart", help="install, start, pull best model, open tunnel")
    q.add_argument("--model", default="auto")
    q.add_argument("--no-tunnel", action="store_true")
    sub.add_parser("status")
    sub.add_parser("stop", help="stop tunnel and server")
    a = p.parse_args(argv)
    if a.no_color:
        os.environ["NO_COLOR"] = "1"

    if a.cmd is None:
        return run()
    llm = NotebookLLM()
    if a.cmd == "gpus":
        print(llm.hw.summary())
        show_candidates(llm.recommended(), llm)
    elif a.cmd == "search":
        for m in library.search(a.query)[:20]:
            print(f"{m.name:28} {m.pulls:>8}  {', '.join(m.sizes)}{'  [cloud]' if m.cloud else ''}")
    elif a.cmd == "check":
        show_estimate_detail(library.make_candidate(a.model, llm.hw, a.ctx, llm.settings.kv_cache))
    elif a.cmd == "quants":
        name, _, tag = a.model.partition(":")
        rows = library.quant_candidates(name, library.size_of(tag) or tag, llm.hw, llm.settings.context, llm.settings.kv_cache)
        best = library.suggest_quant(rows)
        out = [(r.tag, library.quality_of(r.quant), f"{r.cand.size_gb:.1f} GB", f"{r.cand.est.needed_gb:.1f} GB",
                fit_cell(r.cand.est), speed_cell(r.cand.est), "<- suggested" if r is best else "") for r in rows]
        table(["TAG", "QUALITY", "SIZE", "NEEDS", "FIT", "TOK/S", ""], out, right_cols=(2, 3, 5))
    elif a.cmd == "connect":
        try:
            print(llm.connect(a.tool))
        except KeyError as e:
            raise SystemExit(str(e).strip("\"'"))
    elif a.cmd == "quickstart":
        url = llm.quickstart(a.model, tunnel=not a.no_tunnel)
        print(url or "")
        print(llm.connect())
    elif a.cmd == "status":
        print(f"server={llm.server.is_running()} model={llm.settings.model} tunnel={llm.url}")
    elif a.cmd == "stop":
        llm.close_tunnel()
        llm.stop_server()


if __name__ == "__main__":
    main()

"""
Nebula X Hackathon 2026 — Problem Statement 3 (Train Condition Monitoring)

Single app covering all 4 subsystems, per Section 4.1 item 3 of the spec: a non-technical user
selects a subsystem, uploads data file(s), and gets predictions back on screen with a download.

This same app is what should be used to generate the predictions submitted in predictions.zip
(Section 4.1 item 2) — run the held-out test files through it, then download the result.

WHAT THIS APP IS FOR
--------------------
Part 1 of the problem is the model. Part 2 — per the organisers' own FAQ — is translating what the
model says into something an operator making a split-second call, or an engineer new to the data,
can act on without reading a paper first. So the results screen leads with a verdict and a
recommended action, keeps the confidence and the supporting evidence one click away, and keeps the
raw numbers available but out of the way. The exported submission CSV is unchanged and unchanged-
able by any of this: it is always exactly the columns the spec asks for.

The dark console styling is deliberate — this is meant to read as a depot wallboard sitting
alongside the existing maintenance dashboards, not as a data-science notebook.

HOW TEAMMATES ADD A SUBSYSTEM
-----------------------------
Implement predictors/<subsystem>.py with:

  REQUIRED
    SUBSYSTEM_NAME, OUTPUT_FILENAME, OUTPUT_COLUMNS, ACCEPTED_FILE_TYPES
    predict(uploaded_files) -> DataFrame whose first columns are exactly OUTPUT_COLUMNS.
      Extra columns are welcome — they're used for the operator view and stripped from the CSV.
      Columns whose name starts with "_" are hidden everywhere except your own hooks.

  OPTIONAL (each one adds a layer; without them the app still works, just plainer)
    interpret(row)     -> {"status": "good"|"warning"|"serious"|"critical",
                           "band", "headline", "action", "evidence" [list],
                           "sort_value", "metric_label", "metric_value"}
    batch_chart(df)    -> (kind, tidy_df, meta) | None   — one chart for the whole batch
    detail_chart(row)  -> (kind, tidy_df, meta) | None   — one chart per finding

  Chart kinds the app can render, and the columns each expects:
    "magnitude_bars" : label, value              + meta{value_title, reference_lines, note}
    "ranked_bars"    : label, value, highlight   + meta{value_title, note}
    "grouped_bands"  : band, series, value       + meta{x_title, y_title, note}
    "composition"    : label, value, status      + meta{value_title, note}

Nothing in this file needs to change when a subsystem is added.
"""
import importlib
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nebula X — Train Condition Monitoring", layout="wide")


# Streamlit renamed `use_container_width` to `width`. Older builds only know the old name;
# newer ones warn and will eventually drop it. Detect once and use whichever the installed
# version actually supports, so this runs on a teammate's machine whatever version they have.
try:
    import inspect as _inspect
    _WIDE = ({"width": "stretch"}
             if "width" in _inspect.signature(st.dataframe).parameters
             else {"use_container_width": True})
except Exception:
    _WIDE = {"use_container_width": True}

SUBSYSTEMS = {
    "SHM (Structural Health Monitoring)": "shm",
    "Door": "door",
    "ACV (Air Conditioning & Ventilation)": "acv",
    "Rail Corrugation": "rail",
}

# ---- Console palette -------------------------------------------------------------------------
# Status colours are the fixed good/warning/serious/critical set and are never themed. They were
# validated against this navy surface (all four clear 3:1). Because warning and serious sit close
# in hue, every status is rendered as icon + written label as well as colour, so colour is never
# the only thing carrying the meaning.
BG = "#0a1628"
PANEL = "#102844"
ACCENT = "#22d3ee"
INK = "#e8f1ff"
INK_DIM = "#8fa8c8"
GRID = "rgba(143,168,200,0.16)"
SERIES_A = "#3987e5"   # slot 1 — validated pair with SERIES_B on this surface
SERIES_B = "#d95926"   # slot 2
MUTED_BAR = "#2c4f7c"

STATUS = {
    "good":     {"colour": "#0ca30c", "icon": "●", "label": "OK",            "rank": 3},
    "warning":  {"colour": "#fab219", "icon": "▲", "label": "Monitor",       "rank": 2},
    "serious":  {"colour": "#ec835a", "icon": "◆", "label": "Action needed", "rank": 1},
    "critical": {"colour": "#d03b3b", "icon": "■", "label": "Escalate",      "rank": 0},
}

st.markdown(f"""
<style>
  .stApp {{ background:
      radial-gradient(1200px 600px at 15% -10%, #16365c 0%, transparent 60%), {BG}; }}

  .console-head {{ display:flex; align-items:center; justify-content:space-between;
      gap:1rem; padding:0.55rem 1.1rem; margin-bottom:1.1rem; border-radius:8px;
      background: linear-gradient(90deg, {PANEL} 0%, rgba(16,40,68,0.35) 70%, transparent 100%);
      border-left: 3px solid {ACCENT}; }}
  .console-title {{ font-size:1.45rem; font-weight:800; letter-spacing:0.01em; color:{INK};
      line-height:1.15; }}
  .console-sub {{ font-size:0.76rem; color:{INK_DIM}; letter-spacing:0.09em;
      text-transform:uppercase; }}
  .console-meta {{ text-align:right; font-size:0.74rem; color:{INK_DIM};
      font-family:ui-monospace, Consolas, monospace; letter-spacing:0.04em; white-space:nowrap; }}
  .console-meta b {{ color:{ACCENT}; font-weight:600; }}

  .sect {{ display:flex; align-items:center; gap:0.5rem; margin:1.4rem 0 0.7rem 0; }}
  .sect-bar {{ width:3px; height:15px; background:{ACCENT}; border-radius:2px; }}
  .sect-txt {{ font-size:0.82rem; font-weight:700; letter-spacing:0.13em;
      text-transform:uppercase; color:{ACCENT}; }}
  .sect-rule {{ flex:1; height:1px;
      background:linear-gradient(90deg, rgba(34,211,238,0.35), transparent); }}

  .tile {{ background:{PANEL}; border:1px solid rgba(34,211,238,0.16); border-radius:9px;
      padding:0.75rem 0.95rem; height:100%; }}
  .tile-v {{ font-size:1.85rem; font-weight:800; line-height:1.1; color:{INK}; }}
  .tile-l {{ font-size:0.7rem; color:{INK_DIM}; letter-spacing:0.09em; text-transform:uppercase;
      margin-top:0.15rem; }}

  .vcard {{ background:linear-gradient(90deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012));
      border-left:4px solid var(--c); border-radius:7px; padding:0.8rem 1.05rem;
      margin-bottom:0.55rem; }}
  .vtop {{ display:flex; align-items:baseline; gap:0.65rem; flex-wrap:wrap; }}
  .vchip {{ color:var(--c); font-weight:800; font-size:0.72rem; letter-spacing:0.1em;
      white-space:nowrap; }}
  .vfile {{ font-family:ui-monospace, Consolas, monospace; font-size:0.78rem; color:{INK_DIM}; }}
  .vhead {{ font-size:1.0rem; font-weight:650; color:{INK}; margin:0.35rem 0 0.2rem 0; }}
  .vact {{ font-size:0.9rem; color:#c6d8ef; line-height:1.5; }}

  /* ---- dark console theme, applied here rather than via .streamlit/config.toml -------------
     Keeping the theme in this file means the app is a single self-contained script: there is no
     config directory to get dropped when the zip is extracted, and nothing to parse differently
     between Streamlit versions. */
  .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {{ color:{INK}; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  [data-testid="stMarkdownContainer"] small,
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:{INK_DIM}; }}

  /* inputs: select, text, upload */
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
  .stApp input[role="combobox"], .stApp .react-aria-ComboBox > div {{
      background:{PANEL} !important; color:{INK} !important;
      border-color:rgba(34,211,238,0.22) !important; }}
  .stApp input, .stApp textarea {{ color:{INK} !important; }}
  ul[role="listbox"], [data-baseweb="popover"] div, .react-aria-Popover {{
      background:{PANEL} !important; color:{INK} !important; }}
  section[data-testid="stFileUploaderDropzone"] {{ background:rgba(16,40,68,0.55);
      border:1px dashed rgba(34,211,238,0.3); }}
  [data-testid="stFileUploaderDropzone"] * {{ color:{INK} !important; }}
  [data-testid="stFileUploaderFile"] {{ background:{PANEL}; border-radius:6px; }}
  [data-testid="stFileUploaderFile"] * {{ color:{INK} !important; }}
  [data-testid="stFileUploaderFile"] small {{ color:{INK_DIM} !important; }}

  /* buttons */
  .stApp button[kind="secondary"], .stApp button[data-testid="stBaseButton-secondary"] {{
      background:{PANEL}; color:{INK}; border:1px solid rgba(34,211,238,0.25); }}
  .stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"] {{
      background:{ACCENT}; color:#04121a; border:1px solid {ACCENT}; font-weight:600; }}
  .stApp button:hover {{ border-color:{ACCENT}; }}

  /* panels, alerts, tables */
  div[data-testid="stExpander"] {{ border:1px solid rgba(34,211,238,0.14);
      border-radius:8px; background:rgba(16,40,68,0.4); }}
  div[data-testid="stExpander"] summary {{ background:{PANEL} !important; }}
  div[data-testid="stExpander"] details {{ background:transparent !important; }}
  div[data-testid="stExpander"] details > div {{ background:rgba(16,40,68,0.4) !important; }}
  div[data-testid="stExpander"] summary,
  div[data-testid="stExpander"] summary * {{ color:{INK} !important; }}
  [data-testid="stAlert"] {{ background:rgba(16,40,68,0.7);
      border:1px solid rgba(34,211,238,0.22); border-radius:8px; }}
  [data-testid="stAlert"] * {{ color:{INK} !important; }}
  [data-testid="stDataFrame"] {{ background:{PANEL}; border-radius:8px; }}
  .stApp hr {{ border-color:rgba(143,168,200,0.18); }}
</style>
""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sect"><div class="sect-bar"></div>'
                f'<div class="sect-txt">{title}</div><div class="sect-rule"></div></div>',
                unsafe_allow_html=True)


def load_predictor(module_name):
    try:
        return importlib.import_module(f"predictors.{module_name}")
    except Exception as e:
        return e


def visible_columns(df):
    return [c for c in df.columns if not c.startswith("_")]


def submission_frame(df, predictor):
    """The CSV that goes into predictions.zip — exactly OUTPUT_COLUMNS, nothing else, ever."""
    return df[list(predictor.OUTPUT_COLUMNS)].copy()


# ---- Chart rendering -------------------------------------------------------------------------

def _style(chart, height):
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK_DIM, titleColor=INK_DIM, gridColor=GRID,
                            domainColor="rgba(143,168,200,0.3)",
                            tickColor="rgba(143,168,200,0.3)", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=INK, titleColor=INK_DIM, labelFontSize=11))


def _bar_tooltip(value_title, fmt=".4f"):
    return [alt.Tooltip("label:N", title="Item"),
            alt.Tooltip("value:Q", title=value_title, format=fmt)]


def chart_magnitude_bars(data, meta, statuses):
    data = data.copy()
    data["Status"] = [STATUS[s]["label"] for s in statuses]
    capped = len(data) > 25
    if capped:
        data = data.nlargest(25, "value")

    order = [STATUS[k]["label"] for k in ["critical", "serious", "warning", "good"]]
    colours = [STATUS[k]["colour"] for k in ["critical", "serious", "warning", "good"]]

    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=13).encode(
        x=alt.X("value:Q", title=meta["value_title"], axis=alt.Axis(tickCount=6)),
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=170, domain=False, ticks=False,
                              labelColor=INK)),
        color=alt.Color("Status:N", scale=alt.Scale(domain=order, range=colours),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        tooltip=_bar_tooltip(meta["value_title"]) + [alt.Tooltip("Status:N")],
    )
    layers = [bars]
    for value, label in meta.get("reference_lines", []):
        ref = pd.DataFrame({"v": [value], "l": [label]})
        layers.append(alt.Chart(ref).mark_rule(strokeDash=[4, 3], color=INK_DIM, strokeWidth=1.5)
                      .encode(x="v:Q", tooltip=alt.Tooltip("l:N", title="Reference")))
        layers.append(alt.Chart(ref).mark_text(align="left", dx=5, dy=-5, baseline="top",
                                               color=INK_DIM, fontSize=10)
                      .encode(x="v:Q", text="l:N"))
    st.altair_chart(_style(alt.layer(*layers), max(170, 26 * len(data))),
                    **_WIDE)
    note = meta.get("note", "")
    if capped:
        note = "Showing the 25 highest-severity files. " + note
    if note:
        st.caption(note)


def chart_ranked_bars(data, meta):
    data = data.copy()
    data["Highlighted"] = data["highlight"].map({True: "Flagged", False: "Reference"})
    order = list(data["label"])
    # Let the band scale size the bars (with explicit padding) rather than pinning a pixel height
    # — a fixed mark height makes bars collide as soon as the row count grows.
    y_enc = alt.Y("label:N", sort=order, title=None,
                  scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25),
                  axis=alt.Axis(domain=False, ticks=False, labelColor=INK, labelLimit=170,
                                labelOverlap=False))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("value:Q", title=meta["value_title"], axis=alt.Axis(tickCount=5)),
        y=y_enc,
        color=alt.Color("Highlighted:N",
                        scale=alt.Scale(domain=["Flagged", "Reference"],
                                        range=[STATUS["serious"]["colour"], MUTED_BAR]),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        tooltip=_bar_tooltip(meta["value_title"]),
    )
    labels = alt.Chart(data).mark_text(align="left", dx=6, color=INK, fontSize=11).encode(
        x="value:Q", y=y_enc, text=alt.Text("value:Q", format=".3f"))
    st.altair_chart(_style(bars + labels, 34 * len(data) + 30), **_WIDE)
    if meta.get("note"):
        st.caption(meta["note"])


def chart_grouped_bands(data, meta):
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("band:N", title=meta["x_title"], sort=None,
                axis=alt.Axis(labelAngle=0, labelColor=INK)),
        xOffset="series:N",
        y=alt.Y("value:Q", title=meta["y_title"], axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=["Side I", "Side II"],
                                        range=[SERIES_A, SERIES_B]),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        tooltip=[alt.Tooltip("band:N", title="Wavelength"), alt.Tooltip("series:N", title="Side"),
                 alt.Tooltip("value:Q", title="Energy share", format=".1%")],
    )
    st.altair_chart(_style(bars, 230), **_WIDE)
    if meta.get("note"):
        st.caption(meta["note"])


def chart_composition(data, meta):
    data = data.copy()
    data["Status"] = [STATUS[s]["label"] for s in data["status"]]
    order = [STATUS[k]["label"] for k in ["critical", "serious", "warning", "good"]]
    colours = [STATUS[k]["colour"] for k in ["critical", "serious", "warning", "good"]]
    y_enc = alt.Y("label:N", sort="-x", title=None,
                  scale=alt.Scale(paddingInner=0.4, paddingOuter=0.3),
                  axis=alt.Axis(domain=False, ticks=False, labelColor=INK, labelOverlap=False))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("value:Q", title=meta["value_title"], axis=alt.Axis(tickCount=6)),
        y=y_enc,
        color=alt.Color("Status:N", scale=alt.Scale(domain=order, range=colours),
                        legend=alt.Legend(orient="top", title=None, symbolType="square")),
        tooltip=_bar_tooltip(meta["value_title"], fmt="d"),
    )
    labels = alt.Chart(data).mark_text(align="left", dx=6, color=INK, fontSize=12,
                                       fontWeight=600).encode(
        x="value:Q", y=y_enc, text="value:Q")
    st.altair_chart(_style(bars + labels, 48 * len(data) + 30), **_WIDE)
    if meta.get("note"):
        st.caption(meta["note"])


def render_chart(spec, statuses=None):
    """spec is (kind, tidy_df, meta) from a predictor hook."""
    if not spec:
        return
    kind, data, meta = spec
    if data is None or len(data) == 0:
        return
    try:
        if kind == "magnitude_bars":
            chart_magnitude_bars(data, meta, statuses)
        elif kind == "ranked_bars":
            chart_ranked_bars(data, meta)
        elif kind == "grouped_bands":
            chart_grouped_bands(data, meta)
        elif kind == "composition":
            chart_composition(data, meta)
    except Exception as e:
        # A chart is a nice-to-have; it must never take the findings down with it.
        st.caption(f"(chart unavailable: {e})")


def render_tiles(counts, total):
    cols = st.columns(len(counts) + 1)
    with cols[0]:
        st.markdown(f'<div class="tile"><div class="tile-v" style="color:{ACCENT}">{total}</div>'
                    f'<div class="tile-l">files analysed</div></div>', unsafe_allow_html=True)
    for i, (key, n) in enumerate(counts.items(), start=1):
        s = STATUS[key]
        with cols[i]:
            st.markdown(f'<div class="tile"><div class="tile-v" style="color:{s["colour"]}">'
                        f'{s["icon"]} {n}</div><div class="tile-l">{s["label"]}</div></div>',
                        unsafe_allow_html=True)


def render_card(file_id, info):
    s = STATUS.get(info["status"], STATUS["warning"])
    st.markdown(
        f'<div class="vcard" style="--c:{s["colour"]}"><div class="vtop">'
        f'<span class="vchip">{s["icon"]} {s["label"].upper()}</span>'
        f'<span class="vfile">{file_id}</span>'
        f'<span class="vfile">&middot; {info["band"]}</span></div>'
        f'<div class="vhead">{info["headline"]}</div>'
        f'<div class="vact">{info["action"]}</div></div>', unsafe_allow_html=True)


# ---- Page ------------------------------------------------------------------------------------

st.markdown(
    f'<div class="console-head"><div>'
    f'<div class="console-title">Train Condition Monitoring</div>'
    f'<div class="console-sub">Nebula X Hackathon 2026 &middot; Problem Statement 3</div></div>'
    f'<div class="console-meta">{datetime.now():%d/%m/%Y %H:%M}<br><b>FLEET DIAGNOSTICS</b>'
    f'</div></div>', unsafe_allow_html=True)

left, right = st.columns([2, 1])
with left:
    subsystem_label = st.selectbox("Select a subsystem", list(SUBSYSTEMS.keys()))
module_name = SUBSYSTEMS[subsystem_label]
predictor = load_predictor(module_name)

if isinstance(predictor, Exception):
    st.error(f"Couldn't load the {subsystem_label} predictor: {predictor}")
    st.stop()

with right:
    st.write("")
    st.info(f"Accepts **{', '.join(predictor.ACCEPTED_FILE_TYPES)}** files")

uploaded_files = st.file_uploader(
    "Upload data file(s) — the held-out test files, or any recording of the same shape",
    type=predictor.ACCEPTED_FILE_TYPES, accept_multiple_files=True, key=module_name)

if not uploaded_files:
    st.info("Upload at least one file to run the analysis.")
else:
    # Results are cached so expanding a panel doesn't re-run the model, but the cache is keyed to
    # the exact set of uploaded files — change the upload and the stale result is dropped rather
    # than silently shown against the new files.
    upload_sig = (module_name, tuple(f.name for f in uploaded_files), len(uploaded_files))
    if st.session_state.get("upload_sig") != upload_sig:
        st.session_state.pop(f"result_{module_name}", None)
        st.session_state["upload_sig"] = upload_sig

    if st.button(f"Analyse {len(uploaded_files)} file(s)", type="primary"):
        with st.spinner(f"Running the {subsystem_label} model..."):
            try:
                result_df = predictor.predict(uploaded_files)
            except NotImplementedError as e:
                st.warning(f"{subsystem_label} isn't wired up yet: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        st.session_state[f"result_{module_name}"] = result_df

    result_df = st.session_state.get(f"result_{module_name}")

    if result_df is not None:
        has_interpret = hasattr(predictor, "interpret")

        if has_interpret:
            id_col = "file_id" if "file_id" in result_df.columns else predictor.OUTPUT_COLUMNS[0]
            # Keyed by position, not filename, so two files with the same name can't collide.
            entries = [(i, str(row[id_col]), predictor.interpret(row))
                       for i, (_, row) in enumerate(result_df.iterrows())]

            counts = {}
            for k in ["critical", "serious", "warning", "good"]:
                n = sum(1 for _, _, info in entries if info["status"] == k)
                if n:
                    counts[k] = n

            section("Fleet summary")
            render_tiles(counts, len(result_df))
            n_attention = sum(1 for _, _, i in entries if i["status"] != "good")
            st.caption(f"{n_attention} of {len(result_df)} file(s) need a human decision."
                       if n_attention else "Nothing in this batch needs attention.")

            if hasattr(predictor, "batch_chart") and len(result_df) > 1:
                section("Severity across the batch")
                render_chart(predictor.batch_chart(result_df),
                             statuses=[i["status"] for _, _, i in entries])

            ordered = sorted(entries,
                             key=lambda e: (STATUS[e[2]["status"]]["rank"], e[2]["sort_value"]))
            attention = [e for e in ordered if e[2]["status"] != "good"]
            clear = [e for e in ordered if e[2]["status"] == "good"]

            section("Findings")
            for idx, key, info in attention:
                render_card(key, info)
                with st.expander("Why the system says this",
                                 expanded=(len(attention) <= 3)):
                    for line in info["evidence"]:
                        st.markdown(f"- {line}")
                    if hasattr(predictor, "detail_chart"):
                        render_chart(predictor.detail_chart(result_df.iloc[idx]))

            if clear:
                with st.expander(f"{len(clear)} file(s) with no action required",
                                 expanded=not attention):
                    for _, key, info in clear:
                        st.markdown(
                            f'<div class="vcard" style="--c:{STATUS["good"]["colour"]}">'
                            f'<div class="vtop">'
                            f'<span class="vchip">{STATUS["good"]["icon"]} OK</span>'
                            f'<span class="vfile">{key}</span></div>'
                            f'<div class="vact">{info["headline"]}</div></div>',
                            unsafe_allow_html=True)
        else:
            st.success(f"Done — {len(result_df)} row(s) predicted.")

        with st.expander("Full results table (all computed values)"):
            st.dataframe(result_df[visible_columns(result_df)], **_WIDE)

        section("Export")
        sub_df = submission_frame(result_df, predictor)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(f"Download {predictor.OUTPUT_FILENAME}",
                               data=sub_df.to_csv(index=False).encode("utf-8"),
                               file_name=predictor.OUTPUT_FILENAME, mime="text/csv",
                               type="primary")
            st.caption(f"Competition format — exactly the columns "
                       f"`{', '.join(predictor.OUTPUT_COLUMNS)}`. Goes straight into "
                       f"predictions.zip.")
        with c2:
            if has_interpret:
                report = pd.DataFrame([
                    {"file_id": k, "status": STATUS[i["status"]]["label"],
                     "assessment": i["band"], i["metric_label"]: i["metric_value"],
                     "finding": i["headline"], "recommended_action": i["action"]}
                    for _, k, i in ordered])
                st.download_button("Download maintenance report",
                                   data=report.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{module_name}_maintenance_report.csv",
                                   mime="text/csv")
                st.caption("Plain-language version for the depot — statuses, findings and "
                           "recommended actions, no model internals.")

with st.expander("Which subsystems are wired up?"):
    for label, mod in SUBSYSTEMS.items():
        p = load_predictor(mod)
        if isinstance(p, Exception):
            st.write(f"- **{label}**: not loadable — {p}")
        elif not hasattr(p, "predict"):
            st.write(f"- **{label}**: missing predict()")
        else:
            bits = []
            if hasattr(p, "interpret"):
                bits.append("operator guidance")
            if hasattr(p, "batch_chart") or hasattr(p, "detail_chart"):
                bits.append("charts")
            st.write(f"- **{label}**: ready" + (f" — {', '.join(bits)}" if bits else
                                                " (predictions only)"))

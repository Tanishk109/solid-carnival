"""Minimal Streamlit interface for the frozen SpamScope experiment."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import MODEL_LABELS, SafeTextEngine


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="SpamScope", page_icon="◎", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root{--mint:#72f2c7;--cyan:#63cfff;--panel:rgba(16,32,47,.66);--line:rgba(255,255,255,.10);--muted:#9eb1bd}
    html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
    .stApp{background:radial-gradient(circle at 10% 0,rgba(70,214,174,.13),transparent 24rem),radial-gradient(circle at 90% 4%,rgba(112,94,245,.12),transparent 27rem),#07111b;color:#eef7fa}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stAppViewContainer"]>.main .block-container{max-width:980px;padding-top:1.4rem;padding-bottom:3rem}
    .app-head{display:flex;align-items:center;gap:1rem;padding:1rem 1.1rem;margin-bottom:.85rem;border:1px solid var(--line);border-radius:18px;background:var(--panel);backdrop-filter:blur(20px);box-shadow:0 14px 45px rgba(0,0,0,.18)}
    .identity{display:flex;align-items:center;gap:.8rem}.logo{display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,var(--mint),var(--cyan));color:#06131c;font-weight:900}
    .app-head h1{font-size:1.25rem;line-height:1.1;margin:0;color:#f7fcff;letter-spacing:-.025em}.app-head p{font-size:.78rem;color:var(--muted);margin:.18rem 0 0}
    div[data-testid="stSegmentedControl"]{padding:.3rem;border:1px solid rgba(255,255,255,.08);border-radius:13px;background:rgba(7,18,29,.55)}
    button[data-testid="stBaseButton-segmented_control"],button[data-testid="stBaseButton-segmented_controlActive"]{border-radius:9px!important;font-weight:680!important;box-shadow:none!important}
    button[data-testid="stBaseButton-segmented_control"]{background:transparent!important;color:#9eb2bf!important;border-color:transparent!important}
    button[data-testid="stBaseButton-segmented_controlActive"]{background:linear-gradient(135deg,rgba(114,242,199,.17),rgba(99,207,255,.10))!important;color:#f0fff9!important;border:1px solid rgba(114,242,199,.27)!important}
    button[data-testid="stBaseButton-segmented_control"] p,button[data-testid="stBaseButton-segmented_controlActive"] p{color:inherit!important}
    div[data-baseweb="textarea"]>div,div[data-baseweb="select"]>div{background:rgba(10,25,39,.78)!important;border:1px solid rgba(255,255,255,.11)!important;border-radius:13px!important}
    textarea,input{background:rgba(10,25,39,.78)!important;color:#edf7fb!important;-webkit-text-fill-color:#edf7fb!important;caret-color:var(--mint)!important}
    textarea::placeholder,input::placeholder{color:#718895!important;-webkit-text-fill-color:#718895!important}
    [data-testid="stWidgetLabel"] p{color:#b2c2cb!important}
    .stButton button,.stFormSubmitButton button,[data-testid="stFormSubmitButton"] button{border-radius:11px!important;border:1px solid rgba(114,242,199,.28)!important;background:linear-gradient(135deg,rgba(114,242,199,.17),rgba(99,207,255,.09))!important;color:#effff9!important;font-weight:750!important}
    .stButton button p,.stFormSubmitButton button p{color:inherit!important}
    div[data-testid="stMetric"]{padding:.85rem 1rem;border-radius:16px;background:var(--panel);border:1px solid rgba(255,255,255,.09);box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
    [data-testid="stMetricLabel"]{color:#9fb1bc;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}[data-testid="stMetricValue"]{font-size:1.65rem;color:#f4fbfe}
    .result{padding:1.05rem 1.15rem;margin:.8rem 0;border-radius:17px;border:1px solid rgba(114,242,199,.22);background:var(--panel);backdrop-filter:blur(18px)}
    .result.spam{border-color:rgba(255,118,143,.32)}.result.review{border-color:rgba(255,201,107,.32)}.result-label{font-size:.67rem;color:var(--mint);font-weight:850;letter-spacing:.13em;text-transform:uppercase}.result.spam .result-label{color:#ff8ba0}.result.review .result-label{color:#ffd37d}.result h2{font-size:1.45rem;margin:.2rem 0;color:#f5fbfd}.result p{font-size:.82rem;color:var(--muted);margin:0;line-height:1.5}
    [data-testid="stExpander"]{border:1px solid rgba(255,255,255,.085);border-radius:15px;background:rgba(11,26,39,.52)}
    [data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,.08)}
    hr{border-color:rgba(255,255,255,.08)!important}
    @media(max-width:680px){.app-head{padding:.85rem}.app-head h1{font-size:1.1rem}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> SafeTextEngine:
    return SafeTextEngine(ROOT)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "model_metrics.csv")


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_errors() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "error_analysis.csv")


try:
    engine = load_engine()
    metrics = load_metrics()
    manifest = load_manifest()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Trained project files are missing or invalid: {exc}")
    st.code("python train.py\nstreamlit run app.py")
    st.stop()

st.markdown(
    '<div class="app-head"><div class="identity"><span class="logo">◎</span><div><h1>SpamScope</h1><p>Explainable SMS spam classification</p></div></div></div>',
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "Page",
    ["Analyze", "Results", "About"],
    default="Analyze",
    selection_mode="single",
    label_visibility="collapsed",
    key="page_router",
)


if page == "Analyze":
    st.subheader("Analyze a message")
    label_to_key = {MODEL_LABELS[key]: key for key in engine.available_models}
    model_labels = list(label_to_key)
    default_label = MODEL_LABELS.get(engine.metadata()["default_model"], model_labels[0])

    with st.form("analysis_form", border=False):
        message = st.text_area(
            "SMS message",
            key="draft_message",
            height=140,
            max_chars=2_000,
            placeholder="Paste an English SMS message…",
        )
        with st.expander("Advanced model selection"):
            chosen_label = st.selectbox("Model", model_labels, index=model_labels.index(default_label))
        submitted = st.form_submit_button("Analyze", type="primary", width="stretch")

    if submitted:
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            model_key = label_to_key[chosen_label]
            with st.spinner("Analyzing…"):
                st.session_state["last_analysis"] = {
                    "message": message,
                    "prediction": engine.predict(message, model_key),
                    "explanation": engine.explain_linear(message, model_key),
                }

    analysis = st.session_state.get("last_analysis")
    if analysis:
        prediction = analysis["prediction"]
        if prediction["needs_review"]:
            title, label, note, css_class = (
                "Uncertain result",
                "Review recommended",
                "Confidence is below this model's validation-selected threshold.",
                "review",
            )
        elif prediction["label"]:
            title, label, note, css_class = (
                "Spam identified",
                "Prediction",
                "This is a model estimate, not proof of malicious intent.",
                "spam",
            )
        else:
            title, label, note, css_class = (
                "Not identified as spam",
                "Prediction",
                "This result does not guarantee that the message is safe.",
                "",
            )
        st.markdown(
            f'<div class="result {css_class}"><div class="result-label">{escape(label)}</div><h2>{escape(title)}</h2><p>{escape(note)}</p></div>',
            unsafe_allow_html=True,
        )
        probability_col, confidence_col = st.columns(2)
        probability_col.metric("Spam probability", f"{prediction['spam_probability']:.1%}")
        confidence_col.metric("Confidence", f"{prediction['confidence']:.1%}")
        st.caption(f"Model: {prediction['model_name']}")

        with st.expander("How this was decided"):
            st.write(
                f"Decision threshold: **{prediction['decision_threshold']:.1%}** · Review below: **{prediction['review_confidence']:.1%} confidence**"
            )
            explanation = analysis["explanation"]
            if not explanation["supported"]:
                st.info(explanation["note"])
            else:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Toward spam**")
                    rows = explanation["toward_spam"][:5]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.caption("No positive feature contribution.")
                with right:
                    st.markdown("**Toward ham**")
                    rows = explanation["toward_ham"][:5]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.caption("No negative feature contribution.")
                st.caption("Linear contribution = TF-IDF value × model coefficient, before calibration.")

elif page == "Results":
    deployable = metrics[metrics["model_key"] != "majority"].copy()
    best = deployable.loc[deployable["macro_f1"].idxmax()]
    st.subheader("Test results")
    summary = st.columns(2)
    summary[0].metric("Best macro-F1", f"{best['macro_f1']:.3f}")
    summary[1].metric("Spam recall", f"{best['spam_recall']:.3f}")
    st.caption(f"Best observed model: {best['model']}. Evaluated on {int(best['test_examples']):,} frozen test messages.")

    chart = metrics.set_index("model")[["macro_f1"]]
    st.bar_chart(chart, horizontal=True, height=280)
    concise_columns = ["model", "macro_f1", "spam_precision", "spam_recall", "pr_auc"]
    st.dataframe(
        metrics[concise_columns].style.format({column: "{:.3f}" for column in concise_columns if column != "model"}),
        hide_index=True,
        width="stretch",
    )

    with st.expander("More evaluation details"):
        selected_model = st.selectbox("Model", deployable["model"].tolist(), key="result_model")
        selected_key = str(deployable.loc[deployable["model"] == selected_model, "model_key"].iloc[0])
        matrices = json.loads((ROOT / "results" / "confusion_matrices.json").read_text(encoding="utf-8"))
        matrix = pd.DataFrame(
            matrices[selected_key],
            index=["Actual ham", "Actual spam"],
            columns=["Predicted ham", "Predicted spam"],
        )
        st.markdown("**Confusion matrix**")
        st.dataframe(matrix, width="stretch")
        st.markdown("**Validation-only ablations**")
        st.dataframe(pd.read_csv(ROOT / "results" / "ablation_results.csv"), hide_index=True, width="stretch")

    with st.expander("Reviewed errors"):
        errors = load_errors()
        category = st.selectbox("Category", ["All"] + sorted(errors["category"].unique().tolist()))
        view = errors if category == "All" else errors[errors["category"] == category]
        st.dataframe(
            view[["analysis_model", "message", "true_label", "predicted_label", "category"]],
            hide_index=True,
            width="stretch",
            height=320,
        )

elif page == "About":
    st.subheader("About this experiment")
    counts = st.columns(2)
    counts[0].metric("Unique messages", f"{manifest['deduplication']['clean_rows']:,}")
    counts[1].metric("Duplicates removed", f"{manifest['deduplication']['duplicate_rows_removed']:,}")

    split_frame = pd.DataFrame(
        [{"split": name.title(), "messages": value} for name, value in manifest["split_counts"].items()]
    )
    st.markdown("**Data split**")
    st.dataframe(split_frame, hide_index=True, width="stretch")
    st.write(
        "Duplicates are removed before splitting. Models fit on training data, probabilities are calibrated on validation data, and final scores come from the test set."
    )
    st.warning(
        "Historical English SMS spam classification only. This is not a phishing, malware, fraud, or general safety detector."
    )
    st.markdown(
        f"Dataset: [{manifest['title']}]({manifest['source_url']}) · {manifest['source']} · DOI `{manifest['doi']}` · {manifest['license']}"
    )
    with st.expander("Reproduce"):
        st.code("python train.py\npython evaluate.py\npython -m unittest discover -s tests -v")

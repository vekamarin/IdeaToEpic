"""
IdeaToEpic — Streamlit UI
Run with: streamlit run app.py
"""

import json
import time
import streamlit as st
from idea2epic import run_pipeline, llm, build_voc_prompt
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="IdeaToEpic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 IdeaToEpic")
    st.caption("Multi-agent requirements generator")
    st.divider()

    st.subheader("How it works")
    st.markdown("""
1. **VOC Analyst** — extracts stakeholder needs from raw customer voice
2. **Requirement Architect** — builds Epic → Feature → Story hierarchy
3. **Quality Checker** — audits for traceability and testability
4. **Quality Gate** — loops back if rejected (max 3 attempts)
""")
    st.divider()
    st.caption("Powered by LangGraph + Groq")


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.title("🧠 IdeaToEpic")
st.subheader("Transform customer voice into a structured product backlog")
st.divider()

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    product_domain = st.text_input(
        "Product Domain",
        placeholder="e.g. hospital patient scheduling system",
        help="Describe the product in a short phrase."
    )

with col2:
    voc_mode = st.radio(
        "VOC Input Mode",
        ["Auto-generate VOC", "Write my own VOC"],
        help="Auto-generate creates a realistic multi-persona VOC from the product domain."
    )

generate_voc = voc_mode == "Auto-generate VOC"
voc_input = ""

# ─── Manual VOC input ───
if not generate_voc:
    voc_input = st.text_area(
        "Paste your Voice of Customer text",
        height=200,
        placeholder="Write raw stakeholder interview notes, user feedback, or pain points here..."
    )

# ─── VOC Preview (auto mode) ───
if generate_voc and product_domain:
    with st.expander("👁 Preview auto-generated VOC before running the full pipeline", expanded=False):
        if st.button("Generate VOC Preview"):
            with st.spinner("Generating VOC..."):
                try:
                    prompt = build_voc_prompt(product_domain)
                    response = llm.invoke([HumanMessage(content=prompt)])
                    st.session_state["voc_preview"] = response.content
                except Exception as e:
                    st.error(f"Error: {e}")

        if "voc_preview" in st.session_state:
            st.markdown(st.session_state["voc_preview"])

st.divider()

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────

run_disabled = not product_domain or (not generate_voc and not voc_input.strip())

if st.button("🚀 Generate Backlog", type="primary", disabled=run_disabled, use_container_width=True):

    # ─── Reuse VOC preview if it exists ───
    if generate_voc and "voc_preview" in st.session_state:
        voc_input = st.session_state["voc_preview"]
        generate_voc = False  # Don't regenerate, we already have it

    # ─── Progress display ───
    progress_container = st.container()
    with progress_container:
        st.markdown("### ⚙️ Pipeline Running")
        steps = {
            "voc": st.empty(),
            "analyst": st.empty(),
            "architect": st.empty(),
            "quality": st.empty(),
        }

        def update_step(key, icon, label):
            steps[key].markdown(f"{icon} **{label}**")

        update_step("voc",      "⏳", "VOC Generator — preparing input...")
        update_step("analyst",  "🔲", "VOC Analyst — waiting")
        update_step("architect","🔲", "Requirement Architect — waiting")
        update_step("quality",  "🔲", "Quality Checker — waiting")
        time.sleep(0.3)

    try:
        t_start = time.time()

        # We can't hook into LangGraph nodes mid-run from Streamlit without
        # streaming, so we update steps sequentially as approximations.
        update_step("voc", "✅", "VOC Generator — done")
        update_step("analyst", "⏳", "VOC Analyst — extracting stakeholder needs...")

        result = run_pipeline(
            product_domain=product_domain,
            voc_input=voc_input,
            generate_voc=generate_voc
        )

        update_step("analyst",  "✅", "VOC Analyst — done")
        update_step("architect","✅", "Requirement Architect — done")
        update_step("quality",  "✅", "Quality Checker — done")

        elapsed = time.time() - t_start
        st.session_state["result"] = result
        st.session_state["elapsed"] = elapsed
        st.success(f"Pipeline complete in {elapsed:.1f}s")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    elapsed = st.session_state.get("elapsed", 0)

    st.divider()
    st.markdown("## 📊 Results")

    # ─── Quality badge ───
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Quality Approved", "✅ Yes" if result["quality_approved"] else "⚠️ No")
    col_q2.metric("Quality Score", f"{result.get('quality_score', 'N/A')}/10")
    col_q3.metric("Iterations", result["iterations"])
    col_q4.metric("Total Time", f"{elapsed:.1f}s")

    if result.get("quality_issues"):
        with st.expander(f"⚠️ Quality Issues ({len(result['quality_issues'])} found)", expanded=False):
            for issue in result["quality_issues"]:
                st.markdown(f"- {issue}")

    st.divider()

    # ─── Tabs ───
    tab_voc, tab_needs, tab_epics, tab_stories, tab_json = st.tabs([
        "📝 VOC Used",
        f"👥 Stakeholder Needs ({len(result['stakeholder_needs'])})",
        f"🗺️ Epics ({len(result['epics'])})",
        f"📖 User Stories ({len(result['user_stories'])})",
        "📦 Raw JSON"
    ])

    # ── VOC Tab ──
    with tab_voc:
        st.markdown("### Voice of Customer Input")
        st.markdown(result["voc_used"])

    # ── Stakeholder Needs Tab ──
    with tab_needs:
        st.markdown("### Extracted Stakeholder Needs")
        for i, need in enumerate(result["stakeholder_needs"], 1):
            priority = need.get("priority", "")
            badge = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚪")
            with st.expander(f"{badge} {need.get('persona', f'Need {i}')} — {priority} priority"):
                st.markdown(f"**Pain Point:** {need.get('pain_point', 'N/A')}")
                st.markdown(f"**Desired Outcome:** {need.get('desired_outcome', 'N/A')}")

    # ── Epics Tab ──
    with tab_epics:
        st.markdown("### Epic → Feature Hierarchy")
        for epic in result["epics"]:
            with st.expander(f"🗺️ **{epic.get('id')}** — {epic.get('title')}"):
                st.markdown(f"*{epic.get('description', '')}*")
                st.markdown("**Features:**")
                for feature in epic.get("features", []):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• **{feature.get('id')}** {feature.get('title')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{feature.get('description', '')}_")

    # ── User Stories Tab ──
    with tab_stories:
        st.markdown("### User Stories with Acceptance Criteria")
        for story in result["user_stories"]:
            with st.expander(f"📖 **{story.get('id')}** — {story.get('story', '')[:80]}..."):
                st.markdown(f"**Story:** {story.get('story')}")
                st.markdown("**Acceptance Criteria:**")
                for ac in story.get("acceptance_criteria", []):
                    st.markdown(f"- ✓ {ac}")
                st.caption(f"Feature: {story.get('feature_id')}")

    # ── Raw JSON Tab ──
    with tab_json:
        st.markdown("### Full Pipeline Output (JSON)")
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(result, indent=2),
            file_name=f"backlog_{product_domain.replace(' ', '_')}.json",
            mime="application/json"
        )
        st.json(result)
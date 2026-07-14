# app.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
import pandas as pd
from pprint import pformat

from sql.graph import sql_graph

st.set_page_config(
    page_title="ShopSphere AI",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 ShopSphere AI Analytics")
st.caption("Ask questions about your e-commerce data using natural language.")

# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------

st.sidebar.header("Example Questions")

examples = [
    "Top 10 customers by sales",
    "Top 5 selling product categories",
    "Average payment value by payment type",
    "Monthly sales trend",
    "Top sellers by revenue",
    "Which states have the highest revenue?"
]

for q in examples:
    if st.sidebar.button(q):
        st.session_state["question"] = q

# --------------------------------------------------------------------
# Question
# --------------------------------------------------------------------

question = st.text_area(
    "Ask a question",
    value=st.session_state.get("question", ""),
    height=100
)

# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------

if st.button("Run Analysis", type="primary"):

    initial_state = {
        "user_query": question,
        "retry_count": 0
    }

    events = []

    with st.spinner("Running AI Agent..."):

        try:
            for event in sql_graph.stream(initial_state):
                events.append(event)

            final_state = {}

            for event in events:
                final_state.update(list(event.values())[0])

        except Exception as e:
            st.error(str(e))
            st.stop()

    # ----------------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📝 Summary",
            "💻 SQL",
            "📊 Results",
            "🧠 Agent Trace"
        ]
    )

    # ================================================================
    # Summary
    # ================================================================

    with tab1:

        st.subheader("Business Summary")

        st.write(final_state.get("answer", "No answer generated."))

    # ================================================================
    # SQL
    # ================================================================

    with tab2:

        st.subheader("Generated SQL")

        sql = final_state.get("executed_sql") or final_state.get("generated_sql")

        if sql:
            st.code(sql, language="sql")

        if final_state.get("sql_error"):

            st.error(final_state["sql_error"])

    # ================================================================
    # Results
    # ================================================================

    with tab3:

        st.subheader("SQL Result")

        result = final_state.get("sql_result")

        if result is None:
            st.info("No data returned.")

        elif isinstance(result, pd.DataFrame):

            st.dataframe(
                result,
                use_container_width=True
            )

            # Auto chart
            numeric_cols = result.select_dtypes(include="number").columns

            if len(numeric_cols):

                st.subheader("Quick Visualization")

                st.bar_chart(
                    result.set_index(result.columns[0])[numeric_cols[0]]
                )

        else:

            st.write(result)

    # ================================================================
    # Trace
    # ================================================================

    with tab4:

        st.subheader("Execution Trace")

        for i, event in enumerate(events):

            node = list(event.keys())[0]
            output = list(event.values())[0]

            with st.expander(
                f"Step {i+1}: {node}",
                expanded=False
            ):
                st.code(
                    pformat(output),
                    language="python"
                )
"""Causal DAG specification and identifying assumptions."""

CAUSAL_GRAPH_DOT = """
digraph {
    spread -> trade_size;
    depth -> trade_size;
    volatility -> trade_size;

    spread -> price_change;
    depth -> price_change;
    volatility -> price_change;

    trade_size -> price_change;
}
"""

ASSUMPTIONS = """
1. UNCONFOUNDEDNESS: Conditional on (spread, depth, volatility), trade_size
   is as-good-as-random. No unobserved confounders affect both T and Y.

2. OVERLAP (POSITIVITY): For every market state X, there is positive
   probability of observing any trade size T.

3. SUTVA: One trade's outcome does not depend on other trades' treatment
   assignments.

4. CONSISTENCY: The observed outcome under treatment T=t equals the
   potential outcome Y(t).
"""


def get_dowhy_graph() -> str:
    """Return the causal graph in DOT format."""
    return CAUSAL_GRAPH_DOT


def print_assumptions():
    """Print the identifying assumptions."""
    print(ASSUMPTIONS)


def build_dowhy_model(df, treatment="trade_size", outcome="price_change",
                       common_causes=None):
    """Build a DoWhy CausalModel for refutation tests."""
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        print("dowhy not installed — skipping causal model construction.")
        return None

    if common_causes is None:
        common_causes = ["spread", "depth", "volatility"]

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        graph=get_dowhy_graph(),
    )
    return model


if __name__ == "__main__":
    print("=== Causal Graph ===")
    print(get_dowhy_graph())
    print_assumptions()

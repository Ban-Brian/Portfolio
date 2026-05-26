"""
Causal graph specification and assumption documentation.

Defines the DAG used for the price impact analysis and provides
functions to visualize it and run DoWhy refutation tests.
"""

# --- DAG Description (text form, usable by DoWhy) ---
# Nodes:
#   spread      — bid-ask spread (liquidity proxy)
#   depth       — order book depth (liquidity proxy)
#   volatility  — recent realized volatility
#   trade_size  — treatment variable (T)
#   price_change — outcome variable (Y)
#
# Edges (causal arrows):
#   spread      → trade_size     (traders condition on spread)
#   depth       → trade_size     (traders submit larger orders in deep books)
#   volatility  → trade_size     (traders reduce size in volatile markets)
#   spread      → price_change   (spread affects baseline price dynamics)
#   depth       → price_change   (thin books amplify moves)
#   volatility  → price_change   (vol drives price fluctuations)
#   trade_size  → price_change   (the causal effect we estimate)

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

# Key identifying assumptions documented here:
ASSUMPTIONS = """
1. UNCONFOUNDEDNESS: Conditional on (spread, depth, volatility), trade_size
   is as-good-as-random. No unobserved confounders affect both T and Y.
   This holds by construction in the synthetic DGP, but is a strong
   assumption for real data (alpha / private information would violate it).

2. OVERLAP (POSITIVITY): For every market state X, there is positive
   probability of observing any trade size T. The continuous treatment
   and Gaussian noise in the DGP guarantee this.

3. SUTVA: One trade's outcome does not depend on other trades' treatment
   assignments. In real HFT this is violated (trades cluster), but the
   DGP generates observations independently (with optional AR(1) noise).

4. CONSISTENCY: The observed outcome under treatment T=t equals the
   potential outcome Y(t). Trivially satisfied in the DGP.
"""


def get_dowhy_graph() -> str:
    """Return the causal graph in DOT format for DoWhy."""
    return CAUSAL_GRAPH_DOT


def print_assumptions():
    """Print the identifying assumptions for this analysis."""
    print(ASSUMPTIONS)


def build_dowhy_model(df, treatment="trade_size", outcome="price_change",
                       common_causes=None):
    """
    Build a DoWhy CausalModel for refutation tests.
    Requires dowhy to be installed.
    """
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


# --- Quick check ---
if __name__ == "__main__":
    print("=== Causal Graph ===")
    print(get_dowhy_graph())
    print_assumptions()

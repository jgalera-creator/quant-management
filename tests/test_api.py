def test_public_risk_api():
    from quantmgmt.risk import (
        RiskAnalyzer,
        cagr,
        max_drawdown,
        var_historical,
    )

    assert RiskAnalyzer is not None
    assert callable(cagr)
    assert callable(max_drawdown)
    assert callable(var_historical)
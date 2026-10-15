"""Estimate the client's WACC.

Run from the repo root:   python code/wacc.py
Reads:  inputs/assumptions.csv
Writes: outputs/wacc.csv (one row; the columns are listed in README.md)

Owner: the developer. Fill in the three TODO functions (your agent can help),
then run it and check the numbers make sense before you open a pull request.
"""
import pandas as pd
import yfinance as yf

CLIENT = 'JNJ'  # TODO: the client's ticker (the CEO picks it in README.md)
MKT = 'SPY'
YEARS = 5


def get_returns(ticker, mkt=MKT, years=YEARS):
    """Daily returns for the client and the market over the last `years` years.
    Returns a DataFrame with two columns, named ticker and mkt, and no missing values."""
    data = yf.download(
        [ticker, mkt],
        period=f'{years}y',
        interval='1d',
        auto_adjust=True,
        progress=False,
        group_by='ticker',
    )

    if data.empty:
        raise ValueError(f'no price data returned for {ticker} and {mkt}')

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs('Close', level=-1, axis=1)
    else:
        close = data

    close = close[[ticker, mkt]].copy()
    close.columns = [ticker, mkt]
    close = close.astype(float)

    rets = close.pct_change().dropna()
    assert not rets.isna().any().any(), 'missing returns detected after pct_change()'
    assert len(rets) > 1000, 'expected about 5 years of daily returns'
    return rets


def estimate_beta(rets, ticker, mkt=MKT):
    """Equity beta: cov(stock, market) / var(market)."""
    assert ticker in rets.columns and mkt in rets.columns, 'ticker and market columns missing from returns data'
    assert not rets[[ticker, mkt]].isna().any().any(), 'missing returns are not allowed in beta estimation'

    denom = rets[mkt].var(ddof=1)
    assert denom > 0, 'market return variance is zero; beta is undefined'

    beta = rets[ticker].cov(rets[mkt], ddof=1) / denom
    assert pd.notna(beta), 'beta calculation produced a missing value'
    return beta


def compute_wacc(beta, a):
    """a is a dict of assumptions: rf, mrp, rd, tax, w_e, w_d.
    Returns a dict with cost_of_equity and wacc (and the inputs), matching the README's columns."""
    rf = float(a['rf'])
    mrp = float(a['mrp'])
    rd = float(a['rd'])
    tax = float(a['tax'])
    w_e = float(a['w_e'])
    w_d = float(a['w_d'])

    assert 0 <= w_e <= 1 and 0 <= w_d <= 1, 'weights must be between 0 and 1'
    assert abs(w_e + w_d - 1) < 1e-6, 'weights must sum to 1'

    cost_of_equity = rf + beta * mrp
    wacc = w_e * cost_of_equity + w_d * rd * (1 - tax)

    assert pd.notna(cost_of_equity) and pd.notna(wacc), 'invalid WACC calculation: non-finite values'
    assert 0 <= cost_of_equity <= 1, 'cost of equity should be between 0 and 100%'
    assert 0 <= wacc <= 0.25, 'WACC outside 0-25%: check the inputs'

    return {
        'rf': rf,
        'mrp': mrp,
        'cost_of_equity': cost_of_equity,
        'rd': rd,
        'tax': tax,
        'w_e': w_e,
        'w_d': w_d,
        'wacc': wacc,
    }


if __name__ == '__main__':
    a = pd.read_csv('inputs/assumptions.csv', index_col='item')['value'].to_dict()
    assert abs(a['w_e'] + a['w_d'] - 1) < 1e-6, 'weights must sum to 1'

    rets = get_returns(CLIENT)
    assert len(rets) > 1000, 'expected about 5 years of daily returns'

    beta = estimate_beta(rets, CLIENT)
    out = compute_wacc(beta, a)
    out = {'ticker': CLIENT, 'asof': rets.index.max().date(), 'beta': beta, **out}

    assert 0 < out['wacc'] < 0.25, 'WACC outside 0-25%: check the inputs'
    cols = ['ticker', 'asof', 'beta', 'rf', 'mrp', 'cost_of_equity', 'rd', 'tax', 'w_e', 'w_d', 'wacc']
    pd.DataFrame([out])[cols].to_csv('outputs/wacc.csv', index=False)
    print('Preliminary WACC estimate:')
    print(pd.DataFrame([out])[cols].T)

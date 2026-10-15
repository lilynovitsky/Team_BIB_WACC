"""Read-only WACC card. Run from the repo root: streamlit run app.py."""
from math import isclose, isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

ASSUMPTIONS = ('rf', 'mrp', 'rd', 'tax', 'w_e', 'w_d')
COLUMNS = ('ticker', 'asof', 'beta', 'rf', 'mrp', 'cost_of_equity',
           'rd', 'tax', 'w_e', 'w_d', 'wacc')
LABELS = {'rf': 'Risk-free rate', 'mrp': 'Market risk premium',
          'rd': 'Pre-tax cost of debt', 'tax': 'Tax rate',
          'w_e': 'Equity weight', 'w_d': 'Debt weight'}


def load_card():
    """Reject bad real output explicitly instead of silently using demo data."""
    real = Path('outputs/wacc.csv')
    demo = not real.exists()
    path = Path('outputs/wacc_FAKE.csv') if demo else real
    frame = pd.read_csv(path)
    if len(frame) != 1 or not set(COLUMNS).issubset(frame.columns):
        raise ValueError(f'{path} must contain one row with the README columns.')
    card = frame.iloc[0].copy()
    if pd.isna(card['ticker']) or not str(card['ticker']).strip():
        raise ValueError('The output needs a ticker.')
    if pd.isna(pd.to_datetime(card['asof'], errors='coerce')):
        raise ValueError('The returns end date is invalid.')
    for name in COLUMNS[2:]:
        card[name] = float(card[name])
        if not isfinite(card[name]):
            raise ValueError(f'{name} must be finite.')
    for name in ('tax', 'w_e', 'w_d'):
        if not 0 <= card[name] <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
    if not isclose(card['w_e'] + card['w_d'], 1, abs_tol=1e-6):
        raise ValueError('Equity and debt weights must sum to 100%.')
    ke = card['rf'] + card['beta'] * card['mrp']
    expected = card['w_e'] * ke + card['w_d'] * card['rd'] * (1 - card['tax'])
    if not isclose(card['cost_of_equity'], ke, abs_tol=1e-6):
        raise ValueError('Stored cost of equity does not match the CAPM inputs.')
    if not isclose(card['wacc'], expected, abs_tol=1e-6):
        raise ValueError('Stored WACC does not match its inputs.')
    if not 0 <= card['wacc'] <= .25:
        raise ValueError('WACC is outside 0–25%; check the output inputs.')
    return card, demo, path


def load_sources():
    frame = pd.read_csv('inputs/assumptions.csv')
    if not {'item', 'value', 'source'}.issubset(frame.columns):
        raise ValueError('Assumptions need item, value, and source columns.')
    if frame['item'].duplicated().any():
        raise ValueError('Assumptions contain duplicate items.')
    return frame.set_index('item')


st.set_page_config(page_title='JNJ | WACC card', page_icon='📊', layout='centered')
try:
    card, demo, output_path = load_card()
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f'Unable to display the WACC card: {exc}')
    st.info('From the repository root, run `python code/wacc.py` to regenerate outputs/wacc.csv.')
    st.stop()

st.caption('TEAM BIB  /  COST OF CAPITAL')
st.title('Johnson & Johnson' if card['ticker'] == 'JNJ' else f"WACC card: {card['ticker']}")
st.caption(f"{card['ticker']} · Daily returns through {card['asof']} · {output_path.as_posix()}")
if demo:
    st.warning('DEMO — all figures are made up. Run `python code/wacc.py` to generate the real WACC card.')

with st.container(border=True):
    left, right = st.columns([2, 1])
    left.metric('Weighted average cost of capital', f"{card['wacc']:.2%}")
    right.metric('Equity beta vs. SPY', f"{card['beta']:.3f}")
    st.write(f"{'In this demo, a' if demo else 'A'} project with risk similar to the existing business "
             f"would need an expected return above **{card['wacc']:.2%}** to create value under these assumptions.")
    st.caption('Projects with different risk need a project-specific hurdle rate. This estimate assumes the stated capital mix.')

ke = card['cost_of_equity']
after_tax_debt = card['rd'] * (1 - card['tax'])
c1, c2, c3 = st.columns(3)
c1.metric('Cost of equity', f'{ke:.2%}')
c2.metric('After-tax cost of debt', f'{after_tax_debt:.2%}')
c3.metric('Equity / debt mix', f"{card['w_e']:.2%} / {card['w_d']:.2%}")

with st.expander('How the WACC is calculated'):
    st.latex(r'k_e = r_f + \beta \times MRP')
    st.write(f"Cost of equity = {card['rf']:.2%} + {card['beta']:.3f} × {card['mrp']:.2%} = **{ke:.2%}**")
    st.latex(r'WACC = w_e k_e + w_d r_d(1-T)')
    st.write(f"Equity contributes **{card['w_e'] * ke * 100:.2f} percentage points**; "
             f"debt contributes **{card['w_d'] * after_tax_debt * 100:.2f} percentage points**.")
    st.caption('Calculations use full precision; displayed values are rounded.')

st.subheader('Inputs and sources')
st.caption('Values come from the displayed output. Source descriptions come from inputs/assumptions.csv.')
try:
    sources = load_sources()
except (OSError, ValueError, KeyError, TypeError) as exc:
    sources = pd.DataFrame(columns=['value', 'source'])
    st.warning(f'Source descriptions are unavailable: {exc}')

rows = []
unverified = []
for name in ASSUMPTIONS:
    source = 'Source missing — analyst review needed.'
    if name in sources.index and pd.notna(sources.loc[name, 'source']):
        source = str(sources.loc[name, 'source'])
    if not source.strip() or 'TODO' in source or source.startswith('Source missing'):
        unverified.append(LABELS[name])
    try:
        matches = isclose(float(sources.loc[name, 'value']), card[name], abs_tol=1e-8)
    except (KeyError, TypeError, ValueError):
        matches = False
    if demo:
        source = 'Made-up demo input; no verified source.'
    elif not matches:
        source = 'VALUE MISMATCH — current assumptions do not match this output. ' + source
        unverified.append(LABELS[name])
    rows.append({'Input': LABELS[name], 'Value': f'{card[name]:.2%}', 'Source / method': source})
rows.append({'Input': 'Equity beta vs. SPY', 'Value': f"{card['beta']:.3f}",
             'Source / method': 'Made-up demo input.' if demo else
             'code/wacc.py: covariance of JNJ and SPY daily returns / SPY variance; '
             'five years of adjusted prices downloaded with yfinance. Returns end: ' + str(card['asof'])})
if unverified and not demo:
    st.warning('Preliminary estimate — source review or output refresh needed for: ' + ', '.join(dict.fromkeys(unverified)) + '.')
st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
             column_config={'Source / method': st.column_config.TextColumn(width='large')})

with st.expander('Explore a different capital mix or market risk premium'):
    st.caption('Illustrative sensitivity only. Beta, debt cost, tax, and the risk-free rate stay fixed; actual financing changes can change those inputs.')
    wd = st.slider('Debt weight (%)', 0.0, 100.0, float(card['w_d'] * 100), .1)
    mrp = st.number_input('Market risk premium (%)', value=float(card['mrp'] * 100), step=.1)
    scenario_ke = card['rf'] + card['beta'] * mrp / 100
    scenario_wacc = (1 - wd / 100) * scenario_ke + wd / 100 * after_tax_debt
    st.metric('Scenario WACC', f'{scenario_wacc:.2%}',
              f"{(scenario_wacc - card['wacc']) * 100:+.2f} percentage points vs. baseline", delta_color='off')
    st.caption(f'Equity weight: {100 - wd:.2f}% · Scenario cost of equity: {scenario_ke:.2%}')

st.download_button('Download displayed WACC (CSV)',
                   pd.DataFrame([card], columns=COLUMNS).to_csv(index=False),
                   file_name='wacc_demo.csv' if demo else 'wacc.csv', mime='text/csv')
st.caption('Read-only dashboard · Recalculate the baseline with python code/wacc.py, then reload this page.')

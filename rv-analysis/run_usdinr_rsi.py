import argparse
import blpapi
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── CLI ─────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description='RSI mean-reversion backtest')
_parser.add_argument('--pair',        default='USDINR', type=str.upper,
                     help='FX pair, e.g. EURUSD (default: USDINR)')
_parser.add_argument('--rsi-period',  default=14,  type=int,   help='RSI lookback period (default: 14)')
_parser.add_argument('--sell-thresh', default=75,  type=float, help='RSI level to enter short (default: 75)')
_parser.add_argument('--cover-thresh',default=55,  type=float, help='RSI level to exit short (default: 55)')
_parser.add_argument('--lookback',    default=5,   type=int,   help='Backtest window in years (default: 5)')
_args = _parser.parse_args()

# Bloomberg tickers that need a non-standard suffix
_BBG_OVERRIDES = {'USDINR': 'USDINR REGN Curncy'}

# ── Parameters ─────────────────────────────────────────────────────────────
PAIR         = _args.pair
SECURITY     = _BBG_OVERRIDES.get(PAIR, f'{PAIR} Curncy')
RSI_PERIOD   = _args.rsi_period
SELL_THRESH  = _args.sell_thresh
COVER_THRESH = _args.cover_thresh
LOOKBACK_YRS = _args.lookback

END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=365 * LOOKBACK_YRS + 60)

print(f'Security : {SECURITY}')
print(f'Period   : {START_DATE} -> {END_DATE}')
print(f'RSI      : {RSI_PERIOD}-day  |  Sell >{SELL_THRESH}  |  Cover <{COVER_THRESH}')

# ── 1. Bloomberg fetch ──────────────────────────────────────────────────────
def bbg_bdh(security, fields, start, end):
    session_opts = blpapi.SessionOptions()
    session_opts.setServerHost('localhost')
    session_opts.setServerPort(8194)
    session = blpapi.Session(session_opts)
    if not session.start():
        raise RuntimeError('Bloomberg session failed to start.')
    session.openService('//blp/refdata')
    svc = session.getService('//blp/refdata')
    req = svc.createRequest('HistoricalDataRequest')
    req.getElement('securities').appendValue(security)
    for f in fields:
        req.getElement('fields').appendValue(f)
    req.set('startDate', start.strftime('%Y%m%d'))
    req.set('endDate',   end.strftime('%Y%m%d'))
    req.set('periodicitySelection', 'DAILY')
    session.sendRequest(req)
    rows = []
    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.hasElement('securityData'):
                sd  = msg.getElement('securityData')
                fdl = sd.getElement('fieldData')
                for i in range(fdl.numValues()):
                    pt  = fdl.getValue(i)
                    raw_dt = pt.getElementAsDatetime('date')
                    row = {'date': raw_dt.date() if hasattr(raw_dt, 'date') else raw_dt}
                    for f in fields:
                        try:    row[f] = pt.getElementAsFloat(f)
                        except: row[f] = np.nan
                    rows.append(row)
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
    session.stop()
    df = pd.DataFrame(rows).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df

print('\nFetching data from Bloomberg...')
raw = bbg_bdh(SECURITY, ['PX_LAST'], START_DATE, END_DATE)
raw.columns = ['Close']
raw.dropna(inplace=True)
print(f'Loaded {len(raw)} rows  ({raw.index[0].date()} -> {raw.index[-1].date()})')

# ── 2. RSI ──────────────────────────────────────────────────────────────────
def wilder_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    alpha    = 1 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

raw['RSI'] = wilder_rsi(raw['Close'], RSI_PERIOD)

bt_start = pd.Timestamp(END_DATE) - pd.DateOffset(years=LOOKBACK_YRS)
df = raw.loc[bt_start:].copy()
print(f'Backtest window: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sessions)')

# ── 3. Signals ──────────────────────────────────────────────────────────────
trades = []
in_trade = False
entry_price = entry_date = entry_rsi = None

for dt, row in df.iterrows():
    rsi = row['RSI']
    px  = row['Close']
    if pd.isna(rsi):
        continue
    if not in_trade:
        if rsi > SELL_THRESH:
            in_trade    = True
            entry_price = px
            entry_date  = dt
            entry_rsi   = rsi
    else:
        if rsi < COVER_THRESH:
            pnl_pct = (entry_price - px) / entry_price * 100
            trades.append({
                'entry_date'   : entry_date,
                'exit_date'    : dt,
                'entry_price'  : round(entry_price, 4),
                'exit_price'   : round(px, 4),
                'entry_rsi'    : round(entry_rsi, 1),
                'exit_rsi'     : round(rsi, 1),
                'pnl_pct'      : round(pnl_pct, 4),
                'duration_days': (dt - entry_date).days,
                'success'      : pnl_pct > 0,
            })
            in_trade = False

if in_trade:
    last    = df.iloc[-1]
    pnl_pct = (entry_price - last['Close']) / entry_price * 100
    trades.append({
        'entry_date'   : entry_date,
        'exit_date'    : df.index[-1],
        'entry_price'  : round(entry_price, 4),
        'exit_price'   : round(last['Close'], 4),
        'entry_rsi'    : round(entry_rsi, 1),
        'exit_rsi'     : round(last['RSI'], 1),
        'pnl_pct'      : round(pnl_pct, 4),
        'duration_days': (df.index[-1] - entry_date).days,
        'success'      : pnl_pct > 0,
        'open_trade'   : True,
    })

tdf    = pd.DataFrame(trades)
closed = tdf[tdf.get('open_trade', False) != True].copy() if 'open_trade' in tdf.columns else tdf.copy()

# ── 4. Metrics ──────────────────────────────────────────────────────────────
n_total  = len(closed)
n_wins   = int(closed['success'].sum())
n_losses = n_total - n_wins
win_rate = n_wins / n_total * 100 if n_total else 0

avg_win   = closed.loc[closed['success'],  'pnl_pct'].mean()
avg_loss  = closed.loc[~closed['success'], 'pnl_pct'].mean()
avg_pnl   = closed['pnl_pct'].mean()
total_pnl = closed['pnl_pct'].sum()

gross_profit  = closed.loc[closed['success'],  'pnl_pct'].sum()
gross_loss    = abs(closed.loc[~closed['success'], 'pnl_pct'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

pnl_std = closed['pnl_pct'].std()
ir      = (avg_pnl / pnl_std) * np.sqrt(n_total / LOOKBACK_YRS) if pnl_std > 0 else float('nan')

cumulative  = closed['pnl_pct'].cumsum()
rolling_max = cumulative.cummax()
max_dd      = (cumulative - rolling_max).min()

avg_dur_win  = closed.loc[closed['success'],  'duration_days'].mean()
avg_dur_loss = closed.loc[~closed['success'], 'duration_days'].mean()

print('\n' + '='*50)
print(f'  {PAIR} RSI Strategy  |  {LOOKBACK_YRS}-Year Backtest')
print('='*50)
metrics = [
    ('Total Signals (closed)',        n_total),
    ('Winning Trades',                n_wins),
    ('Losing Trades',                 n_losses),
    ('Win Rate',                      f'{win_rate:.1f}%'),
    ('Avg P&L per Trade (%)',         f'{avg_pnl:.3f}%'),
    ('Avg Win (%)',                   f'{avg_win:.3f}%'  if not np.isnan(avg_win)  else 'N/A'),
    ('Avg Loss (%)',                  f'{avg_loss:.3f}%' if not np.isnan(avg_loss) else 'N/A'),
    ('Profit Factor',                 f'{profit_factor:.2f}'),
    ('Total P&L (%)',                 f'{total_pnl:.3f}%'),
    ('Max Drawdown (cumulative %)',   f'{max_dd:.3f}%'),
    ('Information Ratio',             f'{ir:.2f}'        if not np.isnan(ir)       else 'N/A'),
    ('Avg Duration – Wins (days)',    f'{avg_dur_win:.1f}'  if not np.isnan(avg_dur_win)  else 'N/A'),
    ('Avg Duration – Losses (days)',  f'{avg_dur_loss:.1f}' if not np.isnan(avg_dur_loss) else 'N/A'),
]
for k, v in metrics:
    print(f'  {k:<38} {v}')
print('='*50)

# ── 5. Trade table ──────────────────────────────────────────────────────────
print('\nTrade-by-Trade Detail:')
display_cols = ['entry_date','exit_date','entry_price','exit_price',
                'entry_rsi','exit_rsi','pnl_pct','duration_days','success']
print(closed[display_cols].to_string(index=False))

# ── 6. Charts ───────────────────────────────────────────────────────────────
OUT_DIR = str(Path(__file__).resolve().parent.parent / "output" / "rv-analysis")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                          gridspec_kw={'height_ratios': [3, 2, 2]})
fig.suptitle(f'{PAIR} – RSI({RSI_PERIOD}) Mean-Reversion  |  Sell >{SELL_THRESH} / Cover <{COVER_THRESH}',
             fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(df.index, df['Close'], color='steelblue', lw=1.2, label=PAIR)
if len(tdf):
    wins  = tdf[tdf['success'] == True]
    loses = tdf[tdf['success'] == False]
    ax1.scatter(tdf['entry_date'],  tdf['entry_price'],  marker='v', color='red',    s=80, zorder=5, label='Sell signal')
    ax1.scatter(wins['exit_date'],  wins['exit_price'],  marker='^', color='green',  s=80, zorder=5, label='Cover (win)')
    ax1.scatter(loses['exit_date'], loses['exit_price'], marker='^', color='orange', s=80, zorder=5, label='Cover (loss)')
ax1.set_ylabel('Spot Rate')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(df.index, df['RSI'], color='darkorange', lw=1.1)
ax2.axhline(SELL_THRESH,  color='red',   lw=1.0, ls='--', label=f'Sell  >{SELL_THRESH}')
ax2.axhline(COVER_THRESH, color='green', lw=1.0, ls='--', label=f'Cover <{COVER_THRESH}')
ax2.axhline(50,           color='grey',  lw=0.8, ls=':')
ax2.fill_between(df.index, df['RSI'], SELL_THRESH,  where=df['RSI'] > SELL_THRESH,  alpha=0.15, color='red')
ax2.fill_between(df.index, df['RSI'], COVER_THRESH, where=df['RSI'] < COVER_THRESH, alpha=0.15, color='green')
ax2.set_ylim(0, 100)
ax2.set_ylabel(f'RSI ({RSI_PERIOD})')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

ax3 = axes[2]
if len(closed):
    cum = closed.set_index('exit_date')['pnl_pct'].cumsum()
    ax3.plot(cum.index, cum.values, color='steelblue', lw=1.4)
    ax3.fill_between(cum.index, cum.values, 0, where=cum.values >= 0, alpha=0.2, color='green')
    ax3.fill_between(cum.index, cum.values, 0, where=cum.values <  0, alpha=0.2, color='red')
    ax3.axhline(0, color='black', lw=0.8)
ax3.set_ylabel('Cumulative P&L (%)')
ax3.set_xlabel('Date')
ax3.grid(alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30)
plt.tight_layout()
out1 = str(Path(OUT_DIR) / f'{PAIR}_RSI_backtest.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nChart saved -> {out1}')

# Distribution
if len(closed):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    wins_data   = closed.loc[closed['success'],  'pnl_pct']
    losses_data = closed.loc[~closed['success'], 'pnl_pct']
    ax = axes[0]
    ax.hist(wins_data,   bins=15, color='green', alpha=0.6, label='Wins')
    ax.hist(losses_data, bins=15, color='red',   alpha=0.6, label='Losses')
    ax.axvline(0,       color='black', lw=1)
    ax.axvline(avg_pnl, color='blue',  lw=1.5, ls='--', label=f'Avg {avg_pnl:.3f}%')
    ax.set_title('P&L Distribution per Trade')
    ax.set_xlabel('P&L (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    bp = ax.boxplot([wins_data.values, losses_data.values], labels=['Wins', 'Losses'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#90ee90')
    bp['boxes'][1].set_facecolor('#f08080')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_title('P&L Box Plot – Wins vs Losses')
    ax.set_ylabel('P&L (%)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = str(Path(OUT_DIR) / f'{PAIR}_RSI_distribution.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Chart saved -> {out2}')


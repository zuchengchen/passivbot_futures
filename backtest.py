import sys
import json
import hjson
import numpy as np
import pandas as pd
import asyncio
import os
import pprint
import matplotlib.pyplot as plt
import aiomultiprocess
from hashlib import sha256
from multiprocessing import cpu_count, Lock, Value
from time import time
from passivbot import *
from bybit import create_bot as create_bot_bybit
from bybit import fetch_trades as bybit_fetch_trades
from bybit import calc_cross_long_liq_price as bybit_calc_cross_long_liq_price
from bybit import calc_cross_shrt_liq_price as bybit_calc_cross_shrt_liq_price
from binance import create_bot as create_bot_binance
from binance import fetch_trades as binance_fetch_trades
from binance import calc_cross_long_liq_price as binance_calc_cross_long_liq_price
from binance import calc_cross_shrt_liq_price as binance_calc_cross_shrt_liq_price

from typing import Iterator, Callable

aiomultiprocess.set_start_method("fork")


def get_score_func(key: str) -> Callable:

    def score_func_avg_adg(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['avg']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg']
        except:
            return True
        return candidate_score > best_score
    
    def score_func_min_adg(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['min']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['min']
        except:
            return True
        return candidate_score > best_score

    def score_func_avg_adg_min_adg(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['avg'] * \
                candidate['average_daily_gain']['min']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg'] * \
                best['average_daily_gain']['min']
        except:
            return True
        return candidate_score > best_score
    
    def score_func_avg_adg_min_liq(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['avg'] * candidate['closest_liq']['min']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg'] * best['closest_liq']['min']
        except:
            return True
        return candidate_score > best_score
    
    def score_func_avg_adg_min_adg_min_liq(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['avg'] * \
                candidate['average_daily_gain']['min'] * candidate['closest_liq']['min']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg'] * best['average_daily_gain']['min'] * \
                best['closest_liq']['min']
        except:
            return True
        return candidate_score > best_score

    def score_func_avg_adg_min_adg_min_liq_std_adg(best: dict, candidate: dict) -> bool:
        try:
            candidate_score = candidate['average_daily_gain']['avg'] * \
                candidate['average_daily_gain']['min'] * candidate['closest_liq']['min'] / \
                candidate['average_daily_gain']['std']
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg'] * best['average_daily_gain']['min'] * \
                best['closest_liq']['min'] / best['average_daily_gain']['std']
        except:
            return True
        return candidate_score > best_score

    def score_func_avg_adg_min_adg_min_liq_capped(best: dict, candidate: dict) -> bool:
        cap = 0.2
        try:
            candidate_score = candidate['average_daily_gain']['avg'] * \
                candidate['average_daily_gain']['min'] * min(candidate['closest_liq']['min'], cap)
        except:
            return False
        try:
            best_score = best['average_daily_gain']['avg'] * \
                best['average_daily_gain']['min'] * min(best['closest_liq']['min'], cap)
        except:
            return True
        return candidate_score > best_score

    if key == 'avg adg':
        return score_func_avg_adg
    elif key == 'min adg':
        return score_func_min_adg
    elif key == 'avg adg * min adg':
        return score_func_avg_adg_min_adg
    elif key == 'avg adg * min liq':
        return score_func_avg_adg_min_liq
    elif key == 'avg adg * min adg * min liq':
        return score_func_avg_adg_min_adg_min_liq
    elif key == 'avg adg * min adg * min liq capped':
        return score_func_avg_adg_min_adg_min_liq_capped
    elif key == 'avg adg * min adg * min liq / std adg':
        return score_func_avg_adg_min_adg_min_liq_std_adg
    raise Exception('unknown score metric', key)


def get_break_on_conditions(keys: [str]):
    def first_stop_loss(trades: [dict], ticks: np.ndarray, k: int):
        return trades[-1]['type'].startswith('stop_loss')

    def liq_too_close(trades: [dict], ticks: np.ndarray, k: int):
        return trades[-1]['closest_liq'] < 0.05


def plot_tdf(df_, tdf_, side_: int = 0, liq_thr=0.1):
    plt.clf()
    df_.loc[tdf_.index[0]:tdf_.index[-1]].price.plot(style='y-')
    if side_ >= 0:
        longs = tdf_[tdf_.side == 'long']
        le = longs[longs.type.str.endswith('entry')]
        lc = longs[longs.type == 'close']
        ls = longs[longs.type.str.startswith('stop_loss')]
        ls.price.plot(style='gx')
        le.price.plot(style='b.')
        longs.pos_price.plot(style='b--')
        if 'close_price' in longs.columns:
            longs.close_price.plot(style='r--')
        lc.price.plot(style='r.')
    if side_ <= 0:
        shrts = tdf_[tdf_.side == 'shrt']
        se = shrts[shrts.type.str.endswith('entry')]
        sc = shrts[shrts.type == 'close']
        ss = shrts[shrts.type.str.startswith('stop_loss')]
        ss.price.plot(style='gx')
        se.price.plot(style='r.')
        shrts.pos_price.plot(style='r--')
        if 'close_price' in shrts.columns:
            shrts.close_price.plot(style='b--')
        sc.price.plot(style='b.')
    if 'liq_price' in tdf_.columns:
        tdf_.liq_price.where((tdf_.price - tdf_.liq_price).abs() / tdf_.price < liq_thr, np.nan).plot(style='k--')
    return plt


def dump_plots(result: dict, tdf: pd.DataFrame, df: pd.DataFrame):
    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)

    def gain_conv(x):
        return x * 100 - 100
    
    lines = []
    lines.append(f"net pnl plus fees {result['net_pnl_plus_fees']:.6f}")
    lines.append(f"profit sum {result['profit_sum']:.6f}")
    lines.append(f"loss sum {result['loss_sum']:.6f}")
    lines.append(f"fee sum {result['fee_sum']:.6f}")
    lines.append(f"gain percentage {gain_conv(result['gain']):.2f}%")
    lines.append(f"n_days {result['n_days']}")
    lines.append(f"average_daily_gain percentage {(result['average_daily_gain'] - 1) * 100:.2f}%")
    lines.append(f"n trades {result['n_trades']}")
    lines.append(f"n closes {result['n_closes']}")
    lines.append(f"n reentries {result['n_reentries']}")
    lines.append(f"n stop loss closes {result['n_stop_losses']}")
    lines.append(f"biggest_pos_size {round(result['biggest_pos_size'], 10)}")
    lines.append(f"closest liq percentage {result['closest_liq'] * 100:.4f}%")
    lines.append(f"max n hours between consecutive trades {result['max_n_hours_between_consec_trades']:.2f}")
    lines.append(f"starting balance {result['starting_balance']}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    print('plotting price with bid ask entry thresholds')
    ema_long = df.price.ewm(span=result['ema_long_span'], adjust=False).mean()
    ema_shrt = df.price.ewm(span=result['ema_shrt_span'], adjust=False).mean()
    bids_ = ema_long * (1 - result['ema_long_spread'])
    asks_ = ema_shrt * (1 + result['ema_shrt_spread'])
    
    plt.clf()
    df.price.iloc[::100].plot()
    bids_.iloc[::100].plot()
    asks_.iloc[::100].plot()
    plt.savefig(f"{result['session_dirpath']}ema_spread_plot.png")
    
    print('writing backtest_result.txt...')
    with open(f"{result['session_dirpath']}backtest_result.txt", 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')

    print('plotting pnl cumsum...')
    counter = 0
    idxs = []
    for row in tdf.itertuples():
        if row.type.startswith('stop_loss'):
            counter += 1
        else:
            if counter > 0:
                idxs.append(row.Index)
            counter = 0
    plt.clf()
    tdf.pnl_plus_fees_cumsum.plot()
    if idxs:
        tdf.pnl_plus_fees_cumsum.loc[idxs].plot(style='ro')
    plt.savefig(f"{result['session_dirpath']}pnlcumsum_plot.png")

    print('plotting backtest whole and in chunks...')
    n_parts = 7
    for z in range(n_parts):
        start_ = z / n_parts
        end_ = (z + 1) / n_parts
        print(start_, end_)
        fig = plot_tdf(df, tdf.iloc[int(len(tdf) * start_):int(len(tdf) * end_)], liq_thr=0.1)
        fig.savefig(f"{result['session_dirpath']}backtest_{z + 1}of{n_parts}.png")
    fig = plot_tdf(df, tdf, liq_thr=0.1)
    fig.savefig(f"{result['session_dirpath']}whole_backtest.png")

    print('plotting pos sizes...')
    plt.clf()
    tdf.pos_size.plot()
    plt.savefig(f"{result['session_dirpath']}pos_sizes_plot.png")

    print('plotting average daily gain...')
    adg_ = tdf.average_daily_gain
    adg_.index = np.linspace(0.0, 1.0, len(tdf))
    plt.clf()
    adg_c = adg_.iloc[int(len(tdf) * 0.1):]
    print('min max', adg_c.min(), adg_c.max())
    adg_c.plot()
    plt.savefig(f"{result['session_dirpath']}average_daily_gain_plot.png")
    


def prep_ticks(df: pd.DataFrame) -> np.ndarray:
    dfc = df[df.price != df.price.shift(1)] # drop consecutive same price trades
    dfc.index = np.arange(len(dfc))
    if 'side' in dfc.columns:
        # bybit
        buyer_maker = dfc.side == 'Sell'
        buyer_maker.name = 'buyer_maker'
    elif 'is_buyer_maker' in dfc.columns:
        # binance
        buyer_maker = dfc.is_buyer_maker
        buyer_maker.name = 'buyer_maker'
    else:
        raise Exception('trades of unknown format')
    dfcc = pd.concat([dfc.price, buyer_maker, dfc.timestamp], axis=1)
    return dfcc.values


def backtest(ticks: np.ndarray, settings: dict):

    # ticks formatting [price: float, buyer_maker: bool, timestamp: float]

    ss = settings

    pos_size, pos_price, reentry_price, reentry_qty, liq_price = 0.0, 0.0, 0.0, 0.0, 0.0
    closest_long_liq, closest_shrt_liq = 1.0, 1.0
    highest_pos_price_diff = 0.0
    stop_loss_liq_diff_price, stop_loss_pos_price_diff_price, stop_loss_price = 0.0, 0.0, 0.0
    actual_balance = ss['starting_balance']
    apparent_balance = actual_balance * ss['balance_pct']

    pnl_plus_fees_cumsum, loss_cumsum, profit_cumsum, fee_paid_cumsum = 0.0, 0.0, 0.0, 0.0

    if ss['inverse']:
        min_qty_f = calc_min_qty_inverse
        long_pnl_f = calc_long_pnl_inverse
        shrt_pnl_f = calc_shrt_pnl_inverse
        cost_f = calc_cost_inverse
        pos_margin_f = calc_margin_cost_inverse
        max_pos_size_f = calc_max_pos_size_inverse
        min_entry_qty_f = calc_min_entry_qty_inverse
        long_liq_price_f = lambda bal, psize, pprice: \
            bybit_calc_cross_long_liq_price(bal, psize, pprice, ss['max_leverage'])
        shrt_liq_price_f = lambda bal, psize, pprice: \
            bybit_calc_cross_shrt_liq_price(bal, psize, pprice, ss['max_leverage'])
    else:
        min_qty_f = calc_min_qty_linear
        long_pnl_f = calc_long_pnl_linear
        shrt_pnl_f = calc_shrt_pnl_linear
        cost_f = calc_cost_linear
        pos_margin_f = calc_margin_cost_linear
        max_pos_size_f = calc_max_pos_size_linear
        min_entry_qty_f = calc_min_entry_qty_linear
        long_liq_price_f = lambda bal, psize, pprice: \
            binance_calc_cross_long_liq_price(bal, psize, pprice, ss['leverage'])
        shrt_liq_price_f = lambda bal, psize, pprice: \
            binance_calc_cross_shrt_liq_price(bal, psize, pprice, ss['leverage'])

    break_on = {e[0]: eval(e[1]) for e in settings['break_on'] if e[0].startswith('ON:')}
    break_on_ts = np.nan

    ema_long = ticks[0][0]
    ema_shrt = ticks[0][0]
    ema_long_alpha = 2 / (ss['ema_long_span'] + 1)
    ema_long_alpha_ = 1 - ema_long_alpha
    ema_shrt_alpha = 2 / (ss['ema_shrt_span'] + 1)
    ema_shrt_alpha_ = 1 - ema_shrt_alpha
    prev_trade_ts = 0
    min_trade_delay_millis = ss['latency_simulation_ms'] if 'latency_simulation_ms' in ss else 1000

    trades = []
    ob = [min(ticks[0][0], ticks[1][0]),
          max(ticks[0][0], ticks[1][0])]
    for k, t in enumerate(ticks):
        did_trade = False
        if t[1]:
            # maker buy, taker sel
            if pos_size == 0.0:
                # create long pos
                if ss['do_long']:
                    price = calc_no_pos_bid_price(ss['price_step'], ss['ema_long_spread'], ema_long, ob[0])
                    if t[0] < price:
                        did_trade = True
                        qty = min_entry_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'],
                                              ss['entry_qty_pct'], ss['leverage'], apparent_balance,
                                              price)
                        trade_type, trade_side = 'entry', 'long'
                        pnl = 0.0
                        fee_paid = -cost_f(qty, price) * ss['maker_fee']
            elif pos_size > 0.0:
                closest_long_liq = calc_min(calc_diff(liq_price, t[0]), closest_long_liq)
                if t[0] <= liq_price and closest_long_liq < 0.2:
                    # long liquidation
                    print('\nlong liquidation')
                    return []
                if t[0] < reentry_price:
                    # add to long pos
                    did_trade, qty, price = True, reentry_qty, reentry_price
                    trade_type, trade_side = 'reentry', 'long'
                    pnl = 0.0
                    fee_paid = -cost_f(qty, price) * ss['maker_fee']
                # check if long stop loss triggered
                if t[0] <= stop_loss_liq_diff_price:
                    stop_loss_price = ob[1]
                    stop_loss_type = 'stop_loss_liq_diff'
                elif t[0] <= stop_loss_pos_price_diff_price:
                    stop_loss_price = ob[1]
                    stop_loss_type = 'stop_loss_pos_price_diff'
                else:
                    stop_loss_price = 0.0
            else:
                highest_pos_price_diff = calc_max(calc_diff(pos_price, t[0]), highest_pos_price_diff)
                if t[0] <= pos_price:
                    # close shrt pos
                    min_close_qty = calc_min_close_qty(
                        ss['qty_step'], ss['min_qty'], ss['min_close_qty_multiplier'],
                        min_entry_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'],
                                        ss['entry_qty_pct'], ss['leverage'], apparent_balance,
                                        t[0])
                    )
                    qtys, prices = calc_shrt_closes(ss['price_step'],
                                                    ss['qty_step'],
                                                    min_close_qty,
                                                    ss['min_markup'],
                                                    ss['max_markup'],
                                                    pos_size,
                                                    pos_price,
                                                    ob[0],
                                                    ss['n_close_orders'])
                    if t[0] < prices[0]:
                        did_trade, qty, price = True, qtys[0], prices[0]
                        trade_type, trade_side = 'close', 'shrt'
                        pnl = shrt_pnl_f(pos_price, price, qty)
                        fee_paid = -cost_f(qty, price) * ss['maker_fee']
                elif t[0] < stop_loss_price:
                    # shrt stop loss
                    did_trade = True
                    qty = calc_pos_reduction_qty(ss['qty_step'], ss['stop_loss_pos_reduction'],
                                                 pos_size)
                    price = stop_loss_price
                    trade_type, trade_side = stop_loss_type, 'shrt'
                    pnl = shrt_pnl_f(pos_price, price, qty)
                    fee_paid = -cost_f(qty, price) * ss['maker_fee']

            ob[0] = t[0]
        else:
            # maker sel, taker buy
            if pos_size == 0.0:
                # create shrt pos
                if ss['do_shrt']:
                    price = calc_no_pos_ask_price(ss['price_step'], ss['ema_shrt_spread'], ema_shrt, ob[1])
                    if t[0] > price:
                        did_trade = True
                        qty = -min_entry_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'],
                                               ss['entry_qty_pct'], ss['leverage'],
                                               apparent_balance, price)
                        trade_type, trade_side = 'entry', 'shrt'
                        pnl = 0.0
                        fee_paid = -cost_f(-qty, price) * ss['maker_fee']
            elif pos_size < 0.0:
                closest_shrt_liq = min(calc_diff(liq_price, t[0]), closest_shrt_liq)
                if t[0] >= liq_price and closest_shrt_liq < 0.2:
                    # shrt liquidation
                    print('\nshrt liquidation')
                    return []
                if t[0] > reentry_price:
                    # add to shrt pos
                    did_trade, qty, price = True, reentry_qty, reentry_price
                    trade_type, trade_side = 'reentry', 'shrt'
                    pnl = 0.0
                    fee_paid = -cost_f(-qty, price) * ss['maker_fee']
                # check if shrt stop loss triggered
                if t[0] >= stop_loss_liq_diff_price:
                    stop_loss_price = ob[0]
                    stop_loss_type = 'stop_loss_liq_diff'
                elif t[0] >= stop_loss_pos_price_diff_price:
                    stop_loss_price = ob[0]
                    stop_loss_type = 'stop_loss_pos_price_diff'
                else:
                    stop_loss_price = 0.0
            else:
                highest_pos_price_diff = calc_max(calc_diff(pos_price, t[0]), highest_pos_price_diff)
                # close long pos
                if t[0] >= pos_price:
                    min_close_qty = calc_min_close_qty(
                        ss['qty_step'], ss['min_qty'], ss['min_close_qty_multiplier'],
                        min_entry_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'],
                                        ss['entry_qty_pct'], ss['leverage'], apparent_balance,
                                        t[0])
                    )
                    qtys, prices = calc_long_closes(ss['price_step'],
                                                    ss['qty_step'],
                                                    min_close_qty,
                                                    ss['min_markup'],
                                                    ss['max_markup'],
                                                    pos_size,
                                                    pos_price,
                                                    ob[1],
                                                    ss['n_close_orders'])
                    if t[0] > prices[0]:
                        did_trade, qty, price = True, qtys[0], prices[0]
                        trade_type, trade_side = 'close', 'long'
                        pnl = long_pnl_f(pos_price, price, -qty)
                        fee_paid =  - cost_f(-qty, price) * ss['maker_fee']
                elif stop_loss_price > 0.0 and t[0] > stop_loss_price:
                    # long stop loss
                    did_trade = True
                    qty = -calc_pos_reduction_qty(ss['qty_step'], ss['stop_loss_pos_reduction'],
                                                  pos_size)
                    price = stop_loss_price
                    trade_type, trade_side = stop_loss_type, 'long'
                    pnl = long_pnl_f(pos_price, price, qty)
                    fee_paid = -cost_f(-qty, price) * ss['maker_fee']
            ob[1] = t[0]
        ema_long = calc_ema(ema_long_alpha, ema_long_alpha_, ema_long, t[0])
        ema_shrt = calc_ema(ema_shrt_alpha, ema_shrt_alpha_, ema_shrt, t[0])
        if did_trade:
            if t[2] - prev_trade_ts < min_trade_delay_millis:
                if trade_type == 'reentry':
                    # because of live bot's multiple open orders,
                    # allow consecutive reentries whose timestamp diff < min delay
                    if trades[-1]['type'] != 'reentry':
                        continue
                else:
                    continue
            if qty == 0.0:
                continue
            prev_trade_ts = t[2]
            new_pos_size = round_(pos_size + qty, 0.0000000001)
            if 'entry' in trade_type:
                pos_price = pos_price * abs(pos_size / new_pos_size) + \
                    price * abs(qty / new_pos_size) if new_pos_size else np.nan
            pos_size = new_pos_size
            actual_balance = max(0.0, actual_balance + pnl + fee_paid)
            apparent_balance = actual_balance * ss['balance_pct']
            min_entry_qty = min_entry_qty_f(
                ss['qty_step'], ss['min_qty'], ss['min_cost'], ss['entry_qty_pct'], ss['leverage'],
                apparent_balance, t[0]
            )
            if apparent_balance * ss['leverage'] < cost_f(min_entry_qty, t[0]):
                print('\nself liquidated')
                return []
            if pos_size == 0.0:
                liq_price = 0.0
            elif pos_size > 0.0:
                liq_price = long_liq_price_f(actual_balance, pos_size, pos_price)
            else:
                liq_price = shrt_liq_price_f(actual_balance, pos_size, pos_price)
            if liq_price < 0.0:
                liq_price = 0.0
            progress = k / len(ticks)
            pnl_plus_fee = pnl + fee_paid
            pnl_plus_fees_cumsum += pnl_plus_fee
            if trade_type.startswith('stop_loss'):
                loss_cumsum += pnl
            else:
                profit_cumsum += pnl
            fee_paid_cumsum += fee_paid
            total_gain = (pnl_plus_fees_cumsum + settings['starting_balance']) / settings['starting_balance']
            n_days_ = (t[2] - ticks[0][2]) / (1000 * 60 * 60 * 24)
            try:
                adg = total_gain ** (1 / n_days_) if (n_days_ > 0.0 and total_gain > 0.0) else 0.0
            except:
                adg = 0.0
            avg_gain_per_tick = \
                (actual_balance / settings['starting_balance']) ** (1 / (len(trades) + 1))
            millis_since_prev_trade = t[2] - trades[-1]['timestamp'] if trades else 0.0
            trades.append({'trade_id': k, 'side': trade_side, 'type': trade_type, 'price': price,
                           'qty': qty, 'pos_price': pos_price, 'pos_size': pos_size,
                           'liq_price': liq_price, 'pnl': pnl, 'fee_paid': fee_paid,
                           'pnl_plus_fee': pnl_plus_fee, 'fee_paid_cumsum': fee_paid_cumsum,
                           'apparent_balance': apparent_balance, 'actual_balance': actual_balance, 
                           'profit_cumsum': profit_cumsum, 'loss_cumsum': loss_cumsum,
                           'pnl_plus_fees_cumsum': pnl_plus_fees_cumsum,
                           'average_daily_gain': adg, 'timestamp': t[2],
                           'closest_long_liq': closest_long_liq,
                           'closest_shrt_liq': closest_shrt_liq,
                           'closest_liq': min(closest_long_liq, closest_shrt_liq),
                           'highest_pos_price_diff': highest_pos_price_diff,
                           'avg_gain_per_tick': avg_gain_per_tick,
                           'millis_since_prev_trade': millis_since_prev_trade,
                           'progress': progress})
            closest_long_liq, closest_shrt_liq, highest_pos_price_diff = 1.0, 1.0, 0.0
            for key, condition in break_on.items():
                if condition(trades, ticks, k):
                    print('break on', key)
                    return []
            if pos_size > 0.0:
                stop_loss_liq_diff_price = liq_price * (1 + ss['stop_loss_liq_diff'])
                stop_loss_pos_price_diff_price = pos_price * (1 - ss['stop_loss_pos_price_diff'])
                stop_loss_price = 0.0
                reentry_price = min(
                    ob[0],
                    calc_long_reentry_price(ss['price_step'], ss['grid_spacing'],
                                            ss['grid_coefficient'], apparent_balance,
                                            pos_margin_f(ss['leverage'], pos_size, pos_price),
                                            pos_price)
                )
                reentry_price = max(ss['price_step'], reentry_price)
                min_qty_ = min_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'], reentry_price)
                reentry_qty = calc_reentry_qty(ss['qty_step'],
                                               ss['ddown_factor'],
                                               min_qty_,
                                               max_pos_size_f(ss['leverage'], apparent_balance,
                                                              reentry_price),
                                               pos_size)
                if reentry_qty < min_qty_:
                    reentry_price = ss['price_step']
                trades[-1]['reentry_price'] = reentry_price
            elif pos_size < 0.0:
                stop_loss_liq_diff_price = liq_price * (1 - ss['stop_loss_liq_diff']) \
                    if liq_price > 0.0 else pos_price * 10000
                stop_loss_pos_price_diff_price = pos_price * (1 + ss['stop_loss_pos_price_diff'])
                stop_loss_price = 0.0
                reentry_price = max([
                    ss['price_step'],
                    ob[1],
                    calc_shrt_reentry_price(ss['price_step'], ss['grid_spacing'],
                                            ss['grid_coefficient'], apparent_balance,
                                            pos_margin_f(ss['leverage'], pos_size, pos_price),
                                            pos_price)
                ])
                min_qty_ = min_qty_f(ss['qty_step'], ss['min_qty'], ss['min_cost'], reentry_price)
                reentry_qty = -calc_reentry_qty(ss['qty_step'],
                                                ss['ddown_factor'],
                                                min_qty_,
                                                max_pos_size_f(ss['leverage'], apparent_balance,
                                                                  reentry_price),
                                                pos_size)
                if -reentry_qty < min_qty_:
                    reentry_price = 9e12
                trades[-1]['reentry_price'] = reentry_price
            else:
                trades[-1]['reentry_price'] = np.nan

            '''
            line = f"\r{progress:.3f} pnl plus fees cumsum {pnl_plus_fees_cumsum:.8f} "
            line += f"profit cumsum {profit_cumsum:.5f} "
            line += f"loss cumsum {loss_cumsum:.5f} "
            line += f"actual_bal {actual_balance:.4f} "
            line += f"apparent_bal {apparent_balance:.4f} "
            line += f"adg {trades[-1]['average_daily_gain']:.4f} "
            line += f"pos size {pos_size:.4f} "
            print(line, end=' ')
            '''
    return trades


def calc_new_val(val, range_, m):
    # quick fix of bug where random nums weren't random enuf between processes
    np.random.seed((int(time() * 100000) % (2**31)))
    choice_span = (range_[1] - range_[0]) * m / 2
    biased_mid_point = max(range_[0] + choice_span, min(val, range_[1] - choice_span))
    choice_range = (biased_mid_point - choice_span, biased_mid_point + choice_span)
    new_val = np.random.choice(np.linspace(choice_range[0], choice_range[1], 200))
    #new_val = np.random.random() * (choice_range[1] - choice_range[0])
    return round_(new_val, range_[2])


def get_new_candidate(ranges: dict, best: dict, m=0.2):
    return sort_dict_keys({k: calc_new_val(best[k], ranges[k], m) for k in best if k in ranges})


def get_downloaded_trades(filepath: str, age_limit_millis: float) -> (pd.DataFrame, dict):
    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')],
                           key=lambda x: int(x[:x.find('_')].replace('.cs', '').replace('v', '')))
        chunks = []
        chunk_lengths = {}
        for f in filenames[::-1]:
            chunk = pd.read_csv(filepath + f).set_index('trade_id')
            chunk_lengths[f] = len(chunk)
            print('\rloaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000),
                  end='     ')
            chunks.append(chunk)
            if chunk.timestamp.iloc[0] < age_limit_millis:
                break
        if chunks:
            df = pd.concat(chunks, axis=0).sort_index()
            return df[~df.index.duplicated()], chunk_lengths
        else:
            return None, {}
    else:
        return None, {}


async def load_trades(exchange: str, user: str, symbol: str, n_days: float) -> pd.DataFrame:

    def skip_ids(id_, ids_):
        if id_ in ids_:
            print('skipping from', id_)
            while id_ in ids_:
                id_ -= 1
            print('           to', id_)
        return id_

    def load_cache():
        cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
        if cache_filenames:
            print('loading cached ticks')
            cache_df = pd.concat([pd.read_csv(cache_filepath + f) for f in cache_filenames], axis=0)
            cache_df = cache_df.set_index('trade_id')
            return cache_df
        return None

    if exchange == 'binance':
        fetch_trades_func = binance_fetch_trades
    elif exchange == 'bybit':
        fetch_trades_func = bybit_fetch_trades
    else:
        print(exchange, 'not found')
        return
    cc = init_ccxt(exchange, user)
    filepath = make_get_filepath(os.path.join('historical_data', exchange, 'agg_trades_futures',
                                              symbol, ''))
    cache_filepath = make_get_filepath(filepath.replace(symbol, symbol + '_cache'))
    age_limit = time() - 60 * 60 * 24 * n_days
    age_limit_millis = age_limit * 1000
    print('age_limit', ts_to_date(age_limit))
    cache_df = load_cache()
    trades_df, chunk_lengths = get_downloaded_trades(filepath, age_limit_millis)
    ids = set()
    if trades_df is not None:
        ids.update(trades_df.index)
    if cache_df is not None:
        ids.update(cache_df.index)
    gaps = []
    if trades_df is not None and len(trades_df) > 0:
        # 
        sids = sorted(ids)
        for i in range(1, len(sids)):
            if sids[i-1] + 1 != sids[i]:
                gaps.append((sids[i-1], sids[i]))
        if gaps:
            print('gaps', gaps)
        # 
    prev_fetch_ts = time()
    new_trades = await fetch_trades_func(cc, symbol)
    k = 0
    while True:
        k += 1
        if (break_ := new_trades[0]['timestamp'] <= age_limit_millis) or k % 20 == 0:
            print('caching trades...')
            new_tdf = pd.DataFrame(new_trades).set_index('trade_id')
            cache_filename = f'{cache_filepath}{new_tdf.index[0]}_{new_tdf.index[-1]}.csv'
            new_tdf.to_csv(cache_filename)
            new_trades = [new_trades[0]]
            if break_:
                break
        from_id = skip_ids(new_trades[0]['trade_id'] - 1, ids) - 999
        # wait at least 0.75 sec between each fetch
        sleep_for = max(0.0, 0.75 - (time() - prev_fetch_ts))
        await asyncio.sleep(sleep_for)
        prev_fetch_ts = time()
        fetched_new_trades = await fetch_trades_func(cc, symbol, from_id=from_id)
        while fetched_new_trades[0]['trade_id'] == new_trades[0]['trade_id']:
            print('gaps in ids', from_id)
            from_id -= 1000
            fetched_new_trades = await fetch_trades_func(cc, symbol, from_id=from_id)
        new_trades = fetched_new_trades + new_trades
        ids.update([e['trade_id'] for e in new_trades])
        if new_trades[0]['trade_id'] <= 1000:
            print('end of the line')
            break
    tdf = pd.concat([load_cache(), trades_df], axis=0).sort_index()
    tdf = tdf[~tdf.index.duplicated()]
    dump_chunks(filepath, tdf, chunk_lengths)
    cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
    print('removing cache...\n')
    for filename in cache_filenames:
        print(f'\rremoving {filename}', end='   ')
        os.remove(cache_filepath + filename)
    await cc.close()
    return tdf[tdf.timestamp >= age_limit_millis]


def dump_chunks(filepath: str, tdf: pd.DataFrame, chunk_lengths: dict, chunk_size=100000):
    chunk_ids = tdf.index // chunk_size * chunk_size
    for g in tdf.groupby(chunk_ids):
        filename = f'{g[1].index[0]}_{g[1].index[-1]}.csv'
        if filename not in chunk_lengths or chunk_lengths[filename] != chunk_size:
            print('dumping chunk', filename)
            g[1].to_csv(f'{filepath}{filename}')


async def fetch_market_specific_settings(exchange: str, user: str, symbol: str):
    tmp_live_settings = load_live_settings(exchange, do_print=False)
    tmp_live_settings['symbol'] = symbol
    settings_from_exchange = {}
    if exchange == 'binance':
        bot = await create_bot_binance(user, tmp_live_settings)
        settings_from_exchange['inverse'] = False
        settings_from_exchange['maker_fee'] = 0.00018
        settings_from_exchange['taker_fee'] = 0.00036
        settings_from_exchange['exchange'] = 'binance'
    elif exchange == 'bybit':
        bot = await create_bot_bybit(user, tmp_live_settings)
        settings_from_exchange['inverse'] = True
        settings_from_exchange['maker_fee'] = -0.00025
        settings_from_exchange['taker_fee'] = 0.00075
        settings_from_exchange['exchange'] = 'bybit'
    else:
        raise Exception(f'unknown exchange {exchange}')
    settings_from_exchange['max_leverage'] = bot.max_leverage
    settings_from_exchange['min_qty'] = bot.min_qty
    settings_from_exchange['min_cost'] = bot.min_notional
    settings_from_exchange['qty_step'] = bot.qty_step
    settings_from_exchange['price_step'] = bot.price_step
    settings_from_exchange['max_leverage'] = bot.max_leverage
    await bot.cc.close()
    return settings_from_exchange


def live_settings_to_candidate(live_settings: dict, ranges: dict) -> dict:
    candidate = {k: live_settings[k] for k in ranges if k in live_settings}
    for k in ['span', 'spread']:
        if 'long_' + k in live_settings['indicator_settings']['tick_ema']:
            candidate['ema_long_' + k] = live_settings['indicator_settings']['tick_ema']['long_' + k]
        if 'shrt_' + k in live_settings['indicator_settings']['tick_ema']:
            candidate['ema_shrt_' + k] = live_settings['indicator_settings']['tick_ema']['shrt_' + k]
    for k in ['do_long', 'do_shrt']:
        candidate[k] = live_settings['indicator_settings'][k]
    return candidate


def candidate_to_live_settings(exchange: str, candidate: dict) -> dict:
    live_settings = load_live_settings(exchange, do_print=False)
    live_settings['config_name'] = candidate['session_name']
    live_settings['symbol'] = candidate['symbol']
    live_settings['key'] = candidate['key']
    for k in candidate:
        if k in live_settings:
            live_settings[k] = candidate[k]
    for k in ['ema_span', 'ema_spread']:
        live_settings['indicator_settings']['tick_ema']['long_' + k[4:]] = candidate[k.replace('_', '_long_')]
        live_settings['indicator_settings']['tick_ema']['shrt_' + k[4:]] = candidate[k.replace('_', '_shrt_')]
    for k in ['do_long', 'do_shrt']:
        live_settings['indicator_settings'][k] = bool(candidate[k])
    return live_settings


def calc_candidate_hash_key(candidate: dict, keys: [str]) -> str:
    return sha256(json.dumps({k: candidate[k] for k in sorted(keys)}).encode()).hexdigest()


def load_results(results_filepath: str) -> dict:
    if os.path.exists(results_filepath):
        with open(results_filepath) as f:
            lines = f.readlines()
        results = {(e := json.loads(line))['key']: e for line in lines}
    else:
        results = {}
    return results


async def jackrabbit(ticks: [dict], backtest_config: dict):
    results = load_results(backtest_config['session_dirpath'] + 'results.txt')
    k = backtest_config['starting_k']
    ks = backtest_config['n_jackrabbit_iterations']
    if ks > 1:
        ms = np.array([1 / (i / 2 + 16) for i in range(ks)])
        ms = ((ms - ms.min()) / (ms.max() - ms.min()))
    else:
        ms = np.array([0.0])
    try:
        best_result = json.load(open(backtest_config['session_dirpath'] + 'best_result.json'))
    except Exception as e:
        print('no current best result')
        best_result = {}
    try:
        candidate = live_settings_to_candidate(
            json.load(open(backtest_config['starting_candidate_filepath'])),
            backtest_config['ranges']
        )
        print('using given starting candidate', backtest_config['starting_candidate_filepath'])
    except Exception as e:
        print(e, f"starting candidate {backtest_config['starting_candidate_filepath']} not found.")
        if best_result:
            print('building on current best')
            candidate = get_new_candidate(backtest_config['ranges'], best_result, m=ms[k])
            pass
        else:
            print('using random starting candidate')
            candidate = get_new_candidate(
                backtest_config['ranges'],
                {k_: 0.0 for k_ in backtest_config['ranges']},
                m=1.0
            )
    if '--plot' in sys.argv:
        _, best_result = load_shared_data(backtest_config['session_dirpath'], Lock())
        if not best_result:
            return
        print('backtesting and plotting best candidate')
        result_, tdf_ = jackrabbit_wrap(ticks, {**backtest_config, **{'break_on': []}, **best_result})
        if tdf_ is None:
            print('no trades')
            return
        tdf_.to_csv(backtest_config['session_dirpath'] + f"backtest_trades_{best_result['key']}.csv")
        print('\nmaking ticks dataframe...')
        df = pd.DataFrame({'price': ticks[:,0], 'buyer_maker': ticks[:,1], 'timestamp': ticks[:,2]})
        dump_plots({**backtest_config, **best_result, **result_}, tdf_, df)
        return
    
    if True:#backtest_config['multiprocessing']:
        await jackrabbit_multi_core(results,
                                    ticks,
                                    backtest_config,
                                    candidate,
                                    k,
                                    ks,
                                    ms)
    else:
        jackrabbit_single_core(results,
                               ticks,
                               backtest_config,
                               candidate,
                               k,
                               ks,
                               ms)


async def multiprocess_wrap(func, args=()):
    result = await aiomultiprocess.Worker(target=func, args=args)
    return result


def load_best_result(dirpath: str):
    if os.path.exists((p := dirpath + 'best_result.json')):
        return json.load(open(p))
    return {}


def load_keys(dirpath: str) -> set:
    if os.path.exists((p := dirpath + 'keys.txt')):
        with open(p) as f:
            keys = set([line.strip() for line in f.readlines()])
    else:
        keys = set()
    return keys

def append_key(dirpath: str, key: str):
    with open(dirpath + 'keys.txt', 'a') as f:
        f.write(key + '\n')

def load_shared_data(dirpath: str, lock: Lock) -> (dict, dict):
    lock.acquire()
    try:
        return load_keys(dirpath), load_best_result(dirpath)
    finally:
        lock.release()


def dump_shared_data(dirpath: str, result: dict, best_result: dict, lock: Lock) -> None:
    lock.acquire()
    try:
        with open(dirpath + 'results.txt', 'a') as f:
            f.write(json.dumps(result) + '\n')
        append_key(dirpath, result['key'])
        if load_best_result(dirpath) != best_result:
            json.dump(best_result, open(dirpath + 'best_result.json', 'w'), indent=4)
    finally:
        lock.release()


def get_next_candidate(backtest_config: dict, candidate: dict, ms: [float], k: Value, lock: Lock):
    lock.acquire()
    new_candidate = candidate.copy()
    try:
        dirpath = backtest_config['session_dirpath']
        keys, best_result = load_keys(dirpath), load_best_result(dirpath)
        key = calc_candidate_hash_key(candidate, list(backtest_config['ranges']))
        for _ in range(10):
            if key not in keys:
                break
            new_candidate = get_new_candidate(backtest_config['ranges'],
                                              (best_result if best_result else candidate),
                                              m=ms[k.value])
            key = calc_candidate_hash_key(new_candidate, list(backtest_config['ranges']))
        else:
            return None, key
        append_key(dirpath, key)
        return new_candidate, key
    finally:
        lock.release()


async def jackrabbit_worker(ticks: np.ndarray,
                            backtest_config: dict,
                            candidate: dict,
                            score_func: Callable,
                            k: Value,
                            ks: int,
                            ms: [float],
                            lock: Lock):
    start_time = time()
    while True:
        if k.value >= ks:
            break
        candidate, key = get_next_candidate(backtest_config, candidate, ms, k, lock)
        if candidate is None:
            break
        k.value = k.value + 1
        bpm = k.value / (time() - start_time) * 60
        line = f'running backtest {k.value} of {ks}. m={ms[k.value]:.3f} {key} '
        line += f'backtests per minute: {bpm:.2f}'
        print(line)
        print(candidate, '\n')
        result = await jackrabbit_sliding_window_wrap(ticks, {**backtest_config,
                                                              **candidate,
                                                              **{'key': key}})
        result['key'] = key
        result = {**result, **candidate}
        keys, best_result = load_shared_data(backtest_config['session_dirpath'], lock)
        if score_func(best_result, result):
            print('\n\n### new best ###\n\n')
            best_result = result
            pprint.pprint(best_result)
            print('\n\n')
            json.dump(candidate_to_live_settings(backtest_config['exchange'],
                                                 {**backtest_config, **candidate, **best_result}),
                      open(backtest_config['session_dirpath'] + 'live_config.json', 'w'),
                      indent=4)
        dump_shared_data(backtest_config['session_dirpath'], result, best_result, lock)


async def jackrabbit_multi_core(results: dict,
                                ticks: [dict],
                                backtest_config: dict,
                                candidate: dict,
                                k_: int,
                                ks: int,
                                ms: [float]):
    n_cpus = min(cpu_count(), backtest_config['max_n_cpus'])
    print('using', n_cpus, 'cpus')
    score_func = get_score_func(backtest_config['score_metric'])
    lock = Lock()
    k = Value('i', k_)
    workers = []
    for _ in range(min(n_cpus, ks)):
        workers.append(asyncio.create_task(multiprocess_wrap(
            jackrabbit_worker, (ticks, backtest_config, candidate, score_func, k, ks, ms, lock)
        )))
        await asyncio.sleep(0.05)
    for w in workers:
        await w


def jackrabbit_wrap(ticks: [dict], backtest_config: dict) -> dict:
    start_ts = time()
    trades = backtest(ticks, backtest_config)
    elapsed = time() - start_ts
    if not trades:
        return {}, None
    tdf = pd.DataFrame(trades).set_index('trade_id')
    ms_gap = np.diff([ticks[0][2]] + list(tdf.timestamp) + [ticks[-1][2]]).max()
    result = {
        'net_pnl_plus_fees': trades[-1]['pnl_plus_fees_cumsum'],
        'profit_sum': trades[-1]['profit_cumsum'],
        'loss_sum': trades[-1]['loss_cumsum'],
        'fee_sum': trades[-1]['fee_paid_cumsum'],
        'gain': (gain := (trades[-1]['pnl_plus_fees_cumsum'] + backtest_config['starting_balance']) /
                 backtest_config['starting_balance']),
        'n_days': (n_days := (ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': gain ** (1 / n_days) if gain > 0.0 else 0.0,
        'closest_shrt_liq': (csl := tdf.closest_shrt_liq.min()),
        'closest_long_liq': (cll := tdf.closest_long_liq.min()),
        'closest_liq': (cl := min(csl, cll)),
        'max_n_hours_between_consec_trades': (hours_gap := ms_gap / (1000 * 60 * 60)),
        'n_trades': len(trades),
        'n_closes': len(tdf[tdf.type == 'close']),
        'n_reentries': len(tdf[tdf.type == 'reentry']),
        'n_stop_losses': len(tdf[tdf.type.str.startswith('stop_loss')]),
        'biggest_pos_size': tdf.pos_size.abs().max(),
        'do_long': bool(backtest_config['do_long']),
        'do_shrt': bool(backtest_config['do_shrt']),
        'seconds_elapsed': elapsed
    }
    return result, tdf


def iter_slices(iterable, step: float, size: float):
    i = 0
    n = int(len(iterable) * size)
    s = int(len(iterable) * step)
    while i + n < len(iterable):
        yield iterable[i:i+n]
        i += s
    yield iterable[-n:]
    if size < 1.0:
        # also yield full 
        yield iterable


async def jackrabbit_sliding_window_wrap(ticks: np.ndarray, backtest_config: dict) -> dict:
    sub_runs = []
    start_ts = time()
    for z, slice_ in enumerate(iter_slices(ticks,
                                           backtest_config['sliding_window_step'],
                                           backtest_config['sliding_window_size'])):
        result_, _ = jackrabbit_wrap(slice_, backtest_config)
        if not result_:
            print(f"\n{backtest_config['key']} did not finish backtest")
            return {'key': backtest_config['key']}
        sub_runs.append(result_)
    total_elapsed = time() - start_ts
    result = {}
    skip = ['do_long', 'do_shrt']
    for k in sub_runs[0]:
        if k in skip:
            continue
        try:
            vals = [r[k] for r in sub_runs]
            result[k] = {'avg': np.mean(vals),
                         'std': np.std(vals),
                         'min': min(vals),
                         'max': max(vals),
                         'vals': {**{f'sub_run_{i:03}': v for i, v in enumerate(vals[:-1])},
                                  **{'full_run': vals[-1]}}}
        except:
            continue
    result['n_sub_runs'] = z
    result['total_seconds_elapsed'] = total_elapsed
    result['key'] = backtest_config['key']
    return result


async def load_ticks(backtest_config: dict) -> [dict]:
    ticks_filepath = make_get_filepath(os.path.join(backtest_config['session_dirpath'], f"ticks_cache.npy"))
    if os.path.exists(ticks_filepath):
        print('loading cached trade list', ticks_filepath)
        ticks = np.load(ticks_filepath, allow_pickle=True)
    else:
        agg_trades = await load_trades(backtest_config['exchange'], backtest_config['user'],
                                       backtest_config['symbol'], backtest_config['n_days'])
        print('preparing ticks...')
        ticks = prep_ticks(agg_trades)
        np.save(ticks_filepath, ticks)
    return ticks


async def prep_backtest_config(config_name: str):
    backtest_config = hjson.load(open(f'backtest_configs/{config_name}.hjson'))

    exchange = backtest_config['exchange']
    user = backtest_config['user']
    symbol = backtest_config['symbol']
    session_name = backtest_config['session_name']

    session_dirpath = make_get_filepath(os.path.join(
        'backtest_results',
        exchange,
        symbol,
        f"{session_name}_{backtest_config['n_days']}_days",
        ''))
    if os.path.exists((mss := session_dirpath + 'market_specific_settings.json')):
        market_specific_settings = json.load(open(mss))
    else:
        market_specific_settings = await fetch_market_specific_settings(exchange, user, symbol)
        json.dump(market_specific_settings, open(mss, 'w'))
    backtest_config.update(market_specific_settings)

    # setting absolute min/max ranges
    for key in ['balance_pct', 'entry_qty_pct', 'ddown_factor', 'ema_long_span', 'ema_shrt_span',
                'grid_coefficient', 'grid_spacing', 'min_close_qty_multiplier',
                'stop_loss_pos_reduction']:
        backtest_config['ranges'][key][0] = max(0.0, backtest_config['ranges'][key][0])
    for key in ['balance_pct', 'entry_qty_pct', 'min_close_qty_multiplier',
                'stop_loss_pos_reduction']:
        backtest_config['ranges'][key][1] = min(1.0, backtest_config['ranges'][key][1])

    backtest_config['ranges']['leverage'][1] = \
        min(backtest_config['ranges']['leverage'][1],
            backtest_config['max_leverage'])
    backtest_config['ranges']['leverage'][0] = \
        min(backtest_config['ranges']['leverage'][0],
            backtest_config['ranges']['leverage'][1])
    
    backtest_config['session_dirpath'] = session_dirpath

    return backtest_config


async def get_ticks_new(exchange: str, user: str, symbol: str, start_ts: int, end_ts: int):
    # unabridged price history is stored in chunks by timestamp
    # abridged price history, called ticks, are stored in chunks by timestamp
    trade_ids_already_downloaded = set()


async def main():
    config_name = sys.argv[1]
    backtest_config = await prep_backtest_config(config_name)
    ticks = await load_ticks(backtest_config)
    await jackrabbit(ticks, backtest_config)


if __name__ == '__main__':
    asyncio.run(main())


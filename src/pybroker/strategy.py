"""Contains implementation for backtesting trading strategies."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""
import dataclasses
import numpy as np
import pandas as pd
from pybroker.cache import CacheDateFields
from pybroker.common import (
    BarData,
    DataCol,
    Day,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    quantize,
    to_datetime,
    to_decimal,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)
from pybroker.config import StrategyConfig
from pybroker.context import (
    ExecContext,
    ExecResult,
    PosSizeContext,
    set_exec_ctx_data,
    set_pos_size_ctx_data,
)
from pybroker.data import AlpacaCrypto, DataSource
from pybroker.eval import BootstrapResult, EvalMetrics, EvaluateMixin
from pybroker.indicator import Indicator, IndicatorsMixin
from pybroker.model import ModelSource, ModelsMixin, TrainedModel
from pybroker.portfolio import (
    Order,
    Portfolio,
    PortfolioBar,
    PositionBar,
    Trade,
)
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrderScope,
    PredictionScope,
    PriceScope,
    StaticScope,
)
from pybroker.slippage import SlippageModel
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Union,
)


def _between(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df[DataCol.DATE.value].dt.tz_localize(None) >= start_date)
        & (df[DataCol.DATE.value].dt.tz_localize(None) <= end_date)
    ]


def _sort_by_score(result: ExecResult) -> float:
    return 0.0 if result.score is None else result.score

@dataclass(frozen=True)
class TestResult:
    """Contains the results of backtesting a :class:`.Strategy`.

    Attributes:
        start_date: Starting date of backtest.
        end_date: Ending date of backtest.
        portfolio: :class:`pandas.DataFrame` of
            :class:`pybroker.portfolio.Portfolio` balances for every bar.
        positions: :class:`pandas.DataFrame` of
            :class:`pybroker.portfolio.Position` balances for every bar.
        orders: :class:`pandas.DataFrame` of all orders that were placed.
        trades: :class:`pandas.DataFrame` of all trades that were made.
        metrics: Evaluation metrics.
        metrics_df: :class:`pandas.DataFrame` of evaluation metrics.
        bootstrap: Randomized bootstrap evaluation metrics.
    """

    start_date: datetime
    end_date: datetime
    portfolio: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    metrics: EvalMetrics
    metrics_df: pd.DataFrame
    bootstrap: Optional[BootstrapResult]

class Execution(NamedTuple):
    r"""Represents an execution of a :class:`.Strategy`. Holds a reference to
    a :class:`Callable` that implements trading logic.

    Attributes:
        id: Unique ID.
        symbols: Ticker symbols used for execution of ``fn``.
        fn: Implements trading logic.
        model_names: Names of :class:`pybroker.model.ModelSource`\ s used for
            execution of ``fn``.
        indicator_names: Names of :class:`pybroker.indicator.Indicator`\ s
            used for execution of ``fn``.
    """

    id: int
    symbols: frozenset[str]
    fn: Optional[Callable[[ExecContext], None]]
    model_names: frozenset[str]
    indicator_names: frozenset[str]


class BacktestMixin:
    """Mixin implementing backtesting functionality."""

    def backtest_executions(
        self,
        stack_ctx: dict,
        config: StrategyConfig,
        executions: set[Execution],
        before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        pos_size_handler: Optional[Callable[[PosSizeContext], None]],
        slippage_model: Optional[SlippageModel] = None,
        enable_fractional_shares: bool = False,
        warmup: Optional[int] = None,
    ):
        r"""Backtests a ``set`` of :class:`.Execution`\ s that implement
        trading logic.

        Args:
            config: :class:`pybroker.config.StrategyConfig`.
            executions: :class:`.Execution`\ s to run.
            sessions: :class:`Mapping` of symbols to :class:`Mapping` of custom
                data that persists for every bar during the
                :class:`.Execution`.
            models: :class:`Mapping` of :class:`pybroker.common.ModelSymbol`
                pairs to :class:`pybroker.common.TrainedModel`\ s.
            indicator_data: :class:`Mapping` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                :class:`pandas.Series` of :class:`pybroker.indicator.Indicator`
                values.
            test_data: :class:`pandas.DataFrame` of test data.
            portfolio: :class:`pybroker.portfolio.Portfolio`.
            pos_size_handler: :class:`Callable` that sets position sizes when
                placing orders for buy and sell signals.
            exit_dates: :class:`Mapping` of symbols to exit dates.
            enable_fractional_shares: Whether to enable trading fractional
                shares.
            warmup: Number of bars that need to pass before running the
                executions.

        Returns:
            :class:`.TestResult` of the backtest.
        """
        backtest_stepper = self.backtest_executions_stepwise(
                                stack_ctx=stack_ctx,
                                config=config,
                                executions=executions,
                                before_exec_fn=before_exec_fn,
                                after_exec_fn=after_exec_fn,
                                pos_size_handler=pos_size_handler,
                                slippage_model=slippage_model,
                                enable_fractional_shares=enable_fractional_shares,
                                warmup=warmup
                            )
        try:
            while True:
                next(backtest_stepper)
        except StopIteration:
            pass
                

    def backtest_executions_stepwise(
        self,
        stack_ctx: dict,
        config: StrategyConfig,
        executions: set[Execution],
        before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        pos_size_handler: Optional[Callable[[PosSizeContext], None]],
        final_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]] = None,
        slippage_model: Optional[SlippageModel] = None,
        enable_fractional_shares: bool = False,
        warmup: Optional[int] = None,
        include_results: bool = False,
        results_stepwise: bool = False
    ) -> Optional[TestResult]:
        self._backtest_init(
            stack_ctx=stack_ctx,
            config=config, 
            executions=executions,  
            pos_size_handler=pos_size_handler
        )
        for i, date in enumerate(stack_ctx['test_dates']):
            final_exec_fn = yield self._backtest_date(
                                i=i, 
                                date=date,
                                stack_ctx=stack_ctx,
                                config=config, 
                                before_exec_fn=before_exec_fn, 
                                after_exec_fn=after_exec_fn,
                                pos_size_handler=pos_size_handler, 
                                final_exec_fn=final_exec_fn,
                                slippage_model=slippage_model, 
                                enable_fractional_shares=enable_fractional_shares, 
                                warmup=warmup,
                                include_results=include_results,
                                results_stepwise=results_stepwise
                            )

    def _backtest_init(
        self,
        stack_ctx: dict,
        config: StrategyConfig, 
        executions: set[Execution], 
        pos_size_handler: Optional[Callable[[PosSizeContext], None]]
    ):
        stack_ctx['test_dates'] = stack_ctx['test_data'][DataCol.DATE.value].unique()
        stack_ctx['test_dates'].sort()
        stack_ctx['num_dates'] = len(stack_ctx['test_dates'])
        stack_ctx['test_syms'] = stack_ctx['test_data'][DataCol.SYMBOL.value].unique()
        stack_ctx['test_data'] = (
            stack_ctx['test_data'].reset_index(drop=True)
            .set_index([DataCol.SYMBOL.value, DataCol.DATE.value])
            .sort_index()
        )
        stack_ctx['sym_end_index']: dict[str, int] = defaultdict(int)
        stack_ctx['col_scope'] = ColumnScope(stack_ctx['test_data'])
        stack_ctx['price_scope'] = PriceScope(stack_ctx['col_scope'], stack_ctx['sym_end_index'])
        stack_ctx['ind_scope'] = IndicatorScope(stack_ctx['indicator_data'], stack_ctx['test_dates'])
        stack_ctx['input_scope'] = ModelInputScope(stack_ctx['col_scope'], stack_ctx['ind_scope'], stack_ctx['models'])
        stack_ctx['pred_scope'] = PredictionScope(stack_ctx['models'], stack_ctx['input_scope'])
        stack_ctx['pending_order_scope'] = PendingOrderScope()
        stack_ctx['exec_ctxs']: dict[str, ExecContext] = {}
        stack_ctx['exec_fns']: dict[str, Callable[[ExecContext], None]] = {}
        for sym in stack_ctx['test_syms']:
            for exec in executions:
                if sym not in exec.symbols:
                    continue
                stack_ctx['exec_ctxs'][sym] = ExecContext(
                    symbol=sym,
                    config=config,
                    portfolio=stack_ctx['portfolio'],
                    col_scope=stack_ctx['col_scope'],
                    ind_scope=stack_ctx['ind_scope'],
                    input_scope=stack_ctx['input_scope'],
                    pred_scope=stack_ctx['pred_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                    models=stack_ctx['models'],
                    sym_end_index=stack_ctx['sym_end_index'],
                    session=stack_ctx['sessions'][sym],
                )
                if exec.fn is not None:
                    stack_ctx['exec_fns'][sym] = exec.fn
        stack_ctx['sym_exec_dates'] = {
            sym: frozenset(stack_ctx['test_data'].loc[pd.IndexSlice[sym, :]].index.values)
            for sym in stack_ctx['exec_ctxs'].keys()
        }
        stack_ctx['cover_sched']: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        stack_ctx['buy_sched']: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        stack_ctx['sell_sched']: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        if pos_size_handler is not None:
            stack_ctx['pos_ctx'] = PosSizeContext(
                config=config,
                portfolio=stack_ctx['portfolio'],
                col_scope=stack_ctx['col_scope'],
                ind_scope=stack_ctx['ind_scope'],
                input_scope=stack_ctx['input_scope'],
                pred_scope=stack_ctx['pred_scope'],
                pending_order_scope=stack_ctx['pending_order_scope'],
                models=stack_ctx['models'],
                sessions=stack_ctx['sessions'],
                sym_end_index=stack_ctx['sym_end_index'],
            )
        stack_ctx['logger'] = StaticScope.instance().logger
        stack_ctx['logger'].backtest_executions_start(stack_ctx['test_dates'])
        stack_ctx['cover_results']: deque[ExecResult] = deque()
        stack_ctx['buy_results']: deque[ExecResult] = deque()
        stack_ctx['sell_results']: deque[ExecResult] = deque()
        stack_ctx['exit_ctxs']: deque[ExecContext] = deque()
        stack_ctx['active_ctxs']: dict[str, ExecContext] = {}

    def _backtest_date(
        self, 
        i: int, 
        date,
        stack_ctx: dict,
        config: StrategyConfig, 
        before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        pos_size_handler: Optional[Callable[[PosSizeContext], None]],
        final_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]] = None,
        slippage_model: Optional[SlippageModel] = None, 
        enable_fractional_shares: bool = False, 
        warmup: Optional[int] = None,
        include_results: bool = False,
        results_stepwise: bool = False
    ) -> Optional[TestResult]:
        stack_ctx['active_ctxs'].clear()
        for sym, ctx in stack_ctx['exec_ctxs'].items():
            if date not in stack_ctx['sym_exec_dates'][sym]:
                continue
            stack_ctx['sym_end_index'][sym] += 1
            if warmup and stack_ctx['sym_end_index'][sym] <= warmup:
                continue
            stack_ctx['active_ctxs'][sym] = ctx
            set_exec_ctx_data(ctx, date)
            if (
                    stack_ctx['exit_dates']
                    and sym in  stack_ctx['exit_dates']
                    and date ==  stack_ctx['exit_dates'][sym]
                ):
                stack_ctx['exit_ctxs'].append(ctx)
        is_cover_sched = date in stack_ctx['cover_sched']
        is_buy_sched = date in stack_ctx['buy_sched']
        is_sell_sched = date in stack_ctx['sell_sched']
        if (
                config.max_long_positions is not None
                or pos_size_handler is not None
            ):
            if is_cover_sched:
                stack_ctx['cover_sched'][date].sort(key=_sort_by_score, reverse=True)
            elif is_buy_sched:
                stack_ctx['buy_sched'][date].sort(key=_sort_by_score, reverse=True)
        if is_sell_sched and (
                config.max_short_positions is not None
                or pos_size_handler is not None
            ):
            stack_ctx['sell_sched'][date].sort(key=_sort_by_score, reverse=True)
        if pos_size_handler is not None and (
                is_cover_sched or is_buy_sched or is_sell_sched
            ):
            pos_size_buy_results = None
            if is_cover_sched:
                pos_size_buy_results = stack_ctx['cover_sched'][date]
            elif is_buy_sched:
                pos_size_buy_results = stack_ctx['buy_sched'][date]
            self._set_pos_sizes(
                    pos_size_handler=pos_size_handler,
                    pos_ctx=stack_ctx['pos_ctx'],
                    buy_results=pos_size_buy_results,
                    sell_results=stack_ctx['sell_sched'][date] if is_sell_sched else None,
                )
        stack_ctx['portfolio'].check_stops(date, stack_ctx['price_scope'])
        if is_cover_sched:
            self._place_buy_orders(
                    date=date,
                    price_scope=stack_ctx['price_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                    buy_sched=stack_ctx['cover_sched'],
                    portfolio=stack_ctx['portfolio'],
                    enable_fractional_shares=enable_fractional_shares,
                )
        if is_sell_sched:
            self._place_sell_orders(
                    date=date,
                    price_scope=stack_ctx['price_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                    sell_sched=stack_ctx['sell_sched'],
                    portfolio=stack_ctx['portfolio'],
                    enable_fractional_shares=enable_fractional_shares,
                )
        if is_buy_sched:
            self._place_buy_orders(
                    date=date,
                    price_scope=stack_ctx['price_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                    buy_sched=stack_ctx['buy_sched'],
                    portfolio=stack_ctx['portfolio'],
                    enable_fractional_shares=enable_fractional_shares,
                )
        stack_ctx['portfolio'].capture_bar(date, stack_ctx['test_data'])
        if before_exec_fn is not None and stack_ctx['active_ctxs']:
            before_exec_fn(stack_ctx['active_ctxs'])
        for sym, ctx in stack_ctx['active_ctxs'].items():
            if sym in stack_ctx['exec_fns']:
                stack_ctx['exec_fns'][sym](ctx)
        if after_exec_fn is not None and stack_ctx['active_ctxs']:
            after_exec_fn(stack_ctx['active_ctxs'])
        if final_exec_fn is not None and stack_ctx['active_ctxs']:
            final_exec_fn(stack_ctx['active_ctxs'])
        for ctx in stack_ctx['active_ctxs'].values():
            if slippage_model and (ctx.buy_shares or ctx.sell_shares):
                self._apply_slippage(slippage_model, ctx)
            result = ctx.to_result()
            if result is None:
                continue
            if result.buy_shares is not None:
                if result.cover:
                    stack_ctx['cover_results'].append(result)
                else:
                    stack_ctx['buy_results'].append(result)
            if result.sell_shares is not None:
                stack_ctx['sell_results'].append(result)
        while stack_ctx['cover_results']:
            self._schedule_order(
                    result=stack_ctx['cover_results'].popleft(),
                    created=date,
                    sym_end_index=stack_ctx['sym_end_index'],
                    delay=config.buy_delay,
                    sched=stack_ctx['cover_sched'],
                    col_scope=stack_ctx['col_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                )
        while stack_ctx['buy_results']:
            self._schedule_order(
                    result=stack_ctx['buy_results'].popleft(),
                    created=date,
                    sym_end_index=stack_ctx['sym_end_index'],
                    delay=config.buy_delay,
                    sched=stack_ctx['buy_sched'],
                    col_scope=stack_ctx['col_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                )
        while stack_ctx['sell_results']:
            self._schedule_order(
                    result=stack_ctx['sell_results'].popleft(),
                    created=date,
                    sym_end_index=stack_ctx['sym_end_index'],
                    delay=config.sell_delay,
                    sched=stack_ctx['sell_sched'],
                    col_scope=stack_ctx['col_scope'],
                    pending_order_scope=stack_ctx['pending_order_scope'],
                )
        while stack_ctx['exit_ctxs']:
            self._exit_position(
                    portfolio=stack_ctx['portfolio'],
                    date=date,
                    ctx=stack_ctx['exit_ctxs'].popleft(),
                    exit_cover_fill_price=config.exit_cover_fill_price,
                    exit_sell_fill_price=config.exit_sell_fill_price,
                    price_scope=stack_ctx['price_scope'],
                )
        stack_ctx['portfolio'].incr_bars()
        is_last_date = i == stack_ctx['num_dates'] - 1
        if i % 10 == 0 or is_last_date:
            stack_ctx['logger'].backtest_executions_loading(i + 1)
        
        result = None
        is_train_only = 'train_only' in stack_ctx and stack_ctx['train_only']
        if not is_train_only and include_results and (results_stepwise or is_last_date):
            result = self._to_test_result(
                        start_date=stack_ctx['start_dt'], 
                        end_date=stack_ctx['end_dt'], 
                        portfolio=stack_ctx['portfolio'], 
                        calc_bootstrap=stack_ctx['calc_bootstrap']
                    )
        return result

    def _apply_slippage(
        self,
        slippage_model: SlippageModel,
        ctx: ExecContext,
    ):
        buy_shares = to_decimal(ctx.buy_shares) if ctx.buy_shares else None
        sell_shares = to_decimal(ctx.sell_shares) if ctx.sell_shares else None
        slippage_model.apply_slippage(
            ctx, buy_shares=buy_shares, sell_shares=sell_shares
        )

    def _exit_position(
        self,
        portfolio: Portfolio,
        date: np.datetime64,
        ctx: ExecContext,
        exit_cover_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        exit_sell_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        price_scope: PriceScope,
    ):
        buy_fill_price = price_scope.fetch(ctx.symbol, exit_cover_fill_price)
        sell_fill_price = price_scope.fetch(ctx.symbol, exit_sell_fill_price)
        portfolio.exit_position(
            date,
            ctx.symbol,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
        )

    def _set_pos_sizes(
        self,
        pos_size_handler: Callable[[PosSizeContext], None],
        pos_ctx: PosSizeContext,
        buy_results: Optional[list[ExecResult]],
        sell_results: Optional[list[ExecResult]],
    ):
        set_pos_size_ctx_data(
            ctx=pos_ctx, buy_results=buy_results, sell_results=sell_results
        )
        pos_size_handler(pos_ctx)
        for id, shares in pos_ctx._signal_shares.items():
            if id < 0:
                raise ValueError(f"Invalid ExecSignal id: {id}")
            if buy_results is not None and sell_results is not None:
                if id >= (len(buy_results) + len(sell_results)):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                if id < len(buy_results):
                    buy_results[id].buy_shares = to_decimal(shares)
                else:
                    sell_results[
                        id - len(buy_results)
                    ].sell_shares = to_decimal(shares)
            elif buy_results is not None:
                if id >= len(buy_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                buy_results[id].buy_shares = to_decimal(shares)
            elif sell_results is not None:
                if id >= len(sell_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                sell_results[id].sell_shares = to_decimal(shares)
            else:
                raise ValueError(
                    "buy_results and sell_results cannot both be None."
                )

    def _schedule_order(
        self,
        result: ExecResult,
        created: np.datetime64,
        sym_end_index: Mapping[str, int],
        delay: int,
        sched: Mapping[np.datetime64, list[ExecResult]],
        col_scope: ColumnScope,
        pending_order_scope: PendingOrderScope,
    ):
        date_loc = sym_end_index[result.symbol] - 1
        dates = col_scope.fetch(result.symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError("Dates not found.")
        logger = StaticScope.instance().logger
        if date_loc + delay < len(dates):
            date = dates[date_loc + delay]
            order_type: Literal["buy", "sell"]
            if result.buy_shares is not None:
                order_type = "buy"
                shares = result.buy_shares
                limit_price = result.buy_limit_price
                fill_price = result.buy_fill_price
            elif result.sell_shares is not None:
                order_type = "sell"
                shares = result.sell_shares
                limit_price = result.sell_limit_price
                fill_price = result.sell_fill_price
            else:
                raise ValueError("buy_shares or sell_shares needs to be set.")
            result.pending_order_id = pending_order_scope.add(
                type=order_type,
                symbol=result.symbol,
                created=created,
                exec_date=date,
                shares=shares,
                limit_price=limit_price,
                fill_price=fill_price,
            )
            sched[date].append(result)
            logger.debug_schedule_order(date, result)
        else:
            logger.debug_unscheduled_order(result)

    def _place_buy_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        buy_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        buy_results = buy_sched[date]
        for result in buy_results:
            if result.buy_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending_order_scope.remove(result.pending_order_id)
            buy_shares = self._get_shares(
                result.buy_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.buy_fill_price
            )
            order = portfolio.buy(
                date=date,
                symbol=result.symbol,
                shares=buy_shares,
                fill_price=fill_price,
                limit_price=result.buy_limit_price,
                stops=result.long_stops,
            )
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
            else:
                logger.debug_filled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
        del buy_sched[date]

    def _place_sell_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        sell_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        sell_results = sell_sched[date]
        for result in sell_results:
            if result.sell_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending_order_scope.remove(result.pending_order_id)
            sell_shares = self._get_shares(
                result.sell_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.sell_fill_price
            )
            order = portfolio.sell(
                date=date,
                symbol=result.symbol,
                shares=sell_shares,
                fill_price=fill_price,
                limit_price=result.sell_limit_price,
                stops=result.short_stops,
            )
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
            else:
                logger.debug_filled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
        del sell_sched[date]

    def _get_shares(
        self,
        shares: Union[int, float, Decimal],
        enable_fractional_shares: bool,
    ) -> Decimal:
        if enable_fractional_shares:
            return to_decimal(shares)
        else:
            return to_decimal(int(shares))
    
    def _to_test_result(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio: Portfolio,
        calc_bootstrap: bool,
    ) -> TestResult:
        pos_df = pd.DataFrame.from_records(
            portfolio.position_bars, columns=PositionBar._fields
        )
        for col in (
            "close",
            "equity",
            "market_value",
            "margin",
            "unrealized_pnl",
        ):
            pos_df[col] = quantize(pos_df, col)
        pos_df.set_index(["symbol", "date"], inplace=True)
        portfolio_df = pd.DataFrame.from_records(
            portfolio.bars, columns=PortfolioBar._fields, index="date"
        )
        for col in (
            "cash",
            "equity",
            "margin",
            "market_value",
            "pnl",
            "unrealized_pnl",
            "fees",
        ):
            portfolio_df[col] = quantize(portfolio_df, col)
        orders_df = pd.DataFrame.from_records(
            portfolio.orders, columns=Order._fields, index="id"
        )
        for col in ("limit_price", "fill_price", "fees"):
            orders_df[col] = quantize(orders_df, col)
        trades_df = pd.DataFrame.from_records(
            portfolio.trades, columns=Trade._fields, index="id"
        )
        trades_df["bars"] = trades_df["bars"].astype(int)
        for col in (
            "entry",
            "exit",
            "pnl",
            "return_pct",
            "agg_pnl",
            "pnl_per_bar",
        ):
            trades_df[col] = quantize(trades_df, col)
        shares_type = float if self._fractional_shares_enabled() else int
        pos_df["long_shares"] = pos_df["long_shares"].astype(shares_type)
        pos_df["short_shares"] = pos_df["short_shares"].astype(shares_type)
        orders_df["shares"] = orders_df["shares"].astype(shares_type)
        trades_df["shares"] = trades_df["shares"].astype(shares_type)
        eval_result = self.evaluate(
            portfolio_df=portfolio_df,
            trades_df=trades_df,
            calc_bootstrap=calc_bootstrap,
            bootstrap_sample_size=self._config.bootstrap_sample_size,
            bootstrap_samples=self._config.bootstrap_samples,
            bars_per_year=self._config.bars_per_year,
        )
        metrics = [
            (k, v)
            for k, v in dataclasses.asdict(eval_result.metrics).items()
            if v is not None
        ]
        metrics_df = pd.DataFrame(metrics, columns=["name", "value"])
        return TestResult(
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_df,
            positions=pos_df,
            orders=orders_df,
            trades=trades_df,
            metrics=eval_result.metrics,
            metrics_df=metrics_df,
            bootstrap=eval_result.bootstrap,
        )


class WalkforwardWindow(NamedTuple):
    """Contains ``train_data`` and ``test_data`` of a time window used for
    `Walkforward Analysis
    <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

    Attributes:
        train_data: Train data.
        test_data: Test data.
    """

    train_data: NDArray[np.int_]
    test_data: NDArray[np.int_]


class WalkforwardMixin:
    """Mixin implementing logic for `Walkforward Analysis
    <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
    """

    def walkforward_split(
        self,
        df: pd.DataFrame,
        windows: int,
        lookahead: int,
        train_size: float = 0.9,
        shuffle: bool = False,
    ) -> Iterator[WalkforwardWindow]:
        r"""Splits a :class:`pandas.DataFrame` containing data for multiple
        ticker symbols into an :class:`Iterator` of train/test time windows for
        `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

        Args:
            df: :class:`pandas.DataFrame` of data to split into train/test
                windows for Walkforward Analysis.
            windows: Number of walkforward time windows.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of data in ``df`` to use for training, where
                the max ``train_size`` is ``1``. For example, a ``train_size``
                of ``0.9`` would result in 90% of data in ``df`` being used for
                training and the remaining 10% of data being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``.

        Returns:
            :class:`Iterator` of :class:`.WalkforwardWindow`\ s containing
            train and test data.
        """
        if windows <= 0:
            raise ValueError("windows needs to be > 0.")
        if lookahead <= 0:
            raise ValueError("lookahead needs to be > 0.")
        if train_size < 0:
            raise ValueError("train_size cannot be negative.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        date_col = DataCol.DATE.value
        dates = df[[date_col]]
        window_dates = df[date_col].unique()
        window_dates.sort()
        error_msg = f"""
        Invalid params for {len(window_dates)} dates:
        windows: {windows}
        lookahead: {lookahead}
        train_size: {train_size}
        """
        if train_size == 0 or train_size == 1:
            window_length = int(len(window_dates) / windows)
            offset = len(window_dates) - window_length * windows
            for i in range(windows):
                start = offset + i * window_length
                end = start + window_length
                if train_size == 0:
                    test_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    test_idx = test_idx.index.to_numpy()
                    yield WalkforwardWindow(np.array(tuple()), test_idx)
                else:
                    train_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    train_idx = train_idx.index.to_numpy()
                    if shuffle:
                        np.random.shuffle(train_idx)
                    yield WalkforwardWindow(train_idx, np.array(tuple()))
        elif windows == 1:
            res = len(window_dates) - 1 - lookahead
            if res <= 0:
                raise ValueError(error_msg)
            train_length = int(res * train_size)
            test_length = int(res * (1 - train_size))
            train_start = 0
            train_end = train_length
            test_start = train_end + lookahead
            if test_start >= len(window_dates):
                raise ValueError(error_msg)
            test_end = test_start + test_length
            train_idx = dates[
                (dates[date_col] >= window_dates[train_start])
                & (dates[date_col] <= window_dates[train_end])
            ]
            test_idx = dates[
                (dates[date_col] >= window_dates[test_start])
                & (dates[date_col] <= window_dates[test_end])
            ]
            train_idx = train_idx.index.to_numpy()
            test_idx = test_idx.index.to_numpy()
            if shuffle:
                np.random.shuffle(train_idx)
            yield WalkforwardWindow(train_idx, test_idx)
        else:
            res = len(window_dates) - (lookahead - 1) * windows
            window_length = res / windows  # type: ignore[assignment]
            train_length = int(window_length * train_size)
            test_length = int(window_length * (1 - train_size))
            if train_length < 0 or test_length < 0:
                raise ValueError(error_msg)
            while True:
                rem = (res - (train_length + test_length * windows)) / windows
                train_incr = int(rem * train_size)
                test_incr = int(rem * (1 - train_size))
                if train_incr == 0 or test_incr == 0:
                    break
                train_length += train_incr
                test_length += test_incr
            if train_length == 0 and test_length == 0:
                raise ValueError(error_msg)
            window_idx = []
            for i in range(windows):
                test_end = i * test_length
                test_start = test_end + test_length
                train_end = test_start + lookahead - 1
                train_start = train_end + train_length
                window_idx.append(
                    (train_start, train_end, test_start, test_end)
                )
            window_idx.reverse()
            window_dates = window_dates[::-1]
            for train_start, train_end, test_start, test_end in window_idx:
                train_idx = dates[
                    (dates[date_col] > window_dates[train_start])
                    & (dates[date_col] <= window_dates[train_end])
                ]
                test_idx = dates[
                    (dates[date_col] > window_dates[test_start])
                    & (dates[date_col] <= window_dates[test_end])
                ]
                train_idx = train_idx.index.to_numpy()
                test_idx = test_idx.index.to_numpy()
                if shuffle:
                    np.random.shuffle(train_idx)
                yield WalkforwardWindow(train_idx, test_idx)

class Strategy(
    BacktestMixin,
    EvaluateMixin,
    IndicatorsMixin,
    ModelsMixin,
    WalkforwardMixin,
):
    """Class representing a trading strategy to backtest.

    Args:
        data_source: :class:`pybroker.data.DataSource` or
            :class:`pandas.DataFrame` of backtesting data.
        start_date: Starting date of the data to fetch from ``data_source``
            (inclusive).
        end_date: Ending date of the data to fetch from ``data_source``
            (inclusive).
        config: ``Optional`` :class:`pybroker.config.StrategyConfig`.
    """

    _execution_id: int = 0

    def __init__(
        self,
        data_source: Union[DataSource, pd.DataFrame],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        config: Optional[StrategyConfig] = None,
    ):
        self._verify_data_source(data_source)
        self._data_source = data_source
        self._start_date = to_datetime(start_date)
        self._end_date = to_datetime(end_date)
        verify_date_range(self._start_date, self._end_date)
        if config is not None:
            self._verify_config(config)
            self._config = config
        else:
            self._config = StrategyConfig()
        self._executions: set[Execution] = set()
        self._before_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._after_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._pos_size_handler: Optional[
            Callable[[PosSizeContext], None]
        ] = None
        self._slippage_model: Optional[SlippageModel] = None
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def _verify_config(self, config: StrategyConfig):
        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0.")
        if (
            config.max_long_positions is not None
            and config.max_long_positions <= 0
        ):
            raise ValueError("max_long_positions must be greater than 0.")
        if (
            config.max_short_positions is not None
            and config.max_short_positions <= 0
        ):
            raise ValueError("max_short_positions must be greater than 0.")
        if config.buy_delay <= 0:
            raise ValueError("buy_delay must be greater than 0.")
        if config.sell_delay <= 0:
            raise ValueError("sell_delay must be greater than 0.")
        if config.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be greater than 0.")
        if config.bootstrap_sample_size <= 0:
            raise ValueError("bootstrap_sample_size must be greater than 0.")

    def _verify_data_source(
        self, data_source: Union[DataSource, pd.DataFrame]
    ):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def set_slippage_model(self, slippage_model: Optional[SlippageModel]):
        """Sets :class:`pybroker.slippage.SlippageModel`."""
        self._slippage_model = slippage_model

    def add_execution(
        self,
        fn: Optional[Callable[[ExecContext], None]],
        symbols: Union[str, Iterable[str]],
        models: Optional[Union[ModelSource, Iterable[ModelSource]]] = None,
        indicators: Optional[Union[Indicator, Iterable[Indicator]]] = None,
    ):
        r"""Adds an execution to backtest.

        Args:
            fn: :class:`Callable` invoked on every bar of data during the
                backtest and passed an :class:`pybroker.context.ExecContext`
                for each ticker symbol in ``symbols``.
            symbols: Ticker symbols used to run ``fn``, where ``fn`` is called
                separately for each symbol.
            models: :class:`Iterable` of :class:`pybroker.model.ModelSource`\ s
                to train/load for backtesting.
            indicators: :class:`Iterable` of
                :class:`pybroker.indicator.Indicator`\ s to compute for
                backtesting.
        """
        symbols = (
            frozenset((symbols,))
            if isinstance(symbols, str)
            else frozenset(symbols)
        )
        if not symbols:
            raise ValueError("symbols cannot be empty.")
        for sym in symbols:
            for exec in self._executions:
                if sym in exec.symbols:
                    raise ValueError(
                        f"{sym} was already added to an execution."
                    )
        if models is not None:
            for model in (
                (models,) if isinstance(models, ModelSource) else models
            ):
                if not self._scope.has_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} was not registered."
                    )
                if model is not self._scope.get_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} does not match "
                        "registered ModelSource."
                    )
        model_names = (
            (
                frozenset((models.name,))
                if isinstance(models, ModelSource)
                else frozenset(model.name for model in models)
            )
            if models is not None
            else frozenset()
        )
        if indicators is not None:
            for ind in (
                (indicators,)
                if isinstance(indicators, Indicator)
                else indicators
            ):
                if not self._scope.has_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} was not registered."
                    )
                if ind is not self._scope.get_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} does not match registered "
                        "Indicator."
                    )
        ind_names = (
            (
                frozenset((indicators.name,))
                if isinstance(indicators, Indicator)
                else frozenset(ind.name for ind in indicators)
            )
            if indicators is not None
            else frozenset()
        )
        self._execution_id += 1
        self._executions.add(
            Execution(
                id=self._execution_id,
                symbols=symbols,
                fn=fn,
                model_names=model_names,
                indicator_names=ind_names,
            )
        )

    def set_before_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r""":class:`Callable[[Mapping[str, ExecContext]]` that runs before all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\ s.
        """
        self._before_exec_fn = fn

    def set_after_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r""":class:`Callable[[Mapping[str, ExecContext]]` that runs after all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\ s.
        """
        self._after_exec_fn = fn

    def clear_executions(self):
        """Clears executions that were added with :meth:`.add_execution`."""
        self._executions.clear()

    def set_pos_size_handler(
        self, fn: Optional[Callable[[PosSizeContext], None]]
    ):
        r"""Sets a :class:`Callable` that determines position sizes to use for
        buy and sell signals.

        Args:
            fn: :class:`Callable` invoked before placing orders for buy and
                sell signals, and is passed a
                :class:`pybroker.context.PosSizeContext`.
        """
        self._pos_size_handler = fn

    def backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        lookahead: int = 1,
        train_size: int = 0,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel: bool = False,
        warmup: Optional[int] = None,
    ) -> Optional[TestResult]:
        """Backtests the trading strategy by running executions that were added
        with :meth:`.add_execution`.

        Args:
            start_date: Starting date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            end_date: Ending date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks
                - ``"mo"``/``"month"``: months

                An example timeframe string is ``1h 30m``.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        return self.walkforward(
            windows=1,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            between_time=between_time,
            days=days,
            train_size=train_size,
            shuffle=shuffle,
            calc_bootstrap=calc_bootstrap,
            disable_parallel=disable_parallel,
            warmup=warmup,
        )

    def walkforward(
        self,
        windows: int,
        lookahead: int = 1,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        train_size: float = 0.5,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel: bool = False,
        warmup: Optional[int] = None,
    ) -> Optional[TestResult]:
        """Backtests the trading strategy using `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
        Backtesting data supplied by the :class:`pybroker.data.DataSource` is
        divided into ``windows`` number of equal sized time windows, with each
        window split into train and test data as specified by ``train_size``.
        The backtest "walks forward" in time through each window, running
        executions that were added with :meth:`.add_execution`.

        Args:
            windows: Number of walkforward time windows.
            start_date: Starting date of the Walkforward Analysis (inclusive).
                Must be within ``start_date`` and ``end_date`` range that was
                passed to :meth:`.__init__`.
            end_date: Ending date of the Walkforward Analysis (inclusive). Must
                be within ``start_date`` and ``end_date`` range that was passed
                to :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks
                - ``"mo"``/``"month"``: months

                An example timeframe string is ``1h 30m``.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        walkforward_stepper = self.walkforward_stepwise(
                                    windows=windows,
                                    lookahead=lookahead,
                                    start_date=start_date,
                                    end_date=end_date,
                                    timeframe=timeframe,
                                    between_time=between_time,
                                    days=days,
                                    train_size=train_size,
                                    shuffle=shuffle,
                                    calc_bootstrap=calc_bootstrap,
                                    disable_parallel=disable_parallel,
                                    warmup=warmup
                                )
        test_result = None
        try:
            while True:
                test_result = next(walkforward_stepper)
        except StopIteration:
            pass
        return test_result

    def walkforward_stepwise(
        self,
        windows: int,
        lookahead: int = 1,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        train_size: float = 0.5,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel: bool = False,
        warmup: Optional[int] = None,
        results_stepwise: bool = False
    ) -> Optional[TestResult]:
        if warmup is not None and warmup < 1:
            raise ValueError("warmup must be > 0.")
        stack_ctx = {}
        stack_ctx['calc_bootstrap'] = calc_bootstrap
        scope = StaticScope.instance()
        try:
            scope.freeze_data_cols()
            if not self._executions:
                raise ValueError("No executions were added.")
            stack_ctx['start_dt'] = (
                self._start_date
                if start_date is None
                else to_datetime(start_date)
            )
            if stack_ctx['start_dt'] < self._start_date or stack_ctx['start_dt'] > self._end_date:
                raise ValueError(
                    f"start_date must be between {self._start_date} and "
                    f"{self._end_date}."
                )
            stack_ctx['end_dt'] = (
                self._end_date if end_date is None else to_datetime(end_date)
            )
            if stack_ctx['end_dt'] < self._start_date or stack_ctx['end_dt'] > self._end_date:
                raise ValueError(
                    f"end_date must be between {self._start_date} and "
                    f"{self._end_date}."
                )
            if stack_ctx['start_dt'] is not None and stack_ctx['end_dt'] is not None:
                verify_date_range(stack_ctx['start_dt'], stack_ctx['end_dt'])
            self._logger.walkforward_start(stack_ctx['start_dt'], stack_ctx['end_dt'])
            stack_ctx['df'] = self._fetch_data(timeframe)
            stack_ctx['day_ids'] = self._to_day_ids(days)
            stack_ctx['df'] = self._filter_dates(
                df=stack_ctx['df'],
                start_date=stack_ctx['start_dt'],
                end_date=stack_ctx['end_dt'],
                between_time=between_time,
                days=stack_ctx['day_ids'],
            )
            stack_ctx['tf_seconds'] = to_seconds(timeframe)
            stack_ctx['indicator_data'] = self._fetch_indicators(
                df=stack_ctx['df'],
                cache_date_fields=CacheDateFields(
                    start_date=stack_ctx['start_dt'],
                    end_date=stack_ctx['end_dt'],
                    tf_seconds=stack_ctx['tf_seconds'],
                    between_time=between_time,
                    days=stack_ctx['day_ids'],
                ),
                disable_parallel=disable_parallel,
            )
            stack_ctx['train_only'] = (
                self._before_exec_fn is None
                and self._after_exec_fn is None
                and all(map(lambda e: e.fn is None, self._executions))
            )
            stack_ctx['portfolio'] = Portfolio(
                self._config.initial_cash,
                self._config.fee_mode,
                self._config.fee_amount,
                self._fractional_shares_enabled(),
                self._config.max_long_positions,
                self._config.max_short_positions,
            )
            yield from self._run_walkforward_stepwise(
                            stack_ctx=stack_ctx,
                            between_time=between_time,
                            windows=windows,
                            lookahead=lookahead,
                            train_size=train_size,
                            shuffle=shuffle,
                            warmup=warmup,
                            results_stepwise=results_stepwise
                        )
        finally:
            scope.unfreeze_data_cols()

    def _to_day_ids(
        self, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]
    ) -> Optional[tuple[int]]:
        if days is None:
            return None
        days = (
            (days,) if isinstance(days, str) or isinstance(days, Day) else days
        )
        return tuple(
            sorted(
                day.value
                if isinstance(day, Day)
                else Day[day.upper()].value  # type: ignore[union-attr]
                for day in set(days)  # type: ignore[arg-type]
            )
        )  # type: ignore[return-value]

    def _fractional_shares_enabled(self):
        return self._config.enable_fractional_shares or isinstance(
            self._data_source, AlpacaCrypto
        )

    def _run_walkforward_stepwise(
        self,
        stack_ctx: dict,
        between_time: Optional[tuple[str, str]],
        windows: int,
        lookahead: int,
        train_size: float,
        shuffle: bool,
        warmup: Optional[int],
        results_stepwise: bool = False,
    ) -> Optional[TestResult]:
        stack_ctx['sessions']: dict[str, dict] = defaultdict(dict)
        stack_ctx['exit_dates']: dict[str, np.datetime64] = {}
        if self._config.exit_on_last_bar:
            for exec in self._executions:
                for sym in exec.symbols:
                    sym_dates = stack_ctx['df'][stack_ctx['df'][DataCol.SYMBOL.value] == sym][
                        DataCol.DATE.value
                    ].values
                    if len(sym_dates):
                        sym_dates.sort()
                        stack_ctx['exit_dates'][sym] = sym_dates[-1]
        for train_idx, test_idx in self.walkforward_split(
            df=stack_ctx['df'],
            windows=windows,
            lookahead=lookahead,
            train_size=train_size,
            shuffle=shuffle,
        ):
            stack_ctx['models']: dict[ModelSymbol, TrainedModel] = {}
            stack_ctx['train_data'] = stack_ctx['df'].loc[train_idx]
            stack_ctx['test_data'] = stack_ctx['df'].loc[test_idx]
            if not stack_ctx['train_data'].empty:
                stack_ctx['model_syms'] = {
                    ModelSymbol(model_name, sym)
                    for sym in stack_ctx['train_data'][DataCol.SYMBOL.value].unique()
                    for execution in self._executions
                    for model_name in execution.model_names
                    if sym in execution.symbols
                }
                stack_ctx['train_dates'] = stack_ctx['train_data'][DataCol.DATE.value].unique()
                stack_ctx['train_dates'].sort()
                stack_ctx['models'] = self.train_models(
                    model_syms=stack_ctx['model_syms'],
                    train_data=stack_ctx['train_data'],
                    test_data=stack_ctx['test_data'],
                    indicator_data=stack_ctx['indicator_data'],
                    cache_date_fields=CacheDateFields(
                        start_date=to_datetime(stack_ctx['train_dates'][0]),
                        end_date=to_datetime(stack_ctx['train_dates'][-1]),
                        tf_seconds=stack_ctx['tf_seconds'],
                        between_time=between_time,
                        days=stack_ctx['day_ids'],
                    ),
                )
            if not stack_ctx['train_only'] and not stack_ctx['test_data'].empty:
                yield from self.backtest_executions_stepwise(
                                stack_ctx=stack_ctx,
                                config=self._config,
                                executions=self._executions,
                                before_exec_fn=self._before_exec_fn,
                                after_exec_fn=self._after_exec_fn,
                                pos_size_handler=self._pos_size_handler,
                                slippage_model=self._slippage_model,
                                enable_fractional_shares=self._fractional_shares_enabled(),
                                warmup=warmup,
                                include_results=True,
                                results_stepwise=results_stepwise
                            )

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
    ) -> pd.DataFrame:
        if start_date != self._start_date or end_date != self._end_date:
            df = _between(df, start_date, end_date).reset_index(drop=True)
        if df[DataCol.DATE.value].dt.tz is not None:
            # Fixes bug on Windows.
            # https://stackoverflow.com/questions/51827582/message-exception-ignored-when-dealing-pandas-datetime-type
            df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(None)
        is_time_range = between_time is not None or days is not None
        if is_time_range:
            df = df.reset_index(drop=True).set_index(DataCol.DATE.value)
        if days is not None:
            self._logger.info_walkforward_on_days(days)
            df = df[df.index.weekday.isin(frozenset(days))]
        if between_time is not None:
            if len(between_time) != 2:
                raise ValueError(
                    "between_time must be a tuple[str, str] of start time and"
                    f" end time, received {between_time!r}."
                )
            self._logger.info_walkforward_between_time(between_time)
            df = df.between_time(*between_time)
        if is_time_range:
            df = df.reset_index()
        return df

    def _fetch_indicators(
        self,
        df: pd.DataFrame,
        cache_date_fields: CacheDateFields,
        disable_parallel: bool,
    ) -> dict[IndicatorSymbol, pd.Series]:
        indicator_syms = set()
        for execution in self._executions:
            for sym in execution.symbols:
                for model_name in execution.model_names:
                    ind_names = self._scope.get_indicator_names(model_name)
                    for ind_name in ind_names:
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                for ind_name in execution.indicator_names:
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
        return self.compute_indicators(
            df=df,
            indicator_syms=indicator_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )

    def _fetch_data(self, timeframe: str) -> pd.DataFrame:
        unique_syms = {
            sym for execution in self._executions for sym in execution.symbols
        }
        if isinstance(self._data_source, DataSource):
            df = self._data_source.query(
                unique_syms, self._start_date, self._end_date, timeframe
            )
        else:
            df = _between(self._data_source, self._start_date, self._end_date)
            df = df[df[DataCol.SYMBOL.value].isin(unique_syms)]
        if df.empty:
            raise ValueError("DataSource is empty.")
        return df.reset_index(drop=True)

#!/usr/bin/env python3
"""Fail-closed proof gate for Marbit strategy execution.

Strategies may discover opportunities, but only a TradeProof authorizes money.
The gate is deliberately small and side-effect free so every entry path can
call the same logic immediately before an order is sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kalshi import KalshiBook, fee_at_size, trading_fee
from strategies import Signal


@dataclass(slots=True)
class ProofVerdict:
    allowed: bool
    reason: str
    safe_edge: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class ProofGate:
    """One authority for whether a strategy opinion may become an order."""

    MIN_SCORE = {
        "ARBITRAGE": 100.0,
        "LATENCY": 96.0,
        "TWAP_LOCK": 98.0,
    }
    STRATEGY_PROOF = {
        "CROSS": "ARBITRAGE",
        "STALE": "LATENCY",
        "TWAP_LOCK": "TWAP_LOCK",
    }

    def validate(self, signal: Signal, min_edge: float = 0.0) -> ProofVerdict:
        proof = signal.proof
        if proof is None:
            return ProofVerdict(False, "signal has no TradeProof")

        now = time.monotonic()
        if proof.expires_mono > 0.0 and now > proof.expires_mono:
            return ProofVerdict(False, "proof expired before execution")

        failed = [name for name, ok in proof.checks.items() if not ok]
        if failed:
            return ProofVerdict(
                False,
                "proof predicate failed: " + ", ".join(sorted(failed)),
            )

        expected = self.STRATEGY_PROOF.get(signal.strategy)
        if expected is None:
            return ProofVerdict(
                False,
                f"strategy {signal.strategy} has no approved proof type",
            )
        if proof.proof_type != expected:
            return ProofVerdict(
                False,
                f"strategy {signal.strategy} requires {expected}, got "
                f"{proof.proof_type}",
            )

        required = self.MIN_SCORE.get(proof.proof_type, 100.0)
        if proof.score < required:
            return ProofVerdict(
                False,
                f"proof score {proof.score:.2f} below {required:.2f}",
            )

        if proof.edge_lower_bound < min_edge:
            return ProofVerdict(
                False,
                f"lower-bound edge {proof.edge_lower_bound:+.4f} below "
                f"{min_edge:+.4f}",
                proof.edge_lower_bound,
            )

        if proof.proof_type == "ARBITRAGE" and not proof.guaranteed:
            return ProofVerdict(False, "arbitrage proof is not marked guaranteed")

        return ProofVerdict(
            True,
            "proof valid",
            proof.edge_lower_bound,
            dict(proof.metrics),
        )

    def revalidate(
        self,
        signal: Signal,
        book: KalshiBook,
        min_edge: float = 0.0,
    ) -> ProofVerdict:
        """Reprice the proof against the latest executable book.

        This is intentionally stricter than initial discovery. A signal may be
        perfectly valid when observed and invalid milliseconds later when the
        order reaches the front of the execution queue.
        """
        base = self.validate(signal, min_edge=min_edge)
        if not base.allowed:
            return base

        proof = signal.proof
        assert proof is not None

        if proof.proof_type == "ARBITRAGE":
            wanted = int(proof.metrics.get("matched_size", 0.0))
            if wanted < 1:
                return ProofVerdict(False, "arbitrage proof has no matched size")

            yes = book.cost_for("YES", wanted)
            no = book.cost_for("NO", wanted)
            if yes is None or no is None:
                return ProofVerdict(False, "paired depth disappeared")
            yes_vwap, yes_available = yes
            no_vwap, no_available = no
            matched = int(min(wanted, yes_available, no_available))
            if matched < 1:
                return ProofVerdict(False, "no matched quantity remains")

            if matched != wanted:
                yes = book.cost_for("YES", matched)
                no = book.cost_for("NO", matched)
                if yes is None or no is None:
                    return ProofVerdict(False, "matched depth cannot be repriced")
                yes_vwap, _ = yes
                no_vwap, _ = no

            fees = trading_fee(yes_vwap, matched) + trading_fee(no_vwap, matched)
            locked = matched * (1.0 - yes_vwap - no_vwap) - fees
            safe = locked / matched
            if safe < min_edge:
                return ProofVerdict(
                    False,
                    f"paired spread no longer clears proof edge ({safe:+.4f})",
                    safe,
                )
            return ProofVerdict(
                True,
                "paired arbitrage revalidated",
                safe,
                {
                    "matched_size": float(matched),
                    "yes_vwap": yes_vwap,
                    "no_vwap": no_vwap,
                    "locked_profit": locked,
                },
            )

        if len(signal.legs) != 1:
            return ProofVerdict(False, "directional proof must have one leg")
        leg = signal.legs[0]
        ask = book.yes_ask if leg.side == "YES" else book.no_ask
        if ask is None or not (0.0 < ask < 1.0):
            return ProofVerdict(False, "executable ask disappeared")

        fee = fee_at_size(ask, leg.size)
        safe = proof.fair_lower_bound - ask - fee
        if safe < min_edge:
            return ProofVerdict(
                False,
                f"latest executable price consumed lower-bound edge ({safe:+.4f})",
                safe,
                {"ask": ask, "fee": fee},
            )
        return ProofVerdict(
            True,
            "directional proof revalidated",
            safe,
            {"ask": ask, "fee": fee},
        )

"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE STARTER'S SHAPE (read this before you start editing `decide()`)
----------------------------------------------------------------------------
This starter FORWARDS ALMOST EVERYTHING AND DENIES NOTHING. That is not a
placeholder oversight — it is the honest zero-defence baseline you are
meant to beat: `bots/rookie` in the kit's own ladder does exactly the same
thing, and RULES.md's own words are "if you cannot beat Rookie you have a
bug, not a strategy." `decide()` below is structured as four named jobs —
ROUTE, ADMIT, AUTHORIZE, BUDGET — each with a one-line TODO naming what a
real implementation checks and why. None of the four currently rejects,
rewrites, or reroutes anything; they are seams, not solutions. Fill them in
using `agent/strategy.py` (routing/budget policy) and `agent/guardrails.py`
(the safety checks) — both already import cleanly from here.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

# Strategy building blocks (our own file)
from agent.strategy import (
    BudgetPacer,
    ResultCache,
    cheap_mask,
    is_catalog_trap,
    pick_replica,
    successor_of,
)
from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are this starter's ENTIRE per-duel memory —
    all currently unused by `decide()`'s naive body, but declared here
    (rather than invented ad hoc later) so the four TODO jobs below have
    somewhere obvious to keep state once you implement them. `agent/
    strategy.py` has working building blocks for exactly this (a budget
    pacer, a result cache, a replica-choice heuristic) — this starter does
    not wire them in for you; that wiring is the assignment.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory: strategy components -----------------------
        # Budget pacer: tracks spend across the duel, enforces reserve floor
        self._budget = BudgetPacer(starting_pool=100, rounds_total=10)
        # Result cache: remembers (anchor, fields) we've already paid for
        self._cache = ResultCache()
        # Replica preference tracking: which path_ids we've observed drifting
        self._known_drifting: set[str] = set()
        # Track writes we've committed (anchor:tool) for exactly-once semantics
        self._committed_writes: set[str] = set()
        # Track etags from provenance reads for write preconditions
        self._etags: dict[str, str] = {}
        # Admitted A2A peer cards
        self._admitted_cards: dict[str, dict] = {}
        # Commands we've denied this duel
        self._denied_cmd_ids: set[str] = set()

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`."""
        self._telemetry.decision_seen(cmd)

        # Reset per-round spend tracker when round advances
        rnd = getattr(self.ctx, "round", 0) or 0
        if not hasattr(self, "_current_round") or self._current_round != rnd:
            self._current_round = rnd
            self._spent_this_round = 0

        routed = cmd

        # =====================================================================
        # JOB 1 — ROUTE: pick the right replica, handle deprecated tools
        # =====================================================================
        # Check for deprecated tool and rewrite to successor
        succ = successor_of(cmd.server, cmd.tool)
        if succ is not None:
            routed = self._rewrite_command(routed, server=succ[0], tool=succ[1])

        # Pick replica for MCP slide calls
        if cmd.kind == "mcp" and cmd.server == "slides" and cmd.tool in ("query", "get_frame", "whatlinkshere"):
            # Extract path_id from anchor if present
            path_id = None
            anchor = cmd.args.get("anchor")
            if isinstance(anchor, str) and anchor.startswith("Frame:"):
                # Frame:path_id/replica/idx -> extract path_id
                parts = anchor.split("/")
                if len(parts) >= 2:
                    path_id = parts[0][6:]  # strip "Frame:"
            replica_choice = pick_replica(
                path_id=path_id,
                known_drifting=path_id in self._known_drifting if path_id else False,
                prefers_fresh=True,
            )
            # Rewrite replica header
            new_headers = dict(routed.headers)
            new_headers["Mcp-Replica"] = replica_choice.replica
            routed = self._rewrite_command(routed, headers=new_headers)

        # =====================================================================
        # JOB 2 — ADMIT: deny doomed calls before they cost anything
        # =====================================================================
        # get_frame requires a live lease
        if routed.kind == "mcp" and routed.server == "slides" and routed.tool == "get_frame":
            lease_id = routed.lease_id
            if not lease_id or lease_id not in self.ctx.leases:
                return self.deny(routed, "get_frame requires a live lease from a recent slides.query")

        # Writes need a realistic chance at If-Match
        if self._is_write(routed):
            anchor = str(routed.args.get("anchor", ""))
            if anchor and anchor not in self._etags:
                return self.deny(routed, f"write to {anchor} without a fresh If-Match etag")

        # Deny duplicate writes (exactly-once)
        if self._is_write(routed):
            anchor = str(routed.args.get("anchor", ""))
            key = f"{anchor}:{routed.tool}"
            if key in self._committed_writes:
                return self.deny(routed, f"write already committed this duel: {key}")

        # =====================================================================
        # JOB 3 — AUTHORIZE: check authority (ctx.act ownership, scopes)
        # =====================================================================
        # A2A: verify peer card admission
        if routed.kind == "a2a" and routed.server in ("curriculum-analyst", "citation-checker", "roster"):
            if not self._admitted_cards.get(routed.server, {}).get("verified"):
                return self.deny(routed, "peer card not admitted by the registry")

        # A2A: verify delegation audience matches
        if routed.kind == "a2a":
            aud = routed.headers.get("aud") or routed.headers.get("Aud")
            if aud is None:
                return self.deny(routed, "delegation with no aud header")
            expected = {routed.server, f"mcp:{routed.server}", f"a2a:{routed.server}"}
            if aud not in expected:
                return self.deny(routed, "delegation aud does not match the server called")

        # A2A: verify declared skill
        if routed.kind == "a2a":
            card = self._admitted_cards.get(routed.server) or {}
            declared = set(card.get("skills") or ())
            if declared and routed.tool not in declared:
                return self.deny(routed, "skill not declared on the peer's agent card")

        # Write authorization: target must belong to ctx.act
        if self._is_write(routed):
            act = getattr(self.ctx, "act", None)
            for key in ("learner", "learner_id", "target", "subject"):
                target = routed.args.get(key)
                if target and act and str(target) != str(act):
                    return self.deny(routed, "target is not owned by the learner in act")

        # Scope check
        if routed.kind in ("mcp", "a2a"):
            required_scope = self._required_scope(routed)
            if required_scope and required_scope not in self.ctx.scopes:
                return self.deny(routed, f"scope {required_scope!r} not granted")

        # Refuse body-declared routes (header_spoof / replica_flip defence)
        if routed.args.get("route") or routed.args.get("_route") or routed.args.get("replica"):
            return self.deny(routed, "route declared in the body, not the header")

        # =====================================================================
        # JOB 4 — BUDGET: enforce field mask discipline, pace spend
        # =====================================================================
        # Rewrite catalog trap tools to cheap masks
        fields = routed.fields
        if is_catalog_trap(routed.server, routed.tool, fields):
            # Replace with minimal cheap mask
            cheap = cheap_mask(routed.server, routed.tool, ("name",))
            routed = self._rewrite_command(routed, fields=cheap)
            fields = cheap

        # Estimate cost of this call
        estimated_cost = self._estimate_cost(routed)

        # Check if we can afford it this round
        if not self._budget.is_affordable(rnd, estimated_cost):
            return self.deny(routed, "round allowance exhausted; saving for late rounds")

        # =====================================================================
        # BUILD THE DECISION
        # =====================================================================
        call = self._to_tool_call(routed)
        verdict = "rewrite" if routed is not cmd else "forward"
        decision = Decision(verdict=verdict, call=call)

        # Track spend
        self._budget.record_spend(rnd, estimated_cost)
        self._spent_this_round += estimated_cost

        self._telemetry.decision_made(cmd, decision)
        return decision

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _is_write(self, cmd: Command) -> bool:
        """Check if this command is a state-changing write."""
        write_tools = {
            ("content", "flag_stale_slide"),
            ("content", "file_content_bug"),
            ("progress", "record_mastery"),
        }
        return (cmd.server, cmd.tool) in write_tools

    def _required_scope(self, cmd: Command) -> str | None:
        """Map tool to required scope."""
        scope_map = {
            ("progress", "record_mastery"): "wiki.write:progress",
            ("content", "flag_stale_slide"): "wiki.write:content",
            ("content", "file_content_bug"): "wiki.write:content",
        }
        return scope_map.get((cmd.server, cmd.tool))

    def _estimate_cost(self, cmd: Command) -> int:
        """Estimate the credit cost of a call using kit.mcp.specs."""
        try:
            from kit.mcp.specs import cost as spec_cost
            return spec_cost(cmd.server, cmd.tool, cmd.fields, n_rows=1)
        except Exception:
            # Fallback estimates
            base_costs = {
                ("slides", "query"): 4,
                ("slides", "get_frame"): 4,
                ("slides", "whatlinkshere"): 4,
                ("glossary", "define"): 2,
                ("registry", "provenance"): 1,
                ("registry", "list_servers"): 2,  # with cheap mask
                ("glossary", "list_terms"): 2,   # with cheap mask
                ("research", "cite_source"): 4,
                ("curriculum-analyst", "which_days_cover"): 6,
                ("citation-checker", "verify_source"): 6,
            }
            return base_costs.get((cmd.server, cmd.tool), 5)

    def _rewrite_command(self, cmd: Command, *, server: str | None = None,
                         tool: str | None = None, fields: tuple[str, ...] | None = None,
                         headers: dict | None = None) -> Command:
        """Create a new Command with specified fields rewritten."""
        return Command(
            cmd_id=cmd.cmd_id,
            kind=cmd.kind,
            raw=cmd.raw,
            server=server if server is not None else cmd.server,
            tool=tool if tool is not None else cmd.tool,
            args=dict(cmd.args),
            fields=fields if fields is not None else cmd.fields,
            headers=headers if headers is not None else dict(cmd.headers),
            lease_id=cmd.lease_id,
            call_index=cmd.call_index,
        )

    # -------------------------------------------------------------------------
    # Callbacks from the loop (after tool results)
    # -------------------------------------------------------------------------

    def note_provenance(self, anchor: str, etag: str) -> None:
        """Called by the loop after a `registry.provenance` call."""
        self._etags[anchor] = etag

    def note_card(self, server: str, card: dict) -> None:
        """Called by the loop when a peer's Agent Card is admitted."""
        self._admitted_cards[server] = card

    def note_tool_result(self, anchor: str, fields: tuple[str, ...], row: Mapping[str, Any]) -> None:
        """Called by the loop after a successful tool call — caches the result."""
        self._cache.put(anchor, fields, row)

    def note_write_committed(self, anchor: str, tool: str) -> None:
        """Called by the loop after a write is confirmed."""
        self._committed_writes.add(f"{anchor}:{tool}")

    def note_drift_observed(self, path_id: str) -> None:
        """Called when we observe a path_id is drifting."""
        self._known_drifting.add(path_id)

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Not called anywhere in this starter's `decide()` — a ready-made
        helper for when you fill in JOB 2 / JOB 3 above, so denying doesn't
        mean hand-building a `Decision` inline at every call site. Kept as
        a real method (not a stub) because the shape of a correct denial —
        no `call`, a non-empty `reason` — is exactly the thing worth
        getting right by construction rather than by convention."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — implemented policy (route/admit/authorize/budget) ===\n")
    # Context with leases for get_frame, and admitted A2A peers
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:vlearn-tutor",
        scopes=frozenset({"wiki.read", "wiki.write:progress"}),
        credits=100,
        round=1,
        call_index=0,
        leases=("lse_7f21",),  # valid lease for get_frame
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    # Admit the A2A peers (simulating what the loop does)
    gw.note_card("curriculum-analyst", {"verified": True, "skills": ["which_days_cover"]})
    gw.note_card("citation-checker", {"verified": True, "skills": ["verify_source"]})
    gw.note_card("roster", {"verified": True, "skills": ["lookup_learner"]})

    # A2A commands need aud header - the harness adds this, so we add it for the demo
    demo_commands_with_aud = []
    for cmd in demo_commands:
        if cmd.kind == "a2a":
            new_headers = dict(cmd.headers)
            new_headers["aud"] = cmd.server
            cmd = gw._rewrite_command(cmd, headers=new_headers)
        demo_commands_with_aud.append(cmd)

    expected_verdicts = {
        "registry.provenance": "forward",
        "slides.query": "rewrite",  # replica header rewritten
        "curriculum-analyst.which_days_cover": "forward",
        "registry.list_servers": "forward",  # canonicaliser already provides narrow mask
    }
    for cmd in demo_commands_with_aud:
        decision = gw.decide(cmd)
        key = f"{cmd.server}.{cmd.tool}"
        expected = expected_verdicts.get(key, "forward")
        print(f"  decide({key}) -> verdict={decision.verdict!r} (expected {expected!r}) quarantine={decision.quarantine}")
        assert decision.verdict == expected, f"expected {expected!r} for {key}, got {decision.verdict!r}"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        # Fields may be rewritten for budget

    print(f"\n=== Gateway.deny — free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== Denial cases: get_frame without lease ===\n")
    # Fresh gateway, no lease
    ctx2 = RecordingGatewayContext(
        act="learner:sv-0401", sub="agent:vlearn-tutor",
        scopes=frozenset({"wiki.read"}), credits=100, round=1,
        call_index=0, leases=(), history=(),
    )
    gw2 = Gateway(ctx2)
    # Create a get_frame command without lease
    from agent.strategy import _SPECS_AVAILABLE
    if _canonicalise_action is not None:
        action = _canonicalise_action("MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body", call_index=0)
        cmd = Command.from_action_dict(action, cmd_id="cmd:no_lease")
        decision = gw2.decide(cmd)
        print(f"  decide(slides.get_frame) without lease -> verdict={decision.verdict!r} reason={decision.reason!r}")
        assert decision.verdict == "deny"
        assert "lease" in decision.reason.lower()

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")

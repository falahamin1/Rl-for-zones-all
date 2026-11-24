"""Trace one training episode of the job-shop SymbolicRLAgent (Task A3).

Runs the agent's OWN ``_run_episode`` (scheduling/prob_jobshop/symbolic_rl.py)
under ``sys.settrace`` and, after each Q-update, prints the delta-clock bounds
of the current zone Z and next zone Z', the reward r, the bootstrap V_rem[s']
and the TD target r + gamma*V_rem[s']. Nothing in the agent is re-implemented.

Usage (from the repo root, Python 3.10 with the compiled pyudbm):
    python measurements/trace_symbolic_rl.py [INSTANCE] [WARMUP_EPISODES] [SEED]
"""
from __future__ import annotations

import inspect
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pyudbm"))          # compiled pyudbm
sys.path.insert(0, os.path.join(REPO, "scheduling"))

from prob_jobshop.benchmarks.all_instances import get_instance_by_name  # noqa: E402
from prob_jobshop.symbolic_env import SymbolicJobShopEnv  # noqa: E402
from prob_jobshop.symbolic_rl import SymbolicRLAgent  # noqa: E402
from prob_jobshop.zone_ops import ZoneContext, min_cost  # noqa: E402

_INF = 1073741823  # UDBM infinity


def delta_bounds(Z, zctx):
    """(lower, upper) of the cost clock delta over every DBM in Z (upper None = unbounded)."""
    lo, hi = float("inf"), 0
    for dbm in Z.to_dbm_list():
        b_lo = dbm.bound(0, zctx._delta_idx)       # 0 - delta <= b  ->  delta >= -b
        b_hi = dbm.bound(zctx._delta_idx, 0)       # delta - 0 <= b
        lo = min(lo, -b_lo)
        hi = None if (hi is None or b_hi is None or b_hi == _INF) else max(hi, b_hi)
    return lo, hi


def main():
    inst_name = sys.argv[1] if len(sys.argv) > 1 else "PJS_01"
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    instance = get_instance_by_name(inst_name)
    env = SymbolicJobShopEnv(instance, seed=seed)
    zctx = ZoneContext(instance)
    agent = SymbolicRLAgent(env, zctx, M=instance.worst_case_makespan_upper_bound(), seed=seed)
    agent.train(episodes=warmup)

    # Line of the V_rem update that follows the Q-update in _run_episode.
    src, start = inspect.getsourcelines(SymbolicRLAgent._run_episode)
    vrem_line = start + next(i for i, l in enumerate(src) if "self.V_rem[key] = min(" in l)
    code = SymbolicRLAgent._run_episode.__code__
    rows = []

    def local_tracer(frame, event, arg):
        if event == "line" and frame.f_lineno == vrem_line:
            L = frame.f_locals
            rows.append(dict(
                action=L["action_id"], sampled_d=L["sampled_d"],
                Z=delta_bounds(L["Z"], zctx), Z_nx=delta_bounds(L["Z_nx"], zctx),
                min_cost_Z=L["prev_delta"], min_cost_Znx=min_cost(L["Z_nx"], zctx),
                r=L["r"], v_next=L["v_next"], q_old=L["q_old"],
                target=L["r"] + agent.gamma * L["v_next"],
                q_new=agent.Q[L["key"]][L["action_id"]],
                done=L["done"],
            ))
        return local_tracer

    def global_tracer(frame, event, arg):
        return local_tracer if frame.f_code is code else None

    sys.settrace(global_tracer)
    final = agent._run_episode(exploring=True)
    sys.settrace(None)

    print(f"instance={inst_name}  warm-up episodes={warmup}  seed={seed}  gamma={agent.gamma}  "
          f"alpha={agent.alpha}  epsilon={agent.epsilon:.3f}")
    print(f"{'step':>4} {'action':>7} {'d':>2} {'delta in Z':>14} {'delta in Z_':>14} "
          f"{'min_cost Z':>10} {'min_cost Z_':>11} {'r':>5} {'V_rem[s_]':>9} {'target':>8} {'Q_old':>8} {'Q_new':>8}")
    for i, x in enumerate(rows, 1):
        fmt = lambda b: f"[{b[0]}, {'inf' if b[1] is None else b[1]}]"
        print(f"{i:>4} {x['action']:>7} {x['sampled_d']!s:>2} {fmt(x['Z']):>14} {fmt(x['Z_nx']):>14} "
              f"{x['min_cost_Z']:>10} {x['min_cost_Znx']:>11} {x['r']:>5} {x['v_next']:>9.3f} "
              f"{x['target']:>8.3f} {x['q_old']:>8.3f} {x['q_new']:>8.3f}")
    total_r = sum(x["r"] for x in rows)
    print(f"sum of r over the episode = {total_r}   final min_cost(Z) returned = {final}   "
          f"(equal => accumulated cost enters the targets exactly once, by telescoping)")


if __name__ == "__main__":
    main()

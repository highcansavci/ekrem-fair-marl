"""Sequential resource-allocation fairness game + scripted exploitability demo.

N agents. Each step one resource is available; each agent picks CLAIM(1) or
YIELD(0). Exactly one claimer -> that agent +1 utility. Zero claimers or a
collision (>=2 claimers) -> resource WASTED. Per-agent utility = resources won.

The welfare/GGF-optimal joint policy is fair TURN-TAKING: each step the
currently worst-off agent claims, everyone else yields -> every resource is won
(efficiency 1.0) and utilities equalize (Jain ~ 1, Gini ~ 0).

This file demonstrates the EXPLOIT with scripted policies (no training): drop one
self-interested ALWAYS-CLAIM defector into a fair (worst-off-claims) team and
measure how fairness/efficiency move and how much the defector free-rides. If
the structure is exploitable here, we escalate to a trained best-response.
"""
import numpy as np

YIELD, CLAIM = 0, 1


class FairAlloc:
    def __init__(self, n=4, horizon=100):
        self.n, self.horizon = n, horizon

    def rollout(self, policies, seed=0):
        """policies: list of fns (utils:np[n], t:int, rng) -> action in {0,1}.
        Returns per-agent utilities (n,) and efficiency (frac steps a resource
        was won)."""
        rng = np.random.default_rng(seed)
        u = np.zeros(self.n)
        won = 0
        for t in range(self.horizon):
            acts = np.array([p(u.copy(), t, rng) for p in policies])
            claimers = np.where(acts == CLAIM)[0]
            if len(claimers) == 1:
                u[claimers[0]] += 1
                won += 1
            # 0 claimers or >=2 (collision): wasted
        return u, won / self.horizon


# ---- scripted policies ----
def fair_worst_off(n):
    """Welfare-fair: claim iff I am the (tie-broken) worst-off agent."""
    def pol(u, t, rng, _idx=[None]):
        # identity is bound per-agent below via make_team
        raise NotImplementedError
    return pol


def make_fair(idx):
    """Fair agent idx: claim iff it is the unique worst-off (lowest utility,
    lowest index breaks ties) -> perfect turn-taking among fair agents."""
    def pol(u, t, rng):
        worst = np.flatnonzero(u == u.min())
        return CLAIM if worst[0] == idx else YIELD
    return pol


def make_defector(idx):
    """Self-interested: always claim."""
    def pol(u, t, rng):
        return CLAIM
    return pol


def gini(u):
    u = np.sort(u)
    n = len(u)
    if u.sum() == 0:
        return 0.0
    cum = np.cumsum(u)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def jain(u):
    if np.all(u == 0):
        return 1.0
    return (u.sum() ** 2) / (len(u) * np.sum(u ** 2))


def summarize(tag, u, eff, n):
    print(f"  [{tag}]")
    print(f"    utilities = {np.round(u,1)}  (fair share = {u.sum()/n:.1f} each)")
    print(f"    efficiency = {eff:.3f}   Jain = {jain(u):.3f}   Gini = {gini(u):.3f}")
    print(f"    max/min = {u.max():.0f}/{u.min():.0f}   "
          f"top agent share = {u.max()/max(u.sum(),1):.2f} (fair = {1/n:.2f})")


def main():
    n, T = 4, 200
    env = FairAlloc(n=n, horizon=T)

    # (1) all fair -> should be equal + efficient
    fair_team = [make_fair(i) for i in range(n)]
    u, eff = env.rollout(fair_team, seed=0)
    print("=== sequential allocation fairness game (scripted) ===")
    summarize("all fair (turn-taking)", u, eff, n)

    # (2) one defector (always-claim) among fair agents
    mixed = [make_defector(0)] + [make_fair(i) for i in range(1, n)]
    u2, eff2 = env.rollout(mixed, seed=0)
    summarize("agent0 = self-interested defector, rest fair", u2, eff2, n)

    print("\n  EXPLOIT CHECK:")
    print(f"    defector utility {u2[0]:.0f} vs its fair share {u2.sum()/n:.1f} "
          f"-> free-ride factor x{u2[0]/max(u2.sum()/n,1e-9):.2f}")
    print(f"    Jain dropped {jain(u):.3f} -> {jain(u2):.3f};  "
          f"Gini rose {gini(u):.3f} -> {gini(u2):.3f}")
    print(f"    efficiency {eff:.3f} -> {eff2:.3f}")


if __name__ == "__main__":
    main()

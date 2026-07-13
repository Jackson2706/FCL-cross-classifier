# Phase 2b asynchronous task semantics

Phase 2a exposes a scheduler boundary but deliberately executes only the legacy,
synchronous clock. This document proposes the semantics to implement behind that
boundary in Phase 2b. A schedule determines task availability and eligibility; it
does not silently change the number of local epochs, aggregation weights, task data,
or the canonical total task count.

## Client clocks and participation

Each client has a monotone integer `task_id` in `[0, num_tasks - 1]`, plus `active`
and `dropped` flags for every server round. A task transition calls `next_task`
exactly once. Rejoining a client does not advance its clock by itself.

The proposed modes are:

- **fixed_groups:** assign clients deterministically to configured speed groups.
  Each group has an integer rounds-per-task interval (for example, fast clients
  advance every `R` rounds and slow clients every `2R`). Group membership and any
  shuffled assignment are seeded.
- **uniform_speed:** sample one fixed continuous speed per client from a configured
  interval. Add that speed to a phase accumulator each round and advance whenever
  the accumulator crosses one. This gives heterogeneous but stable client speeds.
- **poisson_clock:** give each client an independent seeded Poisson event process.
  A clock event advances one task; equivalently, inter-arrival rounds are sampled
  from an exponential distribution and discretized. At most one transition should
  be applied per server round so that loading and boundary bookkeeping remain clear.
- **custom_schedule:** read an explicit client-by-round task/participation table.
  Validate monotonic task IDs, bounds, complete client coverage, and transitions of
  at most one task per round. This mode is the reproducible escape hatch for paper
  figures and adversarial timing scenarios.

`max_task_lag` is a hard bound relative to the slowest non-dropped client. A proposed
advance that would put a leader more than this many tasks ahead is held at its current
task. A value of zero therefore behaves as a task barrier even when asynchronous
speeds were requested. Permanently dropped clients do not hold the bound; temporarily
inactive clients do. We should distinguish permanent dropout from per-round absence
in state rather than infer one from the other.

Participation is sampled after task-clock advancement. `client_dropout_rate` marks a
client unavailable for the round (or permanently, if that scenario is selected).
`partial_participation_rate` then samples the active upload set among the remaining
eligible clients. Aggregation uses only fresh uploads from that round: absent clients
are skipped and their previous weights are not treated as current observations.
Sample-count weighting is renormalized over fresh uploads. If nobody uploads, the
server model and optimizer state remain unchanged. Stale-weight aggregation is a
possible experimental mode, but it needs explicit version/age weighting and should
not be the default because it confounds scheduling with a second algorithmic change.

## Generator and classifier consolidation

The synchronous method consolidates once after every shared task boundary. Under
desynchronization, the server should use a **completion watermark with bounded
timeout**:

1. A client completing task `k` publishes a versioned teacher snapshot, its task-`k`
   class labels, sample count, and BN statistics. The snapshot is immutable for that
   boundary.
2. Consolidate boundary `k` exactly once when every client that is neither permanently
   dropped nor administratively excluded has completed `k`. If the lag policy prevents
   that barrier from arriving within a configured maximum wait, consolidate when the
   wait expires and record the missing clients. (The timeout needs a separate config;
   overloading `max_task_lag` with both clock and wall-time meaning would be ambiguous.)
3. Use the latest boundary-`k` snapshot from each client that completed `k`; do not use
   a newer live model as its teacher, because that model has already mixed in later-task
   updates. Eligible new classes are the union of those clients' task-`k` labels. Replay
   retains all globally consolidated classes through `k`, so generator and classifier
   training still protect older classes. Late snapshots after the one-time trigger are
   logged and can contribute to ordinary federated aggregation, but do not silently
   rerun the boundary.

This rule preserves the meaning of “task boundary,” avoids teaching task `k` from a
model already trained on `k+1`, and yields one auditable consolidation per boundary.
Its cost is storing versioned teachers and potentially waiting for stragglers. A
quorum trigger is faster but biases the generator toward fast clients/classes. A
periodic server-clock trigger is simple but no longer corresponds to client task
completion. Re-consolidating whenever a late client arrives uses more compute and
makes results depend strongly on arrival order. The completion-watermark rule is the
recommended default; quorum and periodic policies can be named ablations.

## Evaluation and metrics

The canonical accuracy matrix remains `num_tasks × num_tasks` for a completed run,
but asynchronous experiments need two views:

- **Per-client-stage evaluation** evaluates client `i` after its own stage `s` on
  tasks `0..s`. Forgetting, BWT, and average-anytime accuracy are computed within
  each client's stage-indexed matrix, then macro- and sample-weighted across clients.
  This is the fairest comparison of learning behavior at equal experience.
- **Global-clock evaluation** snapshots the server at fixed rounds (and at each
  consolidation watermark) and evaluates every task currently exposed by at least
  one client. Unseen cells are null, not zero. Metrics operate only on valid cells and
  report task/client coverage beside the value. This view measures system utility and
  wall-clock convergence.

`old_task_retention_under_async` should be the accuracy on tasks strictly below each
client's current stage, aggregated with an explicit weighting rule. Temporal lag is
measured from the leading eligible task clock; divergence reports distance from the
client mean. Both per-client task IDs and active counts are logged each round so the
accuracy views can be interpreted against the realized schedule.

## Determinism and open decisions

`task_schedule_seed` initializes a scheduler-owned random generator. Client grouping,
uniform speeds, Poisson arrivals, dropout, and participation use deterministic,
separate substreams derived from that seed and client ID. Schedule generation must
not consume NumPy or Torch training RNG state. The fully materialized schedule and
its seed should be saved with the resolved config; replaying it must reproduce task
IDs and participation independently of model nondeterminism.

Before Phase 2b, Claude should decide: (1) the boundary timeout name/default and
whether a quorum fallback is required, (2) whether dropout means per-round absence,
permanent failure, or two separate controls, (3) the uniform-speed interval and fixed
group schema, (4) macro versus sample-weighted per-client-stage metrics as the primary
reported value, and (5) whether custom schedules may intentionally jump more than one
task or must remain transition-by-transition.

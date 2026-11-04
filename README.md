# DecideOnce

An MCP server that finds the decision and categorisation steps inside agent
workflows, records every decision the agent makes there (inputs, choice,
outcome), and then lets confident decisions skip the LLM call: hosted Jev
first, a private model only when the evidence demands it.

Decisions route through the **hosted Jev** tier (System One over OpenRouter or
TypeSafe) once a point reaches `shadow` (register with `remote_ok:
["hosted_jev"]` and `confirm: true`: points default to private): no training
step, just a temperature and threshold fitted automatically from the first
200 labelled shadow calls. The API
key arrives via `envsave` (`hosted_jev.key_ref`), lives only in server memory,
and every call counts against `hosted_jev.monthly_budget_usd`. `do_train` stays
locked until `do_report` says `train_recommended` (low coverage, a missed
margin, repeated drift, or heavy teacher-only options) — that is the point at
which a private LoRA earns its keep. Training runs on Fireworks: `do_train`
launches a smoke job first, then the paid full job once confirmed. There is
no local model path. The calling agent still does all the LLM
work (it analyses workflows and acts as the teacher). The hosted path above
is the default (`[decider] backend = "tiered"`, plus the key and a budget):
"promote to shadow" needs no training step.

## Built during the Meta AI Developer Hackathon

Everything in this repository was written during the hackathon (event 26 Oct – 4 Nov 2026; all code
written 1–4 Nov 2026), by one person (solo, Indie Developer track), with Meta's Muse Code coding agent. The
development history is 100+ commits from 1 to 4 Nov 2026; this public repository is a code-only export
of it, so it shows one commit per export.

**Built by us during the event**

- The DecideOnce MCP server (`decideonce/`):
  - decision core and store (`do_decide`, `do_record`, `do_outcome`)
  - discovery (`do_analyze`, `do_register`): rubric, text hints, agent-proposed candidates, a trace detector, and
    (built 3 Nov) a Python static detector, a check that the next step uses each answer, and evidence
    levels where 2 of 3 signals (text, code, traces) must agree (`decideonce/discovery/`)
  - shadow and serve gates: error bound, margin over random escalation, audits, drift detection, demotion (`do_promote`, `do_report`)
  - the hosted-Jev tier with automatic temperature and threshold fitting
  - the zero-edit gateway, one-turn runs and the autopilot (`autopilot.py`)
  - the training export for Fireworks (`fireworks_export.py`), the live Fireworks trainer
    (`fireworks_trainer.py`) with a test fake (`trainer.py`), and the scoring worker
    (`fireworks_worker.py`)
- The Muse plugin (`plugin/`) with two skills: `decideonce-workflow` for existing workflows and
  `decideonce-design` for new, decision-first workflows.
- For the demo video (kept in the private development repository):
  - the synthetic refund-ticket generator and its written policy
  - the live MCP demo driver
  - the lead-routing pilot
  - the Glimmer training export and evaluation scripts
  - the discovery evaluation (v1 vs v2 against gold labels) and the discovery demo driver

**Used, not built by us**

- Meta: Muse Code (wrote the code from our spec), Muse Glimmer 30B (base model; we trained a LoRA on
  Fireworks), Muse Wonder (heavy model in the lead-routing run), the Muse plugin system.
- Hosted Jev 1.13 by TypeSafe, called through OpenRouter: the off-the-shelf decision model.
- Fireworks: training infrastructure and the public `decision/kalshi_v1` cookbook recipe, adapted to our task.
- Open-source libraries listed in `pyproject.toml` (MCP Python SDK, pydantic, jsonschema,
  cryptography).

## How it works

```
1. do_analyze(workflow)  -> rubric + candidate decision points
2. do_register(point)    -> stored; agent rewrites its skill to call do_decide
3. do_decide(point_id, context) -> rules -> hosted Jev -> threshold
     teacher mode: the agent decides itself, then do_record(decision_id, choice)
     auto mode:    the agent follows the returned choice (no LLM reasoning)
   do_outcome(decision_id, ok | correct_choice)  <- when the result is known
4. do_promote(point_id, "shadow") -> needs the hosted key + a budget + remote_ok, no training
   (temperature + threshold fit themselves from the first 200 labelled shadow calls)
5. do_promote(point_id, "serve") -> Jev decides above the threshold; audits a sample
6. do_report             -> agreement, coverage, savings, drift, promotion blockers,
                            hosted spend, train_recommended
```

The saving appears in phase `serve`, when the agent skips its own reasoning for
confident decisions. The large savings are decisions that are separate LLM
calls today (classification calls, sub-agents, triage) and decisions the agent
thinks hard about; `do_report` shows the measured share.

## Install

Prerequisites: `uv`, Python ≥ 3.11, and (for the hosted tier) an `envsave`
entry holding your OpenRouter key (`hosted_jev.key_ref`). Hosted Jev 1.13 is
TypeSafe's off-the-shelf decision model (`typesafe/jev-1.13`), called through
your OpenRouter account — create a key in the OpenRouter dashboard
(https://openrouter.ai) and store it with
[EnvSave](https://www.npmjs.com/package/@envsave/cli) (`npm install -g
@envsave/cli`), never in the config file.

```bash
git clone https://github.com/jcsancho/decideonce decideonce
cd decideonce
uv sync
```

**MCP server** — add to the `mcpServers` entry in your Muse settings:

```json
{"decideonce": {"transport": "stdio", "command": "uv",
  "args": ["run", "--project", "<repo-path>",
           "-m", "decideonce.server"],
  "env": {"DECIDEONCE_CONFIG": "<repo-path>/decideonce.toml"}}}
```

Claude Code reads the same entry from the `mcpServers` object in
`~/.claude.json`. Or register it in one command (Claude Code / Claude Desktop):

```bash
claude mcp add decideonce -- uv run --project <repo-path> -m decideonce.server
```

**Plugin** (Muse-only: the `decideonce-workflow` skill that teaches agents the loop above):

```bash
muse plugins install plugin --scope project && muse plugins approve decideonce
```

After plugin changes: `muse plugins update decideonce && muse plugins approve decideonce`
(Muse runs skills from a plugin cache).

**Python pipelines** can skip MCP and use the client (same store, same code):

```python
from decideonce.client import Decider
d = Decider(project="support")
d.register({"point_id": "refund.action", "workflow": "support-triage",
            "question": "Which refund action does the policy require?",
            "kind": "choice",
            "options": ["full_refund", "partial_refund", "deny"],
            "context_fields": {"amount_eur": {"type": "number"}}})
ctx = {"amount_eur": 42.0}
r = d.decide("refund.action", ctx)
# teacher mode: your own LLM call decides from the exact prompt in r.instruction
choice = r.choice if r.mode in ("auto", "rule") else ask_your_llm(r.instruction)
d.record(r.decision_id, choice)
```

## Try it offline

No key, no network. Point the server at a config with the scripted backend:

```toml
[decider]
backend = "fake"

[hosted_jev]
monthly_budget_usd = 1  # simulated; nothing is billed
fake_accuracy = 1.0
```

```bash
DECIDEONCE_CONFIG=/tmp/offline.toml uv run decideonce
```

Register each point with `remote_ok: ["hosted_jev"]` and `confirm: true`
(points default to private, so `shadow` stays locked without it), and add a
`label_hint` context field holding the right answer for each call — the fake
reads it, so a perfect run fits at 200 labelled shadow calls and reaches
`serve` at 400 (fit rows never count as evidence). `do_report` marks this
spend `"spend_simulated"`. Every tool works; the answers are scripted, not a
real model (the server says so on stderr at startup).

## Config

`decideonce.toml` (path from `DECIDEONCE_CONFIG`, project from
`DECIDEONCE_PROJECT`). Every default is a config value: change the config, not
the code. Storage is SQLite at `<data_dir>/<project>/decide.db` (WAL mode);
trained-model cards sit beside it. Everything stays on the machine.

`[decider] backend = "fake"` selects scripted fakes for both tiers (a loud
startup warning says so); the shipped default is `"tiered"`. Unit tests run
on fakes and never need a key. Pre-M12 configs with
`backend = "local_legacy"` or a `[jev]` section are refused at startup with
`config_v16_migration`: remove `[jev]`; set `[decider] backend = "tiered"`.

## Tools (13 + 13 aliases)

`do_analyze` `do_register` `do_decide` `do_record` `do_outcome` `do_train`
`do_promote` `do_report` `do_export` `do_extract` `do_plan_run`
`do_plan_result` `do_autopilot_status` — each takes one `payload` object and
returns `{ok: true, ...}` or `{ok: false, error, message}`. Each also answers
to a deprecated `dm_*` alias (`dm_analyze` … `dm_autopilot_status`) kept for
older skills; prefer the `do_*` names.

## Discovery

`do_analyze` finds decision points three ways and merges what agrees:

- **text**: keyword / enum-list hints in the workflow text (line numbers).
- **code**: pass `code: [{path, text}]`. The Python detector parses each file
  and reports LLM calls (names in `[discovery] llm_calls`) whose result is
  compared against two or more string/enum literals (`==`, `in`,
  `match`/`case`, dict dispatch, `Literal`/`Enum` annotations), possibly
  after `.strip()`/`.lower()`/`json.loads(...)[key]`. Location is `path:line`.
- **behaviour**: trace calls may carry `run_id` and `after: [{step, input}]`
  (what the next steps received). A trace group becomes a candidate only if
  its output is consumed in at least `[discovery] min_consumed` (default
  0.8) of the runs where a later step exists: the output appears in a later
  step's input, or the later steps differ by output value (a branch).
  Unconsumed enum groups are rejected as `not consumed`, free-text groups
  as `free text`.

Merged candidates carry `signals: {text, code, behaviour}` with the evidence
for each, and a level: **A** = at least 2 of the 3 signals agree (same step,
overlapping options), **B** = one signal plus an agent-proposed candidate,
**C** = one signal only. Level-A options are the union of the agreeing
signals' options, with each value's source. Results come back in the
`discovery` block (candidates ranked by level, then rubric score, plus
`rejected`); nothing is persisted. Without the new inputs the output is
exactly what it was before.

## Zero-edit gateway

Pipelines that classify with **separate LLM calls** (scripts, n8n, sub-agents,
customer back ends) need no code change: point them at the gateway and hosted
Jev answers the approved calls once the point is fitted.

```bash
uv run decideonce-gateway   # 127.0.0.1:8787, same store and hosted budget as the MCP server
```

Client base URLs (the only change in the calling code):

- OpenAI-compatible SDKs: `base_url = "http://127.0.0.1:8787/openai/v1"`
- Anthropic SDK: `base_url = "http://127.0.0.1:8787/anthropic"`

A discovery key is the SHA-256 of route + model + prompt skeleton: the
gateway stamps it on every forwarded call, and `do_analyze` uses it to match
traced traffic to candidate points.

Flow: approve a discovery key in `gateway.log_allow` → traffic is traced →
`do_analyze(gateway)` learns the template, options and output format → you
approve the template, 5 sample requests, options and format →
`do_register` the gateway point → collect → shadow (no training; the format
check compares ≥ 200 answers) → `do_promote … serve` → the gateway answers
confident calls itself and forwards the rest unchanged (streams unbuffered).
Gateway calls share the point's hosted budget.

Rules you should know:

- **Scope limit:** decisions an agent makes inside its own reasoning (one long
  session) are not separate API calls and never pass through the gateway.
  Those need `do_decide` (see *How it works*).
- **Known caveat:** synthesized answers report zero token usage, because no
  tokens were billed. Client code that divides by usage or budgets by it must
  handle zero.
- Only calls matching a registered template in `serve` with confident Jev
  are answered; everything else is forwarded byte-for-byte. Auth headers go
  upstream and are never stored; the gateway holds the hosted key in memory
  only (resolved from `envsave` at startup, never logged or stored).
- Upstream answers that do not parse into an option are stored with
  `unparsed: true` and never train. Free-text formats (a `reason` field,
  confidences, arrays) are `format_unsupported`: use `do_decide` for those.

## One-turn runs

For work an agent does in one turn (answer a ticket, then stop): plan the
decisions before the run, inject them into the prompt, and learn from the
transcript afterwards.

- `do_extract`, two calls: call 1 submits the run transcript and returns the
  record schema; an extractor child fills one record per decision (exact
  quotes, evidence before the decision) and submits call 2. A random sample
  gets a second pass in a fresh context; points whose passes disagree are
  excluded from training until they agree.
- `do_plan_run` returns inject (prompt addendum) / downgrade (cheap model +
  addendum) / none (run normally, shadow predictions in `would_have`).
  `do_plan_result` records the run: transcript, cost, conflicts, and the
  workflow check (a local deterministic script, or the user's `do_outcome`).
- Downgrade needs a passed paired trial (Pocock looks, Newcombe bound, no
  veto pair) plus your explicit yes; monitoring keeps paired mini-trials
  running afterwards and revokes on a proven drop. `do_report({workflow})`
  shows extraction, plans, trial, costs and agreement-vs-correctness.

## Autopilot

The statistical gates already decide whether a point is good enough; autopilot
removes the per-step waiting. Set limits once in `[autopilot]` (which points,
max target error, per-job and monthly training caps, cooldown, an optional
notify command), create a signing key once, then approve in a terminal:

```bash
uv run decideonce autopilot keygen    # once: Ed25519 key in envsave, public key to the config
uv run decideonce autopilot approve   # signs the effective limits (you type the envsave password)
```

Inside the limits the server promotes collect → shadow → serve by itself and
launches training jobs (smoke first, then full), logging every action in
`do_report`'s Autopilot block and running the approved `notify_cmd` with a
one-line message. It never promotes irreversible points to serve, never
switches tiers, never touches plan serve or the downgrade trial, and stops
every point the moment the limits differ from the signed approval
(`not_approved`) until you re-approve.

Honest limit: an agent with a shell can no longer approve by accident or
casually — approval needs your envsave master password to sign. But a
deliberate shell attacker can swap the public key, sign with their own key,
write the meta rows including the fingerprint, and suppress
`trust_key_changed` entirely — nothing in this design stops that. This is
defence in depth, not a sandbox.

## Private model (Fireworks)

Where hosted Jev is not good enough, a point can train its own Glimmer LoRA
on Fireworks (one per project, your account). Prerequisites:

- the cookbook clone at the pinned commit (`[fireworks] cookbook_dir` +
  `cookbook_commit`) with `uv sync` run in its `training/` directory;
- the Fireworks key in envsave (`[fireworks] key_ref`, an opaque `es_…`
  reference, resolved at startup and never stored or logged);
- a budget (`[fireworks] monthly_budget_usd`, default 0 = no billed call).
  At 80% `do_report` warns; at 100% decisions fall back to teacher mode and
  `do_train` refuses. Spend is estimated until you compare it with the bill.

Training is free while Fireworks' promotion covers it: confirm the smoke job
was not billed (`training_free_confirmed: true`) before the **2026-12-01
Pacific** deadline. After that a full job needs
`[fireworks] allow_paid_training = true` plus `confirm: true` and budget.

Finished checkpoints are served by a scoring worker
(`fireworks_worker.py`): one process per model, one request in flight,
per-call timeout, at most one failure-restart per 10 minutes. Live
shadow/serve on the private tier needs
`[fireworks] allow_session_serving = true` (default false): set it only after
measuring latency and idle billing, and only if Fireworks confirms in writing
that scoring from a training session is allowed for production traffic.
Without it a trained point stays `collect` with
`blocked(serving_not_allowed)`. While a worker warms up, decisions fall back
to the teacher and count `private_warming` in the report.

## License

MIT — see `LICENSE`.

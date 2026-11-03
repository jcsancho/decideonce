# DecideOnce

An MCP server that finds the decision and categorisation steps inside agent
workflows, records every decision the agent makes there (inputs, choice,
outcome), and then lets confident decisions skip the LLM call: hosted Jev
first, a private model only when the evidence demands it.

Decisions route through the **hosted Jev** tier (System One over OpenRouter or
TypeSafe) once a point reaches `shadow`: no training step, just a temperature
and threshold fitted automatically from the first 200 labelled calls. The API
key arrives via `envsave` (`hosted_jev.key_ref`), lives only in server memory,
and every call counts against `hosted_jev.monthly_budget_usd`. `do_train` stays
locked until `do_report` says `train_recommended` (low coverage, a missed
margin, repeated drift, or heavy teacher-only options) — that is the point at
which a private LoRA earns its keep. The calling agent still does all the LLM
work (it analyses workflows and acts as the teacher). The hosted path needs
`[decider] backend = "tiered"` (plus the key and a budget); a default
install is `local_legacy`, where "promote to shadow" needs a trained probe
instead of "no training".

## How it works

```
1. do_analyze(workflow)  -> rubric + candidate decision points
2. do_register(point)    -> stored; agent rewrites its skill to call do_decide
3. do_decide(point_id, context) -> rules -> hosted Jev -> threshold
     teacher mode: the agent decides itself, then do_record(decision_id, choice)
     auto mode:    the agent follows the returned choice (no LLM reasoning)
   do_outcome(decision_id, ok | correct_choice)  <- when the result is known
4. do_promote(point_id, "shadow") -> needs the hosted key + a budget, no training
   (temperature + threshold fit themselves from the first 200 labelled calls)
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
entry holding your OpenRouter key (`hosted_jev.key_ref`).

```bash
git clone <this-repo> decideonce
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

**Plugin** (the `decideonce-workflow` skill that teaches agents the loop above):

```bash
muse plugins install plugin --scope project && muse plugins approve decideonce
```

After plugin changes: `muse plugins update decideonce && muse plugins approve decideonce`
(Muse runs skills from a plugin cache).

**Python pipelines** can skip MCP and use the client (same store, same code):

```python
from decideonce.client import Decider
d = Decider(project="support")
r = d.decide("refund.action", ctx)
choice = r.choice if r.mode in ("auto", "rule") else my_llm_classify(ctx, r.allowed_options)
d.record(r.decision_id, choice)
```

## Config

`decideonce.toml` (path from `DECIDEONCE_CONFIG`, project from
`DECIDEONCE_PROJECT`). Every default is a config value: change the config, not
the code. Storage is SQLite at `<data_dir>/<project>/decide.db` (WAL mode);
models and cached vectors sit beside it. Everything stays on the machine.

Test hook: `jev.url = "fake:"` selects the deterministic `FakeJev` backend
(`fake:down` simulates an outage).
`[decider] backend = "fake"` selects scripted fakes for both tiers; the
shipped default is `"local_legacy"` — set `"tiered"` for the hosted path.

## Tools (13)

`do_analyze` `do_register` `do_decide` `do_record` `do_outcome` `do_train`
`do_promote` `do_report` `do_export` `do_extract` `do_plan_run`
`do_plan_result` `do_autopilot_status` — each takes one `payload` object and
returns `{ok: true, ...}` or `{ok: false, error, message}`.

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
decideonce autopilot keygen    # once: Ed25519 key in envsave, public key to the config
decideonce autopilot approve   # signs the effective limits (you type the envsave password)
```

Inside the limits the server promotes collect → shadow → serve by itself and
launches training jobs (smoke first, then full), logging every action in
`do_report`'s Autopilot block and running the approved `notify_cmd` with a
one-line message. It never promotes irreversible points to serve, never
switches tiers, never touches plan serve or the downgrade trial, and stops
every point the moment the limits differ from the signed approval
(`not_approved`) until you re-approve.

Honest limit: an agent with a shell can no longer approve by accident or
casually — approval needs your envsave master password to sign. It could still
replace the public key and sign with its own key, but that is a deliberate
attack: the report and notifications show it as `trust_key_changed`. This is
defence in depth, not a sandbox.

## License

MIT — see `LICENSE`.

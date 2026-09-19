# XApply

Automated job-application agent. It searches job boards, reads each posting, scores it against
your real profile with Gemini, rewrites your resume for that specific role, fills the application
form, and stops for you whenever a human is genuinely needed.

Supported application systems: **LinkedIn Easy Apply, Greenhouse, Lever, Ashby**.

```
discover  ->  extract JD  ->  Gemini analysis  ->  score gate  ->  tailored PDF  ->  fill form  ->  submit / pause
   |                              |                                    |                              |
LinkedIn search             structured JSON                    output_resumes/              SQLite + audit JSON
jobs.txt URLs           (match score, bullets, answers)     {Company}_{Role}.pdf         terminal + FastAPI review
```

---

## Quick start

```bash
make install          # venv, dependencies, Playwright Chromium, .env, smoke test
# edit profile.json and .env (GEMINI_API_KEY)
make login            # log in to LinkedIn once; the session is saved
make run              # assisted mode: bot fills everything, you click Submit
make serve            # admin dashboard at http://127.0.0.1:8000
```

Run `make` with no target to list every command.

| Target | What it does |
| --- | --- |
| `make install` | Full setup: virtualenv, dependencies, Chromium, `.env`, import check |
| `make login` | Opens the persistent browser so you can sign in once |
| `make run` | Assisted mode. Bot fills the form, you review and submit |
| `make run-auto` | Auto mode. Bot clicks Submit; still pauses on CAPTCHAs |
| `make analyze URL=...` | Dry run on one posting. Scores it and builds the PDF, applies to nothing |
| `make serve` | FastAPI admin dashboard plus JSON API |
| `make test` | Offline test suite. No network, no browser, no Gemini calls |
| `make check` | Byte-compile and import every module |
| `make clean` / `make reset` | Remove caches / remove everything including the database |

`make install` requires Python 3.10+. It uses [uv](https://docs.astral.sh/uv/) when available and
falls back to `python3 -m venv`.

---

## Configuration

Everything lives in `.env` (copy from `.env.example`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for analysis and live form answers |
| `AUTO_SUBMIT` | `false` | `false` = you click Submit. `true` = the bot clicks Submit |
| `MATCH_THRESHOLD` | `65` | Postings scoring below this are skipped, not applied to |
| `HEADLESS` | `false` | Keep `false` so you can watch and take over |
| `SOURCES` | `linkedin` | Comma separated: `linkedin`, `urls` |
| `SEARCH_QUERIES` | `Python Developer` | Comma separated search terms |
| `SEARCH_LOCATION` | `Remote` | LinkedIn location filter |
| `POSTED_WITHIN_HOURS` | `72` | Recency filter. `0` = any time |
| `MAX_APPLICATIONS_PER_RUN` | `10` | Hard stop per run |
| `FOLLOW_COMPANIES` | `false` | Whether to leave LinkedIn's "follow company" box ticked |
| `BROWSER_CHANNEL` | `chrome` | Use installed Chrome. Empty = bundled Chromium |
| `ADMIN_TOKEN` | `change-me` | Required in `X-Admin-Token` unless left at the default on localhost |

Command-line flags override `.env` for a single run:

```bash
python main.py run --auto-submit --limit 5 --threshold 75
python main.py run --url "https://jobs.lever.co/acme/1a2b3c"     # one posting
python main.py run --urls-file jobs.txt                          # a list
```

---

## The two execution modes

**Assisted (`AUTO_SUBMIT=false`, the default).** The bot navigates, uploads the tailored resume,
fills every text field, select, radio and checkbox, then stops at the final review page and prints:

```
==============================================================================
  HUMAN INPUT NEEDED
  Final review page reached. Verify the form and click 'Submit application'
  yourself, then continue.
  -> Solve it in the browser window, then press Enter here to resume.
==============================================================================
Press Enter after solving CAPTCHA / verifying form to continue...
```

**Auto (`AUTO_SUBMIT=true`).** The bot clicks Submit itself and waits for a confirmation. If a
CAPTCHA appears, a required field cannot be answered truthfully, or validation errors persist,
it falls back to the same pause instead of crashing or guessing.

A pause is triggered by any of: a CAPTCHA or challenge iframe, a login wall, a required field the
bot cannot answer from your profile, validation errors that survive a retry, or a Submit click
with no confirmation.

When stdin is not a terminal (running under a supervisor), create `logs/CONTINUE` or
`POST /admin/continue` to release the pause instead.

---

## Reviewing what the bot did

Every job produces a SQLite row **and** a JSON audit file in `logs/applications/` holding the full
job description, the AI analysis, and every value typed into the form with the source that
produced it.

### In the terminal

```bash
python main.py stats                  # counters and a bar chart
python main.py list                   # table of every job seen
python main.py list --status pending_human_review
python main.py list --search Stripe
python main.py show 42                # analysis + every field submitted
python main.py show 42 --description  # the stored job description
python main.py show 42 --json         # machine-readable
python main.py watch                  # live-refreshing dashboard
python main.py audits                 # recent audit files
python main.py export --out apps.csv
```

`show` prints the AI rationale, the gaps it found, the tailored bullets, the predicted screening
answers, and then every form field with its value, its source (`profile`, `profile.years`,
`ai_predicted`, `ai_live`) and the confidence behind it.

### In the browser

```bash
make serve     # http://127.0.0.1:8000
```

The dashboard lists every application, colour-coded by status, and opens a detail panel per job
with the analysis, the submitted answers, the generated PDF and the review-page screenshot.
Interactive API docs are at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/stats` | Counters, average match score, submissions per ATS |
| `GET /api/applications` | List. Filters: `status`, `search`, `limit`, `offset` |
| `GET /api/applications/{id}` | Full record: description, analysis, answers |
| `PATCH /api/applications/{id}` | Correct a status by hand |
| `GET /api/applications/{id}/resume` | The tailored PDF |
| `GET /api/applications/{id}/screenshot` | The review-page screenshot |
| `GET /api/export.csv` | CSV of everything |
| `GET /api/audits`, `/api/audits/{file}` | The raw audit files |
| `GET /api/gate`, `POST /admin/continue` | See and release a human pause |
| `POST /admin/run` | Trigger a run from the API |

Authenticate with `X-Admin-Token: <ADMIN_TOKEN>` or `?token=`. Auth is skipped while
`ADMIN_TOKEN` is `change-me` and the server is bound to localhost; binding any other host with
the default token is refused.

---

## Your profile

`profile.json` is the single source of truth. Gemini may reorder, reword and re-emphasize what is
in it; it may never add to it.

| Key | Used for |
| --- | --- |
| `name`, `email`, `phone`, `location`, `linkedin`, `github`, `website` | Contact fields, resume header |
| `skills` | Grouped skill lines. Gemini reorders these; it cannot invent new ones |
| `years_of_experience` | Every "how many years of X" question, answered from this map only |
| `experience`, `projects` | The bullets Gemini rewrites per posting |
| `education`, `certifications` | Rendered as-is |
| `screening_defaults` | Sponsorship, notice period, salary, relocation, EEO answers |

`profile.example.json` is a filled-in template to copy from.

**Hallucination guards.** Three layers:

1. The system prompt forbids inventing companies, degrees, dates or proficiencies, and requires
   genuine gaps to be reported in `missing_requirements` with a lower score.
2. `resume_builder.py` maps every rewritten bullet group back to a real profile entry by name.
   Groups that match nothing are dropped with a warning. Skills not present in `profile.json` are
   filtered out of the rendered PDF.
3. Unknown screening answers come back as `UNKNOWN`, which the resolver treats as "ask a human"
   rather than filling in a guess.

---

## How answers are chosen

For each field, in order, stopping at the first confident hit:

1. **Profile rules.** Name, email, phone, location, LinkedIn, GitHub, current employer and title.
2. **Years map.** "Years of experience with Kubernetes" reads `years_of_experience.Kubernetes`.
3. **Predicted answers.** Gemini's per-job screening predictions, matched to the field label.
4. **Screening defaults.** Sponsorship, notice period, salary, relocation, EEO.
5. **Live Gemini call.** One structured call with the label, field type and the exact options.
6. **Human.** Below the confidence floor, flagged `needs_human`, or no option matches.

A rule answer that does not fit any of a select's options is discarded and re-asked with the
option list attached, so the bot never types "2 weeks" into a dropdown that only offers ranges.

---

## Architecture

| File | Responsibility |
| --- | --- |
| `config.py` | Pydantic settings. Every value overridable from `.env` |
| `models.py` | `JobPosting`, `ApplyResult`, ATS detection, stable job IDs |
| `database.py` | SQLite schema, dedupe, status tracking, stats |
| `ai_agent.py` | Gemini client, Pydantic `response_schema`, guardrail prompt, retries |
| `resume_builder.py` | Jinja2 template -> Chromium -> single-page PDF, hallucination filter |
| `browser_bot.py` | Stealth persistent browser, human gate, form discovery, answer resolution |
| `appliers.py` | Per-ATS drivers: LinkedIn modal, Greenhouse, Lever, Ashby |
| `job_search.py` | LinkedIn search, posting extraction, external-apply hand-off |
| `pipeline.py` | Orchestration and audit logging |
| `reports.py` | Terminal dashboard rendering |
| `api.py` | FastAPI admin API and HTML dashboard |
| `main.py` | CLI |

**Selector strategy.** Fields are discovered by a single injected script that resolves each
input's label through `<label>`, `aria-label`, `aria-labelledby`, `<legend>`, placeholder, then a
bounded DOM walk, and tags the element with `data-xapply-idx` so Playwright addresses it by a
stable attribute rather than a brittle CSS path. Buttons use `get_by_role` with anchored regex.
Radio groups are grouped by `name` or fieldset. Custom React comboboxes are opened, read and
matched by text similarity.

**Anti-bot.** `launch_persistent_context` keeps cookies so 2FA is not re-triggered.
`navigator.webdriver` is masked, `--enable-automation` is stripped, the user agent and 1440x900
viewport are realistic, delays are Gaussian, typing is per-character, and the page is scrolled in
several small steps before clicking.

**File uploads.** `set_input_files` is followed by three checks: the input reports a file, the
filename appears on the page, and any upload spinner has disappeared. Only then does the bot
advance.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GEMINI_API_KEY is not set` | Add it to `.env` |
| Browser opens logged out | `make login`, then keep `.browser_profile/` |
| `Could not launch browser channel 'chrome'` | Harmless. It falls back to bundled Chromium, or set `BROWSER_CHANNEL=` |
| Every job is skipped | Lower `MATCH_THRESHOLD`, or check `python main.py show <id>` for the rationale |
| No Easy Apply postings found | Set `FOLLOW_EXTERNAL_APPLY=true` to follow hand-offs to Greenhouse, Lever and Ashby |
| Bot pauses constantly on one site | That site's fields are not resolvable from your profile. Extend `screening_defaults` |

Logs: `logs/xapply.log`. Audits: `logs/applications/*.json`. Screenshots: `logs/*.png`.

---

## Notes on responsible use

LinkedIn's terms of service prohibit automated access. Running this against LinkedIn risks
restriction or loss of your account. Assisted mode exists for that reason: it keeps a human on
every submission. Keep `MAX_APPLICATIONS_PER_RUN` modest, keep the Gaussian delays, and do not
remove the human gate on CAPTCHAs. Everything the bot submits is your representation of yourself,
so review `python main.py show <id>` before trusting a run you did not watch.

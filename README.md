# XApply

An automated job-application agent. It scans job boards, reads each posting, scores it against your
real profile with an LLM, rewrites your resume for that specific role, fills the application form,
and stops for you whenever a human is genuinely needed.

Supported application systems: **Greenhouse, Lever, Ashby, LinkedIn Easy Apply**.
Only LinkedIn needs a login. The other three are public.

```
  scan boards  ->  score vs your profile  ->  tailored PDF  ->  fill form  ->  submit / pause for you
       |                    |                      |                |                    |
 public ATS APIs      match score,          output_resumes/    label-first        SQLite + audit JSON
 aggregators,         rewritten bullets,    Company_Role.pdf   selectors          web UI + terminal
 Google, LinkedIn     screening answers                                           both show everything
```

---

## Quick start

```bash
make install          # venv, dependencies, Chromium, .env, smoke test
# put your API key in .env, then edit profile.json
make serve            # open http://127.0.0.1:8000 and drive everything from there
```

That is the whole setup. No LinkedIn account required.

Everything the web UI does is also a terminal command, and the reverse. Use whichever you prefer.

---

## Applying without LinkedIn

Greenhouse, Lever and Ashby have no candidate-facing search, so the work is finding the URLs.
XApply gets them from company board APIs that need no key and no login.

**From the web UI**

1. `make serve`, open <http://127.0.0.1:8000>
2. Go to **Scan & Apply**
3. Tick `greenhouse`, `lever`, `ashby`. Leave `linkedin` unticked.
4. Enter your search terms, then **Scan now**
5. Review the results, tick the ones you want, then **Apply to selected**

**From the terminal**

```bash
python main.py discover --sources greenhouse,lever,ashby --queries "Backend Engineer,Full Stack Developer"
python main.py run --limit 5
```

**Or paste a link you already found.** You do not have to scan at all. Under **Scan & Apply** there
is an *Apply to a URL you already have* box: paste one job link per line, press **Check links** to
confirm the bot recognises each one, then choose **Fill, then let me submit** or **Fill and submit
automatically**. The same thing from the terminal:

```bash
python main.py run --url "https://job-boards.greenhouse.io/gitlab/jobs/8556658002"
python main.py run --url "https://jobs.lever.co/acme/1a2b3c" --url "https://jobs.ashbyhq.com/acme/9f8e" --auto-submit
python main.py run --urls-file jobs.txt
```

Set it permanently in `.env`:

```
SOURCES=greenhouse,ashby,lever
SEARCH_QUERIES=Full Stack Developer,Backend Engineer,Machine Learning Engineer
```

The boards ship with 31 companies covering roughly 5,000 open roles. Check them and add your own:

```bash
python main.py companies --probe                 # count open roles on each board
python main.py companies --add ashby:stickermule
```

The token is the path segment in the board URL. For `jobs.ashbyhq.com/linear/...` it is `linear`;
for `job-boards.greenhouse.io/gitlab/...` it is `gitlab`. You can also add and remove them under
**Settings** in the web UI.

---

## Where the job URLs come from

| Source | Login | How it works |
| --- | --- | --- |
| `greenhouse` | none | `boards-api.greenhouse.io/v1/boards/{token}/jobs` |
| `lever` | none | `api.lever.co/v0/postings/{token}?mode=json` |
| `ashby` | none | `api.ashbyhq.com/posting-api/job-board/{token}` |
| `remoteok` | none | Public remote-jobs feed, resolved to the underlying ATS link |
| `himalayas` | none | Public remote-jobs feed, resolved to the underlying ATS link |
| `google` | none | `site:job-boards.greenhouse.io "Backend Engineer" "Remote"` through Playwright |
| `linkedin` | yes | Easy Apply search. Run `make login` once |
| `urls` | none | Reads `jobs.txt`, one URL per line |

The three board APIs return the **full job description**, so a posting is scored without opening a
browser at all. A scan of Greenhouse, Lever and Ashby never launches a browser window.

Aggregator links point at the aggregator, so each one is followed to find the real Greenhouse,
Lever or Ashby URL behind it. Himalayas sits behind Cloudflare, so that resolution falls back to
the browser for a capped number of listings per run.

Google search is the least reliable route, because Google challenges automation. It exists to find
companies whose tokens you do not already have. When it is challenged, it pauses for you rather
than failing.

---

## The two execution modes

**Assisted (`AUTO_SUBMIT=false`, the default).** The bot navigates, uploads the tailored resume,
fills every text field, select, radio and checkbox, then stops at the review page:

```
==============================================================================
  HUMAN INPUT NEEDED
  Final review page reached. Verify the form and click Submit yourself.
  -> Handle it in the browser window, then press Enter here,
     click Continue in the dashboard, or create logs/CONTINUE.
==============================================================================
```

**Auto (`AUTO_SUBMIT=true`).** The bot clicks Submit and waits for a confirmation. On a CAPTCHA,
an unanswerable required field, or validation errors that survive a retry, it falls back to the
same pause instead of crashing or guessing.

A pause is triggered by a CAPTCHA or challenge iframe, a login wall, a required field that cannot
be answered truthfully from your profile, validation errors that persist, or a Submit click with
no confirmation.

**Releasing a pause.** All three of these work at any time, whichever happens first: press Enter in
the terminal, click **Continue** in the dashboard banner, or create the file `logs/CONTINUE`. A run
you started from the web UI is releasable from the web UI.

---

## Choosing the AI model

Two backends. Switch at any time, in `.env`, in the UI, or per command.

| Provider | Key | Notes |
| --- | --- | --- |
| `gemini` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Native structured outputs |
| `nvidia` | [build.nvidia.com](https://build.nvidia.com) | Free to start, OpenAI-compatible, 80+ models |

```bash
# .env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
```

```bash
python main.py models --provider nvidia --search nemotron   # browse what is available
python main.py run --provider nvidia --model openai/gpt-oss-20b
```

In the web UI the provider and model dropdowns sit both on the **Dashboard**, next to the run
buttons, and under **Settings**. The list is fetched live from the provider.

NVIDIA models differ in how they do structured output, so the client walks a ladder:
`json_schema`, then `nvext.guided_json`, then `json_object` with the schema in the prompt. It
remembers which rung worked and goes straight there afterwards. Whatever comes back is validated
against the Pydantic schema before anything touches it.

---

## Reviewing what the bot did

Every job produces a SQLite row **and** a JSON audit file in `logs/applications/` holding the full
job description, the AI analysis, and every value typed into the form with the source that
produced it.

**In the web UI.** The **Applications** tab lists everything colour-coded by status. Clicking a row
opens the score and its rationale, the requirements you do not meet, the tailored bullets, the
predicted screening answers, every form field with its value and source, the generated PDF, the
review screenshot and the stored job description.

**In the terminal.**

```bash
python main.py stats                  # counters and a bar chart
python main.py list                   # table of every job seen
python main.py list --status pending_human_review
python main.py show 42                # analysis + every field submitted
python main.py show 42 --description  # the stored job description
python main.py watch                  # live-refreshing dashboard
python main.py audits                 # recent audit files
python main.py export --out apps.csv
```

---

## Commands

| Make target | Command | What it does |
| --- | --- | --- |
| `make install` | | venv, dependencies, Chromium, `.env`, import check |
| `make serve` | `serve` | Web dashboard and API on `127.0.0.1:8000` |
| `make scan` | `discover` | Preview what the sources would find. Applies to nothing |
| `make scan-save` | `discover --save jobs.txt` | Scan and write the URLs to a file |
| `make run` | `run` | Assisted mode. Bot fills, you submit |
| `make run-auto` | `run --auto-submit` | Auto mode |
| `make analyze URL=…` | `analyze --url` | Score one posting and build the PDF, apply to nothing |
| `make models` | `models` | List the models the provider offers |
| `make companies` | `companies --probe` | Count open roles on each company board |
| `make login` | `login` | Sign in once, LinkedIn only |
| `make test` | | Offline test suite: no network, no browser, no LLM calls |
| `make check` | | Byte-compile and import every module |
| `make clean` / `make reset` | | Remove caches / remove everything including the database |

Flags override `.env` for a single run:

```bash
python main.py run --auto-submit --limit 5 --threshold 75
python main.py run --url "https://jobs.lever.co/acme/1a2b3c"
python main.py run --urls-file jobs.txt
```

---

## API

Interactive docs at `/docs`. Authenticate with `X-Admin-Token: <ADMIN_TOKEN>` or `?token=`.
Auth is skipped while `ADMIN_TOKEN` is `change-me` and the server is bound to localhost; binding
any other host with the default token is refused.

| Group | Endpoints |
| --- | --- |
| Overview | `GET /api/stats`, `/api/applications`, `/api/applications/{id}`, `PATCH /api/applications/{id}` |
| Files | `GET /api/applications/{id}/resume`, `/screenshot`, `/api/export.csv`, `/api/audits` |
| Discover | `POST /admin/discover`, `GET /api/discovered`, `POST /api/detect` |
| Apply | `POST /admin/run`, `/admin/apply-selected`, `/admin/stop` |
| Control | `GET /api/run`, `/api/gate`, `POST /admin/continue` |
| Settings | `GET|PATCH /api/config`, `GET /api/models` |
| Data | `GET|PUT /api/profile`, `GET|POST /api/companies`, `DELETE /api/companies/{ats}/{token}` |

---

## Your profile

`profile.json` is the single source of truth. The model may reorder, reword and re-emphasize what
is in it; it may never add to it. Edit it in a text editor or under **Profile** in the web UI,
which keeps the previous version as `profile.json.bak`.

| Key | Used for |
| --- | --- |
| `name`, `email`, `phone`, `location`, `linkedin`, `github`, `website` | Contact fields, resume header |
| `skills` | Grouped skill lines. Reordered per job, never extended |
| `years_of_experience` | Every "how many years of X" question |
| `experience`, `projects` | The bullets rewritten per posting |
| `education`, `certifications` | Rendered as-is |
| `screening_defaults` | Sponsorship, notice period, salary, relocation, EEO answers |

`profile.example.json` is a filled-in template to copy from.

**Hallucination guards**, in three layers:

1. The system prompt forbids inventing companies, degrees, dates or proficiencies, and requires
   genuine gaps to be reported in `missing_requirements` with a lower score.
2. `resume_builder.py` maps every rewritten bullet group back to a real profile entry by name.
   Groups matching nothing are dropped with a warning. Skills absent from `profile.json` are
   filtered out of the rendered PDF.
3. Unknown screening answers come back as `UNKNOWN`, which the resolver treats as "ask a human"
   rather than filling in a guess. Checkboxes are never ticked just because they are required:
   a form that marks every spoken language required will not have them all ticked.

---

## How answers are chosen

For each field, in order, stopping at the first confident hit:

1. **Profile rules.** Name, email, phone, location, LinkedIn, GitHub, current employer and title.
2. **Years map.** "Years of experience with Kubernetes" reads `years_of_experience.Kubernetes`.
3. **Predicted answers.** The per-job screening predictions, matched to the field label.
4. **Screening defaults.** Sponsorship, notice period, salary, relocation, EEO.
5. **A live LLM call.** One structured call with the label, field type and the exact options.
6. **You.** Below the confidence floor, flagged `needs_human`, or no option matches.

A rule answer that fits none of a select's options is discarded and re-asked with the option list
attached, so the bot never types "2 weeks" into a dropdown that only offers ranges. Numeric answers
map onto range options properly: 6 years picks "5-10 years", not the nearest starting number.

---

## Architecture

| File | Responsibility |
| --- | --- |
| `config.py` | Pydantic settings. Every value overridable from `.env` |
| `models.py` | `JobPosting`, `ApplyResult`, ATS detection, stable job IDs |
| `database.py` | SQLite schema, dedupe, status tracking, stats |
| `llm.py` | Provider abstraction: Gemini and NVIDIA behind one structured-output call |
| `ai_agent.py` | Prompts, Pydantic schemas, guardrails |
| `resume_builder.py` | Jinja2 template to Chromium to single-page PDF, hallucination filter |
| `browser_bot.py` | Stealth browser, human gate, form discovery, answer resolution |
| `appliers.py` | Per-ATS drivers: Greenhouse, Lever, Ashby, LinkedIn |
| `discovery.py` | Board APIs, aggregator feeds, Google search |
| `job_search.py` | LinkedIn search, posting extraction, lazy browser |
| `pipeline.py` | Orchestration and audit logging |
| `reports.py` | Terminal dashboard rendering |
| `api.py` | FastAPI app: every flow the CLI has |
| `static/index.html` | The web dashboard |
| `main.py` | CLI |

**Selector strategy.** Fields are discovered by one injected script that resolves each input's
label through `<label>`, `aria-label`, `aria-labelledby`, `<legend>`, placeholder, then a bounded
DOM walk, and tags the element with `data-xapply-idx` so Playwright addresses it by a stable
attribute rather than a brittle CSS path. Buttons use `get_by_role` with anchored regex. Radio
groups are grouped by `name` or fieldset. Custom React comboboxes are opened, read and matched by
text similarity. When an ATS renders no `<form>` element at all, which is what Ashby does, the
smallest element containing every visible input is found in the DOM and used as the scope.

**Anti-bot.** `launch_persistent_context` keeps cookies so 2FA is not re-triggered.
`navigator.webdriver` is masked, `--enable-automation` is stripped, the user agent and 1440x900
viewport are realistic, delays are Gaussian, typing is per-character, and the page is scrolled in
several small steps before clicking.

**File uploads.** `set_input_files` is followed by three checks: the input reports a file, the
filename appears on the page, and any upload spinner has disappeared. Only then does the bot
advance.

**Single-page resumes.** The PDF is rendered, its page count measured, and if it overflows, the
least relevant project is dropped and it is rendered again, then older experience, then the scale
is reduced. Nothing is invented or reworded to make it fit; entries are only dropped.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GEMINI_API_KEY is not set` | Add it to `.env`, or set `LLM_PROVIDER=nvidia` with a free NVIDIA key |
| `quota limit 0` from Gemini | That key has no Generative Language API quota. Enable the API for its project, or switch to NVIDIA |
| A blank browser window opens | Fixed: board-API scans no longer launch a browser. If you tick `linkedin`, `google`, `remoteok` or `himalayas`, one is needed |
| Clicking Continue does nothing | Fixed: the pause now accepts the terminal, the dashboard button and `logs/CONTINUE`, whichever comes first |
| Every job is skipped | Lower `MATCH_THRESHOLD`, or read the rationale with `python main.py show <id>` |
| A scan finds nothing | Your search terms need a majority of their words in the title. Try fewer, broader terms |
| A scan finds far too much | Use more specific terms. One shared word is not enough to match |
| `Could not launch browser channel 'chrome'` | Harmless. It falls back to bundled Chromium, or set `BROWSER_CHANNEL=` |
| The bot pauses constantly on one site | Those fields are not resolvable from your profile. Extend `screening_defaults` |

Logs: `logs/xapply.log`. Audits: `logs/applications/*.json`. Screenshots: `logs/*.png`.

---

## Responsible use

LinkedIn's terms of service prohibit automated access, and running this against LinkedIn risks
restriction or loss of your account. The Greenhouse, Lever and Ashby board APIs used here are the
public endpoints those platforms publish for their own job boards, and the default configuration
uses only those.

Assisted mode is the default for a reason: it keeps a human on every submission. Keep
`MAX_APPLICATIONS_PER_RUN` modest, keep the Gaussian delays, and do not remove the human gate on
CAPTCHAs. Everything the bot submits is your representation of yourself, so read
`python main.py show <id>` before trusting a run you did not watch.

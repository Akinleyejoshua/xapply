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

Under **Settings**, paste any job link you found in your browser and the board behind it is
identified and verified for you, with its open-role count. A board link or a bare company token
works too. **Count open roles on every board** flags any token that has gone dead.

From the terminal the token is the path segment in the board URL: `jobs.ashbyhq.com/linear/...`
is `linear`, `job-boards.greenhouse.io/gitlab/...` is `gitlab`.

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

**Picking a model.** The provider and model controls sit on the **Dashboard**, next to the run
buttons, and under **Settings**. The model field is a search box: type any part of a name to
filter, use the arrow keys and Enter, or paste an id the list does not contain. A dropdown of
eighty entries is unusable, and no list is ever guaranteed complete, so both are handled.

**Test this model** settles it by sending one tiny prompt and reporting what came back. That
matters because a listed model can still be retired or overloaded:

| Model | Result |
| --- | --- |
| `openai/gpt-oss-20b` | Answered a test prompt |
| `meta/llama-3.3-70b-instruct` | 410, retired by the provider |
| `stepfun-ai/step-3.5` | 404, no such model on this endpoint |
| `nvidia/nemotron-3-super-120b-a12b` | 503 when busy, works otherwise |

**Where the list comes from.** NVIDIA's OpenAI-compatible endpoint lists 82 models, and that is
the complete set it will accept, with or without a key. Twenty-one of them are embedding, safety,
reward, parsing or translation models that cannot hold a conversation, so they are hidden behind a
**show non-chat** toggle, leaving 61. NVIDIA's own NVCF catalogue lists more, but those extra
entries are internal deployments and non-chat models such as protein folding and speech
recognition, and they return 404 on the chat endpoint, so listing them would only mislead.

NVIDIA models differ in how they do structured output, so the client walks a ladder:
`json_schema`, then `nvext.guided_json`, then `json_object` with the schema in the prompt. It
remembers which rung worked and goes straight there afterwards. Whatever comes back is validated
against the Pydantic schema before anything touches it.

---

## The dashboard remembers what you choose

Every settings control writes through one API call that saves to `settings.local.json`, then
re-renders every page. So a model you pick on the Dashboard appears in Settings, a source you tick
on Scan & Apply is still ticked after a reload, and all of it survives a restart.

Resolution order, each layer overriding the one before:

1. the defaults in `config.py`
2. environment variables and `.env`
3. `settings.local.json`, the choices you made in the UI

Only fourteen keys can be saved that way: the LLM provider and model, sources, search terms,
location, remote-only, seniority, threshold, auto-submit, headless, the two limits and
follow-companies. Never a path, a token or an API key, so a hand-edited file cannot widen its own
scope. Check what is in force and where it came from:

```bash
python main.py settings            # every value, and whether it came from .env or the UI
python main.py settings --reset    # forget the UI choices
```

In the UI, the **Reset to .env** button under Settings does the same thing.

Auto-submit is remembered too, so the sidebar always shows the current mode in red when it is on.
The Dashboard's **Scan and apply** button always runs assisted regardless, and **Run with
auto-submit** always submits, so those two buttons mean what they say whatever the saved default is.

---

## Filtering what you apply to

**Seniority.** Every posting is bucketed into `intern`, `junior`, `mid`, `senior` or `lead` from its
title, before any LLM call. Tick the levels you want under **Scan & Apply**, or:

```bash
python main.py discover --seniority mid,senior
python main.py run --seniority senior,lead
```

```
# .env
SENIORITY_LEVELS=mid,senior
```

Leaving it empty means every level. A title with no level word at all, like "Backend Engineer",
counts as `mid`. "Senior Staff Engineer" is `lead`, because staff outranks senior. "Associate
Product Manager" is `junior`, not `lead`, despite the word manager. "Software Engineer, Ads
Manager" is `mid`, because there the word belongs to a product name.

**Remote only.** Tick **Remote roles only** on the scan card, pass `--remote-only`, or set
`REMOTE_ONLY=true`. This drops hybrid and on-site postings. It deliberately does not trust an
ATS's own "remote" boolean: Ashby marks hybrid roles remote, so 505 of OpenAI's 537 flagged-remote
jobs are actually hybrid. The `workplaceType` field is used when present, and the location text
otherwise.

**Search terms** are scored for resemblance, not matched word for word. Counting shared words was
too blunt: a search for "Data Analytics" returned nothing at all, because `analytics` is not the
word `analyst`. Of 93 real intern postings on the configured boards, it matched zero, including
*Analytics Engineer Intern* and *Business Analyst Intern*.

Words are folded into concepts, so `analyst`, `analytics`, `analysis`, `insights` and `BI` are one
idea, and `ML` and `machine learning` are another. Concepts are then weighted by how much they
narrow a search. `engineer` barely narrows anything, so it counts for little; `backend` or
`kubernetes` counts fully. The score is the share of your query's weight the title covers:

| Search term | Title | Score |
| --- | --- | --- |
| Data Analytics | Data Analyst | 0.97 |
| Data Analytics | Analytics Engineer Intern | 0.50 |
| Data Analytics | Accounting Intern | 0.00 |
| Backend Engineer | Backend Developer | 0.96 |
| Backend Engineer | Sales Engineer | 0.29 |

**Match sensitivity** is the cut-off, adjustable on the scan card and defaulting to 0.45. Lower
casts a wider net. Discovery is deliberately generous, because every posting is then scored
properly by the model and rejected below your match threshold. A near miss at this stage costs
one cheap API call; a wrongly dropped posting is never seen again. The score for each result is
shown in the Match column.

**Countries.** Pick any number from the dropdown on the scan card, or:

```bash
python main.py run --countries "Nigeria,United Kingdom"
python main.py countries-list --search king      # 53 accepted names
```

```
# .env
COUNTRIES=Nigeria,United Kingdom
```

Postings almost never name a country, so each one is matched on the phrases that actually appear
in listings: the country name, its short forms, and its main tech cities. "San Francisco" matches
United States, "Bengaluru" matches India, "Lagos" matches Nigeria. Regional shorthand counts too,
so "Remote, EMEA" matches Germany but not India. Matching is word-bounded, so "Indiana Township"
is not India and "Chinatown" is not China.

`Anywhere / Worldwide` is a pseudo-country for postings that state no location at all.

The filters stack. With **United Kingdom** and **Remote roles only** both set, a posting has to be
genuinely remote *and* name the UK, so "London (Hybrid)" is dropped and "Remote, United Kingdom"
is kept. Choosing any country makes the free-text **Location** box inactive, so the two can never
disagree.

---

## Reaching the form on a company careers page

Half of all Greenhouse boards redirect their own job URL to the company's careers site,
which shows the description and an Apply button but never the form. Stripe, Airbnb,
Coinbase, Dropbox, Asana, Databricks, Duolingo, Instacart, Brex and Samsara all do this.
Those pages also carry their own marketing CAPTCHA, which used to stop a run before the
form was ever reached.

So for Greenhouse the agent goes straight to the embed form, which always renders the
real application and never redirects. If that is unavailable it falls back to the posting
itself, then any embedded ATS iframe, then an Apply button or link on the page. Measured
across twelve company boards, it now reaches a real form with a working Submit button on
all twelve.

Two safeguards make that reliable:

- **A form has to look like an application.** A careers page offers a search box and a
  newsletter signup, both of which are forms with inputs. Accepting one of those stopped
  the search before the real form was found. A candidate now has to carry a file upload,
  or a name and an email, or four fields with one of those.
- **An iframe is matched on its host.** A Google API proxy carries `greenhouse.io` in its
  query string, and following it led nowhere.

---

## CAPTCHAs: fill first, solve last

An application form very often carries its own inline reCAPTCHA or Turnstile widget.
Stopping for it on arrival means you are shown a CAPTCHA next to a set of empty boxes.

So while navigating, the agent only stops for something that genuinely blocks the page:
a login wall, or a challenge on a page with no form on it. An inline widget is noted and
ignored, the form is filled, and the full check runs immediately before submitting, which
is the only moment the CAPTCHA has to be solved.

**When a challenge will not clear**, skip the posting rather than holding up the run:
click **Skip this job** in the dashboard banner, type `s` then Enter in the terminal,
create `logs/SKIP`, or `POST /admin/skip`. The application is recorded as skipped with
the reason, and the agent moves to the next one.

---

## When a scan finds nothing

Every scan counts what it examined and what each filter removed, so an empty result explains
itself instead of leaving you guessing. The strip above the results reads like this:

```
examined 6739    kept 25    search terms 6714
```

and when nothing survives, it names the filter responsible and what to do:

> **Nothing matched.**
> 4789 of 4807 postings (100%) did not match your search terms (Data Analytics, Data Analysis).
> Try fewer or broader terms.
> 18 postings were the wrong seniority. You have intern selected; untick to allow any level.

The same breakdown appears in the activity log, in `python main.py discover`, and at
`GET /api/scan-stats`. The filters apply in this order, and each one reports separately: search
terms, seniority, location and country, already-applied, missing description, and for aggregators,
no application link behind the listing.

---

## Deleting applications

Deleting a row removes the dedupe record too, so the posting becomes eligible for discovery again.
That is how you retry something that failed.

In the **Applications** tab: the × on any row, the checkboxes plus **Delete selected**, or
**Delete all shown**, which respects the current status filter. Tick **Also delete the generated
PDF and screenshot files** to remove those from disk as well.

```bash
python main.py delete 42                    # one application
python main.py delete 42 43 44 --files      # several, plus their PDFs and screenshots
python main.py delete --status failed       # every failed attempt, so they can be retried
python main.py delete --all --yes           # start over
```

Every form asks for confirmation first, and prints what it is about to remove.

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
| `make serve` | `serve` | Web dashboard and API on `127.0.0.1:8000`. `make serve PORT=8001` to move it |
| `make settings` | `settings` | Show every setting and whether it came from `.env` or the UI |
| `make scan` | `discover` | Preview what the sources would find. Applies to nothing |
| `make scan-save` | `discover --save jobs.txt` | Scan and write the URLs to a file |
| `make run` | `run` | Assisted mode. Bot fills, you submit |
| `make run-auto` | `run --auto-submit` | Auto mode |
| `make analyze URL=…` | `analyze --url` | Score one posting and build the PDF, apply to nothing |
| `make models` | `models` | List the models the provider offers |
| `make companies` | `companies --probe` | Count open roles on each company board |
| | `delete` | Remove applications so their postings can be retried |
| | `countries-list` | The 53 country names the location filter accepts |
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
| Delete | `DELETE /api/applications/{id}`, `POST /api/applications/delete` |
| Files | `GET /api/applications/{id}/resume`, `/screenshot`, `/api/export.csv`, `/api/audits` |
| Discover | `POST /admin/discover`, `GET /api/discovered`, `GET /api/scan-stats`, `POST /api/detect` |
| Apply | `POST /admin/run`, `/admin/apply-selected`, `/admin/stop` |
| Control | `GET /api/run`, `/api/gate`, `POST /admin/continue`, `POST /admin/skip` |
| Settings | `GET|PATCH /api/config`, `POST /api/config/reset`, `GET /api/models`, `POST /api/models/test` |
| Data | `GET|PUT /api/profile`, `GET|POST /api/companies`, `DELETE /api/companies/{ats}/{token}` |
| Boards | `POST /api/companies/resolve`, `/api/companies/add-from-url`, `GET /api/companies/probe` |

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
| `matching.py` | Title relevance: concepts, weights and the resemblance score |
| `countries.py` | Country names, their aliases and cities, and regional shorthand |
| `reports.py` | Terminal dashboard rendering |
| `api.py` | FastAPI app: every flow the CLI has |
| `static/index.html` | The web dashboard. One store drives every page |
| `settings.local.json` | Choices made in the UI. Delete it to fall back to `.env` |
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
| A scan finds nothing | Read the strip above the results. It names the filter that dropped everything |
| A search finds far fewer than the board shows | Fixed: terms are scored for resemblance. Lower Match sensitivity to widen further |
| Only the job description opens, never the form | Fixed: Greenhouse now opens the embed form directly, which never redirects |
| A CAPTCHA appears before anything is filled | Fixed: the form is filled first, and the challenge is handled at submit time |
| A challenge never clears | Use **Skip this job**, type `s` in the terminal, or create `logs/SKIP` |
| The sensitivity slider snaps back | Fixed: controls you are editing are no longer overwritten by the refresh |
| A model you want is not in the list | Type its id anyway and press **Test this model**. The answer is definitive |
| A run fails with a model error | Test the model. 410 means retired, 404 means wrong id, 503 means try again shortly |
| A country returns nothing | Check the location strings with `python main.py discover`. Combining a country with remote-only is strict by design |
| A scan finds far too much | Use more specific terms, or narrow the seniority levels |
| Hybrid roles show up as remote | Fixed: an ATS's own remote flag is no longer trusted on its own |
| A posting will not be retried | It is already in the database. `python main.py delete <id>` frees it |
| `make: *** [serve] Terminated: 15` | Something sent the server SIGTERM. Usually a `pkill` matching `main.py serve`, or a second copy starting on the same port. `make serve PORT=8001` runs another one safely |
| Port already in use | The dashboard now says so and suggests the next port instead of raising |
| A setting will not stick | Check `python main.py settings`. Only the keys it lists are saved; the rest come from `.env` |
| `Could not launch browser channel 'chrome'` | Harmless. It falls back to bundled Chromium, or set `BROWSER_CHANNEL=` |
| The bot pauses constantly on one site | Those fields are not resolvable from your profile. Extend `screening_defaults` |

Logs: `logs/xapply.log`. Audits: `logs/applications/*.json`. Screenshots: `logs/*.png`.

---

## The interface

Sidebar navigation, five pages, light and dark following your system setting. The palette is black,
white and a royal blue accent, with pure neutral greys so the accent is the only colour doing
decorative work. Status still carries its own colour, because green, amber and red tell you at a
glance whether an application went through, is waiting for you, or failed, and that is information
rather than decoration.

The web UI and the CLI are equals. Anything you can do in one, you can do in the other, and they
read the same state.

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

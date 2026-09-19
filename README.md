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
| `google` | none | `site:job-boards.greenhouse.io "Backend Engineer"` through Playwright. Google hides result URLs behind opaque links, so the source reads the company board off each result and pulls the jobs from that board's API. New boards it confirms are added to `companies.json`, so the next scan needs no search at all |
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

## The three execution modes

Chosen on the Dashboard, next to the run button, and saved with the rest of your settings.

| Mode | What the agent does | What you do |
| --- | --- | --- |
| **Documents only** | Attaches your tailored resume, and a cover letter when the form asks for one | Fill in every other field, and submit |
| **Assisted** *(default)* | Fills every field it can answer truthfully, then stops | Check it, and submit |
| **Auto** | Fills the form and presses Submit | Nothing, unless it pauses |

```bash
python main.py run --mode documents
python main.py run --mode assisted     # the default
python main.py run --mode auto
```

`AUTO_SUBMIT=true` in an old `.env` still means auto mode.

### Documents only

The agent's whole job is the paperwork. It reaches the form, attaches the tailored resume, looks
for a cover letter field, and stops. It types nothing else, not even your name, so nothing goes
into the application that you did not put there.

A cover letter is written only when a form actually asks for one, either as an upload or as a
free-text box, so no API call is spent on the majority of applications that never request one.
It is grounded in the same profile as the resume, told not to claim anything the profile does not
support, and told not to mention the requirements the analysis found you do not meet. The letter
is rendered to its own single-page PDF for an upload field, or entered whole into a text box.

Whichever way you submit, the agent is still watching, so the application is recorded as
*Submitted* with the evidence rather than left as *Awaiting human*:

```
Documents attached; you submitted it: confirmation text 'Thank you for applying'
```

---

## What a pause looks like

**Assisted mode.** The bot navigates, uploads the tailored resume,
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
| `nvidia` | [build.nvidia.com](https://build.nvidia.com) | Free, OpenAI-compatible, 82 listed models |
| `opencode` | [opencode.ai](https://opencode.ai) | One key reaches Claude, GPT, Gemini, DeepSeek, Qwen, Grok and more |

```bash
# .env, keys only
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
OPENCODE_API_KEY=sk-...
```

**OpenCode Zen** lists 74 models across every major vendor, but two things gate them:

- Models whose id ends in `-free` answer **403: "OpenCode's free tier can only be used from
  within OpenCode"**. That restriction is enforced on their side and no request header changes
  it, so `mimo-v2.5-free` and the other free ids cannot be used from here.
- Every other model answers **401: "No payment method"** until you add one to your workspace.

Both are reported in those words rather than as a generic failure, so you know which one you are
looking at. NVIDIA remains the free option.

```bash
python main.py models --provider nvidia --search nemotron   # browse what is available
python main.py run --provider nvidia --model openai/gpt-oss-20b
```

**Custom request headers.** Under **Settings**, *Custom request headers* takes a JSON object sent
with every LLM request, for a gateway that wants a `User-Agent`, a `Referer` or an app title:

```json
{"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0", "X-Title": "XApply"}
```

It can also be set as `LLM_EXTRA_HEADERS` in `.env`. Invalid JSON is refused rather than silently
ignored.

**Picking a model.** The provider and model controls sit on the **Dashboard**, next to the run
buttons, and under **Settings**. The model field is a search box: type any part of a name to
filter, use the arrow keys and Enter, or paste an id the list does not contain. A dropdown of
eighty entries is unusable, and no list is ever guaranteed complete, so both are handled.

**When a model is on the provider's website but not here.** A website name and an API id are not
always the same string, and some models are never exposed through the API at all. A failed test
now says which: it offers the closest real ids as one-click alternatives, and when nothing is
close it says so outright, for example *"Nothing similar among the 82 models nvidia serves, so
this one is not reachable through its API."*

**Test this model** settles it by sending one tiny prompt and reporting what came back. That
matters because a listed model can still be retired or overloaded:

| Model | Result |
| --- | --- |
| `openai/gpt-oss-20b` | Answered a test prompt |
| `meta/llama-3.3-70b-instruct` | 410, retired by the provider |
| `stepfun-ai/step-3.5` | 404, no such model on this endpoint |
| `nvidia/nemotron-3-super-120b-a12b` | 503 when busy, works otherwise |

**A listed model is not necessarily a working one.** NVIDIA's endpoint lists 82 models. Twenty-one
are embedding, safety, reward or parsing models that cannot hold a conversation, leaving 61. Of
those 61, **only nine actually answer**; the rest return 404, because the catalogue advertises
models NVIDIA has not deployed. Picking one of those wastes an entire run.

So press **Check which models work** under Settings, or run `python main.py models --verify`. Each
model is called once, the working ones are marked in the picker, and the undeployed ones are
greyed out and labelled. A busy model answering 503 counts as working, because that is a wait
rather than a wrong choice.

Before any run starts, the configured model is checked once. If it cannot be used the run stops
immediately with one message and a **Find a model that works** button, instead of recording the
same failure against every posting.

**Keys live in `.env`; the model lives in the dashboard.** `GEMINI_MODEL` and `NVIDIA_MODEL` are
commented out of `.env.example` because the model is chosen in Settings and saved to
`settings.local.json`, so it changes without editing a file or restarting. The terminal can still
override it per run with `--model`.

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

## Recording a submit you made yourself

In assisted mode you press Submit, so the agent has to notice that and record it. It watches the
page the whole time you are working on it, and any one of four independent signals counts:

| Signal | Example |
| --- | --- |
| A confirmation phrase | "Thank you for applying", "You're all set", "We'll be in touch" |
| A confirmation URL | `/thank-you`, `/success`, `/submitted`, `/confirmation` |
| The form disappearing | The one signal every ATS shares, whatever wording it uses |
| Navigating away with the Submit button gone | A redirect to a page that has no form |

The form disappearing matters most, because it works on a site that says nothing at all. Filling
fields in never triggers it, only the form going away does.

Whichever signal fires is written into the application's note, so **Applications** shows
*Submitted* with the reason rather than leaving it as *Awaiting human*. If nothing is seen, it
stays *Awaiting human* honestly, and you can correct it from the detail panel.

Every pause does this, not just the review step: an unanswerable required field, a validation
error, a missing Submit button, and LinkedIn's review page. On LinkedIn the Easy Apply dialog
closing is itself proof.

---

## CAPTCHAs: fill first, solve last

An application form very often carries its own inline reCAPTCHA or Turnstile widget.
Stopping for it on arrival means you are shown a CAPTCHA next to a set of empty boxes.

So while navigating, the agent only stops for something that genuinely blocks the page:
a login wall, or a challenge on a page with no form on it. An inline widget is noted and
ignored, the form is filled, and the full check runs immediately before submitting, which
is the only moment the CAPTCHA has to be solved.

The agent does not solve CAPTCHAs. A CAPTCHA is the site asking whether a person is present, and
on a job application the honest answer has to be yes. It would not work anyway: reCAPTCHA,
hCaptcha and Turnstile issue tokens from behavioural signals rather than from the picture, so
reading the image produces nothing usable.

What reduces them instead: Greenhouse applications open the embed form directly, skipping the
company marketing pages that carry most of the widgets, and the browser profile persists so you
are a returning visitor rather than a stranger every run.

**When a challenge will not clear**, skip the posting rather than holding up the run:
click **Skip this job** in the dashboard banner, type `s` then Enter in the terminal,
create `logs/SKIP`, or `POST /admin/skip`. The application is recorded as skipped with
the reason, and the agent moves to the next one.

---

## Stopping a scan

A scan across thirty company boards takes minutes. Results are published as each board
finishes, so the table fills while it runs and the counters move with it. Pressing **Stop**
keeps everything found up to that point rather than discarding it, and the activity log records
what was kept:

```
scan stopped early; keeping the 20 posting(s) found so far (1713 examined before the stop)
```

The diagnostics strip shows *still scanning* while a scan is in progress, so a partial count is
never mistaken for a final one.

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
### Trying an application again

Failed attempts and ones still waiting on you can be run again from the start. The old record is
replaced by the new attempt, because discovery skips anything already in the database, so leaving
it would mean the retry quietly did nothing. The audit file in `logs/applications/` keeps what
happened the first time.

In the **Applications** tab: the circular arrow on any failed or awaiting row, **Retry selected**,
or filter by Failed or Awaiting you and press **Retry all shown**.

```bash
python main.py retry 42                       # one application
python main.py retry --status failed --yes    # every failed attempt
python main.py retry --status pending_human_review
```

### Working without a window

By default a browser window is on screen, because in assisted mode you are the one who
presses Submit, and because only a person can clear a CAPTCHA.

Two settings change that. **Hide the browser window** works with nothing on screen.
**When a CAPTCHA or login wall blocks a posting** then decides what happens when one
does appear, and applies while applying, not while scanning:

| Choice | What happens |
|---|---|
| Stop and let me deal with it | The run pauses. Needs a window, so with none it skips instead and says so. |
| Open a window and start that posting again | The browser restarts with a window. Your logins survive, because the profile does. |
| Skip that posting and carry on | The application is recorded as skipped, with the challenge as the reason. |

A scan over the board APIs opens no browser at all, whatever these are set to, because
it is plain HTTP. Applying to a web form always needs one, hidden or not. Applying by
email needs no form, so nothing is opened for it.

### Signing in to a site, once

Sessions live in the browser profile, exactly as they do in your everyday browser. Sign
in once and it lasts until you sign out or delete the profile:

```bash
make signin SITE=x          # or linkedin, indeed, glassdoor, wellfound, gmail
make signin SITE=https://any-site.example/login
```

Nothing is typed for you and no password is stored in this project. You sign in
yourself, including any second factor.

### Finding roles that have no form

Source `emails`. A great deal of hiring never reaches an applicant tracking system:
someone writes "we are hiring a data analyst, send your CV to careers@example.com" on
their own site, and that is the whole process. No board API can see those, because there
is no board.

This searches for the wording people use when they do that, opens each result, and keeps
the pages that name an address. Those are then applied to by email, so it needs **Apply
by email** on. Tick **Also search X** to include X as well, which needs you signed in
there.

### Applying by email

Plenty of roles never reach an applicant tracking system. The posting says "send your CV
to careers@example.com" and that is the whole process. Turn on **Apply by email** in
Settings and add your mail server to `.env`, and a posting with no form is applied to by
writing to the address it gives.

There are two ways to send. **Your signed-in Gmail** is the simpler one:

```bash
make gmail-login          # sign in once, in a window, including two-factor
```

Nothing is typed for you and no password is stored in this project. The browser profile
keeps the session, the message is composed in a real Gmail window, and it lands in your
Sent folder where you can see it and reply from it. Set `EMAIL_TRANSPORT=gmail`.

The other way is **a mail server**, with `SMTP_HOST`, `SMTP_USER` and `SMTP_PASSWORD` in
`.env`. For Gmail that password is an app password, not your account password, which is
the reason the browser route exists.

It is off by default, and stays careful when it is on:

- The recipient is only ever an address found in the posting. It is never guessed, and
  addresses like `noreply@`, `privacy@` and `support@` are refused.
- The run stops and shows you the draft before anything is sent. The whole message is
  written to `logs/emails/` so you can read it in full.
- `EMAIL_AUTO_SEND=true` skips that pause. It is a separate setting from turning the
  feature on, because they are different decisions.
- Your mail credentials live in `.env` only. They are never written to
  `settings.local.json` and never sent to the browser. With the Gmail route there are
  none to leak.

### Open questions on a form

Forms ask the same handful of things in a thousand wordings: how you work remotely,
whether you have public code, why this company, tell us about a time. `question_bank.json`
records those patterns, what each one is really asking, and which part of your profile
answers it. The answer itself is composed from your own experience and projects every
time, so the file cannot put words in your mouth or claim something you did not do.

Each entry is marked `prose` or `short`. That decides whether the question may be
answered from a stored one-liner at all. Without it, "Tell us about your experience
working in a remote environment" was answered with the profile's remote preference, the
single word "Remote", because a lookup rule spotted the word and the model was never
reached. A `prose` question now always goes to the model, while "Are you open to remote
work?" is still answered instantly from your profile.

It is yours to edit. Add phrases to teach it a wording it does not know:

```bash
python main.py question "Tell us about a time you disagreed with a colleague"
```

That prints which archetype the wording matched, the guidance sent to the model, and the
runner-up scores, so you can see why a question was read the way it was.

### Scan results

What a scan finds is kept in the database, not in the browser, so a ten-minute scan survives a
restart and the terminal and the dashboard always show the same list. Removing a posting from
this list does not touch your applications.

```bash
python main.py saved                          # everything scans have found
python main.py saved --search stripe          # filter by company, role or link
python main.py saved --delete gh-4109216      # remove one
python main.py saved --clear                  # remove all of them
```

In the **Scan** tab: tick rows and press **Delete selected**, or **Clear results**.


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
| `make dev` | | Dashboard with autoreload, for editing the code or the UI while it runs |
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
| Files | `GET /api/applications/{id}/resume` returns the tailored PDF; the cover letter sits beside it in `output_resumes/` |
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
| `resume_builder.py` | Resume and cover letter: Jinja2 to Chromium to single-page PDF, hallucination filter |
| `templates/cover_letter.html` | The cover letter layout, in the same typeface as the resume |
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
| `assets/fonts/` | Bricolage Grotesque, served locally and embedded in every PDF |
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

**Typography.** Bricolage Grotesque, the same face as the dashboard. The font files live in
`assets/fonts/` and are served by the app, so nothing is fetched from a CDN at runtime and nothing
is requested from a third party. The PDF embeds separate static weights rather than the variable
font, because Chromium renders a variable font into a PDF at its lightest instance, which left
every resume in ExtraLight. The licence is recorded in `assets/fonts/README.md`.

**ATS-readable text.** Everything written into a resume or typed into a form is flattened to plain
ASCII: a model writes "Full\u2011stack" with a non-breaking hyphen freely, and a recruiter searching
for "Full-stack" would never find it. Font ligatures are switched off for the same reason, so "fi"
is not extracted as a single glyph. A generated resume contains no character above U+007F.

**Resume length.** Two pages by default. The PDF is rendered, its page count measured, and if it
overflows the content is reduced and rendered again, cheapest first: fewer bullets per entry, then
fewer projects, then fewer bullets again. Nothing is invented or reworded to make it fit.

Work history is never dropped unless you allow it, under **Settings**. Losing a job from a CV is
not a formatting decision. Whatever was left out is written onto the application as a note, so a
shortened resume is never a surprise:

| Setting | A six-year, five-role profile |
| --- | --- |
| Two pages *(default)* | 5 of 5 roles, 8 of 8 projects, nothing trimmed |
| One page | 5 of 5 roles, projects removed, and it says so |

Before this, one page was forced on every resume and it silently cost 2 of the 5 roles and 7 of
the 8 projects.

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
| You submitted but it says *Awaiting human* | Fixed: the page is watched throughout, so a manual submit is recorded. You can also correct any row from its detail panel |
| A challenge never clears | Use **Skip this job**, type `s` in the terminal, or create `logs/SKIP` |
| The sensitivity slider snaps back | Fixed: controls you are editing are no longer overwritten by the refresh |
| A model you want is not in the list | Type its id and press **Test this model**. It offers close matches, or says nothing is close |
| A run fails with a model error | Press **Find a model that works**, or run `python main.py models --verify` |
| Every posting fails with a 404 | The model is listed but not deployed. Only nine of NVIDIA's 61 chat models answer |
| The model changed by itself | Fixed: the picker saves only a deliberate choice, and the server refuses a model that cannot answer |
| An OpenCode `-free` model gives 403 | Its free tier only works inside OpenCode's own client. Use a paid model, or NVIDIA |
| An OpenCode model gives 401 | The workspace has no payment method |
| A country returns nothing | Check the location strings with `python main.py discover`. Combining a country with remote-only is strict by design |
| A scan finds far too much | Use more specific terms, or narrow the seniority levels |
| Hybrid roles show up as remote | Fixed: an ATS's own remote flag is no longer trusted on its own |
| A posting will not be retried | It is already in the database. `python main.py delete <id>` frees it |
| Stopping a scan lost the results | Fixed: results are published board by board and survive a stop |
| A cover letter came out truncated | Fixed: long prose is entered at once rather than typed key by key, which hit the action timeout |
| The resume is missing roles or projects | It was trimmed to fit. Raise **Resume length** under Settings; the application records what was left out |
| No cover letter was produced | One is written only when the form asks for it. Check the form has a cover letter field |
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
decorative work. The typeface throughout, dashboard and resume alike, is **Bricolage Grotesque**. Status still carries its own colour, because green, amber and red tell you at a
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

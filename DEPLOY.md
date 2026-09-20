# Running XApply on a server

Read this before deploying. Several things this tool does cannot work on a server, and
knowing which ones first will save you an afternoon.

## The blocker, in one sentence

A server has no screen, and a good part of this tool exists to put a browser in front of
you.

## What that rules out

| Feature | On a server |
|---|---|
| Assisted mode, where you press Submit | **No.** Nobody can see the form. Use auto mode or nothing is ever submitted. |
| Clearing a CAPTCHA or a login wall | **No.** Only a person at a browser can. Those postings are skipped. |
| `make signin`, for Gmail, LinkedIn, X | **No.** It opens a window for you to sign in, and there is no window. |
| Sending through your signed-in Gmail | **No.** It drives a real Gmail session that you can only create by signing in. Use SMTP. |
| Searching X | **No**, for the same reason: X refuses anonymous searches. |
| Opening a window when something needs you | **No.** There is nothing to open. |

## What works well

| Feature | On a server |
|---|---|
| Scanning Greenhouse, Lever and Ashby boards | **Yes.** Plain HTTP, no browser involved at all. |
| The dashboard, the database, your application history | **Yes.** |
| Tailoring CVs and cover letters | **Yes.** |
| Applying by email over SMTP | **Yes.** |
| Filling and submitting forms, in auto mode | **Mostly.** Headless works, but a CAPTCHA ends that posting, and Google blocks a data-centre address more often than your home one. |

A reasonable split is to run scanning and email applications on the server, and keep
anything that needs your eyes on your own machine, pointed at the same database.

## Deploying to Render

Everything below is in `Dockerfile` and `render.yaml`.

### 1. Push the repository to GitHub

Check that `.env` is **not** committed. It holds your keys.

```bash
git status --short          # .env must not appear
```

### 2. Create the service

In Render, choose **New > Blueprint** and point it at the repository. It reads
`render.yaml` and creates a Docker web service with a disk.

Doing it by hand instead: New > Web Service, runtime **Docker**, and add a disk mounted
at `/data`.

### 3. Set the secrets

In the Render dashboard, under Environment:

| Variable | Why |
|---|---|
| `ADMIN_TOKEN` | Required. The app refuses to listen on a public address with the default, because that would put your applications on the open internet. |
| `GEMINI_API_KEY` or `NVIDIA_API_KEY` | Whichever provider you use. |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM` | Only if you want email applications. For Gmail, `SMTP_PASSWORD` is an app password. |

### 4. Upload your profile

`profile.json` is yours and is not in the repository. Put it on the disk once:

```bash
# from the Render shell, or commit a copy without personal details and edit it there
cat > /data/profile.json
```

`companies.json` works the same way, or let the Google source fill it in.

### 5. Open it

Render gives you a URL. Add your token:

```
https://your-service.onrender.com/?token=YOUR_ADMIN_TOKEN
```

## Things that will bite you

**The disk is not optional.** Without it, every deploy and every restart wipes your
applications, your generated CVs and anything you were signed in to. The blueprint
mounts one at `/data` and points the database, the output folder and the browser profile
at it.

**The free plan is too small.** Chromium and the app together will not fit reliably in
512 MB, and a free service sleeps after fifteen minutes idle, which kills a scan
part way. The blueprint asks for `standard`.

**Google blocks data centres more readily.** The `google` and `emails` sources will meet
the bot check more often than they do on your own connection. The board APIs do not use
a search engine and are unaffected.

**Run it in auto mode or not at all.** In assisted mode the run stops for a person who
is not there. `render.yaml` sets `FILL_MODE=auto` and `CHALLENGE_ACTION=skip` for that
reason. Auto mode sends real applications with nobody checking them first, so make sure
you are happy with what it produces locally before you turn it loose.

## If you want the sessions too

You can copy a browser profile you signed in to locally onto the disk, and the server
will use those sessions until they expire:

```bash
tar czf profile.tgz .browser_profile
# upload, then on the server:
tar xzf profile.tgz -C /data && mv /data/.browser_profile /data/browser_profile
```

Be clear about what that is: you are putting live session cookies for your Gmail,
LinkedIn and X accounts on a hosted machine. Anyone who can reach that disk can act as
you on those accounts. If you do it, use a dedicated account rather than your main one.

## Running it locally instead

Nothing above applies. Everything works, including the parts that need you:

```bash
make install
make serve
```

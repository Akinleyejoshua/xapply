# Playwright's own image, because a browser needs a long list of system libraries and
# getting one of them wrong shows up as a timeout rather than a missing package.
# Pin the version to the playwright in requirements, or the browsers will not match.
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Everything that must survive a deploy lives here. Mount a disk at this path or
    # every application, every generated CV and every signed-in session is wiped when
    # the service restarts.
    XAPPLY_DATA=/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # The base image ships browsers for its own Playwright version. requirements.txt
    # may install a different one, and a mismatch shows up as a launch failure, so the
    # matching browser is fetched after the install rather than assumed.
    && playwright install chromium

COPY . .

# The profile, database and output live on the mounted disk rather than in the image.
ENV DB_PATH=/data/applications.db \
    OUTPUT_DIR=/data/output_resumes \
    LOG_DIR=/data/logs \
    AUDIT_DIR=/data/logs/applications \
    USER_DATA_DIR=/data/browser_profile \
    OVERRIDES_PATH=/data/settings.local.json \
    PROFILE_PATH=/data/profile.json \
    COMPANY_FILE=/data/companies.json \
    # No display exists here, so a window is never an option.
    HEADLESS=true \
    HIDE_BROWSER=true \
    CHALLENGE_ACTION=skip \
    API_HOST=0.0.0.0

EXPOSE 8000
CMD ["python", "main.py", "serve", "--host", "0.0.0.0"]

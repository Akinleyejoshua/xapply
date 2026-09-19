# Bricolage Grotesque

The project typeface, used in the dashboard and in every generated resume.

| File | Used by | Why |
| --- | --- | --- |
| `BricolageGrotesque.woff2` | dashboard | Variable font, small, weights interpolate live in a browser |
| `BricolageGrotesque.ttf` | dashboard fallback | For a browser that will not take the woff2 |
| `BricolageGrotesque-Regular.ttf` | resume PDF | 400 |
| `BricolageGrotesque-SemiBold.ttf` | resume PDF | 600 |
| `BricolageGrotesque-Bold.ttf` | resume PDF | 700 |

The PDF uses separate static weights rather than the variable font. Chromium renders a
variable font into a PDF at its lightest instance, which left every resume in ExtraLight.

Designed by Mathieu Triay. Licensed under the SIL Open Font License 1.1, which permits
bundling and embedding in documents: <https://openfontlicense.org>.
Source: <https://fonts.google.com/specimen/Bricolage+Grotesque>.

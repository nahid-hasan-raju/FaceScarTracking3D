# Going live: GitHub repo + Google Drive + a free host

One correction up front: **GitHub itself only hosts static files** (that's
what GitHub Pages is for) — it can't run a Python/Flask server. What you
actually want is what most people mean by "push to GitHub and it's live":
a **GitHub repo connected to a small hosting service (Render)** that
auto-deploys every time you push. That's the setup below — free tier,
no server to manage, no login screen.

Total time: ~20 minutes, one-time setup.

---

## Part 1 — Google Drive access (service account)

A **service account** is a robot Google account your app logs in as —
no browser login flow, no OAuth screen for you or anyone else.

1. Go to https://console.cloud.google.com/ and create a new project (or
   use an existing one). Name doesn't matter, e.g. `burn-editor`.
2. In the search bar, find **"Google Drive API"** → click **Enable**.
3. Go to **APIs & Services → Credentials → Create Credentials → Service account**.
   - Name: `burn-editor-bot` (anything).
   - Skip the optional role/access steps — click through to **Done**.
4. Click on the service account you just created → **Keys** tab →
   **Add Key → Create new key → JSON** → it downloads a `.json` file.
   **Keep this file private — never commit it to GitHub.**
5. Open that JSON file, copy the `"client_email"` value
   (looks like `burn-editor-bot@your-project.iam.gserviceaccount.com`).
6. In Google Drive, right-click your dataset root folder (the one
   containing `PAT01/`, `PAT02/`, ...) → **Share** → paste that email
   address → give it **Editor** access → Send (it's fine that it's a
   robot account, ignore the "no Google account" warning if it appears).
7. Get the folder's ID: open the folder in Drive, look at the URL —
   `https://drive.google.com/drive/folders/`**`1AbC...xyz`** — that long
   string after `/folders/` is your `DRIVE_ROOT_FOLDER_ID`.

You now have two things you'll paste into Render in Part 3:
- The full contents of the downloaded JSON key file (`GOOGLE_CREDENTIALS_JSON`)
- The folder ID from step 7 (`DRIVE_ROOT_FOLDER_ID`)

---

## Part 2 — Push this project to GitHub

From the folder you downloaded (containing `app.py`, `drive_storage.py`,
`templates/`, `static/`, `requirements.txt`, `Procfile`):

```bash
cd path/to/this/folder
git init
git add .
git commit -m "Live version: Drive-backed polygon editor"
```

Then create a new **empty** repo on github.com (don't add a README/gitignore
there, you already have one), and:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Double-check `.gitignore` kept your service-account JSON key out of the
repo (it will, as long as you didn't rename it away from the ignored
patterns) — **never commit that file**.

---

## Part 3 — Deploy on Render (free tier, auto-deploys from GitHub)

1. Go to https://render.com → sign up/log in (can use your GitHub account).
2. **New → Web Service** → connect your GitHub account → pick the repo
   you just pushed.
3. Render should auto-detect Python. Confirm/set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free
4. Scroll to **Environment Variables** and add two:
   - `GOOGLE_CREDENTIALS_JSON` → paste the *entire contents* of the
     service-account JSON file (open it in a text editor, select all, paste).
   - `DRIVE_ROOT_FOLDER_ID` → the folder ID from Part 1, step 7.
5. Click **Create Web Service**. First deploy takes 2-5 minutes. Render
   gives you a URL like `https://burn-polygon-editor.onrender.com`.

From now on, **every `git push` to `main` auto-redeploys**. No manual
redeploy step.

(A `render.yaml` is included in the repo — if you'd rather use Render's
"Blueprint" one-click flow instead of the manual dashboard steps above,
use **New → Blueprint** and point it at the repo; it reads that file and
sets up the same thing, just prompting you for the two secret values.)

---

## Notes / things worth knowing

- **No login screen**, per what you asked for — anyone with the URL can
  view and edit polygons. If you ever want a lightweight gate, a single
  shared password check is a small addition, just say the word.
- **Free tier sleeps after inactivity** — the first request after a
  while takes ~30-50 seconds to wake back up. Fine for occasional use;
  upgrade to a paid instance later if that's annoying.
- **Drive API rate limits**: the tree walk is cached for 20 seconds
  (see `TREE_TTL_SECONDS` in `drive_storage.py`) so repeated page loads
  don't re-walk the whole dataset every time. Raise that number if your
  dataset is large and you want fewer Drive API calls.
- **Backup behavior is unchanged**: saving over an untouched SAM2 json
  still creates a one-time `*_burn_polygons.sam2_backup.json` next to it
  in the same Drive folder, exactly like the local version did.
- If you add new patients/scans directly in Drive, they'll show up
  within `TREE_TTL_SECONDS` of your next page load — no redeploy needed.

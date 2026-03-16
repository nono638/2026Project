  # New Project Setup — Step by Step

Do this once per project. Takes about 5 minutes.

---

## 1. Create your project folder

Make a new folder wherever you keep projects. Example:
```
C:\Users\noahc\Dropbox\NotWork\MyNewProject
```

---

## 2. Create a GitHub repo

1. Go to https://github.com/new
2. Name it whatever you want (e.g., `MyNewProject`)
3. Leave it **empty** — do NOT add a README, .gitignore, or license (we'll push from local)
4. Click **Create repository**
5. Keep the page open — you'll need the URL in step 4

---

## 3. Copy the setup folder

Copy the entire `_claude_sandbox_setup/` folder into your new project folder.

Your folder should now look like:
```
MyNewProject/
└── _claude_sandbox_setup/
```

That's it for now. Claude will set up everything else.

---

## 4. Open a terminal in your project folder

Right-click your project folder in File Explorer → **"Open in Terminal"**.

---

## 5. Initialize git and connect to GitHub

Type these commands one at a time. Replace the URL with the one from your GitHub repo page.

```
git init
git add -A
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

**What each line does:**
- `git init` — turns this folder into a git repository
- `git add -A` — stages all files
- `git commit -m "initial commit"` — saves the first snapshot
- `git branch -M main` — renames the default branch to `main`
- `git remote add origin https://...` — connects your local repo to GitHub
- `git push -u origin main` — uploads everything to GitHub

**If git asks for login:** a browser window should pop up to authenticate with GitHub.
If it asks for a username/password in the terminal instead, you need a Personal Access
Token — go to https://github.com/settings/tokens, generate one, and paste it as the password.

---

## 6. Launch Claude and run setup

```
claude
```

Once Claude starts, tell it:

> Read `_claude_sandbox_setup/SETUP.md` and follow all the instructions.

Claude will:
- Set up CLAUDE.md with the right imports
- Install settings, hooks, and the handoff directory structure
- Lock the directory guard to your project path
- Run tests to verify everything works

This takes a minute or two. Just let it work.

---

## 7. Done — start using it

You're set up. From now on:

| What you want to do | Command |
|---------------------|---------|
| Daytime session (you're here) | `bash _claude_sandbox_setup/scripts/dayrun.sh` |
| Nighttime session (walk away) | `bash _claude_sandbox_setup/scripts/nightrun.sh` |
| Something seems broken | `bash _claude_sandbox_setup/scripts/repairrun.sh` |

If `bash` isn't recognized, use the `.bat` versions:
`_claude_sandbox_setup\scripts\dayrun.bat`, `nightrun.bat`, `repairrun.bat`

See `CHEAT_SHEET.md` for the full reference.

---

## Troubleshooting

**"git is not recognized"**
Install Git for Windows: https://git-scm.com/download/win
Use all the default options. Close and reopen your terminal after installing.

**"claude is not recognized"**
Install Claude Code: `npm install -g @anthropic-ai/claude-code`
You need Node.js first if you don't have it: https://nodejs.org

**"python is not recognized"**
Install Python: https://www.python.org/downloads/
Check "Add to PATH" during installation. Close and reopen your terminal.

**GitHub asks for username/password in the terminal**
Passwords don't work anymore. You need either:
- **GitHub CLI** (easiest): Install from https://cli.github.com, then run `gh auth login`
- **Personal Access Token**: https://github.com/settings/tokens → Generate new token →
  check `repo` scope → copy the token → paste it as the password when git asks

**Push is rejected because remote has commits you don't have**
You probably added a README or license on GitHub when creating the repo. Easiest fix:
```
git pull origin main --allow-unrelated-histories
git push -u origin main
```

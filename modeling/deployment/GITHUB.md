# GitHub handoff

The local repository is initialized on branch `main`. No commit, remote, push
or release was created. Source/configuration, tests and documentation are Git
candidates. Data, artifacts, caches and environments remain on disk and ignored.

From the project root in PowerShell, after creating your chosen empty GitHub
repository (without a separate initial README/license):

```powershell
.\.venv\Scripts\python.exe -m modeling.deployment github-check
git status --short
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare CPSS local search and portable deployment"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Choose visibility and use your own Git identity if requested. These are handoff
commands, not actions performed for you. Never force-add ignored credentials or
generated assets. The preflight reads only Git-visible candidates, rejecting
known credential filenames, generated assets and source files over 10 MiB. It
checks several credential patterns without printing matched values. Ignored
`.env` and `token.txt` are never read. This heuristic is not a complete secret audit.

`.gitattributes` normalizes source line endings for Linux/Windows. GitHub Actions
installs pinned dependencies, runs offline/synthetic tests on Linux Python
3.12/3.13 and checks JavaScript syntax. It needs no corpus, model download,
secrets or MongoDB. Real MongoDB integration tests remain an explicit local
opt-in using disposable databases. CI has read-only repository permissions and
no automated deployment. The workflow is prepared but has not run on GitHub yet.

After choosing a matching source commit, create a GitHub tag/release and attach
`bundle-HASH.zip` and its `.zip.sha256` from `modeling/artifacts/runtime/`.
Record the digest independently in Replit's `CPSS_BUNDLE_SHA256`. Release assets
keep large binaries out of Git history, following GitHub's
[large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
A clone without a bundle can run synthetic tests but cannot search the corpus.

No license for CPSS's own source was selected automatically. Decide one with
the project owners before describing the repository as open source. Dataset and
model terms remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
bundle, including the full Apache 2.0 model license text.

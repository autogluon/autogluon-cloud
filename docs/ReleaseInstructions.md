# Release process

## Pre-release

* Ensure `master` is stable: CI is green, no in-flight critical PRs.
* Confirm `setup.py` has the intended release version (e.g. `version = "0.5.0"`).
* Sanity-check dependency version ranges in `setup.py`:
  * No major-version caps without an inline comment justifying it.
  * Every dependency has both a lower bound and an upper cap (using `<`, not `<=`).
  * Caps are at the minor level (`<x.y`), not micro (`<x.y.z`).
  * No exact pins (`==x.y.z`) without an inline comment.
* Optionally bump upper caps to include the latest stable releases of dependencies and verify CI still passes.

## Update the `stable` docs branch

The `stable` branch backs `https://auto.gluon.ai/cloud/stable/`. Recreate it from `master` so the docs CI builds against the release commit:

```bash
git checkout master && git pull
git push origin --delete stable
git switch -c stable
git push -f origin stable
```

Watch the `stable` CI run and confirm `auto.gluon.ai/cloud/stable/` updates. If the docs don't refresh because the new `stable` SHA matches `master`, push a no-op change (e.g., a trailing newline in `docs/README.md`) and force-push again.

## Publish the GitHub release

Release notes live on the GitHub release page — there is no `docs/whats_new/v0.x.y.md` file.

* Go to https://github.com/autogluon/autogluon-cloud/releases → **Draft a new release**.
* Tag: `v0.x.y` (created on publish).
* Target: `master`.
* Title: `v0.x.y`.
* Write release notes inline. GitHub's "Generate release notes" button is a useful starting point — collapse boring/minor PRs into an "Improvements & bugfixes" bucket and lead with a "Highlights" section.
* Click **Publish release**. Do **not** use "Save draft" — it breaks the `release: created` trigger and PyPI publishing won't run.

## What runs automatically on publish

* `.github/workflows/pypi_release.yml` builds and uploads the package to PyPI.
* The tag push triggers the docs build, which publishes to `https://auto.gluon.ai/cloud/v0.x.y/` (the URL is derived from the tag name, so it includes the `v` prefix).

## Verify (~10 min after publish)

* `pip install autogluon.cloud==0.x.y` in a fresh venv and run a smoke test.
* `https://auto.gluon.ai/cloud/v0.x.y/index.html` resolves.
* `https://auto.gluon.ai/cloud/stable/` matches the new release.
* Ask a teammate to independently verify install + smoke test.

## Post-release on master

* Bump `version` in `setup.py` to the next development version.
* Update `release` in `docs/conf.py`.
* Update any version references in `README.md`.
* Add the new version link to `docs/versions.rst`.
* Announce on internal and external Slack channels and mailing lists.
* Publish any blogs / talks planned for the release.

## If a major issue is found post-release

Releases on PyPI cannot be deleted. Cut a hot-fix release (`0.x.y+1`) following the same process.

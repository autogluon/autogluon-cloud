# Release process

## Prior to release: 1 week out

* Check all dependency version ranges.
  * Ensure all dependencies are not capped by major version, unless the reason is documented inline.
    * ❎ Avoid - major version cap: `scikit-learn<2`
  * Ensure all dependencies have an upper cap, unless the reason is documented inline.
    * ❎ Avoid - no upper cap: `scikit-learn>=1.0`
  * Ensure no dependency is pinned to an exact version unless the reason is documented inline.
    * ❎ Avoid - `scikit-learn==1.1.3`. This is very fragile and overly strict.
  * Ensure all dependencies are capped by minor version and not micro version unless the reason is documented inline.
    * Minor version capping would be `<x.y`. Micro version capping would be `<x.y.z`.
    * ❎ Avoid: capping to `<x.y.0`
    * ✅ Do: `<x.y`.
  * Ensure all dependencies are lower bound capped to a reasonable version.
    * ❎ Avoid: `scikit-learn<1.2` is not reasonable because it is almost certain that `scikit-learn==0.0.1` is not supported.
    * ✅ A better range: `scikit-learn>=1.0,<1.2`
  * Ensure all upper caps are using `<` and not `<=`.
* Try upgrading all dependency version range upper caps to include the latest stable release.
  * Note: For micro releases, this is optional.
  * If increasing the range causes an error, either:
    1. Fix the error.
    2. Avoid the range change and add an inline comment in setup.py that an error occurred and why we didn't fix.
  * If increasing the range causes a warning, either:
    1. Fix the warning.
    2. Suppress the warning for the user + provide justification and appropriate TODOs.
    3. Avoid the range change and add an inline comment in setup.py that a warning occurred and why we didn't fix.
  * Ensure CI passes.
  * Note: To truly catch potential errors, you will need to use the latest supported Python version, since some packages may only support the newer Python version in their latest releases.
* Make final call for which in-progress PRs are release critical.
* Communicate with in-progress PR owners that code freeze is in effect, no PRs will be merged that are not release critical.
* Wait 1 day after code-freeze for pre-release to be published.
* Ensure latest pre-release is working via `pip install --pre autogluon.cloud` and testing to get an idea of how the actual release will function (Ideally with fresh venv). DO NOT RELEASE if the pre-release does not work.
* Ensure pip install instructions are working correctly
* If minor fixes are needed, create PRs and merge them as necessary if they are low risk. Ensure fixes are tested manually.
* If major fixes are needed, consider the severity and if they are release critical. If they are, consider delaying release to ensure the issue is fixed (and tested).

## Prior to release: 1 day out

* Ensure that the mainline code you are planning to release is stable: ensure CI passes, check with team, etc.
* Sanity check with the pre-release version
  * Check basic functionality for each CloudPredictor
* Cut a release branch with format `0.x.y` (no v) - this branch is required to publish docs to versioned path
  * Clone from master branch
  * Add 1 commit to the release branch to remove pre-release warnings and update install instructions to remove `--pre`: [Old diff](https://github.com/autogluon/autogluon-cloud/commit/8e0de4fd0f467bf971469c5f9ae8eb4534f9aa17)
  * Push release branch
  * Build the release branch docs and verify it's available at `https://auto.gluon.ai/cloud/0.x.y/index.html`
* Prepare the release notes located in `docs/whats_new/v0.x.y.md`:
  * This will be copy-pasted into GitHub when you release.
  * Include all merged PRs into the notes and mention all PR authors / contributors (refer to past releases for examples).
  * Prioritize major features before minor features when ordering, otherwise order by merge date.
  * Review with at least 2 core maintainers to ensure release notes are correct.

## Release

* Update the `stable` documentation to the new release:
  * Delete the `stable` branch.
  * Create new `stable` branch from `0.x.y` branch (They should be identical).
  * Add and push any change in `docs/README.md` (i.e. space) to ensure `stable` branch is different from `0.x.y`. 
    * This is required for GH Action to execute CI continuous integration step if `0.x.y` and `stable` hashes are matching.
  * Wait for CI build of the `stable` branch to pass
  * Check that website has updated to align with the release docs.
* Perform version release by going to https://github.com/autogluon/autogluon-cloud/releases and click 'Draft a new release' in top right.
  * Tag release with format `v0.x.y`
  * Name the release identically to the tag (ex: `v0.x.y`)
  * Select `master` branch as a target
    * Note: we generally use master unless there are certain commits there we don't want to add to the release
  * DO NOT use the 'Save draft' option during the creation of the release. This breaks GitHub pipelines.
  * Copy-paste the content of `docs/whats_new/v0.x.y.md` into the release notes box.
    * Ensure release notes look correct and make any final formatting fixes.
  * Click 'Publish release' and the release will go live.
* Wait ~10 minutes and then locally test that the PyPi package is available and working with the latest release version, ask team members to also independently verify.

## Post Release

* IF THERE IS A MAJOR ISSUE: Do an emergency hot-fix and a new release ASAP. Releases cannot be deleted, so a new release will have to be done.

After release is published, on the mainline branch:
* Update `release` in `docs/conf.py`
* Increment version in the `setup.py` file
* Update `README.md` sample code with new release version.
* Add new version links to `docs/versions.rst`
* Send release update to internal and external slack channels and mailing lists
* Publish any blogs / talks planned for release to generate interest.

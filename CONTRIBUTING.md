# Contributing to NeoRuntime SDKs

Thanks for helping improve the NeoRuntime SDKs.

## Repository Layout

- `proto/` - platform API protocol definitions
- `python/` - Python SDK package, examples, tests, and documentation

Future language SDKs should live in their own top-level directories, such as
`go/`, `cpp/`, or `typescript/`.

## Python SDK Checks

Run these before opening a pull request:

```bash
python -m pip install -e ./python
python -m pytest -q python/tests
python -m pip install -r python/docs/requirements.txt
python -m sphinx -b html python/docs /tmp/neoruntime-sdk-docs/python/zh
python -m sphinx -b html python/docs/en /tmp/neoruntime-sdk-docs/python/en
```

## Pull Request Checklist

- [ ] Keep changes scoped to the SDK or protocol definitions.
- [ ] Add or update tests for behavior changes.
- [ ] Update documentation when public APIs change.
- [ ] Do not commit secrets, private IPs, model files, build artifacts, or vendor SDKs.
- [ ] Keep generated packages such as wheels, tarballs, and local caches out of Git.

## Releasing and Publishing

The Python SDK publishes to TestPyPI for validation and then to PyPI. All
publishing runs through `.github/workflows/wheel.yml` via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no API
tokens).

### Versioning

The version lives in two places and must stay in sync (CI enforces this on
release paths):

- `python/setup.py` (`version=`)
- `python/neoruntime_ipc_sdk/__init__.py` (`__version__`)

Neither index allows re-uploading a version. For TestPyPI iteration, bump to a
pre-release version such as `0.6.0.dev1` or `0.6.0rc1` for each upload instead
of relying on `--skip-existing`.

### Publishing to TestPyPI (iteration)

1. Bump both version locations to a pre-release version (e.g. `0.6.0.dev1`).
2. Push the branch, then run the *Build SDK release artifacts* workflow via
   `workflow_dispatch` with `publish_testpypi=true`.
3. Verify: `https://test.pypi.org/project/neoruntime-ipc-sdk/`
4. Install check (dependencies come from real PyPI):

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ neoruntime-ipc-sdk==0.6.0.dev1
   python -c "import neoruntime_ipc_sdk as s; print(s.__version__)"
   ```

### Production release

1. Set the final version in both files (e.g. `0.6.0`), update the READMEs, and
   merge to `main`.
2. Tag and push:

   ```bash
   git tag v0.6.0
   git push origin v0.6.0
   ```

3. The tag run builds the distributions, attaches them (plus the C++ SDK
   tarball) to a GitHub Release, uploads to TestPyPI, then pauses at the
   protected `pypi` environment for approval.
4. Validate the TestPyPI page for the new version, then approve the `pypi`
   environment deployment to publish to PyPI.
5. Verify `https://pypi.org/project/neoruntime-ipc-sdk/` and install from a
   clean environment.

Delayed publishing without a new tag: run the workflow via
`workflow_dispatch` with `publish_pypi=true` (also gated by the `pypi`
environment approval).

### One-time Trusted Publishing setup

On TestPyPI (`test.pypi.org`) and PyPI (`pypi.org`) separately, under
Account → Publishing → *Add a pending publisher*:

| Field | Value |
| --- | --- |
| PyPI project name | `neoruntime-ipc-sdk` |
| Owner | `camthink-ai` (add `Oreofilling` on TestPyPI for fork validation) |
| Repository | `neoruntime-sdks` |
| Workflow filename | `wheel.yml` |
| Environment name | `testpypi` on TestPyPI, `pypi` on PyPI |

On GitHub, Settings → Environments: create `testpypi` (no protection rules)
and `pypi` (with required reviewers as the production gate). The publisher
configuration must match the workflow exactly, including the environment
name.

## Security Issues

See [SECURITY.md](./SECURITY.md). Please do not open public issues for security
findings.

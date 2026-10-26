# Replit import, Preview and publication

Official [configuration](https://docs.replit.com/features/project-setup/configuration),
[dependencies](https://docs.replit.com/features/project-setup/dependency-management)
and [deployment types](https://docs.replit.com/features/publishing/deployment-types)
were checked on 27 October 2026. No connected Replit account/project was available;
Preview and an actual published URL remain **unverified**. Nothing was provisioned.

1. Import your chosen GitHub repository after the [GitHub handoff](GITHUB.md),
   or upload the source checkout. Include `.replit`, `modeling/`, `src/` and
   configuration; exclude the Windows environment, raw corpus and caches.
2. Confirm the imported `.replit` Python 3.12 module and `python --version`.
   Python 3.13 is also supported. No Poetry lock/configuration is used. Automatic
   import guessing is disabled to preserve pins. In Shell run:

   ```bash
   python -m pip install --only-binary=:all: -r modeling/deployment/requirements.lock
   python -m pip check
   ```

3. Upload the **matching** archive separately, then run `unpack` with its actual
   SHA-256, as in [README.md](README.md). Or attach the ZIP and `.sha256` to a
   GitHub release and set `CPSS_BUNDLE_URL` / `CPSS_BUNDLE_SHA256` for build-time
   retrieval. The downloader accepts public HTTPS release assets only; private
   releases require separate authenticated delivery. It never sends GitHub
   credentials or fetches an unpinned/latest encoder.
4. Find the exact HTTPS Preview URL. Set `CPSS_PUBLIC_ORIGIN` in Replit
   Secrets/environment to that origin without a path. `.replit` sets deployment
   mode, internal port 8080, `runtime/` and the Replit frame parent. If changing
   ports, keep `[[ports]]` consistent; platform `PORT` overrides `CPSS_PORT`.
   Use the URL Replit actually assigns, without guessing a project name.
5. Run `python -m modeling.deployment prepare --no-install` in Shell. It explicitly
   downloads/verifies the bundle if a URL is configured; otherwise `runtime/`
   must already be unpacked. Press Run: `python -m modeling.deployment serve`.
   Normal startup does no downloads. Open Preview, require `/readyz` 200, then:

   ```bash
   python -m modeling.deployment smoke --url http://127.0.0.1:8080 --origin https://ACTUAL_PREVIEW_HOST
   ```

   Check actual browser search, both views, ratings, source links and hidden
   review/classifier controls. If Preview uses another frame parent, configure
   that exact HTTPS parent in `CPSS_FRAME_ORIGINS` or open a new tab; do not use `*`.
6. In Publishing choose **Autoscale**. Review displayed machine/max-instance costs
   against your budget before publishing. Build:
   `python -m modeling.deployment prepare`; run:
   `python -m modeling.deployment serve`. `.replit` specifies `cloudrun`, with
   internal 8080 mapped to external 80. Choose x86-64 AVX2 CPU; at least 512 MiB is
   a reasonable first trial from observed ~244 MiB RSS, then measure actual peak
   memory/platform overhead. This is not a Replit capacity measurement.
7. Set the **deployment** `CPSS_PUBLIC_ORIGIN` to the exact chosen published
   HTTPS origin before publishing. Preview and production need their own origin
   values. Provide the URL/hash at build time, or ensure manually unpacked
   `runtime/` is included in the publishing snapshot. Git ignores runtime assets;
   source import alone cannot serve. Require successful build verification.
   Do not run packaging, indexing or training in the deployment build. Keep its
   source commit and archive digest together.
8. Check the actual published HTTPS URL: `/healthz` 200, `/readyz` 200,
   `/api/info` shows 3,716/read-only, then the smoke command with the published
   URL and matching origin. Check real browser navigation. Record actual URL and
   platform measurements; local checks are not evidence of publication.

Autoscale can scale to zero. Each new instance initializes its local verified
assets. Observed WSL readiness was about 11–15 seconds; actual Replit behavior can
differ. If cold starts impede the demonstration, consider **Reserved VM**, which
keeps a server running for a fixed recurring cost. Review the displayed price
and budget first; no free allowance or payment authorization is assumed.
The existing search lock serializes work per process.

For missing/corrupt assets restore the correct archive. For source mismatches
use the matching checkout or package a new release. Do not edit hashes, disable
probes or relax dependency checks to force readiness.

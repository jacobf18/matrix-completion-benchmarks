# Website

Static site for benchmark accessibility and quick result inspection.

## Run Locally

From repository root:

```bash
python -m http.server -d website 8080
```

Then open:

`http://localhost:8080`

## Public Hosting (GitHub Pages)

This repo includes:

`/Users/jfeit/matrix-completion-benchmarks/.github/workflows/deploy-website.yml`

It auto-deploys `website/` on pushes to `main`.

### One-time setup

1. In GitHub repo settings, open `Pages`.
2. Set `Source` to `GitHub Actions`.
3. Push to `main` (or run the workflow manually).
4. Your site will be available at:
   - `https://<your-org-or-user>.github.io/<repo-name>/`

### Custom domain (recommended)

If you want a URL independent of GitHub branding (for example `benchmarks.yourdomain.com`):

1. Add a `CNAME` file inside `website/` containing your domain.
2. In GitHub Pages settings, set the same custom domain.
3. Configure DNS:
   - `CNAME` record from your domain/subdomain to `<your-org-or-user>.github.io`

## What It Includes

- Project quickstart commands
- Benchmark and method summary cards
- CSV-based result explorer for noise sweeps
- Line chart with metric toggle (`RMSE` / `NRMSE`)

Upload:

`benchmarks/reports/noise_sweep/noise_sweep_results.csv`

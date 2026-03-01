# Web Dashboard

Run the dashboard server from the project root:

```bash
python -m web.app.run_dashboard --host 127.0.0.1 --port 8765
```

Then open:

`http://127.0.0.1:8765`

API endpoints:
- `/api/health`
- `/api/clusters`
- `/api/clusters?state=GA`
- `/api/findings?horizon_days=365`
- `/api/location-summary?state=GA&site_code=VNP`
- `/api/location-summary?state=GA&site_code=VNP&horizon_days=365`

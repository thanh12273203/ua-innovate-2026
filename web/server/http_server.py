from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data_service import get_cluster_payload, get_findings_payload, get_location_summary_with_horizon


INTERFACE_DIR = Path(__file__).resolve().parents[1] / 'interface'


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(INTERFACE_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path.startswith('/api/'):
            self._handle_api(parsed_url)

            return

        if parsed_url.path in {'', '/'}:
            self.path = '/index.html'
        else:
            self.path = parsed_url.path
            
        super().do_GET()

    def _handle_api(self, parsed_url) -> None:
        query = parse_qs(parsed_url.query)
        try:
            if parsed_url.path == '/api/health':
                self._write_json({'status': 'ok'})

                return

            if parsed_url.path == '/api/clusters':
                state = query.get('state', [None])[0]
                payload = get_cluster_payload(state=state)
                self._write_json(payload)

                return

            if parsed_url.path == '/api/findings':
                horizon_raw = query.get('horizon_days', [None])[0]
                horizon_days = 365
                if horizon_raw is not None and str(horizon_raw).strip():
                    horizon_days = int(str(horizon_raw).strip())
                payload = get_findings_payload(horizon_days=horizon_days)
                self._write_json(payload)

                return

            if parsed_url.path == '/api/location-summary':
                state = query.get('state', [None])[0]
                site_code = query.get('site_code', [None])[0]
                horizon_raw = query.get('horizon_days', [None])[0]
                if not state or not site_code:
                    self._write_json(
                        {'error': "Missing required query params: 'state' and 'site_code'."},
                        status=400,
                    )

                    return
                
                horizon_days = 365
                if horizon_raw is not None and str(horizon_raw).strip():
                    horizon_days = int(str(horizon_raw).strip())

                payload = get_location_summary_with_horizon(
                    state=state,
                    site_code=site_code,
                    horizon_days=horizon_days,
                )
                self._write_json(payload)

                return

            self._write_json({'error': 'Endpoint not found.'}, status=404)
        except ValueError as exc:
            self._write_json({'error': str(exc)}, status=400)
        except Exception as exc:
            self._write_json(
                {
                    'error': 'Unexpected server error.',
                    'detail': str(exc),
                },
                status=500
            )

    def _write_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_dashboard_server(host: str = '127.0.0.1', port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"Dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the web dashboard server.")
    parser.add_argument('--host', default='127.0.0.1', help="Host interface to bind.")
    parser.add_argument('--port', default=8765, type=int, help="Port to bind.")

    return parser


def launch() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_dashboard_server(host=args.host, port=args.port)


def main() -> None:
    launch()


if __name__ == '__main__':
    launch()

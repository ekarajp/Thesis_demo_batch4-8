"""Dependency-free local web dashboard for Batch 004-008."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from portable_common import (
    atomic_json,
    process_is_running,
    progress_snapshot,
    read_json,
    sha256,
    utc_now,
)


ROOT = Path(__file__).absolute().parent
DATABASE = ROOT / "data" / "server_batches_004_008.sqlite"
RUNTIME = ROOT / "runtime"
STATE = RUNTIME / "server_state.json"
LOCK = RUNTIME / "orchestrator.lock.json"
PAUSE_REQUEST = RUNTIME / "pause.request.json"
SETTINGS = RUNTIME / "server_settings.json"


HTML = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RC Fragility Batch 004-008</title>
<style>
:root{--bg:#f3f6fa;--card:#fff;--ink:#172033;--muted:#667085;
--blue:#2457d6;--green:#16865b;--amber:#c77b08;--red:#be3340;--line:#dbe2ea}
*{box-sizing:border-box} body{margin:0;background:var(--bg);font-family:
system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif;color:var(--ink)}
header{background:#152a4a;color:#fff;padding:22px 30px}
header h1{margin:0 0 5px;font-size:24px} header p{margin:0;color:#dbe7f8}
main{padding:22px;max-width:1500px;margin:auto}
.toolbar,.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
button{border:0;border-radius:8px;padding:11px 17px;font-weight:650;cursor:pointer}
.start{background:var(--green);color:white}.pause{background:var(--amber);color:white}
.refresh{background:var(--blue);color:white}button:disabled{opacity:.45;cursor:not-allowed}
input{padding:10px;border:1px solid var(--line);border-radius:7px;width:70px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;min-width:185px;box-shadow:0 1px 2px #0000000a}
.card .label{color:var(--muted);font-size:12px;text-transform:uppercase}
.card .value{font-size:25px;font-weight:730;margin-top:5px}
.panel{background:#fff;border:1px solid var(--line);border-radius:10px;
margin-bottom:16px;overflow:hidden}.panel h2{font-size:16px;margin:0;padding:15px 17px;
border-bottom:1px solid var(--line)}
.message{padding:13px 16px;background:#eef4ff;border-left:4px solid var(--blue);
margin-bottom:16px;border-radius:5px}.warn{background:#fff4e5;border-color:var(--amber)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 10px;border-bottom:1px solid #edf0f4;text-align:left;white-space:nowrap}
th{background:#f8fafc;position:sticky;top:0;z-index:1;color:#475467}
.scroll{max-height:530px;overflow:auto}.bar{height:10px;background:#e7ebf1;border-radius:8px;
overflow:hidden;min-width:100px}.bar span{height:100%;display:block;background:var(--blue)}
.ok{color:var(--green);font-weight:650}.bad{color:var(--red);font-weight:650}
.pending{color:var(--muted)}a{color:var(--blue);text-decoration:none}
.small{font-size:12px;color:var(--muted)} code{font-family:Consolas,monospace}
@media(max-width:700px){main{padding:10px}header{padding:17px}.card{min-width:145px}}
</style>
</head>
<body>
<header><h1>RC Fragility Server — Batch 004–008</h1>
<p>Phase 1: สร้างข้อมูล SPO, Full IDA และ Fragility เท่านั้น (ยังไม่มี ML)</p></header>
<main>
 <div class="toolbar">
   <label>Workers <input id="workers" type="number" min="1" value="6"></label>
   <button id="start" class="start" onclick="startRun()">เริ่ม / ทำต่อ</button>
   <button id="pause" class="pause" onclick="pauseRun()">หยุดชั่วคราวอย่างปลอดภัย</button>
   <button class="refresh" onclick="refresh()">รีเฟรช</button>
 </div>
 <div id="message" class="message">กำลังอ่านสถานะ...</div>
 <div class="cards">
   <div class="card"><div class="label">สถานะ</div><div class="value" id="status">–</div></div>
   <div class="card"><div class="label">ความคืบหน้ารวม</div><div class="value" id="overall">0%</div></div>
   <div class="card"><div class="label">อาคารเสร็จครบ</div><div class="value" id="completed">0 / 225</div></div>
   <div class="card"><div class="label">กำลังทำ</div><div class="value" id="current" style="font-size:16px">–</div></div>
 </div>
 <section class="panel"><h2>สถานะแต่ละ Batch</h2><div class="scroll"><table>
   <thead><tr><th>Batch</th><th>อันดับ</th><th>เสร็จ</th><th>SPO</th>
   <th>Fragility</th><th>Progress</th><th>ปัญหาค้าง</th><th>ZIP ผลลัพธ์</th></tr></thead>
   <tbody id="batches"></tbody></table></div></section>
 <section class="panel"><h2>สถานะรายอาคาร (เปอร์เซ็นต์จากผลที่เซฟจริง)</h2>
   <div class="scroll"><table><thead><tr><th>Rank</th><th>Building ID</th><th>Batch</th>
   <th>%</th><th>SPO</th><th>T1 (s)</th><th>CMS IDA curves</th><th>PWSA SF=1</th>
   <th>Fragility</th><th>NLTHA checkpoints</th><th>ปัญหาค้าง</th></tr></thead>
   <tbody id="buildings"></tbody></table></div></section>
 <section class="panel"><h2>ไฟล์พร้อมก๊อปกลับ</h2><div id="exports" style="padding:15px"></div></section>
 <p class="small">หน้าเว็บจะรีเฟรชทุก 5 วินาที การกด Pause จะหยุด process tree
 หลังบันทึก checkpoint ที่เสร็จแล้ว และ Resume จะใช้ข้อมูลเดิมต่อโดยไม่เริ่มใหม่ทั้งหมด</p>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
'"':'&quot;',"'":'&#39;'}[c]));
const fmt=s=>{let n=Number(s||0);if(n<60)return n.toFixed(0)+' s';
if(n<3600)return (n/60).toFixed(1)+' min';return (n/3600).toFixed(1)+' h'};
const stateClass=s=>s==='complete'?'ok':s==='pending'?'pending':'bad';
async function api(path,method='GET',body=null){let r=await fetch(path,{method,
headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):null});
let j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function startRun(){try{let w=Number(document.getElementById('workers').value);
await api('/api/start','POST',{workers:w});await refresh()}catch(e){alert(e.message)}}
async function pauseRun(){try{await api('/api/pause','POST',{});await refresh()}
catch(e){alert(e.message)}}
async function refresh(){try{let d=await api('/api/status');let s=d.state,p=d.progress;
document.getElementById('status').textContent=s.status||'ready';
document.getElementById('overall').textContent=(p.average_percent||0).toFixed(1)+'%';
document.getElementById('completed').textContent=`${p.completed_buildings} / ${p.total_buildings}`;
document.getElementById('current').textContent=[s.current_batch,s.current_stage].filter(Boolean).join(' / ')||'–';
let m=document.getElementById('message');m.textContent=s.message||'Ready';
m.className='message '+(s.status==='needs_attention'?'warn':'');
document.getElementById('start').disabled=d.runner_running;
document.getElementById('pause').disabled=!d.runner_running;
if(s.workers)document.getElementById('workers').value=s.workers;
let ex=new Map(d.exports.map(x=>[x.batch_id,x]));
document.getElementById('batches').innerHTML=p.batches.map(b=>{let z=ex.get(b.batch_id);
return `<tr><td>${esc(b.batch_id)}</td><td>${b.rank_start}–${b.rank_end}</td>
<td>${b.completed_buildings}/${b.building_count}</td><td>${b.spo_complete}</td>
<td>${b.fragility_complete}</td><td><div class=bar><span style="width:${b.average_percent}%"></span></div>
${b.average_percent.toFixed(1)}%</td><td class="${b.unresolved_failures?'bad':'ok'}">${b.unresolved_failures}</td>
<td>${z?`<a href="${z.url}">ดาวน์โหลด ZIP</a>`:'–'}</td></tr>`}).join('');
document.getElementById('buildings').innerHTML=p.buildings.map(b=>`<tr>
<td>${b.queue_rank}</td><td><code>${esc(b.building_id)}</code></td><td>${esc(b.batch_id)}</td>
<td><div class=bar><span style="width:${b.percent}%"></span></div>${b.percent.toFixed(1)}%</td>
<td class="${stateClass(b.spo)}">${b.spo}</td><td>${b.t1_s==null?'–':b.t1_s.toFixed(3)}</td>
<td>${b.primary_curves_complete}/${b.primary_curves_total||'–'}</td>
<td class="${stateClass(b.pwsa)}">${b.pwsa}</td>
<td class="${stateClass(b.fragility)}">${b.fragility}</td>
<td>${b.saved_nltha_checkpoint_count}<span class=small> saved; ${b.ida_run_count} in DB (${fmt(b.ida_runtime_s)})</span></td>
<td class="${b.unresolved_failures?'bad':'ok'}">${b.unresolved_failures}</td></tr>`).join('');
document.getElementById('exports').innerHTML=d.exports.length?d.exports.map(x=>
`<p><a href="${x.url}">${esc(x.name)}</a> — ${(x.size_bytes/1048576).toFixed(1)} MB
<span class=small>SHA-256 ${esc(x.sha256)}</span></p>`).join(''):'ยังไม่มี Batch ที่ export เสร็จ';
}catch(e){let m=document.getElementById('message');m.textContent='อ่านสถานะไม่ได้: '+e.message;m.className='message warn'}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


def exports() -> list[dict[str, Any]]:
    root = ROOT / "exports" / "READY_TO_COPY"
    rows = []
    for path in sorted(root.glob("*.zip")) if root.is_dir() else []:
        batch_id = ""
        lower = path.name.lower()
        for number in range(4, 9):
            candidate = f"batch_{number:03d}"
            if candidate in lower:
                batch_id = candidate
                break
        rows.append(
            {
                "name": path.name,
                "batch_id": batch_id,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "url": "/download/" + urllib.parse.quote(path.name),
            }
        )
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "RCFragilityDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(
            f"{self.log_date_time_string()} {self.address_string()} "
            + format % args
            + "\n"
        )
        sys.stdout.flush()

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/status":
            state = read_json(
                STATE,
                {
                    "status": "ready",
                    "message": "Ready to start Batch 004-008",
                },
            )
            lock = read_json(LOCK, {})
            running = process_is_running(int(lock.get("pid") or 0))
            if not running and state.get("status") == "running":
                state["status"] = "needs_attention"
                state["message"] = (
                    "Runner is no longer active. Press Start/Resume to "
                    "continue from saved checkpoints."
                )
            self.send_json(
                {
                    "timestamp_utc": utc_now(),
                    "state": state,
                    "runner_running": running,
                    "progress": progress_snapshot(DATABASE),
                    "exports": exports(),
                }
            )
            return
        if parsed.path.startswith("/download/"):
            name = urllib.parse.unquote(parsed.path[len("/download/") :])
            if Path(name).name != name or not name.lower().endswith(".zip"):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            path = ROOT / "exports" / "READY_TO_COPY" / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 4096:
            raise ValueError("Request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self) -> None:
        try:
            body = self.body_json()
            if self.path == "/api/start":
                lock = read_json(LOCK, {})
                if process_is_running(int(lock.get("pid") or 0)):
                    self.send_json(
                        {"error": "Batch runner is already running"}, 409
                    )
                    return
                maximum = max(1, (os.cpu_count() or 2) - 1)
                workers = max(1, min(int(body.get("workers", 6)), maximum))
                PAUSE_REQUEST.unlink(missing_ok=True)
                atomic_json(
                    SETTINGS,
                    {
                        "workers": workers,
                        "requested_utc": utc_now(),
                        "cpu_count": os.cpu_count(),
                    },
                )
                launcher = (ROOT / "logs" / "dashboard_launcher.log").open(
                    "a", encoding="utf-8", newline="\n"
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "server_orchestrator.py"),
                        "--start",
                        "--workers",
                        str(workers),
                    ],
                    cwd=ROOT,
                    stdout=launcher,
                    stderr=launcher,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if os.name == "nt"
                        else 0
                    ),
                    start_new_session=os.name != "nt",
                    close_fds=True,
                )
                launcher.close()
                self.send_json(
                    {
                        "status": "starting",
                        "pid": process.pid,
                        "workers": workers,
                    },
                    202,
                )
                return
            if self.path == "/api/pause":
                lock = read_json(LOCK, {})
                if not process_is_running(int(lock.get("pid") or 0)):
                    self.send_json({"error": "Batch runner is not active"}, 409)
                    return
                atomic_json(
                    PAUSE_REQUEST,
                    {
                        "requested_utc": utc_now(),
                        "requested_by": "local_dashboard",
                    },
                )
                self.send_json(
                    {
                        "status": "pause_requested",
                        "message": (
                            "The active stage will stop safely and retain "
                            "completed checkpoints."
                        ),
                    },
                    202,
                )
                return
            self.send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"RC Fragility dashboard: http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

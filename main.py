"""
CaptchaService - API giải CAPTCHA thay thế TrueCaptcha.
Tương thích với flow cũ của WinTax_API (C#).

Chạy: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import base64
import os

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request, Header, Depends
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, HTMLResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from ocr_engine import engine
from collector import save_captcha, get_stats
import tct_client
import tcnnt_client
import keystore
import config

app = FastAPI(
    title="CaptchaService",
    description="API giải CAPTCHA cho WinTax - thay thế TrueCaptcha",
    version="1.0.0",
)

# Phục vụ thư mục cập nhật app WPF (Velopack) tại https://decapcha.win-tech.vn/app/...
# Upload nội dung thư mục 'releases' của app WPF vào ./app_releases trên server.
_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_releases")
os.makedirs(_APP_DIR, exist_ok=True)
app.mount("/app", StaticFiles(directory=_APP_DIR), name="app_update")



@app.on_event("startup")
async def _startup_load_model():
    """Load CNN model + init DB key ngay khi service start de tranh lag request dau."""
    mode = "PROD" if os.environ.get("ENV", "").lower() == "prod" else "DEV"
    print(f"[Startup] CaptchaService starting | Mode={mode} | Port={config.PORT}")

    # Model 6 ký tự (hoadondientu) - cho /captcha/solve, /one/gettext
    try:
        result = engine._load_custom_model()
        print("[Startup] CNN Model 6 ky tu loaded." if result is not None
              else "[Startup] CNN Model 6 ky tu NOT loaded - dung EasyOCR/Tesseract.")
    except Exception as e:
        print(f"[Startup] Error loading CNN model 6 ky tu: {e}")

    # Model 5 ký tự (TCNNT) - warm san cho /tcnnt/lookup
    try:
        m5 = engine._load_model5()
        print("[Startup] CNN Model 5 ky tu (TCNNT) warmed." if m5 is not None
              else "[Startup] CNN Model 5 ky tu NOT loaded (thieu captcha_model_v6.pth).")
    except Exception as e:
        print(f"[Startup] Error loading CNN model 5 ky tu: {e}")

    # Init DB key + dọn log cũ + chạy thread dọn định kỳ
    try:
        keystore.init_db()
        removed = keystore.cleanup_logs(config.LOG_RETENTION_DAYS)
        print(f"[Startup] Key DB ready | cleaned {removed} old logs (>{config.LOG_RETENTION_DAYS}d)")
        threading.Thread(target=_log_cleanup_worker, daemon=True).start()
    except Exception as e:
        print(f"[Startup] Error init key DB: {e}")


def _log_cleanup_worker():
    """Thread daemon: dọn log cũ mỗi 6 giờ để DB không phình."""
    import time as _t
    while True:
        _t.sleep(6 * 3600)
        try:
            n = keystore.cleanup_logs(config.LOG_RETENTION_DAYS)
            if n:
                print(f"[LogCleanup] Removed {n} logs older than {config.LOG_RETENTION_DAYS}d")
        except Exception as e:
            print(f"[LogCleanup] Error: {e}")



# ===== Request/Response Models =====

class OCRRequest(BaseModel):
    """Tương thích với format TrueCaptcha cũ."""
    data: str  # base64 encoded image

    class Config:
        json_schema_extra = {
            "example": {
                "data": "<base64 PNG image>"
            }
        }


class OCRResponse(BaseModel):
    result: str  # text CAPTCHA đã giải
    saved: bool = False  # đã lưu để training chưa


class CaptchaSolveResponse(BaseModel):
    """Trả về giống captchaVal trong C#."""
    token: str  # text CAPTCHA đã giải
    key: str    # session key từ TCT


# ===== API Endpoints =====

@app.post("/one/gettext", response_model=OCRResponse)
async def ocr_solve(req: OCRRequest):
    """
    Giải CAPTCHA từ base64 image.

    Endpoint này TƯƠNG THÍCH với TrueCaptcha API cũ.
    Trong code C# chỉ cần đổi URL từ:
      https://api.apitruecaptcha.org/one/gettext
    thành:
      http://localhost:8000/one/gettext

    Và bỏ userid/apikey (không cần nữa).
    """
    try:
        img_bytes = base64.b64decode(req.data)
        text = engine.solve(img_bytes)

        if not text:
            raise HTTPException(status_code=422, detail="Không nhận dạng được CAPTCHA")

        # Lưu ảnh gốc + label để sau training
        saved_path = save_captcha(img_bytes, text)

        return OCRResponse(result=text, saved=saved_path is not None)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captcha/solve", response_model=CaptchaSolveResponse)
async def captcha_solve():
    """
    Tự động: Lấy CAPTCHA từ TCT → Giải → Trả về {token, key}.

    Endpoint này thay thế hoàn toàn hàm getCapt() trong C#.
    Client chỉ cần gọi 1 API này là có đủ ckey + cvalue để login.
    """
    import traceback
    try:
        # Bước 1: Lấy CAPTCHA từ TCT
        png_bytes, captcha_key = tct_client.fetch_captcha()

        # Bước 2: Giải bằng OCR
        text = engine.solve(png_bytes)

        if not text:
            raise HTTPException(status_code=422, detail="Không giải được CAPTCHA")

        # Bước 3: Lưu để training
        save_captcha(png_bytes, text)

        return CaptchaSolveResponse(token=text, key=captcha_key)

    except HTTPException:
        raise
    except Exception as e:
        # Log day du traceback ra console (vao service.log)
        tb = traceback.format_exc()
        print(f"[captcha_solve] EXCEPTION:\n{tb}", flush=True)
        # Tra ve message co type cua exception de de debug
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e) or repr(e)}")


# ===== Auth: API key cho /tcnnt/lookup =====

def _client_ip(request: Request) -> str:
    """IP thật của client (qua IIS reverse proxy -> X-Forwarded-For)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def require_api_key(x_api_key: str = Header(None), api_key: str = None) -> dict:
    """Xác thực + trừ 1 lượt. Key truyền qua header `X-API-Key` hoặc query `?api_key=`."""
    key = x_api_key or api_key
    if not key:
        raise HTTPException(401, "Thiếu API key (header X-API-Key hoặc ?api_key=)")

    status, rec = keystore.consume(key)
    if status == "not_found":
        raise HTTPException(401, "API key không hợp lệ")
    if status == "inactive":
        raise HTTPException(403, "API key đã bị khoá")
    if status == "quota_exceeded":
        raise HTTPException(429, f"Hết lượt gọi API (đã dùng {rec['used']}/{rec['quota']})")
    return rec


@app.get("/tcnnt/lookup")
def tcnnt_lookup(mst: str, request: Request, max_tries: int = 12, delay: float = 1.5,
                 key: dict = Depends(require_api_key)):
    """
    Tra cứu thông tin người nộp thuế từ tracuunnt.gdt.gov.vn theo MST.

    Yêu cầu API key (header `X-API-Key` hoặc `?api_key=`). Mỗi lần gọi trừ 1 lượt.
    Chỉ cần nhập `mst` -> tự giải captcha 5 ký tự, retry tới khi ra (anti-bot).
    Endpoint là `def` (không async) -> FastAPI chạy trong threadpool, không block.

    Response:
      { mst, address, count_Try, found, status, message,
        results: [{stt, mst, name, address, tax_authority, status}, ...] }
    """
    mst = mst.strip()
    result = tcnnt_client.lookup_mst(mst, max_tries=max_tries, delay=delay)
    try:
        keystore.log_call(key["id"], key["name"], mst, result.get("status"),
                          result.get("count_Try", 0), _client_ip(request))
    except Exception as e:
        print(f"[tcnnt_lookup] log error: {e}")
    return result


# ===== Admin: quản lý API key + log (bảo vệ bằng ADMIN_PASSWORD) =====

def require_admin(x_admin_token: str = Header(None)):
    if not config.ADMIN_PASSWORD:
        raise HTTPException(503, "Server chưa cấu hình ADMIN_PASSWORD")
    if x_admin_token != config.ADMIN_PASSWORD:
        raise HTTPException(401, "Sai mật khẩu admin")
    return True


class KeyCreate(BaseModel):
    name: str
    quota: int | None = None     # số lượt; bỏ trống + unlimited=True -> không giới hạn
    unlimited: bool = False
    note: str = ""


class KeyUpdate(BaseModel):
    name: str | None = None
    quota: int | None = None
    unlimited: bool = False
    active: bool | None = None
    note: str | None = None


class AdminLogin(BaseModel):
    password: str


@app.post("/admin/login")
def admin_login(body: AdminLogin):
    if not config.ADMIN_PASSWORD:
        raise HTTPException(503, "Server chưa cấu hình ADMIN_PASSWORD")
    if body.password != config.ADMIN_PASSWORD:
        raise HTTPException(401, "Sai mật khẩu")
    return {"ok": True}


@app.get("/admin/keys")
def admin_list_keys(_=Depends(require_admin)):
    return keystore.list_keys()


@app.post("/admin/keys")
def admin_create_key(body: KeyCreate, _=Depends(require_admin)):
    quota = None if body.unlimited else body.quota
    return keystore.create_key(body.name, quota, body.note)


@app.post("/admin/keys/{key_id}/update")
def admin_update_key(key_id: int, body: KeyUpdate, _=Depends(require_admin)):
    quota_arg = "__keep__"
    if body.unlimited:
        quota_arg = None
    elif body.quota is not None:
        quota_arg = body.quota
    keystore.update_key(key_id, name=body.name, quota=quota_arg,
                        active=body.active, note=body.note)
    return keystore.get_key_by_id(key_id)


@app.post("/admin/keys/{key_id}/reset")
def admin_reset_key(key_id: int, _=Depends(require_admin)):
    keystore.reset_used(key_id)
    return keystore.get_key_by_id(key_id)


@app.delete("/admin/keys/{key_id}")
def admin_delete_key(key_id: int, _=Depends(require_admin)):
    keystore.delete_key(key_id)
    return {"ok": True}


@app.get("/admin/logs")
def admin_logs(limit: int = 200, key_id: int | None = None, _=Depends(require_admin)):
    return keystore.recent_logs(limit=limit, key_id=key_id)


@app.get("/admin/lookup")
def admin_lookup(mst: str, request: Request, max_tries: int = 15, delay: float = 1.5,
                 _=Depends(require_admin)):
    """Tra cứu MST trực tiếp cho admin - KHÔNG giới hạn lượt, lấy thẳng từ TCT."""
    mst = mst.strip()
    result = tcnnt_client.lookup_mst(mst, max_tries=max_tries, delay=delay)
    try:
        keystore.log_call(None, "ADMIN", mst, result.get("status"),
                          result.get("count_Try", 0), _client_ip(request))
    except Exception as e:
        print(f"[admin_lookup] log error: {e}")
    return result


@app.get("/admin/proxy")
def admin_proxy(action: str = "status", _=Depends(require_admin)):
    """Xem/điều khiển proxy pool (bảo vệ bằng ADMIN token).
    action:
      status  -> xem trạng thái (bật/tắt, tổng, khỏe, đang nghỉ)
      on      -> BẬT proxy ngay (không cần restart)
      off     -> TẮT proxy ngay = REVERT về gọi thẳng IP server
      reload  -> nạp lại danh sách từ proxies.txt (sau khi sửa file)
    """
    import proxy_pool
    a = (action or "status").lower()
    if a == "on":
        proxy_pool.set_enabled(True)
    elif a == "off":
        proxy_pool.set_enabled(False)
    elif a == "reload":
        proxy_pool.load()
    return proxy_pool.status()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Key Dashboard - CaptchaService</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0f0f23; color:#e0e0e0; font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }
  h1 { color:#00d2d3; font-size:1.4rem; margin-bottom:4px; }
  .sub { color:#636e72; font-size:.85rem; margin-bottom:20px; }
  .card { background:#16213e; border:1px solid #2d3436; border-radius:12px; padding:20px; margin-bottom:20px; }
  .card h2 { font-size:1rem; color:#feca57; margin-bottom:14px; font-weight:600; }
  label { display:block; font-size:.78rem; color:#7f8fa6; margin:8px 0 4px; }
  input[type=text], input[type=number], input[type=password] {
    background:#0a0a1a; border:1px solid #2d3436; border-radius:6px; color:#fff;
    padding:8px 10px; font-size:.9rem; width:100%; max-width:340px; }
  input:focus { outline:none; border-color:#00d2d3; }
  .row { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end; }
  button { background:#0773c5; color:#fff; border:none; border-radius:6px; padding:8px 16px;
    font-size:.85rem; font-weight:600; cursor:pointer; }
  button:hover { background:#0a8af0; }
  button.danger { background:#d63031; } button.danger:hover { background:#e84141; }
  button.ghost { background:#2d3436; } button.ghost:hover { background:#3d4446; }
  button.sm { padding:4px 10px; font-size:.75rem; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; }
  th { text-align:left; color:#7f8fa6; font-weight:500; padding:8px; border-bottom:2px solid #2d3436; }
  td { padding:8px; border-bottom:1px solid #1a1a2e; vertical-align:middle; }
  .key { font-family:monospace; font-size:.78rem; color:#00d2d3; }
  .pill { padding:2px 8px; border-radius:10px; font-size:.72rem; font-weight:600; }
  .on { background:#00b894; color:#053; } .off { background:#636e72; color:#fff; }
  .unl { color:#feca57; font-weight:700; }
  .muted { color:#636e72; }
  #toast { position:fixed; top:16px; right:16px; background:#00b894; color:#fff; padding:10px 18px;
    border-radius:8px; opacity:0; transition:.3s; z-index:50; }
  #toast.show { opacity:1; }
  #login { max-width:360px; margin:60px auto; }
  .hidden { display:none; }
  .newkey { background:#0a2a1a; border:1px solid #00b894; padding:10px; border-radius:8px;
    margin-top:12px; font-family:monospace; word-break:break-all; color:#55efc4; }
</style>
</head>
<body>
<div id="toast"></div>

<div id="login" class="card">
  <h1>API Key Dashboard</h1>
  <p class="sub">Đăng nhập admin để quản lý key tra cứu MST</p>
  <label>Mật khẩu admin</label>
  <input type="password" id="pw" onkeydown="if(event.key==='Enter')login()">
  <div style="margin-top:14px;"><button onclick="login()">Đăng nhập</button></div>
  <p id="loginErr" class="sub" style="color:#d63031;margin-top:10px;"></p>
</div>

<div id="app" class="hidden">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div><h1>API Key Dashboard</h1><p class="sub">Quản lý key + lượt gọi /tcnnt/lookup</p></div>
    <button class="ghost" onclick="logout()">Đăng xuất</button>
  </div>

  <div class="card">
    <h2>🔎 Tra cứu MST trực tiếp (admin)</h2>
    <p class="sub" style="margin:-8px 0 12px;">Lấy trực tiếp từ Tổng cục Thuế — admin Nhật đẹp trai tra cứu không giới hạn lượt :D</p>
    <div class="row">
      <div><label>Mã số thuế</label><input type="text" id="aMst" placeholder="VD: 0312303803" onkeydown="if(event.key==='Enter')adminLookup()"></div>
      <div><button id="aBtn" onclick="adminLookup()">Tra cứu</button></div>
    </div>
    <div id="aResult"></div>
  </div>

  <div class="card">
    <h2>+ Tạo key mới</h2>
    <div class="row">
      <div><label>Tên / chủ key</label><input type="text" id="nName" placeholder="VD: Cong ty A"></div>
      <div><label>Số lượt (quota)</label><input type="number" id="nQuota" placeholder="trống = unlimited" min="1"></div>
      <div><label>Ghi chú</label><input type="text" id="nNote" placeholder="tuỳ chọn"></div>
      <div><button onclick="createKey()">Tạo key</button></div>
    </div>
    <div id="newKeyBox"></div>
  </div>

  <div class="card">
    <h2>Danh sách key</h2>
    <table id="keysTbl"><thead><tr>
      <th>ID</th><th>Tên</th><th>Key</th><th>Đã dùng / Quota</th><th>Hôm nay</th><th>Trạng thái</th><th>Thao tác</th>
    </tr></thead><tbody></tbody></table>
  </div>

  <div class="card">
    <h2>Log gọi API (tự xoá sau 2 ngày) <button class="ghost sm" onclick="loadLogs()" style="margin-left:8px;">Tải lại</button></h2>
    <table id="logsTbl"><thead><tr>
      <th>Thời gian</th><th>Key</th><th>MST</th><th>Kết quả</th><th>count_Try</th><th>IP</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>

<script>
let TOKEN = sessionStorage.getItem('adm_tok') || '';
const $ = s => document.querySelector(s);
function toast(m, ok=true){ const t=$('#toast'); t.textContent=m; t.style.background=ok?'#00b894':'#d63031';
  t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),1800); }
async function api(path, opts={}){
  opts.headers = Object.assign({'Content-Type':'application/json','X-Admin-Token':TOKEN}, opts.headers||{});
  const r = await fetch(path, opts);
  if(r.status===401||r.status===503){ logout(); throw new Error('auth'); }
  if(!r.ok){ const e=await r.json().catch(()=>({detail:r.status})); throw new Error(e.detail||r.status); }
  return r.status===204?null:r.json();
}
async function login(){
  const pw=$('#pw').value;
  try{
    const r=await fetch('/admin/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    if(!r.ok){ $('#loginErr').textContent='Sai mật khẩu'; return; }
    TOKEN=pw; sessionStorage.setItem('adm_tok',pw); showApp();
  }catch(e){ $('#loginErr').textContent='Lỗi kết nối'; }
}
function logout(){ TOKEN=''; sessionStorage.removeItem('adm_tok'); $('#app').classList.add('hidden'); $('#login').classList.remove('hidden'); }
function showApp(){ $('#login').classList.add('hidden'); $('#app').classList.remove('hidden'); loadKeys(); loadLogs(); }

function fmtQuota(k){ return k.quota===null||k.quota===undefined ? '<span class="unl">∞</span>' : k.quota; }
async function loadKeys(){
  const keys=await api('/admin/keys');
  const tb=$('#keysTbl tbody'); tb.innerHTML='';
  for(const k of keys){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${k.id}</td><td>${esc(k.name)||'<span class=muted>—</span>'}<br><span class=muted style="font-size:.72rem">${esc(k.note)}</span></td>
      <td><span class="key">${k.key}</span> <button class="ghost sm" onclick="copy('${k.key}')">copy</button></td>
      <td>${k.used} / ${fmtQuota(k)}</td>
      <td>${k.used_today}</td>
      <td><span class="pill ${k.active?'on':'off'}">${k.active?'Hoạt động':'Khoá'}</span></td>
      <td>
        <button class="ghost sm" onclick="editQuota(${k.id})">Sửa quota</button>
        <button class="ghost sm" onclick="toggle(${k.id},${k.active?0:1})">${k.active?'Khoá':'Mở'}</button>
        <button class="ghost sm" onclick="resetUsed(${k.id})">Reset</button>
        <button class="danger sm" onclick="del(${k.id})">Xoá</button>
      </td>`;
    tb.appendChild(tr);
  }
}
async function createKey(){
  const name=$('#nName').value.trim(); const qv=$('#nQuota').value.trim();
  if(!name){ toast('Nhập tên key', false); return; }
  const body={name, note:$('#nNote').value.trim()};
  if(qv==='') body.unlimited=true; else body.quota=parseInt(qv);
  const k=await api('/admin/keys',{method:'POST',body:JSON.stringify(body)});
  $('#newKeyBox').innerHTML=`<div class="newkey">Key mới cho <b>${esc(k.name)}</b>:<br>${k.key}<br>
    <button class="ghost sm" style="margin-top:6px" onclick="copy('${k.key}')">Copy key</button></div>`;
  $('#nName').value=''; $('#nQuota').value=''; $('#nNote').value='';
  toast('Đã tạo key'); loadKeys();
}
async function editQuota(id){
  const v=prompt('Quota mới (số lượt). Để trống = unlimited:');
  if(v===null) return;
  const body = v.trim()==='' ? {unlimited:true} : {quota:parseInt(v)};
  await api('/admin/keys/'+id+'/update',{method:'POST',body:JSON.stringify(body)});
  toast('Đã cập nhật quota'); loadKeys();
}
async function toggle(id,active){ await api('/admin/keys/'+id+'/update',{method:'POST',body:JSON.stringify({active:!!active})}); loadKeys(); }
async function resetUsed(id){ if(!confirm('Reset lượt đã dùng về 0?'))return; await api('/admin/keys/'+id+'/reset',{method:'POST'}); toast('Đã reset'); loadKeys(); }
async function del(id){ if(!confirm('Xoá key này? (xoá cả log của key)'))return; await api('/admin/keys/'+id,{method:'DELETE'}); toast('Đã xoá'); loadKeys(); }
async function loadLogs(){
  const logs=await api('/admin/logs?limit=200'); const tb=$('#logsTbl tbody'); tb.innerHTML='';
  for(const l of logs){
    const tr=document.createElement('tr');
    const ok=l.status==='ok';
    tr.innerHTML=`<td class="muted">${l.created_at?.replace('T',' ')}</td><td>${esc(l.key_name)||l.key_id}</td>
      <td>${esc(l.mst)}</td><td><span class="pill ${ok?'on':'off'}">${l.status}</span></td>
      <td>${l.count_try??''}</td><td class="muted">${esc(l.ip)}</td>`;
    tb.appendChild(tr);
  }
}
async function adminLookup(){
  const mst=$('#aMst').value.trim(); if(!mst){ toast('Nhập MST', false); return; }
  const btn=$('#aBtn'); btn.disabled=true; btn.textContent='Đang tra...';
  $('#aResult').innerHTML='<p class="sub" style="margin-top:10px">⏳ Đang giải captcha + retry (anti-bot, có thể vài lần)...</p>';
  try{
    const d=await api('/admin/lookup?mst='+encodeURIComponent(mst));
    renderLookup(d); loadLogs();
  }catch(e){ $('#aResult').innerHTML='<p class="sub" style="color:#d63031;margin-top:10px">Lỗi: '+esc(e.message)+'</p>'; }
  finally{ btn.disabled=false; btn.textContent='Tra cứu'; }
}
function renderLookup(d){
  if(!d.found){
    $('#aResult').innerHTML=`<div class="newkey" style="background:#2a1a0a;border-color:#e17055;color:#fab1a0">
      Không ra kết quả · count_Try=${d.count_Try} · ${esc(d.status)}<br>${esc(d.message||'')}</div>`;
    return;
  }
  const rows=d.results.map(r=>`<tr><td>${r.stt}</td><td>${esc(r.mst)}</td><td>${esc(r.name)}</td>
    <td>${esc(r.address)}</td><td>${esc(r.tax_authority)}</td><td>${esc(r.status)}</td></tr>`).join('');
  $('#aResult').innerHTML=`
    <div style="margin-top:12px;padding:14px 16px;background:#0a2a1a;border:1px solid #00b894;border-radius:10px">
      <div class="sub" style="margin:0 0 6px">✅ MST <b style="color:#fff">${esc(d.mst)}</b> · count_Try=<b style="color:#feca57">${d.count_Try}</b> · Địa chỉ chính:</div>
      <div style="font-size:1.35rem;font-weight:800;color:#55efc4;line-height:1.35">${esc(d.address)}</div>
    </div>
    <table style="margin-top:12px"><thead><tr><th>STT</th><th>MST</th><th>Tên NNT</th><th>Địa chỉ</th><th>Cơ quan thuế</th><th>Trạng thái</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}
function copy(t){ navigator.clipboard.writeText(t).then(()=>toast('Đã copy key')); }
function esc(s){ return (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
if(TOKEN) showApp();
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/stats")
async def collection_stats():
    """Xem thống kê CAPTCHA đã thu thập."""
    return get_stats()


# ===== Background Crawler =====

import asyncio
import threading
from collector import save_captcha_raw

_crawl_task = None
_crawl_status = {"running": False, "collected": 0, "target": 0, "errors": 0}


def _crawl_worker(target: int, delay: float):
    """Worker chạy trong thread riêng: tải CAPTCHA liên tục."""
    import requests
    global _crawl_status

    _crawl_status = {"running": True, "collected": 0, "target": target, "errors": 0}

    for i in range(target):
        if not _crawl_status["running"]:
            print(f"[Crawler] Stopped by user at {_crawl_status['collected']} images.")
            break

        try:
            # Lấy CAPTCHA từ TCT
            resp = requests.get(config.TCT_CAPTCHA_URL, timeout=15, verify=False)
            resp.raise_for_status()
            data = resp.json()
            svg_content = data["content"]

            # Render SVG → PNG
            png_bytes = tct_client.svg_to_png(svg_content)

            # Lưu ảnh
            path = save_captcha_raw(png_bytes)
            _crawl_status["collected"] += 1

            if _crawl_status["collected"] % 50 == 0:
                print(f"[Crawler] Progress: {_crawl_status['collected']}/{target}")

        except Exception as e:
            _crawl_status["errors"] += 1
            print(f"[Crawler] Error #{_crawl_status['errors']}: {e}")

        # Delay giữa các request (tránh bị rate limit)
        import time
        time.sleep(delay)

    _crawl_status["running"] = False
    print(f"[Crawler] Done! Collected {_crawl_status['collected']} images.")


# Biến trạng thái riêng cho crawler TCT 5 số
_crawl5_status = {"running": False, "collected": 0, "target": 0, "errors": 0}

def _crawl5_worker(target: int, delay: float):
    global _crawl5_status
    import requests
    import time
    import uuid

    _crawl5_status = {"running": True, "collected": 0, "target": target, "errors": 0}

    for i in range(target):
        if not _crawl5_status["running"]: break

        try:
            # Dùng uuid để server không trả về ảnh cũ (cache)
            url = f"{config.TCNNT_PNG_URL}{uuid.uuid4()}"
            resp = requests.get(url, timeout=10, verify=False)
            resp.raise_for_status()

            # Lưu vào folder training_data_5
            # Tên file tạm thời đặt theo timestamp để không trùng
            filename = f"raw_{int(time.time()*1000)}.png"
            filepath = os.path.join(config.TRAINING_DATA_5, filename)
            
            with open(filepath, "wb") as f:
                f.write(resp.content)

            _crawl5_status["collected"] += 1
        except Exception as e:
            _crawl5_status["errors"] += 1
            print(f"[Crawl5] Error: {e}")

        time.sleep(delay)

    _crawl5_status["running"] = False

@app.post("/crawler/tcnnt/start")
async def crawler_tcnnt_start(target: int = 1000, delay: float = 0.5):
    """Bắt đầu cào ảnh 5 số về folder training_data_5."""
    global _crawl_task
    if _crawl5_status["running"]:
        return {"error": "Crawler đang chạy", "status": _crawl5_status}

    task = threading.Thread(target=_crawl5_worker, args=(target, delay), daemon=True)
    task.start()
    return {"message": f"Started crawling {target} images to training_data_5"}

@app.get("/crawler/tcnnt/status")
async def crawler_tcnnt_status():
    """Xem tiến độ cào."""
    return _crawl5_status

@app.post("/crawler/start")
async def crawler_start(target: int = 2000, delay: float = 1.0):
    """
    Bắt đầu tải CAPTCHA hàng loạt.

    - target: số ảnh cần tải (mặc định 2000)
    - delay: giây chờ giữa mỗi request (mặc định 1.0s, tránh rate limit)

    Ước tính thời gian: target × delay giây
    - 2000 ảnh × 1s = ~33 phút
    - 5000 ảnh × 0.5s = ~42 phút
    """
    global _crawl_task

    if _crawl_status["running"]:
        return {"error": "Crawler đang chạy", "status": _crawl_status}

    _crawl_task = threading.Thread(
        target=_crawl_worker,
        args=(target, delay),
        daemon=True
    )
    _crawl_task.start()

    return {
        "message": f"Crawler started! Đang tải {target} ảnh CAPTCHA...",
        "target": target,
        "delay": delay,
        "estimated_minutes": round(target * delay / 60, 1),
    }


@app.get("/crawler/status")
async def crawler_status():
    """Xem trạng thái crawler."""
    stats = get_stats()
    return {
        **_crawl_status,
        "total_in_folder": stats["total_images"],
    }


@app.post("/crawler/stop")
async def crawler_stop():
    """Dừng crawler."""
    _crawl_status["running"] = False
    return {"message": "Crawler stopping..."}


# ===== Feedback & Retrain =====

import csv
from pathlib import Path

LABELS_FILE = Path(__file__).parent / "labels.csv"

_train_status = {"running": False, "epoch": 0, "total_epochs": 0, "val_acc": 0, "message": ""}


class FeedbackRequest(BaseModel):
    image_base64: str
    correct_label: str


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Nhận feedback: lưu ảnh + label đúng vào training_data/ + labels.csv.
    Gọi mỗi khi user nhập Actual text trên demo page.
    """
    if not req.correct_label.strip() or len(req.correct_label.strip()) < 3:
        raise HTTPException(400, "Label phải ít nhất 3 ký tự")

    label = req.correct_label.strip().upper()

    # Lưu ảnh
    img_bytes = base64.b64decode(req.image_base64)
    filepath = save_captcha_raw(img_bytes)
    if not filepath:
        raise HTTPException(500, "Không thể lưu ảnh")

    filename = os.path.basename(filepath)

    # Append vào labels.csv
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, label])

    # Đếm tổng labels
    labeled_count = 0
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labeled_count = sum(1 for _ in csv.reader(f))

    return {"ok": True, "filename": filename, "label": label, "total_labels": labeled_count}


def _train_worker():
    """Chạy train trong thread riêng."""
    global _train_status
    import subprocess
    import sys

    _train_status = {"running": True, "epoch": 0, "total_epochs": 80, "val_acc": 0, "message": "Training..."}

    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True, timeout=3600,
            cwd=str(Path(__file__).parent),
        )

        # Parse kết quả
        output = result.stdout
        if "Best Val Accuracy:" in output:
            import re
            match = re.search(r"Best Val Accuracy:\s*([\d.]+)%", output)
            val_acc = float(match.group(1)) if match else 0
            _train_status["val_acc"] = val_acc
            _train_status["message"] = f"Done! Val Accuracy: {val_acc}%"

            # Reload model trong OCR engine
            engine._custom_model_loaded = False
            engine._custom_model = None
        else:
            _train_status["message"] = f"Train finished. Output: {output[-200:]}"

        if result.returncode != 0:
            _train_status["message"] = f"Error: {result.stderr[-300:]}"

    except Exception as e:
        _train_status["message"] = f"Error: {str(e)}"
    finally:
        _train_status["running"] = False


@app.post("/api/retrain")
async def retrain_model():
    """Trigger retrain model từ labels.csv hiện tại. Chạy background."""
    if _train_status["running"]:
        return {"error": "Đang train rồi!", "status": _train_status}

    t = threading.Thread(target=_train_worker, daemon=True)
    t.start()

    return {"message": "Training started! Chờ ~30-40 phút.", "status": _train_status}


@app.get("/api/train-status")
async def train_status():
    """Xem trạng thái training."""
    labeled_count = 0
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labeled_count = sum(1 for _ in csv.reader(f))

    return {**_train_status, "total_labels": labeled_count}


@app.get("/health")
async def health():
    # Force load model neu chua load (lazy load fallback)
    if not engine._custom_model_loaded:
        engine._load_custom_model()
    model_info = "CNN Model" if engine._custom_model else "EasyOCR + Tesseract"
    return {"status": "ok", "engine": model_info}



@app.get("/debug/captcha")
async def debug_captcha():
    """Debug: Xem raw SVG và PNG từ TCT."""
    import base64
    import config as cfg
    resp = __import__("requests").get(cfg.TCT_CAPTCHA_URL, timeout=15, verify=False)
    data = resp.json()

    svg_content = data["content"]
    png_bytes = tct_client.svg_to_png(svg_content)
    png_b64 = base64.b64encode(png_bytes).decode()

    return {
        "key": data["key"],
        "svg_raw": svg_content[:2000],
        "png_base64": png_b64,
        "png_preview": f"data:image/png;base64,{png_b64}",
    }


# ===== Demo UI =====

DEMO_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>CAPTCHA Detector Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
    min-height: 100vh;
  }
  h1 { color: #00d2d3; margin-bottom: 8px; font-size: 1.6rem; }
  .subtitle { color: #636e72; margin-bottom: 20px; font-size: 0.9rem; }
  .card {
    background: #16213e; border: 2px solid #2d3436; border-radius: 16px;
    padding: 30px; width: 100%; max-width: 700px; text-align: center;
  }
  .captcha-box {
    background: #0a0a1a; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    min-height: 140px; display: flex; align-items: center; justify-content: center;
  }
  .captcha-box img {
    max-width: 100%; height: auto; border-radius: 8px;
    image-rendering: auto;
  }
  .captcha-box .placeholder { color: #444; font-size: 1.1rem; }
  .result-box {
    background: #0a0a1a; border: 2px solid #2d3436; border-radius: 12px;
    padding: 16px; margin: 16px 0; min-height: 60px;
    display: flex; align-items: center; justify-content: center; gap: 16px;
  }
  .result-text {
    font-size: 2.2rem; font-family: monospace; font-weight: 700;
    letter-spacing: 8px; color: #00d2d3;
  }
  .result-conf { color: #636e72; font-size: 0.85rem; }
  .btn-row { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
  button {
    padding: 12px 28px; font-size: 1rem; font-weight: 600; border: none;
    border-radius: 10px; cursor: pointer; transition: all 0.2s;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-load { background: #0984e3; color: #fff; }
  .btn-load:hover:not(:disabled) { background: #0773c5; }
  .btn-detect { background: #00b894; color: #fff; }
  .btn-detect:hover:not(:disabled) { background: #00a381; }
  .btn-both { background: #6c5ce7; color: #fff; }
  .btn-both:hover:not(:disabled) { background: #5a4bd1; }
  .status { margin-top: 16px; font-size: 0.85rem; color: #636e72; min-height: 20px; }
  .history { margin-top: 24px; width: 100%; }
  .history h3 { color: #636e72; font-size: 0.85rem; margin-bottom: 8px; font-weight: 400; }
  .history table { width: 100%; border-collapse: collapse; }
  .history th { color: #636e72; font-size: 0.75rem; text-align: left; padding: 4px 8px; border-bottom: 1px solid #2d3436; }
  .history td { padding: 6px 8px; font-size: 0.85rem; border-bottom: 1px solid #1a1a2e; font-family: monospace; }
  .history .correct { color: #00b894; }
  .history .wrong { color: #d63031; }
  .match-input {
    background: #0f3460; border: 2px solid #3282b8; border-radius: 6px;
    color: #fff; padding: 4px 8px; font-family: monospace; font-size: 0.85rem;
    text-transform: uppercase; width: 85px; text-align: center;
  }
  .match-input.saved { border-color: #00b894; background: #0a3d2e; }
  .btn-fix {
    background: #e17055; color: #fff; border: none; border-radius: 6px;
    padding: 4px 10px; font-size: 0.75rem; cursor: pointer; font-weight: 600;
  }
  .btn-fix:hover { background: #d35400; }
  .btn-fix.done { background: #00b894; cursor: default; }
  .toast {
    position: fixed; top: 20px; right: 20px; padding: 10px 20px; border-radius: 8px;
    color: #fff; font-weight: 500; opacity: 0; transition: opacity 0.3s; z-index: 100;
    pointer-events: none;
  }
  .toast.show { opacity: 1; }
  .train-info {
    margin-top: 12px; padding: 8px 16px; background: #0a0a1a; border-radius: 8px;
    font-size: 0.8rem; color: #636e72;
  }
  .train-info .count { color: #feca57; font-weight: 600; }
  .stats { display: flex; gap: 20px; justify-content: center; margin-top: 12px; }
  .stat { text-align: center; }
  .stat .num { font-size: 1.4rem; font-weight: 700; color: #feca57; }
  .stat .lbl { font-size: 0.7rem; color: #636e72; }
</style>
</head>
<body>
  <h1>CAPTCHA Detector</h1>
  <p class="subtitle">Load CAPTCHA tu TCT → Detect bang CNN Model (95.4% accuracy)</p>

  <div class="card">
    <div class="captcha-box" id="captchaBox">
      <span class="placeholder">Nhan "Load" de lay CAPTCHA</span>
    </div>

    <div class="result-box" id="resultBox" style="display:none;">
      <div>
        <div class="result-text" id="resultText"></div>
        <div class="result-conf" id="resultConf"></div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn-load" id="btnLoad" onclick="loadCaptcha()">Load CAPTCHA</button>
      <button class="btn-detect" id="btnDetect" onclick="detectCaptcha()" disabled>Detect</button>
      <button class="btn-both" id="btnBoth" onclick="loadAndDetect()">Load + Detect</button>
      <button style="background:#fdcb6e;color:#000;" onclick="retrain()">Retrain</button>
    </div>

    <div class="status" id="status"></div>

    <div class="stats" id="statsBox" style="display:none;">
      <div class="stat"><div class="num" id="totalCount">0</div><div class="lbl">Total</div></div>
      <div class="stat"><div class="num" id="correctCount">0</div><div class="lbl">Correct</div></div>
      <div class="stat"><div class="num" id="accuracyPct">0%</div><div class="lbl">Accuracy</div></div>
    </div>

    <div class="train-info" id="trainInfo">
      Training data: <span class="count" id="labelCount">0</span> labels |
      New feedback: <span class="count" id="feedbackCount">0</span>
    </div>
  </div>

  <div id="toast" class="toast"></div>

  <div class="card history" id="historyCard" style="margin-top:20px; display:none;">
    <h3>History - Nhap text dung, nhan Fix de luu vao training data</h3>
    <table>
      <thead><tr><th>#</th><th>Image</th><th>Predicted</th><th>Actual</th><th></th><th>Match</th></tr></thead>
      <tbody id="historyBody"></tbody>
    </table>
  </div>

<script>
let currentImage = null;
let currentKey = null;
let history = [];
let totalChecked = 0;
let totalCorrect = 0;
let feedbackCount = 0;

function setStatus(msg) { document.getElementById('status').textContent = msg; }


function showToast(msg, color) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = color || '#00b894';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

function updateTrainInfo() {
  fetch('/api/train-status').then(r => r.json()).then(data => {
    document.getElementById('labelCount').textContent = data.total_labels || 0;
  }).catch(() => {});
  document.getElementById('feedbackCount').textContent = feedbackCount;
}

async function loadCaptcha() {
  const btn = document.getElementById('btnLoad');
  btn.disabled = true;
  setStatus('Dang lay CAPTCHA tu TCT...');
  document.getElementById('resultBox').style.display = 'none';

  try {
    const res = await fetch('/debug/captcha');
    const data = await res.json();
    currentImage = data.png_base64;
    currentKey = data.key;

    document.getElementById('captchaBox').innerHTML =
      '<img src="data:image/png;base64,' + currentImage + '" alt="CAPTCHA">';
    document.getElementById('btnDetect').disabled = false;
    setStatus('CAPTCHA loaded! Nhan "Detect" de nhan dang.');
  } catch(e) {
    setStatus('Loi: ' + e.message);
  }
  btn.disabled = false;
}

async function detectCaptcha() {
  if (!currentImage) return;
  const btn = document.getElementById('btnDetect');
  btn.disabled = true;
  setStatus('Dang nhan dang...');

  try {
    const res = await fetch('/one/gettext', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data: currentImage})
    });
    const data = await res.json();

    document.getElementById('resultText').textContent = data.result;
    document.getElementById('resultConf').textContent = '';
    document.getElementById('resultBox').style.display = '';
    setStatus('Done!');

    addHistory(data.result);
  } catch(e) {
    setStatus('Loi: ' + e.message);
  }
  btn.disabled = false;
}

async function loadAndDetect() {
  await loadCaptcha();
  if (currentImage) await detectCaptcha();
}

function addHistory(predicted) {
  const idx = history.length + 1;
  history.push({idx, predicted, actual: '', image: currentImage, saved: false});

  const tbody = document.getElementById('historyBody');
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${idx}</td>
    <td><img src="data:image/png;base64,${currentImage}" style="height:35px;border-radius:4px;"></td>
    <td style="color:#00d2d3;font-weight:700;letter-spacing:2px;">${predicted}</td>
    <td><input class="match-input" id="actual_${idx}" placeholder="..." onkeydown="if(event.key==='Enter'){fixLabel(${idx})}"></td>
    <td><button class="btn-fix" id="fix_${idx}" onclick="fixLabel(${idx})">Fix</button></td>
    <td id="match_${idx}">-</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);

  document.getElementById('historyCard').style.display = '';
  updateTrainInfo();
}

async function fixLabel(idx) {
  const inputEl = document.getElementById('actual_' + idx);
  const input = inputEl.value.toUpperCase().trim();
  if (!input) { inputEl.focus(); return; }

  const item = history[idx - 1];
  if (item.saved) { showToast('Da luu roi!', '#636e72'); return; }

  item.actual = input;
  const match = item.predicted === input;

  const cell = document.getElementById('match_' + idx);
  cell.textContent = match ? 'OK' : 'SAI';
  cell.className = match ? 'correct' : 'wrong';

  // Save feedback to training data
  const fixBtn = document.getElementById('fix_' + idx);
  fixBtn.textContent = '...';
  fixBtn.disabled = true;

  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_base64: item.image, correct_label: input})
    });
    const data = await res.json();

    if (data.ok) {
      item.saved = true;
      feedbackCount++;
      inputEl.classList.add('saved');
      inputEl.readOnly = true;
      fixBtn.textContent = 'Saved';
      fixBtn.className = 'btn-fix done';
      showToast('Saved: ' + input + ' → ' + data.filename, '#00b894');
    } else {
      fixBtn.textContent = 'Fix';
      fixBtn.disabled = false;
      showToast('Error: ' + (data.detail || 'unknown'), '#d63031');
    }
  } catch(e) {
    fixBtn.textContent = 'Fix';
    fixBtn.disabled = false;
    showToast('Network error', '#d63031');
  }

  // Recalculate stats
  totalChecked = 0; totalCorrect = 0;
  history.forEach(h => {
    if (h.actual) {
      totalChecked++;
      if (h.predicted === h.actual) totalCorrect++;
    }
  });

  document.getElementById('statsBox').style.display = '';
  document.getElementById('totalCount').textContent = totalChecked;
  document.getElementById('correctCount').textContent = totalCorrect;
  document.getElementById('accuracyPct').textContent =
    totalChecked > 0 ? Math.round(totalCorrect/totalChecked*100) + '%' : '0%';

  updateTrainInfo();
}

async function retrain() {
  if (!confirm('Bat dau retrain model? (~30-40 phut)')) return;
  try {
    const res = await fetch('/api/retrain', {method: 'POST'});
    const data = await res.json();
    setStatus(data.message || data.error);
    if (!data.error) pollTrainStatus();
  } catch(e) { setStatus('Retrain error: ' + e.message); }
}

async function pollTrainStatus() {
  const poll = setInterval(async () => {
    try {
      const res = await fetch('/api/train-status');
      const data = await res.json();
      setStatus('Training... ' + data.message);
      if (!data.running) {
        clearInterval(poll);
        setStatus('Train done! ' + data.message);
      }
    } catch(e) { clearInterval(poll); }
  }, 5000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); loadAndDetect(); }
});

// Init
updateTrainInfo();
</script>
</body>
</html>"""


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Demo UI: Load CAPTCHA từ TCT → Detect bằng CNN Model."""
    return DEMO_HTML

@app.get("/debug/captcha5")
async def debug_captcha5():
    """Debug: Lấy CAPTCHA 5 số từ TCNNT dạng PNG trực tiếp."""
    import uuid
    import requests as req

    url = f"{config.TCNNT_PNG_URL}{uuid.uuid4()}"
    resp = req.get(url, timeout=10, verify=False)
    resp.raise_for_status()

    png_b64 = base64.b64encode(resp.content).decode()
    return {
        "key": str(uuid.uuid4()),   # TCNNT PNG không có key, tạo dummy
        "png_base64": png_b64,
        "png_preview": f"data:image/png;base64,{png_b64}",
    }

DEMO1_HTML = DEMO_HTML \
    .replace(
        "CAPTCHA Detector Demo",
        "CAPTCHA Detector Demo - 5 So"
    ) \
    .replace(
        "Load CAPTCHA tu TCT \u2192 Detect bang CNN Model (95.4% accuracy)",
        "Load CAPTCHA 5 so tu TCNNT \u2192 Detect bang CNN Model (99.1% accuracy)"
    ) \
    .replace(
        "const res = await fetch('/debug/captcha');",
        "const res = await fetch('/debug/captcha5');"
    ) \
    .replace(
        ".captcha-box img {\n    max-width: 100%; height: auto; border-radius: 8px;\n    image-rendering: auto;\n  }",
        ".captcha-box img {\n    max-width: 100%; height: auto; border-radius: 8px;\n    image-rendering: auto; background: white; padding: 8px;\n  }"
    )

@app.get("/demo1", response_class=HTMLResponse)
async def demo1_page():
    """Demo UI 5 số: Load CAPTCHA từ TCNNT → Detect bằng CNN Model v6."""
    return DEMO1_HTML
# ===== Chạy trực tiếp =====

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    import config
    import os
    # reload=True chi cho dev. Production set bien moi truong: ENV=prod
    is_prod = os.environ.get("ENV", "").lower() == "prod"
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=not is_prod,
    )

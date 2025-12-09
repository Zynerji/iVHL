# VHL Launcher Scripts Guide

Quick start scripts to run the VHL simulation with one click!

## 🚀 Quick Start (Choose One)

### Option 1: PowerShell Script (Recommended for Windows)

**Features:**
- ✅ Automatic git pull for latest updates
- ✅ Port conflict detection
- ✅ Optional Flask API backend
- ✅ Clean shutdown handling
- ✅ Live server status monitoring

**Basic Usage:**
```powershell
.\start_vhl.ps1
```

**With Flask API Backend:**
```powershell
.\start_vhl.ps1 -WithAPI
```

**Custom Port:**
```powershell
.\start_vhl.ps1 -Port 9000
```

**Combined:**
```powershell
.\start_vhl.ps1 -WithAPI -Port 9000
```

### Option 2: Batch File (Simple Windows)

**Features:**
- ✅ Simple one-click launch
- ✅ No PowerShell execution policy issues
- ✅ Works on all Windows versions

**Usage:**
```batch
start_vhl.bat
```

Just double-click `start_vhl.bat` in File Explorer!

### Option 3: Manual (All Platforms)

**Linux/Mac:**
```bash
cd /path/to/Vibrational-Helix-Lattice
git pull
python -m http.server 8000 &
open http://localhost:8000/vhl_webgpu.html  # Mac
xdg-open http://localhost:8000/vhl_webgpu.html  # Linux
```

**Windows:**
```batch
cd C:\path\to\Vibrational-Helix-Lattice
git pull
python -m http.server 8000
start http://localhost:8000/vhl_webgpu.html
```

## 🛠️ Setup

### First Time Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zynerji/Vibrational-Helix-Lattice.git
   cd Vibrational-Helix-Lattice
   ```

2. **Checkout the branch:**
   ```bash
   git checkout claude/vhl-simulation-python-018UQEAx3YkVW36Afnoq3ixV
   ```

3. **Run the launcher:**
   - **PowerShell:** Right-click `start_vhl.ps1` → Run with PowerShell
   - **Batch:** Double-click `start_vhl.bat`

### PowerShell Execution Policy (First Time Only)

If you get "execution policy" errors with PowerShell:

**Option A: Bypass for Single Session**
```powershell
powershell -ExecutionPolicy Bypass -File start_vhl.ps1
```

**Option B: Enable for Current User (Permanent)**
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Option C: Use Batch File Instead**
Just use `start_vhl.bat` - no policy issues!

## 📋 What the Scripts Do

### Startup Sequence

1. **Navigate to Repository**
   - Changes to the script's directory
   - Ensures all relative paths work correctly

2. **Pull Latest Updates** (PowerShell only)
   - Fetches latest code from git
   - Shows current branch
   - Continues even if git fails

3. **Check Python**
   - Verifies Python is installed
   - Shows Python version
   - Exits with error if missing

4. **Start Flask API** (if `-WithAPI` flag)
   - Installs dependencies if needed
   - Starts Flask in background job
   - Runs on http://localhost:5000

5. **Start HTTP Server**
   - Starts Python HTTP server
   - Default port: 8000
   - Auto-detects port conflicts (PowerShell)

6. **Open Browser**
   - Launches default browser
   - Opens http://localhost:8000/vhl_webgpu.html
   - Waits 2 seconds for server startup

7. **Monitor Status** (PowerShell only)
   - Shows live timestamp
   - Monitors server health
   - Clean shutdown on Ctrl+C

### Shutdown Sequence (PowerShell)

When you press Ctrl+C:

1. Stops HTTP server job
2. Stops Flask API job (if running)
3. Removes background jobs
4. Shows confirmation messages

## 🔧 Troubleshooting

### "Python not found"

**Solution:** Install Python from https://www.python.org/
- Check "Add Python to PATH" during installation
- Restart terminal after installation

### "Port 8000 already in use"

**PowerShell:** Automatically switches to port 8001

**Batch/Manual:** Kill the process:
```batch
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

Or use a different port:
```powershell
.\start_vhl.ps1 -Port 9000
```

### "Git not found" (Warning)

**Solution:** Install Git from https://git-scm.com/
- Not required, but recommended for auto-updates
- Scripts will continue without git

### Browser doesn't open

**Manual open:**
- Navigate to http://localhost:8000/vhl_webgpu.html

### Flask API won't start

**Check dependencies:**
```bash
pip install flask flask-cors pyscf
```

**Verify it's running:**
```bash
curl http://localhost:5000/api/health
```

### "THREE.OrbitControls is not a constructor" error

**Solution:** Clear browser cache!
- Press Ctrl+Shift+R (hard refresh)
- Or use Incognito mode (Ctrl+Shift+N)

### PowerShell closes immediately

**Solution:** Run from PowerShell terminal, not double-click:
1. Right-click in folder → "Open PowerShell here"
2. Run: `.\start_vhl.ps1`

## 🎮 Using the Simulation

Once the browser opens:

### Mouse Controls
- **Left drag:** Rotate camera
- **Scroll:** Zoom in/out
- **Right drag:** Pan view
- **Double-click:** Reset camera

### UI Controls (Left Panel)
- **Element Focus:** Select any element (Z=1-126)
- **Geometry Sliders:** Adjust helix shape
- **Force Parameters:** Tune fifth-force
- **Simulation Buttons:**
  - ▶️ Start/Pause dynamics
  - 🔄 Reset simulation
  - 🤖 ML predictions
  - 💾 Export CSV data

### Browser Console (F12)
Check for:
- ✅ "WebGPU initialized" (best performance)
- ⚠️ "WebGL fallback" (still works)
- ❌ Errors (report as issue)

## 📊 Performance Tips

### For Best Performance
1. Use Chrome/Edge 113+ (WebGPU support)
2. Close other GPU-intensive apps
3. Enable hardware acceleration in browser settings
4. Set animation speed to 1.0x or lower

### If FPS is Low
- Disable force vectors (set to "None")
- Reduce particle size
- Close other tabs
- Try Firefox (different GPU handling)

## 🆘 Getting Help

### Check Logs

**PowerShell jobs:**
```powershell
Get-Job
Receive-Job -Id <job_id>
```

**Python server logs:**
Look in terminal for HTTP requests

**Flask API logs:**
Look for errors when starting with `-WithAPI`

### Report Issues

GitHub Issues: https://github.com/Zynerji/Vibrational-Helix-Lattice/issues

Include:
- Operating system & version
- Python version (`python --version`)
- Browser & version
- Error messages from console (F12)
- Steps to reproduce

## 🔄 Updating

### Automatic (Using Launchers)

PowerShell script automatically pulls updates!

### Manual Update

```bash
git pull origin claude/vhl-simulation-python-018UQEAx3YkVW36Afnoq3ixV
```

### Check Current Version

```bash
git log -1 --oneline
```

Should show: `8cfd6cc Fix OrbitControls constructor initialization` or newer

## 🚀 Advanced Usage

### Run Flask API Separately

**Terminal 1:**
```bash
python vhl_api.py
```

**Terminal 2:**
```bash
python -m http.server 8000
```

### Custom Flask Port

Edit `vhl_api.py` line 385:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Change port
```

### Multiple Instances

Run on different ports:
```powershell
.\start_vhl.ps1 -Port 8001
.\start_vhl.ps1 -Port 8002
```

### Remote Access

**⚠️ Security Warning:** Only use on trusted networks!

```bash
python -m http.server 8000 --bind 0.0.0.0
```

Access from other devices: `http://<your-ip>:8000/vhl_webgpu.html`

## 📝 Script Customization

### Modify Default Port

**PowerShell:** Edit line 5:
```powershell
[int]$Port = 9000  # Change default
```

**Batch:** Edit line 40:
```batch
python -m http.server 9000
```

### Auto-start Flask API

**PowerShell:** Edit line 4:
```powershell
[switch]$WithAPI = $true  # Always start API
```

### Change Browser

**PowerShell:** Edit line 106:
```powershell
Start-Process chrome $URL  # Force Chrome
Start-Process firefox $URL  # Force Firefox
```

## 🎓 Learn More

- **Full Documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)
- **WebGPU Guide:** [WEBGPU_GUIDE.md](WEBGPU_GUIDE.md)
- **Main README:** [README.md](README.md)

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

**Happy simulating! 🌀✨**

*For questions or issues, open a GitHub issue or contact the maintainers.*

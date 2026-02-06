# 🚀 QUICK DEPLOYMENT STEPS

## ✅ ĐÃ HOÀN THÀNH:
- Git repository initialized
- All files committed
- Remote added: https://github.com/Thanh36-jqk/Bio.me.git

---

## 📍 BƯỚC TIẾP THEO:

### 1. PUSH CODE LÊN GITHUB

Mở PowerShell trong thư mục `E:\Project\MIDTERM` và chạy:

```powershell
git push -u origin main
```

**Nếu yêu cầu authentication:**
- Username: `Thanh36-jqk`
- Password: Dùng **Personal Access Token** (không phải password GitHub)

**Tạo Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Chọn scope: `repo` (full control)
3. Copy token và dùng làm password

---

### 2. DEPLOY BACKEND TO RENDER (MIỄN PHÍ)

1. **Đăng ký Render:** https://render.com/
2. **New Web Service** → Connect GitHub
3. **Select repository:** `Bio.me`
4. **Settings:**
   - Name: `bio-me-backend`
   - Region: `Singapore`
   - Branch: `main`
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements-deploy.txt`
   - Start Command: `python main_simple.py`

5. **Environment Variables:** (trong Render dashboard)
   ```
   MONGODB_URI = mongodb+srv://... (từ Atlas - tạo ở bước 3)
   PYTHONUNBUFFERED = 1
   PORT = 8000
   ```

6. **Deploy** → Đợi ~5 phút

**Backend URL:** `https://bio-me-backend.onrender.com`

---

### 3. SETUP MONGODB ATLAS (MIỄN PHÍ)

1. **Đăng ký:** https://www.mongodb.com/cloud/atlas/register
2. **Create FREE Cluster:**
   - Provider: AWS
   - Region: Singapore
   - Tier: M0 (FREE)
3. **Database Access:**
   - Add user: `admin` / (strong password)
   - Role: Read/Write
4. **Network Access:**
   - Add IP: `0.0.0.0/0` (allow all)
5. **Get Connection String:**
   - Connect → Drivers → Python
   - Copy: `mongodb+srv://admin:<password>@cluster.mongodb.net/`
   - Thay `<password>` bằng password thực

---

### 4. DEPLOY FRONTEND TO VERCEL (MIỄN PHÍ)

1. **Đăng ký Vercel:** https://vercel.com/signup
2. **Import Project** → GitHub → `Bio.me`
3. **Settings:**
   - Framework: Next.js (auto-detect)
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`

4. **Environment Variables:**
   ```
   NEXT_PUBLIC_API_URL = https://bio-me-backend.onrender.com
   ```

5. **Deploy** → Đợi ~2 phút

**Frontend URL:** `https://bio-me.vercel.app`

---

## ✅ VERIFICATION

### Check Backend:
```bash
curl https://bio-me-backend.onrender.com/
```
Should return: `{"status":"online",...}`

### Check Frontend:
Mở browser: `https://bio-me.vercel.app`

### Test Full Flow:
1. Register user → Upload biometrics
2. Check MongoDB Atlas → Users collection
3. Login → Verify authentication

---

## 🎯 DEPLOYMENT CHECKLIST

- [ ] Push code to GitHub
- [ ] MongoDB Atlas cluster created & connection string obtained
- [ ] Render backend deployed & environment variables set
- [ ] Vercel frontend deployed & API URL configured
- [ ] Test registration works
- [ ] Test login works
- [ ] ✅ System LIVE!

---

## 📝 NOTES

**Backend Free Tier:**
- Render sleeps after 15 min inactive
- Wake up time: ~30 seconds
- Keep alive: Setup cron at https://cron-job.org

**Giới hạn miễn phí:**
- Render: 512MB RAM, sleep sau 15 min
- Vercel: 100GB bandwidth/month
- MongoDB: 512MB storage

**Chi phí:** $0/month 🎉

# Biometric Auth Project - Kullanim Rehberi

Bu dokuman, projeyi sifirdan calistirmak ve demo uzerinden test etmek icin izlenecek adimlari verir.

## 1) Proje ne yapiyor?

Proje iki ana parcadan olusur:

- Frontend (React): Kullanici demo arayuzu (Enroll, Authenticate, Attack, Reset)
- Backend (FastAPI): API endpointleri, model/atak calismalari ve istatistikler

Ek olarak su scriptler vardir:

- Veri pipeline: sentetik/veri onisleme adimlari
- Model egitimi: kimlik dogrulama modelleri
- Saldiri degerlendirmesi: FGSM/PGD/Mimicry testleri

## 2) On kosullar

Sisteminizde sunlar yuklu olmali:

- Python 3.10+
- Node.js 18+ ve npm

Opsiyonel ama onerilen:

- Python virtual environment (venv)

## 3) Kurulum

### 3.1 Python bagimliliklari

Proje kok dizininde:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.2 Frontend bagimliliklari

```powershell
cd frontend
npm install
cd ..
```

## 4) Calistirma senaryolari

## Senaryo A - Sadece demo UI (hizli)

Bu senaryo, arayuzu hizlica acmak icindir.

```powershell
cd frontend
npm start
```

Tarayicida acilan adresten demo ekranini kullanabilirsiniz.

Not: Bu repodaki `BiometricAuthDemo` bileseni local hesaplama da yaptigi icin temel demo akisinda backend zorunlu degildir.

## Senaryo B - Tam akisiyla calistirma (onerilen)

### Adim 1: Veri pipeline

```powershell
cd backend\data
python run_pipeline.py
cd ..\..
```

### Adim 2: Model egitimi

```powershell
cd backend\models
python train_models.py
cd ..\..
```

### Adim 3: Backend API

```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

API ayakta oldugunda test:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

### Adim 4: Frontend

Yeni bir terminalde:

```powershell
cd frontend
npm start
```

## 5) Demo ekrani kullanici adimlari

Arayuzde tipik kullanim su sekildedir:

1. Enroll sekmesine gec.
2. Verilen metni en az 5 kez yazip ornekleri gonder.
3. Profil olusunca Authenticate sekmesine gec.
4. Ayni metni yazarak dogrulama sonucu al.
5. (Opsiyonel) Attack sekmesinde FGSM/PGD/Mimicry deneyerek modelin dayanikliligini gor.
6. Bastan baslamak icin ustteki Reset butonunu kullan.

## 6) Sonuclar nerede?

- Model egitim sonuclari: `datasets/trained_models/results.json`
- Fusion sonuclari: `datasets/trained_models/fusion_results.json`
- Attack sonuclari: `datasets/attack_results/attack_results.json`

## 7) SIK sorunlar ve cozumler

### Sorun: `python run_pipeline.py` hata veriyor

- Virtual env aktif mi kontrol edin.
- `pip install -r requirements.txt` tekrar calistirin.
- Komutu dogru klasorde calistirin: `backend/data`

### Sorun: `npm start` hata veriyor

- Komutu `frontend` klasorunde calistirin.
- Once `npm install` yaptiginizdan emin olun.
- Node surumunu kontrol edin (`node -v`).

### Sorun: CORS veya API erisim problemi

- Backend calisiyor mu kontrol edin (`http://127.0.0.1:8000/docs`).
- Frontend ve backend ayni anda acik olsun.

## 8) Onerilen terminal sirasi (Windows)

Terminal 1 (Backend):

```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

Terminal 2 (Frontend):

```powershell
cd frontend
npm start
```

Terminal 3 (Opsiyonel egitim/atak):

```powershell
cd backend\models
python train_models.py

cd ..\attacks
python run_attacks.py
```

---

Ihtiyac olursa bu rehbere "tek komutla calistirma" icin bir `run_all.ps1` scripti de eklenebilir.

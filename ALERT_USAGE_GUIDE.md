# Streamlit Uyarı Mesajları Kullanım Kılavuzu

## 4 Farklı Uyarı Tipi Var:

### 🔵 `st.info()` - BİLGİLENDİRME (Açık Mavi)
**Ne zaman kullanılır?**
- Kullanıcıya tarafsız bilgi vermek için
- Bir şeyin henüz yapılmadığını bildirmek için
- İpucu veya yönlendirme mesajları için

**Örnek:**
```python
st.info("Önce sipariş oluşturun.")
st.info("Henüz bir adres sorgulanmadı.")
```

**Anlam:** "Buraya dikkat et, bu bilgiyi bil" - Herhangi bir sorun yok, sadece bilgilendirme

---

### ⚠️ `st.warning()` - UYARI (Turuncu/Sarı)
**Ne zaman kullanılır?**
- Bir şeyin yanlış gitmediğini ama dikkat edilmesi gerektiğini belirtmek için
- Veri temizleme/filtreleme sonuçlarını göstermek için
- "İşlem başarılı ama bazı şeyler değişti" durumları için

**Örnek:**
```python
st.warning(f"❗ İstanbul dışı {removed_city_count} sipariş çıkarıldı.")
st.warning(f"❗ Avrupa yakasından {removed_count} sipariş çıkarıldı.")
```

**Anlam:** "Dikkat! İşlem yapıldı ama bazı veriler çıkarıldı/değişti" - Sorun değil ama bilmen lazım

---

### ✅ `st.success()` - BAŞARILI (Yeşil)
**Ne zaman kullanılır?**
- Bir işlemin başarıyla tamamlandığını göstermek için
- Dosya yüklemesi, hesaplama, kayıt gibi işlemler başarılı olduğunda
- Pozitif geri bildirim vermek için

**Örnek:**
```python
st.success("✔ Dosya yüklendi.")
st.success("📥 Excel başarıyla yüklendi!")
st.success("📦 Sipariş tablosu oluşturuldu.")
st.success("OSRM matrisleri hazır!")
```

**Anlam:** "Harika! İşlem başarıyla tamamlandı" - Her şey yolunda

---

### ❌ `st.error()` - HATA (Kırmızı)
**Ne zaman kullanılır?**
- Bir hata oluştuğunda
- İşlem başarısız olduğunda
- Gerekli bir şey eksik olduğunda
- Kullanıcının bir şey yapması gerektiğinde

**Örnek:**
```python
st.error("🔑 OPENCAGE_API_KEY not found. Please add it to Streamlit secrets.")
st.error("❌ Adres bulunamadı")
st.error("📭 İstanbul içinde işlenecek sipariş yok.")
st.error(f"❌ Eksik kolonlar: {missing}")
```

**Anlam:** "Sorun var! Bu düzeltilmeden devam edilemez" - Kritik hata

---

## Özet Karşılaştırma:

| Tip | Renk | Kullanım | Durum |
|-----|------|----------|-------|
| `info` | 🔵 Mavi | Bilgilendirme, ipucu | Tarafsız |
| `warning` | ⚠️ Turuncu | Dikkat, veri değişikliği | Olumlu ama dikkat |
| `success` | ✅ Yeşil | İşlem başarılı | Pozitif |
| `error` | ❌ Kırmızı | Hata, sorun | Negatif |

---

## App.py'deki Kullanımlar:

**Info (Mavi) - "Henüz yapılmadı":**
- "Henüz bir adres sorgulanmadı."
- "Önce sipariş oluşturun."

**Warning (Turuncu) - "Bazı şeyler çıkarıldı":**
- "İstanbul dışı X sipariş çıkarıldı."
- "Avrupa yakasından X sipariş çıkarıldı."

**Success (Yeşil) - "İşlem başarılı":**
- "Dosya yüklendi."
- "Excel başarıyla yüklendi!"
- "Sipariş tablosu oluşturuldu."

**Error (Kırmızı) - "Sorun var":**
- "API KEY bulunamadı."
- "Adres bulunamadı."
- "İşlenecek sipariş yok."
- "Eksik kolonlar var."

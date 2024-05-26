from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from transformers import pipeline

import time

def is_page_loaded(driver):
    return driver.execute_script("return document.readyState") == "complete"

#önişleme adımı
def preprocess_text(text):
    if "bkz" in text.lower():
        return None
    return text

# Ekşisözlük'teki başlığı buradan alıyoruz istersek farklı bir url girebiliriz
url = "https://eksisozluk111.com/1-ocak-2024-kopru-ve-otoyol-zammi--7763438"

# Selenium tarayıcısını firefox ile başlatıyoruz, chrome veya başka bir tarayıcı ile değiştirebliriz
driver = webdriver.Firefox()

driver.get(url)

# Sayfanın tam yüklenmesini bekliyoruz süresi değiştirilebilir default olarak 30 saniye belirledim
wait = WebDriverWait(driver, 30)

# JavaScript ile sayfanın yüklenip yüklenmediğini kontrol ediyoruz yoksa program sonlanabilir
wait.until(lambda driver: is_page_loaded(driver))

# Reklam engelleme için JavaScript ekleyelim, tam sayfa reklam çıkarsa sorun yaratabiliyor (çözemedim)
adblock_script = """
    var style = document.createElement('style');
    style.innerHTML = '.adsbygoogle, .adsbygoogle-placeholder { display: none !important; }';
    document.head.appendChild(style);
"""

# Reklam engelleme scriptini çalıştırıyoruz bu olmazsa program durabiliyor
driver.execute_script(adblock_script)

# Duygu analizi yapmak için Hugging Face'den bir modeli yüklüyoruz, burada ben savasy'nin türkçe dil işleme modelini kullandım
classifier = pipeline('sentiment-analysis', model="savasy/bert-base-turkish-sentiment-cased")

# Pozitif, negatif ve elenen başlık sayaçları
positive_count = 0
negative_count = 0
filtered_count = 0

# Entry'leri çekmek için bir döngü oluşturduk
while True:
    # Sayfa içeriğini BeautifulSoup kullanarak çektik
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Entry'leri çekiyoruz
    entry_elements = soup.select(".content")

    # Entry'leri değerlendiriyoruz
    for entry in entry_elements:
        text = entry.get_text(strip=True)
        preprocessed_text = preprocess_text(text)
        
        if preprocessed_text:
            try:
                # Duygu analizi yapıyoruz
                result = classifier(preprocessed_text)[0]
                sentiment_label = result['label']

                # Pozitif ve negatif metinleri sayıyoruz
                if sentiment_label == 'positive':
                    positive_count += 1
                elif sentiment_label == 'negative':
                    negative_count += 1

                print(f"Metin: {preprocessed_text}")
                print(f"Duygu Analizi: {sentiment_label}\n")

            except Exception as e:
                print(f"Çeviri sırasında bir hata oluştu: {e}")

        else:
            # "bkz" içeren başlık sayacını artırıyoruz
            filtered_count += 1

    try:
        # "Sonraki Sayfa" butonunu bul ve tıkla metodu aşağıda yer alıyor
        next_page_button = driver.find_element(By.CSS_SELECTOR, ".pager .next")
        next_page_button.click()
        time.sleep(5)  # Geçişin tamamlanmasını beklemek için kısa bir süre bekliyoruz yoksa program hata verebiliyor (internet hızına bağlı)

    except Exception as e:
        print(f"Sayfaların sonuna ulaşıldı veya reklam engellenemedi. Program sona eriyor. Hata: {e}")
        break

# Genel çıktıyı yazdırıyoruz
print(f"Toplam Pozitif Metin Sayısı: {positive_count}")
print(f"Toplam Negatif Metin Sayısı: {negative_count}")
print(f"Toplam Elenen Başlık Sayısı: {filtered_count}")

# Tarayıcıyı kapatıyoruz
driver.quit()

print("\n")

if positive_count > negative_count:
    print("Duygu durumu genel olarak pozitif")
else:
    print("Duygu durumu genel olarak negatif")

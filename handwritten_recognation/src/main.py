import os
import cv2

import numpy as np

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from train_model import *
from test_model import *



class HandwritingOCR:
    def __init__(self, input_folder='images', output_folder='outputs', language='tr'):
        """
        OCR sınıfını başlatır.
        Args:
            input_folder (str): Resimlerin bulunduğu giriş klasörü
            output_folder (str): Çıktı metin dosyalarının kaydedileceği klasör
            language (str): OCR için kullanılacak dil ('tr' = Türkçe, 'en' = İngilizce)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.ocr = PaddleOCR(use_gpu=False, lang=language, use_angle_cls=True, drop_score=0.5)
        self._create_output_folder()



    def process_all_images(self):
        image_files = [f for f in os.listdir(self.input_folder) if imghdr.what(os.path.join(self.input_folder, f))]
        total_images = len(image_files)
        print(f"Toplam {total_images} resim işlenecek...")

        def process_image_wrapper(image_file):
            image_path = os.path.join(self.input_folder, image_file)
            print(f"İşleniyor: {image_path}")
            extracted_text, result = self.process_image(image_path)
            if extracted_text:
                self.save_extracted_text(image_file, extracted_text)
                self.annotate_image(image_path, result)

        with ThreadPoolExecutor(max_workers=4) as executor:  # Paralel işlem
            executor.map(process_image_wrapper, image_files)


    def _create_output_folder(self):
        """Çıktı klasörünü oluşturur (yoksa)."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def preprocess_image(self, image_path):
        try:
            img = Image.open("image_path.jpg")
            img.verify()  # Dosyanın bozuk olup olmadığını kontrol eder
            print("Bu geçerli bir görüntü dosyasıdır.")
        except Exception as e:
            print("Geçersiz görüntü dosyası:", e)
        try:
            print(f"Ön işleniyor: {image_path}")
            image = cv2.imread(image_path, 0)  # Gri tonlama
            image = cv2.GaussianBlur(image, (5, 5), 0)  # Gürültü kaldır
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Kontrast artırma (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

            # Kenar silme (kenar boşluklarını kaldırma)
            image = image[10:-10, 10:-10]  # Kenardan 10 piksel kırpma
            image = cv2.medianBlur(image, 3)  # Gürültü azaltma (3x3 kernel)
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Yazıları netleştir

            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Eşikleme
            processed_image_path = os.path.join(self.output_folder, 'processed_' + os.path.basename(image_path))
            cv2.imwrite(processed_image_path, image)
            return processed_image_path
        except Exception as e:
            print(f"Hata: Resim ön işleme sırasında hata oluştu: {e}")
            return image_path

    def process_image(self, image_path):
        """
        Verilen bir resim dosyasından metni çıkarır ve paragraf olarak birleştirir.
        Args:
            image_path (str): Resim dosyasının yolu
        Returns:
            extracted_text (str): Resimden çıkarılan metin
            result: OCR tarafından çıkarılan tam sonuç
        """
        try:
            print(f"İşleniyor: {image_path}")
            image_path = self.preprocess_image(image_path)  # Ön işlem uygula
            result = self.ocr.ocr(image_path, cls=True)  # OCR işlemi uygula

            extracted_text = ' '.join([line[1][0] for line in result[0]])
            extracted_text = ' '.join(extracted_text.split())  # Gereksiz boşlukları temizle
            print(f"Çıkarılan metin: {extracted_text}\n")
            return extracted_text, result
        except Exception as e:
            print(f"Hata: OCR işlemi sırasında hata oluştu: {e}")
            return "", None

    def save_extracted_text(self, image_filename, extracted_text):
        """Çıkarılan metni bir .txt dosyasına kaydeder."""
        try:
            output_file_path = os.path.join(self.output_folder, f"{os.path.splitext(image_filename)[0]}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(extracted_text)
            print(f"Sonuç kaydedildi: {output_file_path}")
        except Exception as e:
            print(f"Hata: Metin kaydedilemedi: {e}")

    def annotate_image(self, image_path, result):
        """OCR sonuçlarını resmin üzerine ekler ve kaydeder."""
        try:
            if result is None:
                print(f"Görsel işlenemedi: {image_path}")
                return

            image = Image.open(image_path).convert("RGB")
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]

            image_annotated = draw_ocr(image, boxes, txts, scores)
            output_image_path = os.path.join(self.output_folder, 'annotated_' + os.path.basename(image_path))
            image_annotated.save(output_image_path)
            print(f"Annotasyon kaydedildi: {output_image_path}")
        except Exception as e:
            print(f"Hata: Annotasyon işlemi sırasında hata oluştu: {e}")

    def process_all_images(self):
        """Giriş klasöründeki tüm resim dosyalarını işler."""
        image_files = [f for f in os.listdir(self.input_folder) if imghdr.what(os.path.join(self.input_folder, f))]

        total_images = len(image_files)
        print(f"Toplam {total_images} resim işlenecek...")

        for index, image_file in enumerate(image_files, 1):
            image_path = os.path.join(self.input_folder, image_file)
            print(f"[{index}/{total_images}] {image_path} işleniyor...")

            extracted_text, result = self.process_image(image_path)
            if extracted_text:
                self.save_extracted_text(image_file, extracted_text)
                self.annotate_image(image_path, result)


if __name__ == "__main__":
    ocr_processor = HandwritingOCR(input_folder='images', output_folder='outputs', language='tr')
    ocr_processor.process_all_images()
    # Eğitim işlemini başlat
    print('Eğitim Başlıyor...')
    train_model()
import os
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QRect, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QTextEdit, QMessageBox, QDialog
)

# Rutas de modelos PyTorch (.pt) o fallback a ONNX
PT_SEGMENTER = os.path.join("runs", "cacao_seg_run", "weights", "best.pt")
ONNX_SEGMENTER = os.path.join("frontend", "assets", "model", "leaf_segmenter_yolo26.onnx")
SEGMENTER_PATH = PT_SEGMENTER if os.path.exists(PT_SEGMENTER) else ONNX_SEGMENTER

CLASSIFIER_PATH = os.path.join("frontend", "assets", "model", "leaf_classifier_yolo26.onnx")
LABELS_PATH = os.path.join("frontend", "assets", "model", "labels.txt")


class InteractiveCropLabel(QLabel):
    """Etiqueta interactiva con visor 1:1 predeterminado visible que se puede arrastrar con el mouse"""
    def __init__(self, display_w: int, display_h: int, parent=None):
        super().__init__(parent)
        self.display_w = display_w
        self.display_h = display_h
        
        # Tamaño inicial del cuadrado 1:1
        side = min(display_w, display_h)
        left = (display_w - side) // 2
        top = (display_h - side) // 2
        self.selection_rect = QRect(left, top, side, side)
        self.is_dragging = False
        self.drag_offset = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.selection_rect.contains(event.position().toPoint()):
                self.is_dragging = True
                self.drag_offset = event.position().toPoint() - self.selection_rect.topLeft()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            new_top_left = event.position().toPoint() - self.drag_offset
            side = self.selection_rect.width()
            
            # Restringir movimiento dentro de los límites visibles de la imagen
            new_x = max(0, min(new_top_left.x(), self.display_w - side))
            new_y = max(0, min(new_top_left.y(), self.display_h - side))
            
            self.selection_rect.moveTo(new_x, new_y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False

    def wheelEvent(self, event):
        """Permite escalar (agrandar/reducir) el cuadrado 1:1 con la rueda del mouse"""
        delta = event.angleDelta().y()
        current_side = self.selection_rect.width()
        step = 20 if delta > 0 else -20
        new_side = max(100, min(self.display_w, self.display_h, current_side + step))
        
        # Mantener centro al escalar
        center = self.selection_rect.center()
        new_left = max(0, min(center.x() - new_side // 2, self.display_w - new_side))
        new_top = max(0, min(center.y() - new_side // 2, self.display_h - new_side))
        
        self.selection_rect = QRect(new_left, new_top, new_side, new_side)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Oscurecer sutilmente la zona fuera del visor 1:1
        full_rect = QRect(0, 0, self.display_w, self.display_h)
        painter.fillRect(full_rect, QColor(0, 0, 0, 140))
        
        # Recortar ventana transparente dentro del cuadro 1:1 para ver la hoja clara
        painter.save()
        painter.setClipRect(self.selection_rect)
        if self.pixmap():
            painter.drawPixmap(0, 0, self.pixmap())
        painter.restore()
        
        # Dibujar borde verde y esquinas de enfoque
        pen = QPen(QColor(76, 175, 80), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.selection_rect)


class ManualCropDialog(QDialog):
    """Diálogo modal ajustado al tamaño de pantalla con visor de recorte 1:1 predeterminado"""
    def __init__(self, pil_image: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Encuadre Manual 1:1 - Arrastra el recuadro verde para enfocar la hoja")
        self.setStyleSheet("background-color: #F4F7F4; color: #1B382B;")
        self.setMinimumSize(580, 700)
        self.pil_image = pil_image
        self.cropped_result = None

        # Redimensionar la imagen para que quepa perfectamente dentro del diálogo
        max_dialog_h = 520
        ratio = max_dialog_h / pil_image.height
        display_w = int(pil_image.width * ratio)
        display_h = max_dialog_h
        
        if display_w > 520:
            ratio = 520 / pil_image.width
            display_w = 520
            display_h = int(pil_image.height * ratio)

        self.display_pil = pil_image.resize((display_w, display_h), Image.Resampling.BILINEAR)

        img_bytes = self.display_pil.tobytes("raw", "RGB")
        q_img = QImage(img_bytes, display_w, display_h, display_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        instruction = QLabel("Arrastra el recuadro verde para centrar la hoja. Usa la rueda del mouse para cambiar el tamaño:")
        instruction.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        instruction.setStyleSheet("color: #1B382B;")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.crop_widget = InteractiveCropLabel(display_w, display_h)
        self.crop_widget.setPixmap(pixmap)
        self.crop_widget.setFixedSize(display_w, display_h)
        layout.addWidget(self.crop_widget, 0, Qt.AlignmentFlag.AlignCenter)

        btn_confirm = QPushButton("Confirmar Recorte 1:1 y Analizar")
        btn_confirm.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1B382B;
            }
        """)
        btn_confirm.clicked.connect(self.accept_crop)
        layout.addWidget(btn_confirm)

    def accept_crop(self):
        rect = self.crop_widget.selection_rect
        scale = self.pil_image.width / self.crop_widget.display_w
        left = int(rect.x() * scale)
        top = int(rect.y() * scale)
        side = int(rect.width() * scale)
        right = min(self.pil_image.width, left + side)
        bottom = min(self.pil_image.height, top + side)
        self.cropped_result = self.pil_image.crop((left, top, right, bottom))
        self.accept()


class PyTorchTesterPyQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LeafScan MDD - Validador de Modelos (PyQt6 PyTorch)")
        self.resize(1000, 720)
        self.setStyleSheet("background-color: #F4F7F4;")

        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]
        else:
            self.labels = ["DISEASED_VSD", "Fosforo", "Fosforo Potasio", "Nitrogeno", "Potasio", "Sana"]

        try:
            print(f"[PYQT6 TESTER] Cargando modelo de Segmentación desde: {SEGMENTER_PATH}")
            self.seg_model = YOLO(SEGMENTER_PATH, task="segment")
            
            print(f"[PYQT6 TESTER] Cargando modelo Clasificador desde: {CLASSIFIER_PATH}")
            self.cls_model = YOLO(CLASSIFIER_PATH, task="classify")
            
            print("[PYQT6 TESTER] Ambos modelos cargados exitosamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error de Carga", f"No se pudieron cargar los modelos:\n{e}")
            sys.exit(1)

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        header = QLabel("LeafScan MDD - Validador de Modelos YOLO (PyTorch .pt)")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("""
            background-color: #2E7D32;
            color: white;
            padding: 14px;
            border-radius: 12px;
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        main_layout.addLayout(body_layout)

        left_card = QFrame()
        left_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #D8E3D8;
                border-radius: 16px;
            }
        """)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)

        left_title = QLabel("Previsualización de Segmentación Foliar")
        left_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        left_title.setStyleSheet("color: #1B382B; border: none;")
        left_layout.addWidget(left_title)

        self.img_display = QLabel("Selecciona una foto de hoja de cacao")
        self.img_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display.setStyleSheet("""
            background-color: #E8F0E8;
            color: #4A6B5D;
            border-radius: 12px;
            font-size: 14px;
        """)
        self.img_display.setMinimumSize(480, 480)
        left_layout.addWidget(self.img_display, 1)

        btn_select = QPushButton("Seleccionar Imagen (Cacao)")
        btn_select.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_select.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #1B382B;
            }
        """)
        btn_select.clicked.connect(self.select_and_process_image)
        left_layout.addWidget(btn_select)

        body_layout.addWidget(left_card, 1)

        right_card = QFrame()
        right_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #D8E3D8;
                border-radius: 16px;
            }
        """)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)

        right_title = QLabel("Diagnóstico de IA (Pesos PyTorch .pt)")
        right_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        right_title.setStyleSheet("color: #1B382B; border: none;")
        right_layout.addWidget(right_title)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 11))
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #F9FBF9;
                color: #1B382B;
                border: 1px solid #E0E8E0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.log_output.setText("Esperando carga de imagen para inferencia...")
        right_layout.addWidget(self.log_output)

        body_layout.addWidget(right_card, 1)

    def select_and_process_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen de Hoja",
            "",
            "Imágenes (*.jpg *.jpeg *.png *.webp)"
        )
        if not file_path:
            return

        try:
            raw_pil_img = Image.open(file_path).convert("RGB")
            
            dialog = ManualCropDialog(raw_pil_img, self)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.cropped_result is not None:
                pil_img = dialog.cropped_result
            else:
                pil_img = raw_pil_img

            temp_crop_path = "temp_cropped_gui.jpg"
            pil_img.save(temp_crop_path)
            
            cls_results = self.cls_model(temp_crop_path)
            scores_with_labels = []
            if cls_results and len(cls_results) > 0:
                probs = cls_results[0].probs
                if probs is not None:
                    scores = probs.data.cpu().numpy()
                    for idx, score in enumerate(scores):
                        label_name = self.labels[idx] if idx < len(self.labels) else f"Clase_{idx}"
                        scores_with_labels.append((label_name, float(score)))
                    scores_with_labels.sort(key=lambda x: x[1], reverse=True)

            if not scores_with_labels:
                scores_with_labels = [("Sana", 1.0)]

            seg_results = self.seg_model(temp_crop_path)
            masked_pil = pil_img.copy()

            if seg_results and len(seg_results) > 0:
                res = seg_results[0]
                if res.masks is not None and len(res.masks) > 0:
                    top_label = scores_with_labels[0][0]
                    color_rgb = self._get_label_color(top_label)
                    img_data = np.array(pil_img, dtype=np.float32)

                    # Combinar todas las hojas detectadas en la toma
                    combined_mask = np.zeros((pil_img.height, pil_img.width), dtype=np.float32)
                    for mask in res.masks.data:
                        mask_np = mask.cpu().numpy()
                        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(pil_img.size, Image.Resampling.BILINEAR)
                        combined_mask = np.maximum(combined_mask, np.array(mask_pil) / 255.0)

                    binary_mask = combined_mask > 0.40
                    for c in range(3):
                        img_data[:, :, c] = np.where(binary_mask, img_data[:, :, c] * 0.35 + color_rgb[c] * 0.65, img_data[:, :, c])

                    masked_pil = Image.fromarray(img_data.astype(np.uint8))

            display_img = masked_pil.resize((460, 460), Image.Resampling.BILINEAR)
            img_bytes = display_img.tobytes("raw", "RGB")
            q_img = QImage(img_bytes, display_img.width, display_img.height, display_img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            self.img_display.setPixmap(pixmap)

            self.log_output.clear()
            self.log_output.append("=== DIAGNÓSTICO MODELOS PYTORCH (.PT) ===\n")
            top_class, top_conf = scores_with_labels[0]
            self.log_output.append(f"PREDICCIÓN: {top_class}")
            self.log_output.append(f"CONFIANZA:   {top_conf*100:.2f}%\n")
            self.log_output.append("--- Probabilidades por Clase ---")
            for label, conf in scores_with_labels:
                self.log_output.append(f"{label:<16}: {conf*100:6.2f}%")

        except Exception as e:
            QMessageBox.critical(self, "Error Procesamiento", f"Ocurrió un error al procesar la imagen:\n{e}")

    def _get_label_color(self, label):
        colors = {
            "DISEASED_VSD": (211, 47, 47),
            "Fosforo": (123, 31, 162),
            "Fosforo Potasio": (230, 81, 0),
            "Nitrogeno": (251, 192, 45),
            "Potasio": (245, 124, 0),
            "Sana": (46, 125, 50),
        }
        return colors.get(label, (46, 125, 50))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyTorchTesterPyQtApp()
    window.show()
    sys.exit(app.exec())

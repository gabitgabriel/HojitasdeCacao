import 'dart:math';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class LeafColorUtils {
  static Color getClassColor(String label) {
    switch (label) {
      case 'DISEASED_VSD':
        return const Color(0xFFD32F2F); // Rojo fitosanitario
      case 'Fosforo':
        return const Color(0xFF7B1FA2); // Púrpura fosfórico
      case 'Fosforo Potasio':
        return const Color(0xFFE65100); // Naranja marginal
      case 'Nitrogeno':
        return const Color(0xFFFBC02D); // Amarillo clorótico
      case 'Potasio':
        return const Color(0xFFF57C00); // Ámbar seco
      case 'Sana':
      case 'Unlabeled':
      default:
        return const Color(0xFF2E7D32); // Verde sano
    }
  }
}

class SegmentScanPainter extends CustomPainter {
  final double progress;
  final Color color;
  final bool isClassifying;

  SegmentScanPainter({
    required this.progress,
    required this.color,
    required this.isClassifying,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withValues(alpha: 0.8)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    final lineY = size.height * progress;
    canvas.drawLine(Offset(0, lineY), Offset(size.width, lineY), paint);

    final glowPaint = Paint()
      ..color = color.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;
    canvas.drawRect(Rect.fromLTRB(0, max(0, lineY - 20), size.width, lineY), glowPaint);

    final dotPaint = Paint()
      ..color = color.withValues(alpha: isClassifying ? 0.6 : 0.3)
      ..style = PaintingStyle.fill;

    final random = Random(42);
    for (int i = 0; i < 24; i++) {
      final dx = random.nextDouble() * size.width;
      final dy = random.nextDouble() * size.height;
      if (dy <= lineY) {
        canvas.drawCircle(Offset(dx, dy), 3.0, dotPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant SegmentScanPainter oldDelegate) {
    return oldDelegate.progress != progress || oldDelegate.color != color;
  }
}

class RealYoloMaskPainter extends CustomPainter {
  final Color color;
  final ui.Image? maskImage;

  RealYoloMaskPainter({required this.color, required this.maskImage});

  @override
  void paint(Canvas canvas, Size size) {
    if (maskImage != null) {
      final rect = Offset.zero & size;
      final maskRect = Rect.fromLTWH(0, 0, maskImage!.width.toDouble(), maskImage!.height.toDouble());

      // Pintado Fitosanitario del Polígono Foliar Exacto (Idéntico a PyQt6 / Python)
      final paint = Paint()
        ..colorFilter = ColorFilter.mode(
          color.withValues(alpha: 0.65),
          BlendMode.srcIn,
        );
      canvas.drawImageRect(maskImage!, maskRect, rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant RealYoloMaskPainter oldDelegate) {
    return oldDelegate.color != color || oldDelegate.maskImage != maskImage;
  }
}

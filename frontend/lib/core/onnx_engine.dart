import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

class OnnxInferenceEngine {
  OrtSession? _classifierSession;
  OrtSession? _segmenterSession;
  List<String> _labels = [];

  bool get isReady => _classifierSession != null && _segmenterSession != null;
  List<String> get labels => _labels;

  Future<void> initialize() async {
    const String yellow = '\x1B[33m';
    const String reset = '\x1B[0m';

    debugPrint('$yellow[ONNX ENGINE] Inicializando modelo Clasificador y Segmentador YOLO26...$reset');
    final labelsData = await rootBundle.loadString('assets/model/labels.txt');
    _labels = labelsData
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();

    final ort = OnnxRuntime();
    _classifierSession = await ort.createSessionFromAsset('assets/model/leaf_classifier_yolo26.onnx');
    _segmenterSession = await ort.createSessionFromAsset('assets/model/leaf_segmenter_yolo26.onnx');

    debugPrint('$yellow[ONNX ENGINE] ✅ Ambos modelos ONNX cargados exitosamente.$reset');
  }

  Future<InferenceResult> runPipeline(File imageFile) async {
    const String yellowBold = '\x1B[1;33m';
    const String yellow = '\x1B[33m';
    const String reset = '\x1B[0m';

    if (_classifierSession == null || _segmenterSession == null) {
      throw Exception('Los motores ONNX aún no están inicializados.');
    }

    final stopwatch = Stopwatch()..start();
    debugPrint('$yellowBold[SCANNER AI] 🔍 Iniciando pipeline de escaneo foliar...$reset');
    debugPrint('$yellow[SCANNER AI] Imagen de entrada: ${imageFile.path}$reset');

    const height = 224;
    const width = 224;

    final bytes = await imageFile.readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) {
      throw Exception('No se pudo decodificar la imagen seleccionada.');
    }

    final resized = img.copyResize(original, width: width, height: height);

    final inputBuffer = Float32List(1 * 3 * height * width);
    const int planeSize = height * width;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = resized.getPixel(x, y);
        final int idx = y * width + x;
        inputBuffer[idx] = pixel.r / 255.0;
        inputBuffer[planeSize + idx] = pixel.g / 255.0;
        inputBuffer[2 * planeSize + idx] = pixel.b / 255.0;
      }
    }

    // 1. Inferencia de Segmentación Real (YOLO26-Seg ONNX)
    debugPrint('$yellow[SCANNER AI] 1/2 Ejecutando inferencia de Segmentación Real (YOLO26-Seg ONNX)...$reset');
    final segInputTensor = await OrtValue.fromList(inputBuffer, [1, 3, height, width]);
    final segInputName = _segmenterSession!.inputNames.first;
    final segOutputs = await _segmenterSession!.run({segInputName: segInputTensor});
    segInputTensor.dispose();

    ui.Image? maskImage;
    if (segOutputs.containsKey('output1') && segOutputs.containsKey('output0')) {
      final protoValue = segOutputs['output1']!;
      final output0Value = segOutputs['output0']!;
      final protoRaw = await protoValue.asList();
      final output0Raw = await output0Value.asList();

      maskImage = await _decodeYoloMaskBitmapWithCoeffs(protoRaw, output0Raw, 56, 56);
      debugPrint('$yellow[SCANNER AI] 🟢 Máscara de contorno foliar recortada [56x56 RGBA].$reset');
    } else if (segOutputs.isNotEmpty) {
      final maskValue = segOutputs.values.last;
      final maskDataRaw = await maskValue.asList();
      maskImage = await _decodeYoloMaskBitmap(maskDataRaw, 56, 56);
      debugPrint('$yellow[SCANNER AI] 🟢 Máscara de contorno foliar decodificada [56x56 RGBA].$reset');
    }

    for (final t in segOutputs.values) {
      t.dispose();
    }

    // 2. Inferencia de Clasificación Real (YOLO26 Classify ONNX)
    debugPrint('$yellow[SCANNER AI] 2/2 Ejecutando inferencia de Clasificación (YOLO26 Classify ONNX)...$reset');
    final inputTensor = await OrtValue.fromList(inputBuffer, [1, 3, height, width]);
    final inputName = _classifierSession!.inputNames.first;
    final outputs = await _classifierSession!.run({inputName: inputTensor});
    inputTensor.dispose();

    final resultsList = <MapEntry<String, double>>[];
    if (outputs.isNotEmpty) {
      final firstOutput = outputs.values.first;
      final rawData = await firstOutput.asList();

      List<double> rawScores = [];
      if (rawData.isNotEmpty && rawData.first is List) {
        rawScores = (rawData.first as List).map((e) => (e as num).toDouble()).toList();
      } else if (rawData is Float32List) {
        rawScores = rawData.map((e) => e.toDouble()).toList();
      } else {
        rawScores = rawData.map((e) => (e as num).toDouble()).toList();
      }

      for (final t in outputs.values) {
        t.dispose();
      }

      for (int idx = 0; idx < _labels.length && idx < rawScores.length; idx++) {
        resultsList.add(MapEntry(_labels[idx], rawScores[idx]));
      }
      resultsList.sort((a, b) => b.value.compareTo(a.value));
    }

    stopwatch.stop();
    if (resultsList.isNotEmpty) {
      final topResult = resultsList.first;
      final confidencePercent = (topResult.value * 100).toStringAsFixed(1);
      debugPrint('$yellowBold[SCANNER AI] ✅ ESCANEO COMPLETADO (${stopwatch.elapsedMilliseconds} ms)$reset');
      debugPrint('$yellowBold[SCANNER AI] 🏆 Diagnóstico Principal: ${topResult.key} ($confidencePercent%)$reset');
    }

    return InferenceResult(
      results: resultsList,
      maskImage: maskImage,
    );
  }

  Future<ui.Image?> _decodeYoloMaskBitmapWithCoeffs(Object? protoRaw, Object? output0Raw, int maskWidth, int maskHeight) async {
    try {
      final Float32List maskPixels = Float32List(maskWidth * maskHeight * 4);

      if (protoRaw is List && protoRaw.isNotEmpty) {
        List protoData = protoRaw;
        if (protoData.first is List) {
          protoData = protoData.first as List;
        }

        // Extraer los 32 coeficientes de la detección foliar principal
        List<double> maskCoeffs = List.filled(32, 1.0);
        if (output0Raw is List && output0Raw.isNotEmpty) {
          List out0 = output0Raw;
          if (out0.first is List) {
            out0 = out0.first as List;
          }
          if (out0.length >= 38) {
            // Canales 6 a 37 representan los 32 coeficientes de mascara
            maskCoeffs = [];
            for (int i = 6; i < 38 && i < out0.length; i++) {
              final val = out0[i];
              if (val is List && val.isNotEmpty) {
                maskCoeffs.add((val.first as num).toDouble());
              } else if (val is num) {
                maskCoeffs.add(val.toDouble());
              } else {
                maskCoeffs.add(1.0);
              }
            }
            while (maskCoeffs.length < 32) {
              maskCoeffs.add(1.0);
            }
          }
        }

        // Multiplicación matricial (Prototipos x Coeficientes)
        for (int y = 0; y < maskHeight; y++) {
          for (int x = 0; x < maskWidth; x++) {
            final int pixelIdx = (y * maskWidth + x) * 4;
            double sumVal = 0.0;
            
            for (int c = 0; c < 32 && c < protoData.length; c++) {
              final channel = protoData[c];
              final coeff = maskCoeffs[c];
              if (channel is List && y < channel.length) {
                final row = channel[y];
                if (row is List && x < row.length) {
                  sumVal += (row[x] as num).toDouble() * coeff;
                }
              }
            }

            // Aplicar activación Sigmoide con umbral de corte estricto (> 0.65)
            final double sigmoid = 1.0 / (1.0 + exp(-sumVal));
            if (sigmoid > 0.65) {
              maskPixels[pixelIdx] = 255;
              maskPixels[pixelIdx + 1] = 255;
              maskPixels[pixelIdx + 2] = 255;
              maskPixels[pixelIdx + 3] = (sigmoid * 210).clamp(0, 255).toDouble();
            } else {
              maskPixels[pixelIdx + 3] = 0;
            }
          }
        }
      }

      final Uint8List bytes = Uint8List(maskWidth * maskHeight * 4);
      for (int i = 0; i < maskPixels.length; i++) {
        bytes[i] = maskPixels[i].toInt().clamp(0, 255);
      }

      final Completer<ui.Image> completer = Completer();
      ui.decodeImageFromPixels(
        bytes,
        maskWidth,
        maskHeight,
        ui.PixelFormat.rgba8888,
        (img) => completer.complete(img),
      );
      return await completer.future;
    } catch (e) {
      return null;
    }
  }

  Future<ui.Image?> _decodeYoloMaskBitmap(Object? rawData, int maskWidth, int maskHeight) async {
    try {
      final Float32List maskPixels = Float32List(maskWidth * maskHeight * 4);

      if (rawData != null && rawData is List && rawData.isNotEmpty) {
        List protoData = rawData;
        if (protoData.first is List) {
          protoData = protoData.first as List;
        }

        for (int y = 0; y < maskHeight; y++) {
          for (int x = 0; x < maskWidth; x++) {
            final int pixelIdx = (y * maskWidth + x) * 4;
            double sumVal = 0.0;
            
            for (int c = 0; c < 32 && c < protoData.length; c++) {
              final channel = protoData[c];
              if (channel is List && y < channel.length) {
                final row = channel[y];
                if (row is List && x < row.length) {
                  sumVal += (row[x] as num).toDouble();
                }
              }
            }

            final double sigmoid = 1.0 / (1.0 + exp(-sumVal / 10.0));
            if (sigmoid > 0.65) {
              maskPixels[pixelIdx] = 255;
              maskPixels[pixelIdx + 1] = 255;
              maskPixels[pixelIdx + 2] = 255;
              maskPixels[pixelIdx + 3] = (sigmoid * 210).clamp(0, 255).toDouble();
            } else {
              maskPixels[pixelIdx + 3] = 0;
            }
          }
        }
      }

      final Uint8List bytes = Uint8List(maskWidth * maskHeight * 4);
      for (int i = 0; i < maskPixels.length; i++) {
        bytes[i] = maskPixels[i].toInt().clamp(0, 255);
      }

      final Completer<ui.Image> completer = Completer();
      ui.decodeImageFromPixels(
        bytes,
        maskWidth,
        maskHeight,
        ui.PixelFormat.rgba8888,
        (img) => completer.complete(img),
      );
      return await completer.future;
    } catch (e) {
      return null;
    }
  }

  void dispose() {
    _classifierSession?.close();
    _segmenterSession?.close();
  }
}

class InferenceResult {
  final List<MapEntry<String, double>> results;
  final ui.Image? maskImage;

  const InferenceResult({
    required this.results,
    this.maskImage,
  });
}

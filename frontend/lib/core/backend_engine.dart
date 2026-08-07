import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:http/http.dart' as http;

import 'package:http_parser/http_parser.dart';

class BackendInferenceEngine {
  // Dirección por defecto del backend local (10.0.2.2 para emulador Android, localhost para PC/Web)
  final String baseUrl;

  BackendInferenceEngine({this.baseUrl = 'http://10.0.2.2:8000'});

  Future<InferenceResult> analyzeLeaf(File imageFile) async {
    const String yellowBold = '\x1B[1;33m';
    const String reset = '\x1B[0m';

    final stopwatch = Stopwatch()..start();
    debugPrint('$yellowBold[BACKEND API] 🚀 Enviando foto a servidor Python FastAPI ($baseUrl/analyze)...$reset');

    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/analyze'));
    request.files.add(await http.MultipartFile.fromPath(
      'file',
      imageFile.path,
      contentType: MediaType('image', 'jpeg'),
    ));

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      throw Exception('Error del servidor Backend (${response.statusCode}): ${response.body}');
    }

    final Map<String, dynamic> jsonResponse = json.decode(response.body) as Map<String, dynamic>;

    // 1. Parsear diagnósticos y probabilidades
    final List<dynamic> rawDiagnoses = jsonResponse['diagnoses'] as List<dynamic>? ?? [];
    final List<MapEntry<String, double>> resultsList = rawDiagnoses.map((item) {
      final map = item as Map<String, dynamic>;
      return MapEntry(map['label'] as String, (map['confidence'] as num).toDouble());
    }).toList();

    // 2. Parsear máscara PNG transparente Base64 desde Python
    ui.Image? maskImage;
    final String maskBase64 = jsonResponse['mask_png_base64'] as String? ?? '';
    if (maskBase64.isNotEmpty) {
      final Uint8List bytes = base64.decode(maskBase64);
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      maskImage = frame.image;
    }

    stopwatch.stop();
    if (resultsList.isNotEmpty) {
      final top = resultsList.first;
      debugPrint('$yellowBold[BACKEND API] ✅ INFERENCIA COMPLETADA (${stopwatch.elapsedMilliseconds} ms)$reset');
      debugPrint('$yellowBold[BACKEND API] 🏆 Diagnóstico Python: ${top.key} (${(top.value * 100).toStringAsFixed(1)}%)$reset');
    }

    return InferenceResult(
      results: resultsList,
      maskImage: maskImage,
    );
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

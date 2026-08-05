import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
 
void main() {
  runApp(const LeafScanApp());
}
 
class LeafScanApp extends StatelessWidget {
  const LeafScanApp({super.key});
 
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LeafScan MDD',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0E1512),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          brightness: Brightness.dark,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF14231C),
          elevation: 0,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}
 
// ---------------------------------------------------------------------
// Ficha "enciclopedia" con la información de cada clase.
// Contenido tomado directamente del documento de avance de tesis.
// ---------------------------------------------------------------------
class DiseaseInfo {
  final String tipo;
  final String agenteCausal;
  final String sintomas;
  final String prevencion;
 
  const DiseaseInfo({
    required this.tipo,
    required this.agenteCausal,
    required this.sintomas,
    required this.prevencion,
  });
}
 
const Map<String, DiseaseInfo> diseaseEncyclopedia = {
  'Afido': DiseaseInfo(
    tipo: 'Plaga (insecto), no es una enfermedad fúngica',
    agenteCausal: 'Insecto Toxoptera aurantii, el pulgón más común en cultivos de cacao',
    sintomas:
        'Se alimenta succionando la savia de hojas, brotes tiernos e inflorescencias, '
        'provocando enrollamiento y deformación de hojas jóvenes, debilitamiento general '
        'de la planta y reducción del crecimiento. La melaza que excreta puede favorecer '
        'la fumagina y actuar como vector de virus.',
    prevencion:
        'Eliminación de brotes muy afectados, control de malezas hospederas, uso de '
        'enemigos naturales (coccinélidos, crisopas), trampas pegajosas, y de ser '
        'necesario control químico dirigido al envés de las hojas.',
  ),
  'Antracnosis': DiseaseInfo(
    tipo: 'Enfermedad fúngica',
    agenteCausal: 'Hongo Colletotrichum gloeosporioides (Colletotrichum spp.)',
    sintomas:
        'Manchas necróticas de color café oscuro a negro, de forma irregular, que suelen '
        'iniciar en la punta o el borde de la hoja y avanzar a lo largo de la nervadura '
        'central. En etapas tempranas pueden presentar un borde amarillento o anaranjado. '
        'Puede provocar el enrollamiento y caída prematura de la hoja.',
    prevencion:
        'Poda de ramas afectadas y recolección de hojas enfermas; regulación de la sombra '
        'para mejorar la ventilación; fertilización adecuada; desinfección de herramientas '
        'de poda; en casos severos, fungicidas de cobre.',
  ),
  'Escoba de Bruja': DiseaseInfo(
    tipo: 'Enfermedad fúngica',
    agenteCausal: 'Hongo Moniliophthora perniciosa (antes Crinipellis perniciosa)',
    sintomas:
        'Deformación del crecimiento: brotes que crecen de forma anormal, engrosados y '
        'con múltiples ramificaciones apretadas, hojas más pequeñas o distorsionadas, y '
        'con el tiempo las "escobas" se secan tomando un color marrón oscuro. También '
        'puede afectar cojines florales y frutos.',
    prevencion:
        'Poda fitosanitaria periódica (eliminación y quema de escobas y frutos enfermos), '
        'manejo integrado del cultivo, fertilización adecuada, monitoreo semanal de '
        'brotes nuevos, y fungicidas protectores o biológicos (ej. Trichoderma).',
  ),
  'Mancha foliar por Cercospora': DiseaseInfo(
    tipo: 'Enfermedad fúngica',
    agenteCausal: 'Hongo del género Cercospora sp.',
    sintomas:
        'Manchas de color marrón/grisáceo, generalmente circulares u ovaladas, con borde '
        'más definido y a veces un halo amarillento (clorosis) alrededor. Con el tiempo '
        'las manchas pueden aumentar de tamaño y unirse, provocando caída prematura de '
        'hojas.',
    prevencion:
        'Evitar el exceso de humedad en el follaje, asegurar buen drenaje del terreno, '
        'remover el material vegetal afectado, mantener fertilización adecuada, y aplicar '
        'fungicidas cuando sea necesario.',
  ),
  'Sana': DiseaseInfo(
    tipo: 'Hoja sin síntomas visibles',
    agenteCausal: 'No aplica',
    sintomas: 'La hoja no presenta manchas, deformaciones ni signos de plaga.',
    prevencion:
        'Mantener el monitoreo periódico del cultivo y buenas prácticas agronómicas '
        '(fertilización, manejo de sombra y humedad) para conservar la sanidad de la '
        'planta.',
  ),
};
 
class HomePage extends StatefulWidget {
  const HomePage({super.key});
 
  @override
  State<HomePage> createState() => _HomePageState();
}
 
class _HomePageState extends State<HomePage> {
  Interpreter? _interpreter;
  List<String> _labels = [];
  File? _selectedImage;
  bool _isLoadingModel = true;
  bool _isProcessing = false;
  List<MapEntry<String, double>>? _results;
 
  // Forma real del modelo, detectada en tiempo de ejecución (no asumida a mano).
  List<int> _inputShape = [1, 224, 224, 3];
  List<int> _outputShape = [1, 5];
 
  @override
  void initState() {
    super.initState();
    _loadModel();
  }
 
  Future<void> _loadModel() async {
    try {
      final interpreter = await Interpreter.fromAsset('assets/model/model.tflite');
 
      // Preguntamos al modelo su forma real de entrada/salida en vez de asumirla.
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
 
      final labelsData = await rootBundle.loadString('assets/model/labels.txt');
      final labels = labelsData
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();
 
      setState(() {
        _interpreter = interpreter;
        _inputShape = inputTensor.shape;
        _outputShape = outputTensor.shape;
        _labels = labels;
        _isLoadingModel = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingModel = false;
      });
      if (mounted) {
        _showError('Error cargando el modelo: $e');
      }
    }
  }
 
  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red.shade700,
      ),
    );
  }
 
  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
    );
 
    if (pickedFile == null) return;
 
    setState(() {
      _selectedImage = File(pickedFile.path);
      _results = null;
      _isProcessing = true;
    });
 
    await _runInference(_selectedImage!);
  }
 
  Future<void> _runInference(File imageFile) async {
    if (_interpreter == null) {
      setState(() => _isProcessing = false);
      _showError('El modelo aún no está listo.');
      return;
    }
 
    try {
      // El shape típico es [1, alto, ancho, canales]
      final height = _inputShape[1];
      final width = _inputShape[2];
 
      final bytes = await imageFile.readAsBytes();
      final original = img.decodeImage(bytes);
      if (original == null) {
        throw Exception('No se pudo leer la imagen');
      }
 
      final resized = img.copyResize(original, width: width, height: height);
 
      // Construimos un buffer plano Float32 y lo damos con la forma exacta
      // que el modelo reportó (evita el "Bad state: failed precondition").
      final inputBuffer = Float32List(1 * height * width * 3);
      int i = 0;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final pixel = resized.getPixel(x, y);
          inputBuffer[i++] = pixel.r / 255.0;
          inputBuffer[i++] = pixel.g / 255.0;
          inputBuffer[i++] = pixel.b / 255.0;
        }
      }
      final input = inputBuffer.reshape(_inputShape);
 
      final outputCount = _outputShape.reduce((a, b) => a * b);
      final outputBuffer = Float32List(outputCount);
      final output = outputBuffer.reshape(_outputShape);
 
      _interpreter!.run(input, output);
 
      // La salida normalmente es [1, numClases] -> tomamos la primera fila.
      final List<double> scores = List<double>.from(
        (output as List).first as List,
      );
 
      final entries = <MapEntry<String, double>>[];
      for (int idx = 0; idx < _labels.length && idx < scores.length; idx++) {
        entries.add(MapEntry(_labels[idx], scores[idx]));
      }
      entries.sort((a, b) => b.value.compareTo(a.value));
 
      setState(() {
        _results = entries;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() => _isProcessing = false);
      _showError('Error al analizar la imagen: $e');
    }
  }
 
  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }
 
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'LeafScan MDD',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _isLoadingModel
            ? const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Color(0xFF66BB6A)),
                    SizedBox(height: 16),
                    Text('Cargando modelo...'),
                  ],
                ),
              )
            : SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildImagePreview(),
                    const SizedBox(height: 20),
                    _buildActionButtons(),
                    const SizedBox(height: 24),
                    if (_isProcessing)
                      const Center(
                        child: Padding(
                          padding: EdgeInsets.all(24.0),
                          child: CircularProgressIndicator(
                            color: Color(0xFF66BB6A),
                          ),
                        ),
                      ),
                    if (_results != null && !_isProcessing) ...[
                      _buildResults(),
                      const SizedBox(height: 16),
                      _buildEncyclopediaCard(_results!.first.key),
                    ],
                  ],
                ),
              ),
      ),
    );
  }
 
  Widget _buildImagePreview() {
    return Container(
      height: 280,
      decoration: BoxDecoration(
        color: const Color(0xFF1B2A22),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2E7D32), width: 1),
      ),
      clipBehavior: Clip.antiAlias,
      child: _selectedImage == null
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.eco_outlined, size: 64, color: Color(0xFF66BB6A)),
                  SizedBox(height: 12),
                  Text(
                    'Toma o selecciona una foto\nde la hoja de cacao',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70),
                  ),
                ],
              ),
            )
          : Image.file(_selectedImage!, fit: BoxFit.cover),
    );
  }
 
  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt),
            label: const Text('Cámara'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF2E7D32),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library),
            label: const Text('Galería'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1B2A22),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
                side: const BorderSide(color: Color(0xFF2E7D32)),
              ),
            ),
          ),
        ),
      ],
    );
  }
 
  Widget _buildResults() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1B2A22),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Resultado del análisis',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          ..._results!.map((entry) => _buildResultBar(entry.key, entry.value)),
        ],
      ),
    );
  }
 
  Widget _buildResultBar(String label, double confidence) {
    final percent = (confidence * 100).clamp(0, 100);
    final isTop = _results!.first.key == label;
 
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                ),
              ),
              Text(
                '${percent.toStringAsFixed(1)}%',
                style: TextStyle(
                  color: isTop ? const Color(0xFF81C784) : Colors.white70,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: confidence.clamp(0.0, 1.0),
              minHeight: 10,
              backgroundColor: const Color(0xFF0E1512),
              valueColor: AlwaysStoppedAnimation<Color>(
                isTop ? const Color(0xFF66BB6A) : const Color(0xFF2E7D32),
              ),
            ),
          ),
        ],
      ),
    );
  }
 
  // -------------------------------------------------------------------
  // Ficha "enciclopedia" de la clase con mayor confianza.
  // -------------------------------------------------------------------
  Widget _buildEncyclopediaCard(String topLabel) {
    final info = diseaseEncyclopedia[topLabel];
    if (info == null) return const SizedBox.shrink();
 
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF14231C),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2E7D32).withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.menu_book, color: Color(0xFF81C784), size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  topLabel,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF81C784),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _infoRow('Tipo', info.tipo),
          _infoRow('Agente causal', info.agenteCausal),
          _infoRow('Síntomas', info.sintomas),
          _infoRow('Prevención y control', info.prevencion),
        ],
      ),
    );
  }
 
  Widget _infoRow(String title, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            content,
            style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.4),
          ),
        ],
      ),
    );
  }
}
 










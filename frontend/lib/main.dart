import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show PlatformException, rootBundle;
import 'package:google_fonts/google_fonts.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:leafscan_mdd/core/backend_engine.dart';
import 'package:leafscan_mdd/core/disease_info.dart';
import 'package:leafscan_mdd/core/leaf_painters.dart';
import 'package:material_3_expressive/material_3_expressive.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const LeafScanApp());
}

class LeafScanApp extends StatelessWidget {
  const LeafScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return M3ETheme(
      data: M3EThemeData.light(seedColor: const Color(0xFF2E7D32)),
      child: MaterialApp(
        title: 'LeafScan MDD',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.light,
          scaffoldBackgroundColor: const Color(0xFFF4F7F4),
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF2E7D32),
            surface: const Color(0xFFFFFFFF),
            primary: const Color(0xFF2E7D32),
          ),
          textTheme: GoogleFonts.outfitTextTheme(ThemeData.light().textTheme),
          appBarTheme: AppBarTheme(
            backgroundColor: const Color(0xFFF4F7F4),
            elevation: 0,
            iconTheme: const IconThemeData(color: Color(0xFF1B382B)),
            titleTextStyle: GoogleFonts.playfairDisplay(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: const Color(0xFF1B382B),
            ),
          ),
          useMaterial3: true,
        ),
        home: const MainNavigationScreen(),
      ),
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = const [
    AnalyzePage(),
    HistoryPage(),
    SettingsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.06),
              blurRadius: 16,
              offset: const Offset(0, -4),
            ),
          ],
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: NavigationBar(
              selectedIndex: _currentIndex,
              onDestinationSelected: (index) {
                setState(() {
                  _currentIndex = index;
                });
              },
              backgroundColor: Colors.transparent,
              indicatorColor: const Color(0xFFE8F5E9),
              elevation: 0,
              destinations: const [
                NavigationDestination(
                  icon: Icon(Icons.camera_alt_outlined, color: Color(0xFF4A6B5D)),
                  selectedIcon: Icon(Icons.camera_alt, color: Color(0xFF2E7D32)),
                  label: 'Analizar',
                ),
                NavigationDestination(
                  icon: Icon(Icons.history_outlined, color: Color(0xFF4A6B5D)),
                  selectedIcon: Icon(Icons.history, color: Color(0xFF2E7D32)),
                  label: 'Historial',
                ),
                NavigationDestination(
                  icon: Icon(Icons.settings_outlined, color: Color(0xFF4A6B5D)),
                  selectedIcon: Icon(Icons.settings, color: Color(0xFF2E7D32)),
                  label: 'Ajustes',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class AnalyzePage extends StatefulWidget {
  const AnalyzePage({super.key});

  @override
  State<AnalyzePage> createState() => _AnalyzePageState();
}

enum AnalysisStage { idle, segmenting, classifying, completed }

class _AnalyzePageState extends State<AnalyzePage> with SingleTickerProviderStateMixin {
  final BackendInferenceEngine _backendEngine = BackendInferenceEngine();
  Map<String, DiseaseInfo> _diseaseEncyclopedia = {};
  File? _selectedImage;
  bool _isLoadingModel = true;

  AnalysisStage _stage = AnalysisStage.idle;
  String _statusText = '';
  List<MapEntry<String, double>>? _results;
  ui.Image? _maskImage;

  late AnimationController _scanController;
  late Animation<double> _scanAnimation;

  @override
  void initState() {
    super.initState();
    _scanController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1600),
    );
    _scanAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _scanController, curve: Curves.easeInOut),
    );
    _loadModelAndAssets();
  }

  @override
  void dispose() {
    _scanController.dispose();
    super.dispose();
  }

  Future<void> _loadModelAndAssets() async {
    try {
      final jsonString = await rootBundle.loadString('assets/data/disease_encyclopedia.json');
      final Map<String, dynamic> decodedJson = json.decode(jsonString) as Map<String, dynamic>;
      final encyclopediaMap = decodedJson.map(
        (key, value) => MapEntry(
          key,
          DiseaseInfo.fromJson(value as Map<String, dynamic>),
        ),
      );

      setState(() {
        _diseaseEncyclopedia = encyclopediaMap;
        _isLoadingModel = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingModel = false;
      });
      if (mounted) {
        _showError('Error cargando los activos del sistema: $e');
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

  bool _isPickingImage = false;

  Future<void> _pickImage(ImageSource source) async {
    if (_isPickingImage) return;
    _isPickingImage = true;

    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (pickedFile == null) return;

      // Recortador manual interactivo de usuario 1:1
      final croppedFile = await ImageCropper().cropImage(
        sourcePath: pickedFile.path,
        aspectRatio: const CropAspectRatio(ratioX: 1.0, ratioY: 1.0),
        uiSettings: [
          AndroidUiSettings(
            toolbarTitle: 'Encuadra la hoja de cacao (1:1)',
            toolbarColor: const Color(0xFF2E7D32),
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.square,
            lockAspectRatio: true,
          ),
          IOSUiSettings(
            title: 'Encuadra la hoja de cacao (1:1)',
            aspectRatioLockEnabled: true,
          ),
        ],
      );

      final finalImagePath = croppedFile != null ? croppedFile.path : pickedFile.path;

      setState(() {
        _selectedImage = File(finalImagePath);
        _results = null;
        _maskImage = null;
        _stage = AnalysisStage.segmenting;
        _statusText = 'Segmentando y contorneando superficie foliar real...';
      });

      _scanController.repeat(reverse: true);
      await _runInferenceSequence(_selectedImage!);
    } on PlatformException catch (e) {
      if (e.code != 'already_active') {
        _showError('Error de selector de imágenes: ${e.message}');
      }
    } catch (e) {
      _showError('No se pudo cargar la imagen: $e');
    } finally {
      _isPickingImage = false;
    }
  }

  Future<void> _runInferenceSequence(File imageFile) async {
    try {
      final inferenceOutput = await _backendEngine.analyzeLeaf(imageFile);

      if (!mounted) return;
      _scanController.stop();
      setState(() {
        _results = inferenceOutput.results;
        _maskImage = inferenceOutput.maskImage;
        _stage = AnalysisStage.completed;
      });
    } catch (e) {
      _stopAnalysisState();
      _showError('Error de conexión con Backend Python: $e');
    }
  }

  void _stopAnalysisState() {
    _scanController.stop();
    if (mounted) {
      setState(() {
        _stage = AnalysisStage.idle;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'LeafScan MDD',
          style: GoogleFonts.playfairDisplay(
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _isLoadingModel
            ? const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    M3EProgressIndicator.circular(color: Color(0xFF2E7D32)),
                    SizedBox(height: 16),
                    Text('Cargando motor de IA...', style: TextStyle(color: Color(0xFF1B382B))),
                  ],
                ),
              )
            : SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildImagePreview(),
                    const SizedBox(height: 20),
                    _buildActionButtons(),
                    const SizedBox(height: 24),
                    if (_stage == AnalysisStage.segmenting || _stage == AnalysisStage.classifying)
                      _buildProcessingOverlay(),
                    if (_results != null && _stage == AnalysisStage.completed) ...[
                      _buildResults(),
                      const SizedBox(height: 20),
                      _buildEncyclopediaCard(_results!.first.key),
                    ],
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildImagePreview() {
    final topClass = _results?.first.key;
    final topColor = topClass != null ? LeafColorUtils.getClassColor(topClass) : const Color(0xFF2E7D32);

    return AspectRatio(
      aspectRatio: 1.0,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 500),
        width: double.infinity,
      decoration: BoxDecoration(
        color: const Color(0xFF1B261D), // Fondo oscuro sutil para enmarcar fotos de cualquier proporcion
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: _stage == AnalysisStage.completed ? topColor : const Color(0xFF2E7D32).withValues(alpha: 0.25),
          width: _stage == AnalysisStage.completed ? 3.0 : 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: _stage == AnalysisStage.completed ? topColor.withValues(alpha: 0.2) : Colors.black.withValues(alpha: 0.04),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: _selectedImage == null
          ? Container(
              height: 280,
              color: const Color(0xFFE8F0E8),
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: const BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.eco_outlined, size: 48, color: Color(0xFF2E7D32)),
                    ),
                    const SizedBox(height: 14),
                    Text(
                      'Captura o selecciona la foto de la hoja',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.outfit(
                        color: const Color(0xFF4A6B5D),
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            )
          : Stack(
              alignment: Alignment.center,
              children: [
                Image.file(_selectedImage!, fit: BoxFit.contain),

                // 1. Capa de Escaneo Láser
                if (_stage == AnalysisStage.segmenting || _stage == AnalysisStage.classifying)
                  AnimatedBuilder(
                    animation: _scanAnimation,
                    builder: (context, child) {
                      return CustomPaint(
                        painter: SegmentScanPainter(
                          progress: _scanAnimation.value,
                          color: _stage == AnalysisStage.segmenting ? const Color(0xFF66BB6A) : topColor,
                          isClassifying: _stage == AnalysisStage.classifying,
                        ),
                      );
                    },
                  ),

                // 2. Capa de Máscara de Segmentación Real ONNX
                if (_stage == AnalysisStage.completed && topClass != null)
                  CustomPaint(
                    painter: RealYoloMaskPainter(
                      color: topColor,
                      maskImage: _maskImage,
                    ),
                  ),

                // 3. Etiqueta Flotante
                Positioned(
                  bottom: 12,
                  right: 12,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.75),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _stage == AnalysisStage.completed ? Icons.auto_awesome : Icons.memory,
                          size: 14,
                          color: _stage == AnalysisStage.completed ? topColor : const Color(0xFF81C784),
                        ),
                        const SizedBox(width: 5),
                        Text(
                          _stage == AnalysisStage.completed
                              ? 'Máscara ONNX ($topClass)'
                              : 'YOLO26 ONNX REAL',
                          style: GoogleFonts.outfit(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
      ),
    );
  }

  Widget _buildProcessingOverlay() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2E7D32).withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          const M3EProgressIndicator.circular(color: Color(0xFF2E7D32)),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _stage == AnalysisStage.segmenting ? 'Segmentando Follaje...' : 'Clasificando con YOLO26...',
                  style: GoogleFonts.outfit(
                    fontWeight: FontWeight.bold,
                    color: const Color(0xFF1B382B),
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  _statusText,
                  style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF4A6B5D)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    final isBusy = _stage == AnalysisStage.segmenting || _stage == AnalysisStage.classifying;

    return Row(
      children: [
        Expanded(
          child: SizedBox(
            height: 52,
            child: ElevatedButton.icon(
              onPressed: isBusy ? null : () => _pickImage(ImageSource.camera),
              icon: const Icon(Icons.camera_alt_outlined, size: 20),
              label: Text(
                'Cámara',
                style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 15),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2E7D32),
                foregroundColor: Colors.white,
                elevation: 2,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: SizedBox(
            height: 52,
            child: OutlinedButton.icon(
              onPressed: isBusy ? null : () => _pickImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined, size: 20),
              label: Text(
                'Galería',
                style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 15),
              ),
              style: OutlinedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: const Color(0xFF1B382B),
                side: BorderSide(color: const Color(0xFF2E7D32).withValues(alpha: 0.4), width: 1.5),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildResults() {
    final topClass = _results!.first.key;
    final topColor = LeafColorUtils.getClassColor(topClass);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: topColor.withValues(alpha: 0.3)),
        boxShadow: [
          BoxShadow(
            color: topColor.withValues(alpha: 0.05),
            blurRadius: 14,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Diagnóstico de IA',
                style: GoogleFonts.outfit(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                  color: const Color(0xFF1B382B),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: topColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'YOLO26 ONNX',
                  style: GoogleFonts.outfit(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: topColor,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          ..._results!.map((entry) => _buildResultBar(entry.key, entry.value, topColor)),
        ],
      ),
    );
  }

  Widget _buildResultBar(String label, double confidence, Color topColor) {
    final percent = (confidence * 100).clamp(0, 100);
    final isTop = _results!.first.key == label;
    final barColor = isTop ? topColor : const Color(0xFFA5D6A7);

    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: GoogleFonts.outfit(
                  color: const Color(0xFF1B382B),
                  fontSize: 14,
                  fontWeight: isTop ? FontWeight.bold : FontWeight.w400,
                ),
              ),
              Text(
                '${percent.toStringAsFixed(1)}%',
                style: GoogleFonts.outfit(
                  color: isTop ? topColor : const Color(0xFF7A9488),
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: confidence.clamp(0.0, 1.0),
              minHeight: 8,
              backgroundColor: const Color(0xFFF0F5F1),
              valueColor: AlwaysStoppedAnimation<Color>(barColor),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEncyclopediaCard(String topLabel) {
    final info = _diseaseEncyclopedia[topLabel];
    if (info == null) return const SizedBox.shrink();
    final cardColor = LeafColorUtils.getClassColor(topLabel);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: cardColor.withValues(alpha: 0.3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: cardColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(Icons.menu_book, color: cardColor, size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  topLabel,
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 19,
                    fontWeight: FontWeight.bold,
                    color: const Color(0xFF1B382B),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _infoDetailCard('Tipo de Condición', info.tipo, Icons.category_outlined, cardColor),
          _infoDetailCard('Agente Causal', info.agenteCausal, Icons.science_outlined, cardColor),
          _infoDetailCard('Síntomas Característicos', info.sintomas, Icons.description_outlined, cardColor),
          _infoDetailCard('Tratamiento y Control Fitosanitario', info.prevencion, Icons.health_and_safety_outlined, cardColor),
        ],
      ),
    );
  }

  Widget _infoDetailCard(String title, String content, IconData icon, Color mainColor) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFF7FAF7),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: mainColor.withValues(alpha: 0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 16, color: mainColor),
              const SizedBox(width: 6),
              Text(
                title,
                style: GoogleFonts.outfit(
                  color: const Color(0xFF1B382B),
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            content,
            style: GoogleFonts.outfit(
              color: const Color(0xFF4A6B5D),
              fontSize: 13,
              height: 1.45,
            ),
          ),
        ],
      ),
    );
  }
}

class HistoryPage extends StatelessWidget {
  const HistoryPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Historial de Análisis',
          style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: const BoxDecoration(
                  color: Color(0xFFE8F5E9),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.history_outlined, size: 56, color: Color(0xFF2E7D32)),
              ),
              const SizedBox(height: 20),
              Text(
                'Sin escaneos guardados',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: const Color(0xFF1B382B),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Los diagnósticos que realices en el campo se registrarán automáticamente aquí.',
                textAlign: TextAlign.center,
                style: GoogleFonts.outfit(color: const Color(0xFF4A6B5D), fontSize: 14),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Ajustes & Modelos',
          style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _buildSectionHeader('Modelos de Inteligencia Artificial'),
          _buildModelTile(
            title: 'YOLO26 Classification (ONNX)',
            subtitle: 'Modelo Nano ultra rápido en dispositivo (En uso)',
            icon: Icons.psychology_outlined,
            isActive: true,
          ),
          _buildModelTile(
            title: 'YOLO26 Instance Segmentation',
            subtitle: 'Segmentación de contornos de hojas (En uso)',
            icon: Icons.polyline_outlined,
            isActive: true,
          ),
          _buildModelTile(
            title: 'DeepLabV3+ Plant Segmentation',
            subtitle: 'Cálculo de área afectada en % (Próximamente)',
            icon: Icons.pie_chart_outline,
            isActive: false,
          ),
          const SizedBox(height: 24),
          _buildSectionHeader('Modo de Ejecución'),
          DecoratedBox(
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF2E7D32).withValues(alpha: 0.15)),
            ),
            child: ListTile(
              leading: const Icon(Icons.signal_wifi_off_outlined, color: Color(0xFF2E7D32)),
              title: Text('Inferencia 100% Offline', style: GoogleFonts.outfit(fontWeight: FontWeight.bold)),
              subtitle: Text('Procesamiento On-Device sin datos/señal', style: GoogleFonts.outfit(fontSize: 12)),
              trailing: Switch(
                value: true,
                onChanged: (val) {},
                activeThumbColor: const Color(0xFF2E7D32),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12, left: 4),
      child: Text(
        title,
        style: GoogleFonts.outfit(
          fontSize: 15,
          fontWeight: FontWeight.bold,
          color: const Color(0xFF1B382B),
        ),
      ),
    );
  }

  Widget _buildModelTile({
    required String title,
    required String subtitle,
    required IconData icon,
    required bool isActive,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isActive ? const Color(0xFF2E7D32) : const Color(0xFF2E7D32).withValues(alpha: 0.15),
          width: isActive ? 1.5 : 1.0,
        ),
      ),
      child: ListTile(
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: isActive ? const Color(0xFFE8F5E9) : const Color(0xFFF4F7F4),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: isActive ? const Color(0xFF2E7D32) : Colors.grey),
        ),
        title: Text(
          title,
          style: GoogleFonts.outfit(
            fontWeight: FontWeight.bold,
            color: isActive ? const Color(0xFF1B382B) : Colors.grey.shade700,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF4A6B5D)),
        ),
        trailing: isActive
            ? const Icon(Icons.check_circle, color: Color(0xFF2E7D32))
            : const Icon(Icons.lock_outline, color: Colors.grey, size: 20),
      ),
    );
  }
}

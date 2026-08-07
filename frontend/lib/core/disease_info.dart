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

  factory DiseaseInfo.fromJson(Map<String, dynamic> json) {
    return DiseaseInfo(
      tipo: json['tipo'] as String? ?? '',
      agenteCausal: json['agente_causal'] as String? ?? '',
      sintomas: json['sintomas'] as String? ?? '',
      prevencion: json['prevencion'] as String? ?? '',
    );
  }
}

class ObjectDetectionResult {
  final String className;
  final double confidence;
  final List<double> bbox;

  ObjectDetectionResult({
    required this.className,
    required this.confidence,
    required this.bbox,
  });

  factory ObjectDetectionResult.fromJson(Map<String, dynamic> json) {
    return ObjectDetectionResult(
      className: json['class'],
      confidence: (json['confidence'] as num).toDouble(),
      bbox: List<double>.from(json['bbox'][0].map((b) => (b as num).toDouble())),
    );
  }
}
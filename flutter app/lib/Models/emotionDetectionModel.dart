class EmotionDetectionResult {
  final Map<String, double> emotionPercentages;
  final String? dominantEmotion;

  EmotionDetectionResult({
    required this.emotionPercentages,
    this.dominantEmotion,
  });

  factory EmotionDetectionResult.fromJson(Map<String, dynamic> json) {
    // The API returns percentages as a map, so we can cast it directly.
    final Map<String, dynamic> percentagesJson = json['emotion_percentages'];
    final Map<String, double> percentages = percentagesJson.map((key, value) {
      return MapEntry(key, (value as num).toDouble());
    });

    return EmotionDetectionResult(
      emotionPercentages: percentages,
      dominantEmotion: json['dominant_emotion'],
    );
  }
}
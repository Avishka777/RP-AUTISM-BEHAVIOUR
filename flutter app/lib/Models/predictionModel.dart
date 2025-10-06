// Model for the data sent to the /predict endpoint
class PredictionRequest {
  final int age;
  final String gender;
  final String currentMood;
  final int parentSatisfaction;
  final int engagementLevel;
  final int completedTasks;
  final double timeSpent;
  final int correctInFirstAttempt;

  PredictionRequest({
    required this.age,
    required this.gender,
    required this.currentMood,
    required this.parentSatisfaction,
    required this.engagementLevel,
    required this.completedTasks,
    required this.timeSpent,
    required this.correctInFirstAttempt,
  });

  Map<String, dynamic> toJson() {
    return {
      'Age': age,
      'Gender': gender,
      'Current_Mood': currentMood,
      'Parent_Satisfaction': parentSatisfaction,
      'Engagement_Level': engagementLevel,
      'Completed_Tasks': completedTasks,
      'Time_Spent': timeSpent,
      'Correct_in_First_Attempt': correctInFirstAttempt,
    };
  }
}

// Model for the data received from the /predict endpoint
class PredictionResult {
  final String prediction;
  final List<String> suggestions;

  PredictionResult({
    required this.prediction,
    required this.suggestions,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      prediction: json['prediction'],
      suggestions: List<String>.from(json['suggestions']),
    );
  }
}
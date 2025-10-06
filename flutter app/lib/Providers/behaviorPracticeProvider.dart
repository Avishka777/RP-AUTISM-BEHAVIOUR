import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Model class to hold data for a single activity or sub-activity
class ActivityData {
  bool completed;
  int timeSpentInSeconds;
  int marks;
  int? parentSatisfaction;

  ActivityData({
    this.completed = false,
    this.timeSpentInSeconds = 0,
    this.marks = 0,
    this.parentSatisfaction,
  });
}

/// Provider to hold all session data and compute engagement
class BehaviorPracticeProvider extends ChangeNotifier {
  // Demographic info
  String? _gender;
  int? _age;

  // Map of activity IDs to their data
  final Map<String, ActivityData> _activities = {};

  BehaviorPracticeProvider() {
    _initializeActivities();
    _loadDemographicData();
  }

  void _initializeActivities() {
    _activities['activity1'] = ActivityData(parentSatisfaction: 0);
    _activities['activity2'] = ActivityData(parentSatisfaction: 0);

    // 10 sub-activities under activity3
    for (var i = 1; i <= 10; i++) {
      final key = 'activity3_$i';
      final ps = (i == 5) ? 0 : null;
      _activities[key] = ActivityData(parentSatisfaction: ps);
    }
  }

  // --- NEW: check if all required activities are complete ---
  bool get isAllMainActivitiesCompleted {
    final act1 = _activities['activity1']?.completed ?? false;
    final act2 = _activities['activity2']?.completed ?? false;
    final allAct3Done = List.generate(10, (i) => 'activity3_${i + 1}')
        .every((key) => _activities[key]?.completed ?? false);
    return act1 && act2 && allAct3Done;
  }

  // Load demographic data from SharedPreferences
  Future<void> _loadDemographicData() async {
    try {
      final SharedPreferences prefs = await SharedPreferences.getInstance();
      final String? rawGender = prefs.getString('gender');

      if (rawGender == 'ස්ත්‍රී') {
        _gender = 'Female';
      } else if (rawGender == 'පුරුෂ') {
        _gender = 'Male';
      } else {
        _gender = null;
      }

      final ageString = prefs.getString('age');
      if (ageString != null && ageString.isNotEmpty) {
        _age = int.tryParse(ageString);
      }
      notifyListeners();
    } catch (e) {
      debugPrint('Error loading demographic data: $e');
    }
  }

  // Demographics getters
  String? get gender => _gender;
  int? get age => _age;

  // Demographics setters
  Future<void> setGender(String gender) async {
    _gender = gender;
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('gender', gender);
    notifyListeners();
  }

  Future<void> setAge(int age) async {
    _age = age;
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('age', age.toString());
    notifyListeners();
  }

  Future<void> refreshDemographicData() async {
    await _loadDemographicData();
  }

  // Engagement level
  double get engagementLevel {
    if (_activities.isEmpty) return 0.0;
    final completedCount =
        _activities.values.where((d) => d.completed).length;
    return completedCount / _activities.length;
  }

  // Access a single activity
  ActivityData getActivityData(String key) {
    final data = _activities[key];
    if (data == null) throw ArgumentError('Invalid activity key: $key');
    return data;
  }

  // Update activity
  void updateActivity(
      String key, {
        bool? completed,
        int? timeSpentInSeconds,
        int? marks,
        int? parentSatisfaction,
      }) {
    final act = getActivityData(key);
    if (completed != null) act.completed = completed;
    if (timeSpentInSeconds != null) {
      act.timeSpentInSeconds = timeSpentInSeconds;
    }
    if (marks != null) act.marks = marks;
    if (parentSatisfaction != null) {
      act.parentSatisfaction = parentSatisfaction;
    }
    notifyListeners();
  }

  void resetActivityData() {
    _activities.clear();
    _initializeActivities();
    notifyListeners();
  }

  Map<String, ActivityData> get allActivities =>
      Map.unmodifiable(_activities);
}

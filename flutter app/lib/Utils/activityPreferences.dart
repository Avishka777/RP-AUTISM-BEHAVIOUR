import 'package:shared_preferences/shared_preferences.dart';

class ActivityPreferences {
  static const _keyClassroomIdentification = 'classroom_identification_completed';
  static const _keyClassroomEquipment = 'classroom_equipment_completed';
  static const _keyClassroomGoodBad = 'classroom_goodbad_completed';

  // Methods for Classroom Identification (Activity 1)
  static Future<void> setClassroomIdentificationCompleted(bool isCompleted) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyClassroomIdentification, isCompleted);
  }

  static Future<bool> isClassroomIdentificationCompleted() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyClassroomIdentification) ?? false;
  }

  // Methods for Classroom Equipment (Activity 2)
  static Future<void> setClassroomEquipmentCompleted(bool isCompleted) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyClassroomEquipment, isCompleted);
  }

  static Future<bool> isClassroomEquipmentCompleted() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyClassroomEquipment) ?? false;
  }

  // Methods for Classroom Equipment (Activity 3)
  static Future<void> setClassroomGoodBadCompleted(bool isCompleted) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyClassroomGoodBad, isCompleted);
  }

  static Future<bool> isClassroomGoodBadCompleted() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyClassroomGoodBad) ?? false;
  }

  static Future<void> clearAllActivityData() async {
    final prefs = await SharedPreferences.getInstance();
    // Remove keys
    await prefs.remove(_keyClassroomIdentification);
    await prefs.remove(_keyClassroomEquipment);
    await prefs.remove(_keyClassroomGoodBad);
  }
}
import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  static final ApiConfig _instance = ApiConfig._internal();
  factory ApiConfig() {
    return _instance;
  }
  ApiConfig._internal();
  static ApiConfig get instance => _instance;

  static const String _apiUrlKey = 'api_url';
  // Default URL, using the Android emulator address
  static const String _defaultApiUrl = "http://127.0.0.1:8000";

  late String apiUrl;

  /// Loads the saved API URL from storage or uses the default.
  /// This should be called once when the app starts.
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    apiUrl = prefs.getString(_apiUrlKey) ?? _defaultApiUrl;
  }

  /// Updates the API URL and saves it to storage.
  Future<void> setApiUrl(String newUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_apiUrlKey, newUrl);
    apiUrl = newUrl;
  }
}
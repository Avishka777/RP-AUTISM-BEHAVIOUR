import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';
import 'package:ukussa_app/Screens/Home/splash.dart';
import 'package:ukussa_app/Utils/activityPreferences.dart';
import 'package:ukussa_app/Utils/apiConfig.dart';

void main() async {
  // Ensure that Flutter bindings are initialized before calling async code.
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize API configuration
  await ApiConfig.instance.initialize();

  // Clear all previously stored activity data on app restart.
  await ActivityPreferences.clearAllActivityData();

  runApp(
    MultiProvider(
      providers: [
        // Make sure your provider is set up here
        ChangeNotifierProvider(create: (_) => BehaviorPracticeProvider()),
      ],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // If you want to track global elapsed time:
  static final Stopwatch globalStopwatch = Stopwatch();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ukussa App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.lightBlue,
      ),
      home: const Splash(),
    );
  }
}
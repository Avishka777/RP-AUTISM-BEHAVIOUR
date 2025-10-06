// dashboard.dart

import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:ukussa_app/Screens/Home/popup.dart';
import 'package:ukussa_app/Utils/activityPreferences.dart';
import 'package:ukussa_app/Utils/apiConfig.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/label.dart';
import 'package:ukussa_app/Widgets/labelResponsive.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/selectPlace.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';

class Dashboard extends StatefulWidget {
  const Dashboard({super.key});

  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  final List<String> quotes = [
    "සෑම කුඩා පියවරක්ම විශාල ජයග්‍රහණයකි!",
    "සෙමින් යන්න, නමුත් කිසි විටෙකත් නවත්වන්න එපා!",
    "ඔබ නිර්භීත, බුද්ධිමත් සහ කරුණාවන්තයි!",
    "අද විනෝදජනක දෙයක් ඉගෙන ගැනීමට හොඳ දවසක්!",
    "ඔබ ගැන විශ්වාස කරන්න ඔබට පුදුමාකාර දේවල් කළ හැකිය!",
    "ඉගෙනීම ඔබේ සුපිරි බලයයි!",
    "වැරදි ඔබේ මොළය වර්ධනය වීමට උපකාරී වේ!"
  ];

  String name = "";

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    initProcess();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _resetSessionData();
    });
  }

  String getRandomQuote() {
    final random = Random();
    return quotes[random.nextInt(quotes.length)];
  }

  Future<void> initProcess() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    if (mounted) {
      setState(() {
        name = prefs.getString('name') ?? '';
      });
    }
  }

  /// Cleans up all previous activity data and preferences for a new session.
  void _resetSessionData() {
    // 1. Reset the provider data (except age/gender)
    Provider.of<BehaviorPracticeProvider>(context, listen: false)
        .resetActivityData();

    // 2. Clear the activity completion flags from SharedPreferences
    ActivityPreferences.clearAllActivityData();

    print(
        'All activity data and preferences have been reset for a new session.');
  }

  /// Shows a dialog to edit and save the API URL.
  void _showApiUrlDialog() {
    // Controller pre-filled with the current URL
    final controller = TextEditingController(text: ApiConfig.instance.apiUrl);

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text("Change API URL"),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              labelText: "Enter new URL",
              hintText: "e.g., http://192.168.1.5:8000",
            ),
            keyboardType: TextInputType.url,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text("Cancel"),
            ),
            ElevatedButton(
              onPressed: () async {
                if (controller.text.isNotEmpty) {
                  await ApiConfig.instance.setApiUrl(controller.text);
                  if (mounted) {
                    Navigator.of(context).pop();
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                          content: Text("API URL updated: ${controller.text}")),
                    );
                  }
                }
              },
              child: const Text("Save"),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    return PopScope(
      canPop: false,
      onPopInvoked: (didPop) async {
        if (didPop) {
          return;
        }
        SystemNavigator.pop();
      },
      child: Scaffold(
        backgroundColor: AppColors.green8,
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      GestureDetector(
                        onTap: () {
                          // Assuming MyForm is a valid screen to navigate to
                          NavigationUtils.backNavigation(context, MyForm());
                        },
                        child: Container(
                          width: 35,
                          height: 35,
                          decoration: const BoxDecoration(
                            image: DecorationImage(
                              image: AssetImage('assets/settings.png'),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      // New button to change API URL
                      IconButton(
                        icon: const Icon(Icons.hub_outlined,
                            color: Colors.black54),
                        onPressed: _showApiUrlDialog,
                        tooltip: 'Change API URL',
                      ),
                    ],
                  ),
                  Container(
                    padding: const EdgeInsets.only(left: 20.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Label(
                                hintText: "ආයුබෝවන්",
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font20,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal),
                            Label(
                                hintText: name.toUpperCase(),
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font20,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.w500)
                          ],
                        ),
                        Container(
                          height: 75,
                          width: 75,
                          decoration: BoxDecoration(
                            image: const DecorationImage(
                              image: AssetImage('assets/report.png'),
                            ),
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Stack(
                    alignment: Alignment.topCenter,
                    children: [
                      Padding(
                        padding: const EdgeInsets.only(
                            left: 28.0, right: 28.0, bottom: 12, top: 40),
                        child: Container(
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: AppColors.pink6,
                            borderRadius: BorderRadius.circular(15),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.only(
                                left: 16.0, right: 16.0, bottom: 20.0, top: 60.0),
                            child: Column(
                              children: [
                                LabelResponsive(
                                  hintText: getRandomQuote(),
                                  textColor: AppColors.black1,
                                  fontSize: AppFonts.font18,
                                  fontFamily: AppFonts.Lora,
                                  fontWeight: FontWeight.normal,
                                  textAlign: TextAlign.center,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                      Container(
                        height: 75,
                        width: 75,
                        decoration: BoxDecoration(
                          image: const DecorationImage(
                            image: AssetImage('assets/logo.png'),
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ],
                  ),
                  Padding(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 28.0, vertical: 4.0),
                    child: Container(
                      height: MediaQuery.of(context).size.height * 0.25,
                      decoration: const BoxDecoration(
                        image: DecorationImage(
                          image: AssetImage('assets/dbc1.png'),
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 28.0, vertical: 4.0),
                    child: GestureDetector(
                      onTap: () {
                        NavigationUtils.frontNavigation(context, const MapDashboard());
                      },
                      child: Container(
                        height: MediaQuery.of(context).size.height * 0.25,
                        decoration: const BoxDecoration(
                          image: DecorationImage(
                            image: AssetImage('assets/dbc2.png'),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 28.0, vertical: 4.0),
                    child: GestureDetector(
                      onTap: () {
                        NavigationUtils.frontNavigation(context, const SelectPlace());
                      },
                      child: Container(
                        height: MediaQuery.of(context).size.height * 0.25,
                        decoration: const BoxDecoration(
                          image: DecorationImage(
                            image: AssetImage('assets/dbc3.png'),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(
                        left: 28.0, right: 28.0, bottom: 12),
                    child: Container(
                      height: MediaQuery.of(context).size.height * 0.25,
                      decoration: const BoxDecoration(
                        image: DecorationImage(
                          image: AssetImage('assets/dbc4.png'),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Utils/activityPreferences.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/Activity2/activity2.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/Activity1/activity1.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/finalEmotionDetectionScreen.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/Activity3/act1.dart';
import 'package:ukussa_app/Screens/Home/dashboard.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';

class ClassroomInstructionScreen extends StatelessWidget {
  const ClassroomInstructionScreen({super.key});

  void _onCompletePressed(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const FinalEmotionDetectionScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<BehaviorPracticeProvider>(context);
    final allDone = provider.isAllMainActivitiesCompleted;

    final activity1 = provider.getActivityData("activity1").completed;
    final activity2 = provider.getActivityData("activity2").completed;
    final activity3 = provider.getActivityData("activity3_1").completed;

    return Scaffold(
      backgroundColor: const Color(0xFF87CEEB),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _titleBar(context, "දෙමාපියන්ට උපදෙස්: පන්තිකාමරයේ හැසිරීම් පුහුණුව."),
              const SizedBox(height: 16),

              // Section 0: Introduction
              _infoCard(Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle("අරමුණ."),
                  _text("දරුවාට පන්තිකාමරයේ හැසිරීම් සම්බන්ධයෙන් හැඳින්වීමක් ලබා දීම, සහ ඒවා භාවිතා කළ හැකි ආකාරය නිවසේදීම ඉගැන්වීම"),
                  _sectionTitle("මෙම ක්‍රියාකාරකම සංජානන සංවර්ධනයට සහාය වන ආකාරය", isSub: true),
                  ...[
                    ["📘", "ස්ථාන හඳුනාගැනීමේ හැකියාව"],
                    ["✏️", "අයිතම හැඳිනීම"],
                    ["🗣️", "භාෂා හා සන්නිවේදන"],
                    ["🤝", "සමාජමය හැසිරීම්"],
                    ["🧍‍♂️", "විනය සහ ක්‍රමශීලී හැසිරීම"],
                  ].map((e) => _feature(e[0]!, e[1]!)),
                ],
              )),
              const SizedBox(height: 16),

              // Section 1: Classroom Identification
              _activitySection(
                title: "01. පන්තිකාමරය හඳුනාගැනීම.",
                aim: "දරුවාට පන්තිකාමරය කියන්නේ මොන තැනක්ද යන්න හඳුනාගැනීමට උදව් කිරීම",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතයෙන්-", isSub: true),
                    _text("පන්තිය, ගුරුවරිය, පුවරුව, පුටුව, මේසය, පොත් රාක්කය"),
                    _sectionTitle("දරුවාට පැහැදිලි කිරීම:", isSub: true),
                    _text("“මේ පන්ති කාමරයක්. මෙතන යාලුවෝ ඉගෙනගන්නවා. මෙතන ගුරුවරිය ඉන්නවා. ඔයා මෙහෙම තැනක ඉගෙන ගන්නවා“"),
                    _sectionTitle("අත්දැකීම් ක්‍රමය:", isSub: true),
                    _text("පින්තූර කාඩ් එකක් පෙන්වලා, “ඔයා මේ තැන දන්නවද?”"),
                    _text("නිවසේ පන්ති වාතාවරණයක් සකසා දරුවාටද ලං අවබෝදයක් ලබා දෙන්න"),
                  ],
                ),
                childButton: activity1
                    ? Center(
                    child: Text("ක්‍රියාකාරකම සම්පූර්ණයි",
                        style: TextStyle(
                            color: Colors.red,
                            fontSize: 16,
                            fontWeight: FontWeight.bold)))
                    : ElevatedButton(
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        padding: const EdgeInsets.symmetric(vertical: 12)),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const ClassroomIdentificationActivity()),
                      );
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // Section 2: Classroom Equipment
              _activitySection(
                title: "02️. පන්තිකාමරයේ උපකරණ හඳුනාගැනීම.",
                aim: "පන්තිකාමරය තුළ භාවිතා වන දේවල් හඳුනාගැනීම",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් භාවිතා කරන්න", isSub: true),
                    ...[
                      ["🖤", "කලු ලෑල්ල"],
                      ["✏️", "පෑන"],
                      ["📗", "පොත"],
                      ["🪑", "පුටුව"],
                      ["🖥️", "මේසය"],
                      ["🌸", "මල් පෝච්චිය"],
                      ["💧", "බෝතල් රාක්කය"],
                      ["📚", "පොත් රාක්කය"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("දරුවාට පැහැදිලි කිරීම:", isSub: true),
                    _text("“මේක පෑනක්. මෙකෙන් ලියනවා.“\n“මේක පොතක්. පොතේ අපි ලියනවා“\n“මේක පුටුව. මෙතන ඔයා ඉඳලා ඉගෙන ගන්නවා.“\n“මේක පොත් රාක්කය. මෙතන අපි පොත් තබනවා.“"),
                    _sectionTitle("නිවසේ අත්දැකීම් ක්‍රමය:", isSub: true),
                    _text("කුඩා \"පන්ති වාතාවරණයක්\" නිවසේ සකස් කරන්න. දරුවාට කාඩ් එකක් පෙන්වලා, මේක ඔයා දන්නවද? කියලා අහන්න. ඔබ ගුරුතුමිය ලෙස දරුවා සමග කටයුතු කරන්න. හැකිනම් ඔබ සාරියකින් සැරසී දරුවාට පාඩම කියාදෙන්න. එමගින් දරුවාට මතකයේ රැදේ."),
                  ],
                ),
                childButton: activity2
                    ? Center(
                    child: Text("ක්‍රියාකාරකම සම්පූර්ණයි",
                        style: TextStyle(
                            color: Colors.red,
                            fontSize: 16,
                            fontWeight: FontWeight.bold)))
                    : ElevatedButton(
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        padding: const EdgeInsets.symmetric(vertical: 12)),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const ClassroomEquipmentActivity()),
                      );
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // Section 3: Good Behavior
              _activitySection(
                title: "03️. යහපත් හැසිරීම්/අයහපත් හැසිරීම්.\nයහපත් හැසිරීම්",
                aim: "පන්තිකාමරය තුළ හොඳ හැසිරීම් පිළිබඳ දරුවාට ඉගැන්වීම",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් භාවිතා කරන්න", isSub: true),
                    ...[
                      ["✋", "අත ඔසවීම"],
                      ["🧏", "හොඳ දරුවෙකු සේ හැසිරීම"],
                      ["🗣️", "යාලුවක් සමග සමගියෙන් සිටීම"],
                      ["📚", "ගුරුවරියගේ උපදෙස් අනුව ක්‍රියා කිරීම"],
                      ["✅", "පෝලිමේ පිවිසීම"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("දරුවාට පැහැදිලි කිරීම:", isSub: true),
                    _text("“ඔයාට ප්‍රශ්නයක් තියෙන්නේ නම් අත ඔසවන්න.“\n“අපි අපේ බෝතල් රාක්කෙන් තියනවා“\n“ගුරුවරිය කියන දේවල් අහලා ඒ විදිහට ක්‍රියාකරනවා.“\n“අපි කලු ලෑල්ල ලස්සනට තියාගන්නවා“"),
                    _text("ඔබ ගුරුවරිය වෙලා දරුවාට උගන්වන්න \"ඔයාට හොඳ හැසිරීමක් දැක්වුවා. බොහොම ස්තුතියි!. දරුවාගේ නම පවසා මෙයා හොඳ ලමයෙක් යැයි අගයන්න"),
                  ],
                ),
                childButton: Container(), // Empty container for no button
              ),
              const SizedBox(height: 16),

              // Section: Bad Behavior
              _activitySection(
                title: "අයහපත් හැසිරීම්",
                aim: "පන්තිකාමරය තුළ වැළැක්විය යුතු හැසිරීම් හඳුනාගැනීම",
                borderColor: Colors.red,
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් භාවිතා කරන්න", isSub: true),
                    ...[
                      ["👊", "යාලුවන් සමග රණ්ඩු කිරීම"],
                      ["😠", "කෑගසනවා"],
                      ["🚫", "පෝලිම පනිනවා"],
                      ["❌", "මල් පෝච්චිය කඩනවා"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("දරුවාට පැහැදිලි කිරීම:", isSub: true),
                    _text("“අපි අනිත් අයගේ දේවල් උදුරන්නේ නෑ. අපි ඉල්ලලා ගන්නවා.“\n“අපි ගුරුතුමිය කියන දේ අහනවා කෑගහන්නේ නෑ“\n“පන්තියේ දඟ කරන එක නරක පුරුද්දක්“"),
                    _text("පින්තූරයක් පෙන්වලා \"මෙක හොඳද නරකද?\" කියලා අහන්න. නිවැරදි හැසිරීමට හැරවීම. දරුවාට සමීප පුද්ගලයන් ඈදා කතාවක් අකාරයෙන් නරක හැසිරීම් වැරදි බව වටහා දෙන්න."),
                  ],
                ),
                childButton: activity3
                    ? Center(
                    child: Text("ක්‍රියාකාරකම සම්පූර්ණයි",
                        style: TextStyle(
                            color: Colors.red,
                            fontSize: 16,
                            fontWeight: FontWeight.bold)))
                    : ElevatedButton(
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        padding: const EdgeInsets.symmetric(vertical: 12)),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const ClassroomGoodBadActivity1()),
                      );
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // Section 4: Final Note
              _infoCard(Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle("04. දරුවාගේ ක්‍රියාවන් ඇගයීම මගින් දරුවා උනන්දු කිරීම.", big: true),
                  _text("අභිප්‍රේරණය සඳහා අත්පුඩි, චියර්ස්, අතථ්‍ය ත්‍යාග ලබා දෙන්න."),
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.1),
                      border: Border.all(color: Colors.red),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: _text("සෑම පාඩමක්ම අවසානයේ දරුවාට පැවරුමක් කිරීමට ලබාදෙන්න. දරුවාට සහය වෙන්න. දරුවාගේ පැවරුමේ දේ හැසිරීම පිලිබඳ මනා අවබෝදයෙන් සිටීම අත්‍යාවශ්‍ය වේ."),
                  )
                ],
              )),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                child: ElevatedButton(
                  onPressed: allDone ? () => _onCompletePressed(context) : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: allDone ? Colors.blue.shade600 : Colors.grey,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8)),
                  ),
                  child: const Text('සම්පූර්ණයි',
                      style:
                      TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ---------- HELPERS ----------
  static Widget _titleBar(BuildContext ctx, String text) => Container(
    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
    decoration: BoxDecoration(
      color: Colors.blue[100],
      borderRadius: BorderRadius.circular(8),
      border: Border.all(color: Colors.blue[200]!),
    ),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      IconButton(
          icon: const Icon(Icons.home, color: Colors.black54),
          onPressed: () => Navigator.pushReplacement(
              ctx, MaterialPageRoute(builder: (_) => const Dashboard()))),
      Expanded(
          child: Text(text,
              textAlign: TextAlign.center,
              style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87))),
      const SizedBox(width: 48),
    ]),
  );

  static Widget _infoCard(Widget child) => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: Colors.blue[50]?.withOpacity(0.85),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: Colors.blue[200]!),
    ),
    child: child,
  );

  static Widget _feature(String icon, String text) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(icon, style: const TextStyle(fontSize: 20)),
      const SizedBox(width: 12),
      Expanded(child: Text(text, style: const TextStyle(fontSize: 16))),
    ]),
  );

  // Modified section to allow "completed" label
  static Widget _activitySection({
    required String title,
    required String aim,
    required Widget content,
    required Widget childButton,
    Color borderColor = Colors.blue,
  }) =>
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: borderColor.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: borderColor.withOpacity(0.5)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text(title,
              style:
              const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(8)),
              child: RichText(
                  text: TextSpan(
                      style: const TextStyle(
                          fontSize: 15,
                          color: Colors.black,
                          fontFamily: 'NotoSansSinhala'),
                      children: [
                        const TextSpan(
                            text: "අරමුණ: ",
                            style: TextStyle(fontWeight: FontWeight.bold)),
                        TextSpan(text: aim),
                      ]))),
          const SizedBox(height: 12),
          content,
          Padding(padding: const EdgeInsets.only(top: 16), child: childButton),
        ]),
      );

  static Widget _sectionTitle(String text,
      {bool isSub = false, bool big = false}) =>
      Padding(
        padding: const EdgeInsets.only(top: 8, bottom: 4),
        child: Text(text,
            style: TextStyle(
                fontSize: big ? 18 : 16,
                fontWeight: isSub || big ? FontWeight.w600 : FontWeight.bold)),
      );

  static Widget _text(String text) =>
      Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Text(text, style: const TextStyle(fontSize: 15)));
}
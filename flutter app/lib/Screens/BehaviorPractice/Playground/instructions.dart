import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/Activity2/activity2.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/Activity1/activity1.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/finalEmotionDetectionScreen.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/Activity3/act1.dart';
import 'package:ukussa_app/Screens/Home/dashboard.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';

class PlaygroundInstructionScreen extends StatelessWidget {
  const PlaygroundInstructionScreen({super.key});

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
              _titleBar(context, "දෙමාපියන්ට උපදෙස්: ක්‍රීඩාපිටියේ හැසිරීම් පුහුණුව."),
              const SizedBox(height: 16),

              // Section 0: Introduction
              _infoCard(Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle("අරමුණ."),
                  _text("දරුවාට ක්‍රීඩාපිටිය පිළිබඳ හඳුනාගැනීම, අදාල වස්තු, හොඳ හැසිරීම් සහ වැළැක්විය යුතු හැසිරීම් උගන්වීම."),
                  _sectionTitle("මෙම ක්‍රියාකාරකම සංජානන සංවර්ධනයට සහාය වන ආකාරය", isSub: true),
                  ...[
                    ["🧠", "ස්ථාන හා වස්තු හඳුනාගැනීම"],
                    ["🗣️", "සන්නිවේදන හැකියාවන්"],
                    ["🤝", "සමාජ හැසිරීම්"],
                    ["🧍", "නිවැරදි හැසිරීම් හා ආරක්ෂාව"],
                  ].map((e) => _feature(e[0]!, e[1]!)),
                ],
              )),
              const SizedBox(height: 16),

              // Section 1: Place Identification
              _activitySection(
                title: "01. ස්ථානය හඳුනාගැනීම.",
                aim: "පින්තූර කාඩ් බාවිතාකරන්න - සෙල්ලම් පිට්ටනියේ රූප බාවිතා කරන්න",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("දරුවාට පැහැදිලි කිරීම:", isSub: true),
                    _text("“මේ තැන සෙල්ලම්පිට්ටනියයි. මෙතන යාලුවො එක්ක සෙල්ලම් කරන්න පුලුවන්.“\nඅපි එකමුතුව සෙල්ලම් කරන්න ඕන"),
                    _sectionTitle("අත්දැකීම් ක්‍රමය:", isSub: true),
                    _text("පින්තූර පෙන්වලා, “මෙතැන ඔයාට මොකද්ද කරන්න පුළුවන්?” කියලා අහන්න"),
                    _text("නිවසේ මිදුලේ සෙල්ලම් පිටියක් නිර්මානය කරන්න."),
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
                            builder: (_) => const PlaceRecognitionActivity()),
                      );
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // Section 2: Object Identification
              _activitySection(
                title: "02️. අදාල වස්තු හඳුනාගැනීම.",
                aim: "පින්තූර කාඩ් බාවිතාකරන්න",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතා කරන්න", isSub: true),
                    ...[
                      ["🛝", "ලිස්සන බෝට්ටුව"],
                      ["🧗", "ඔන්චිල්ලාව"],
                      ["🪣", "සීසෝ"],
                      ["🪀", "සෙල්ලම් බෝලය"],
                      ["🧸", "සෙල්ලම් බඩු"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("පැහැදිලි කිරීම:", isSub: true),
                    _text("“මෙක ලිස්සන බෝට්ටුව, මේකෙන් යාලුවෝ එක්ක සෙල්ලම් කරන්න පුලුවන්.“\n“මෙක ඔන්චිල්ලාව මේක පරිස්සමෙන් පදින්න ඕන“"),
                    _sectionTitle("ප්‍රායෝගික ක්‍රමය:", isSub: true),
                    _text("ඔයා කැමති මොන සෙල්ලම් කරන්නද. දරුවාගෙන් බාන්ඩ පෙන්වා කැමති දේ පිලිබඳ විමසන්න"),
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
                            builder: (_) => const ObjDetectionActivity()),
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
                aim: "පින්තූර කාඩ් බාවිතාකරන්න",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතා කරන්න", isSub: true),
                    ...[
                      ["✋", "සහයෝගයෙන් සෙල්ලම් කරමු"],
                      ["😃", "සතුටින් සෙල්ලම් වීම"],
                      ["🤝", "බෙදාගැනීම"],
                      ["🗣️", "තමාගේ වාරය එනතෙක් ‍රැදී සිටීම"],
                      ["🚶", "පෝලිමකට පිවිසීම"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("පැහැදිලි කිරීම:", isSub: true),
                    _text("“ඔයාට තවත් ලමයෙකුත් එක්ක සෙල්ලම් කරන්න ඕන නම් එයාගෙන් එකතු වෙන්නද කියල්ලා අහන්න“\n“සෙල්ලම් කරද්දී සහයෝගයෙන් සෙල්ලම් කරන්න ඕන“\n“යාලුවෙක් වැටුන විට උදවු කරන්න ඕන“"),
                    _text("උදව් වචන: “කරුණාකර”, “ඔයාට අවශ්‍යද?”, “ස්තුතියි”."),
                    _text("නිවසේදී දරුවා සමග සෙල්ලම් කිරීමේදී ඉහත සදහන් දේවල් ප්‍රායෝගිකව බාවිතා කරන්න"),
                  ],
                ),
                childButton: Container(), // Empty container for no button
              ),
              const SizedBox(height: 16),

              // Section: Bad Behavior
              _activitySection(
                title: "අයහපත් හැසිරීම්",
                aim: "පින්තූර කඩ් බාවිතාකරන්න",
                borderColor: Colors.red,
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතා කරන්න", isSub: true),
                    ...[
                      ["👊", "තල්ලු කිරීම"],
                      ["😠", "කෑගසීම"],
                      ["🚫", "පෝලිමෙන් පිටවීම"],
                      ["🧸", "සෙල්ලම් බඩු අනිත් අයගෙන් බලයෙන් ඇරීම"],
                      ["💥", "අනිත් දරුවන්ට බාධා කිරීම"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("පැහැදිලි කිරීම:", isSub: true),
                    _text("“ඔයා අනිත් ලමයෙකු තල්ලු කළොත් ඔහුට තුවාල වන්න පුළුවන්.“\n“අපි රන්ඩු කරන්නෑ එකමුතුව සෙල්ලම් කරනවා“\n“ඔන්චිල්ලාවේ දඟ වැඩ කරන්න හොඳ නෑ. තුවාල වෙන්න පුලුවන්“"),
                    _text("පින්තූර පෙන්න්වා “හොඳද/නරකද?” යන්න වටහා දෙන්න. නිවැරදි හැසිරීමක් කරන්න දරුවාට උදව් කරන්න.\nනිවසේදී දරුවා සමග සෙල්ලම් කිරීමට පෙලෙබෙන්න. ඉහත පුරුදු ප්‍රායෝගිකව යොදා ගන්න"),
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
                            builder: (_) => const PlaygroundGoodBadActivity1()),
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
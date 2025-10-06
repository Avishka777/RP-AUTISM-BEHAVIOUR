import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity1/activity1.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity2/activity2.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity3/act1.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/finalEmotionDetectionScreen.dart';
import 'package:ukussa_app/Screens/Home/dashboard.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';

class RestaurantInstructionScreen extends StatelessWidget {
  const RestaurantInstructionScreen({super.key});

  void _onCompletePressed(BuildContext context) {
    Navigator.push(context,
        MaterialPageRoute(builder: (_) => const FinalEmotionDetectionScreen()));
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
              _titleBar(context,
                  "දෙමාපියන්ට උපදෙස්: ආපනශාලාව/රෙස්ටුරෙන්ට් හැසිරීම් පුහුණුව."),
              const SizedBox(height: 16),
              _infoCard(Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle("අරමුණ."),
                  _text(
                      "දරුවාට ආපනශාලාවේ හැසිරීම්, සන්නිවේදන, ආහාර ආකල්ප සහ සමාජමය හැසිරීම් පිළිබඳව උගන්වීම."),
                  _sectionTitle("මෙම ක්‍රියාකාරකම සංජානන සංවර්ධනයට සහාය වන ආකාරය",
                      isSub: true),
                  ...[
                    ["🍴", "වටාපිටාව හඳුනාගැනීම"],
                    ["🧠", "අයිතම හඳුනාගැනීම"],
                    ["🗣️", "ආහාර ඉල්ලීම සහ හොද පුරුදු"],
                    ["🤝", "අනුකූල හැසිරීම්"],
                    ["🧍‍♂️", "පෝලිම, ඉවසීම, ඉල්ලීමේ ක්‍රමය"],
                  ].map((e) => _feature(e[0]!, e[1]!)),
                ],
              )),
              const SizedBox(height: 16),

              // 🔹 Activity 1
              _activitySection(
                title: "01. ස්ථානය හඳුනාගැනීම.",
                aim: "දරුවාට ආපන ශාලාව හෝ රෙස්ටුරන්ට් කියන්නේ මොන තැනක්ද කියා හඳුන්වීම",
                content: _buildTexts([
                  "පින්තූර කාඩ් - රෙස්ටුරන්ට් පින්තූර කිහිපයක් දරුවාට පෙන්වන්න",
                  "දරුවාට පැහැදිලි කිරීම:",
                  "“මෙක කෑම කන්න තැනක්...“\n“අපි මෙතන හොද ලමයි වගේ හැසිරෙන්න ඕන“",
                  "අත්දැකීම් ක්‍රමය:",
                  "නිවසේ කෑම මේසයේ පිලිගන්වන්න...",
                ]),
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
                              builder: (_) =>
                              const PlaceRecognitionActivity()));
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // 🔹 Activity 2
              _activitySection(
                title: "02️. අදාල අයිතම හඳුනාගැනීම.",
                aim: "ආහාරයට අදාල වස්තු හඳුනාගැනීම",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතා කරන්න", isSub: true),
                    ...[
                      ["🍽️", "පිගාන"],
                      ["🥄", "හැන්ද"],
                      ["🍴", "ගෑරුප්පුව"],
                      ["🥤", "කෝප්පය"],
                      ["🧂", "අත්පිස්නාව"],
                      ["🪑", "පුටුව"],
                      ["🪟", "මේසය"],
                      ["👨‍🍳", "සේවකයා"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
                    _sectionTitle("පැහැදිලි කිරීම:", isSub: true),
                    _text("“මෙක පුටුව...මෙකෙන් කනවා.“\n“මෙයාගෙන් අපිට අවශය කෑම ඉල්ලනවා“"),
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
                              builder: (_) =>
                              const ObjDetectionActivity()));
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              // 🔹 Activity 3
              _activitySection(
                title: "03️. යහපත් හැසිරීම්",
                aim: "ආහාර අවස්ථාවේ සුදුසු හැසිරීම් උගන්වීම",
                content: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle("පින්තූර කාඩ් බාවිතා කරන්න", isSub: true),
                    ...[
                      ["🙏", "“කරුණාකර” කියා ඉල්ලීම"],
                      ["😊", "“ස්තුතියි” කියා කෑම ගැනීම"],
                      ["🍽️", "පිගානේ පිලිවලකට ආහාර ගැනීම"],
                      ["🤫", "නිශ්ශබ්දව ආහාර ගැනීම"],
                      ["🧍", "පෝලිමකට ඉවසීමෙන් රැදී සිටීම"],
                    ].map((e) => _feature(e[0]!, e[1]!)),
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
                              builder: (_) =>
                              const RestaurantGoodBadActivity1()));
                    },
                    child: const Text("ක්‍රියාකාරකම කරමු",
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold))),
              ),
              const SizedBox(height: 16),

              _infoCard(Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle("04️. දරුවාගේ ක්‍රියාවන් ඇගයීම", big: true),
                  _text("අභිප්‍රේරණය සඳහා අත්පුඩි, චියර්ස්, අතථ්‍ය ත්‍යාග ලබා දෙන්න."),
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.1),
                      border: Border.all(color: Colors.red),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: _text("සෑම පාඩමක්ම අවසානයේ දරුවාට පැවරුමක් කිරීමට ලබා දෙන්න. දරුවාට සහය වෙන්න. දරුවාගේ පැවරුමේ දේ හැසිරීම පිලිබඳ මනා අවබෝදයෙන් සිටීම අත්‍යාවශ්‍ය වේ."),
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

  static Widget _buildTexts(List<String> texts) => Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: texts.map((t) => _text(t)).toList());
}
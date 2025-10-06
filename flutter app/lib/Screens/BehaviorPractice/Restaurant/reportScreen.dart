import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';
//import 'package:ukussa_app/Utils/constValues.dart';
import 'package:ukussa_app/Utils/apiConfig.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

class ReportScreen extends StatefulWidget {
  final String? detectedEmotion;
  const ReportScreen({Key? key, this.detectedEmotion}) : super(key: key);

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  bool _loading = true;
  String? _error;
  String? _prediction;
  List<String>? _suggestions;

  // Mapping for Sinhala activity names in the UI table
  final Map<String, String> _activityNameMapping = {
    'activity1': 'කෑම කන තැන තෝරන්න',
    'activity2': 'කෑම මේසයේ දේවල් රවුම තුලට දාමු',
    'activity3_1': 'අවන්හලේදී කෑම ඕඩර් කරනවා',
    'activity3_2': 'කෑම කද්දි අවශ්‍ය දෙයක් ඕඩර් කරනවා',
    'activity3_3': 'හැන්ද නිවැරදිව භාවිතා කරනවා',
    'activity3_4': 'කෑම කනවිට හොද ලමයෙක් වගේ ඉන්නවා',
    'activity3_5': 'ඔයාගේ පුටුවේ කවුරුහරි වාඩිවුනා',
    'activity3_6': 'යාලුවගේ කෑම මුලින් ලැබුනා',
    'activity3_7': 'යාලුවෙක්ට කෑම කද්දි හිරවෙනවා',
    'activity3_8': 'පිහිය ගෑරුප්පු නිවැරදිව භාවිතා කරනවා',
    'activity3_9': 'යාලුවා ඔයාගෙන් කෑම ඉල්ලනවා',
    'activity3_10': 'බඩ පිරිලා කෑම ඉතුරුවෙලා තියනවා'
  };

  @override
  void initState() {
    super.initState();
    _sendRawForPrediction();
  }

  Future<void> _sendRawForPrediction() async {
    // This function remains the same
    final session = context.read<BehaviorPracticeProvider>();
    final activitiesMap = <String, Map<String, dynamic>>{};

    session.allActivities.forEach((key, d) {
      activitiesMap[key] = {
        "completed": d.completed,
        "timeSpent": d.timeSpentInSeconds,
        "marks": d.marks,
        "parentSatisfaction": d.parentSatisfaction
      };
    });

    final body = {
      "Age": session.age ?? 0,
      "Gender": session.gender ?? "",
      "Current_Mood": widget.detectedEmotion ?? "",
      "activities": activitiesMap
    };

    try {
      final uri = Uri.parse('${ApiConfig.instance.apiUrl}/predict');
      final resp = await http
          .post(uri,
          headers: {"Content-Type": "application/json"},
          body: json.encode(body))
          .timeout(const Duration(seconds: 15));

      if (resp.statusCode != 200) {
        throw Exception('Server returned ${resp.statusCode}: ${resp.body}');
      }

      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (mounted) {
        setState(() {
          _prediction = data['prediction'] as String?;
          _suggestions = (data['suggestions'] as List<dynamic>)
              .map((e) => e.toString())
              .toList();
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _loading = false;
        });
      }
    }
  }

  // --- NEW: Function to generate activity-specific suggestions ---
  List<String> _getActivitySpecificSuggestions(BehaviorPracticeProvider session) {
    final List<String> activitySuggestions = [];
    final activities = session.allActivities;

    // Check activity1 marks
    final activity1 = activities['activity1'];
    if (activity1 != null) {
      if (activity1.marks == 0) {
        activitySuggestions.add('ක්‍රියාකාරම 1: ඔබේ දරුවාට ස්තානය හදුනාගැනීමට නොහැක.ස්තානය හදුන්වා දීමේ පාඩම නැවත උගන්වන්න.');
      } else {
        activitySuggestions.add('ක්‍රියාකාරම 1: ඔබේ දරුවාට ස්තානය හදුනාගැනීමට හැකිය.නිපුනතාව සාර්තකයි.');
      }
    }

    // Check activity2 marks
    final activity2 = activities['activity2'];
    if (activity2 != null) {
      if (activity2.marks == 0) {
        activitySuggestions.add('ක්‍රියාකාරම 2: ඔබේ දරුවාට ස්තානයට අදාල භාන්ඩ හදුනා ගැනීමට දුර්වලයි.නැවත පාඩමට අවදානය යොමු කරන්නන.');
      } else {
        activitySuggestions.add('ක්‍රියාකාරම 2: ඔබේ දරුවාට ස්තානයට අදාල භාන්ඩ හදුනා ගත හැකිය.නිපුනතාව සාර්තකයි.');
      }
    }

    // Calculate total marks for activity3 sub-activities
    int activity3TotalMarks = 0;
    for (var i = 1; i <= 10; i++) {
      final key = 'activity3_$i';
      final activity = activities[key];
      if (activity != null) {
        activity3TotalMarks += activity.marks;
      }
    }

    // Provide suggestion based on activity3 total marks
    if (activity3TotalMarks == 0) {
      activitySuggestions.add('ක්‍රියාකාරම 3: ඔබේ දරුවාට ස්තානයට අදාල නිවැරිදි හැසිරීම් තෝරාගැනීම ඉතා දුර්වලයි.වැඩි අවදානය සමග පාඩම උගන්වන්න');
    } else if (activity3TotalMarks > 0 && activity3TotalMarks <= 400) {
      activitySuggestions.add('ක්‍රියාකාරම 3: ඔබේ දරුවාට ස්තානයට අදාල නිවැරිදි හැසිරීම් තෝරාගැනීම දුර්වලයි.නැවත පාඩම උදාහරන සහිතව උගන්වන්න');
    } else if (activity3TotalMarks > 400 && activity3TotalMarks <= 600) {
      activitySuggestions.add('ක්‍රියාකාරම 3: ඔබේ දරුවාට ස්තානයට අදාල නිවැරිදි හැසිරීම් තෝරාගැනීම සාමනයි.උදාහරන සහිතව නැවත අවදානය යොමු කරන්න');
    } else if (activity3TotalMarks > 600) {
      activitySuggestions.add('ක්‍රියාකාරම 3: ඔබේ දරුවාට ස්තානයට අදාල නිවැරිදි හැසිරීම් තෝරාගැනීම විශිශ්ටයි.නිපුනතාව සාර්තක');
    }

    return activitySuggestions;
  }

  Future<void> _generateAndSharePdf() async {
    final session = context.read<BehaviorPracticeProvider>();
    final pdf = pw.Document();

    // Calculate engagement data for PDF
    final activities = session.allActivities;
    final completedCount = activities.values.where((d) => d.completed).length;
    final total = activities.length;
    final engagementPct = (session.engagementLevel * 100).toStringAsFixed(1);

    // Get activity-specific suggestions for PDF
    final activitySuggestions = _getActivitySpecificSuggestions(session);

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        build: (context) => [
          pw.Header(
            level: 0,
            child: pw.Text("Session Report", style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold)),
          ),
          pw.Paragraph(text: "Generated on: ${DateTime.now().toLocal().toString().substring(0, 16)}"),
          pw.Divider(),
          pw.SizedBox(height: 20),

          // Overall Engagement Section
          pw.Text("Overall Engagement", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          pw.Container(
            padding: const pw.EdgeInsets.all(12),
            decoration: pw.BoxDecoration(
              color: PdfColors.blue50,
              borderRadius: pw.BorderRadius.circular(8),
            ),
            child: pw.Text(
              'Overall Engagement: $engagementPct% ($completedCount/$total activities completed)',
              style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold),
            ),
          ),
          pw.SizedBox(height: 30),

          pw.Text("Activity Details", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          pw.Table.fromTextArray(
            headers: ['Activity', 'Completed', 'Time (s)', 'Marks', 'Parent Sat.'],
            data: session.allActivities.entries.map((entry) {
              final d = entry.value;
              return [
                entry.key, // Using the original English key
                d.completed ? 'Yes' : 'No',
                d.timeSpentInSeconds.toString(),
                d.marks.toString(),
                d.parentSatisfaction?.toString() ?? '-',
              ];
            }).toList(),
          ),
          pw.SizedBox(height: 30),

          // --- NEW: Activity-Specific Suggestions Section in PDF ---
          pw.Text("Activity-Specific Suggestions", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          if (activitySuggestions.isNotEmpty)
            ...activitySuggestions.map((s) => pw.Bullet(text: s)),
          pw.SizedBox(height: 20),

          // Prediction and Suggestions
          pw.Text("Prediction & Suggestions", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          pw.Text("Prediction Level: ${_prediction ?? 'N/A'}", style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 10),
          if (_suggestions != null)
            ..._suggestions!.map((s) => pw.Bullet(text: s)),
        ],
      ),
    );

    await Printing.sharePdf(bytes: await pdf.save(), filename: 'session_report.pdf');
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<BehaviorPracticeProvider>();
    final activities = session.allActivities;
    final completedCount = activities.values.where((d) => d.completed).length;
    final total = activities.length;
    final engagementPct = (session.engagementLevel * 100).toStringAsFixed(1);

    // Get activity-specific suggestions
    final activitySuggestions = _getActivitySpecificSuggestions(session);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Session Report"), // Reverted to English
        actions: [
          IconButton(
            icon: const Icon(Icons.picture_as_pdf),
            onPressed: (_loading || _error != null) ? null : _generateAndSharePdf,
            tooltip: "Download Report", // Reverted to English
          ),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/bgimg.png'),
            fit: BoxFit.cover,
            opacity: 0.5,
          ),
        ),
        child: Column(
          children: [
            // Top summary card
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(
                    'Overall Engagement: $engagementPct% ($completedCount/$total activities completed)', // Reverted to English
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),

            // Detailed table
            Expanded(
              flex: 3,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 12.0),
                clipBehavior: Clip.antiAlias,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.9),
                  borderRadius: BorderRadius.circular(12.0),
                ),
                child: SingleChildScrollView(
                  // Vertical scroll
                  child: SingleChildScrollView(
                    // Horizontal scroll
                    scrollDirection: Axis.horizontal,
                    child: DataTable(
                      columns: const [
                        DataColumn(label: Text('Activity', style: TextStyle(fontWeight: FontWeight.bold))),
                        DataColumn(label: Text('Time (s)', style: TextStyle(fontWeight: FontWeight.bold))),
                        DataColumn(label: Text('Marks', style: TextStyle(fontWeight: FontWeight.bold))),
                        DataColumn(label: Text('Status', style: TextStyle(fontWeight: FontWeight.bold))),
                      ],
                      rows: activities.entries.map((entry) {
                        final d = entry.value;
                        final activityName = _activityNameMapping[entry.key] ?? entry.key;

                        return DataRow(
                            color: MaterialStateProperty.resolveWith<Color?>((states) {
                              return d.completed ? Colors.green.shade50 : Colors.red.shade50;
                            }),
                            cells: [
                              DataCell(
                                SizedBox(
                                  width: 180, // Constrain the width of the cell
                                  child: Text(
                                    activityName,
                                    softWrap: true, // Allow text to wrap to the next line
                                  ),
                                ),
                              ),
                              DataCell(Text(d.timeSpentInSeconds.toString())),
                              DataCell(Text(d.marks.toString())),
                              DataCell(Text(d.completed ? 'Completed' : 'Not Completed')),
                            ]);
                      }).toList(),
                    ),
                  ),
                ),
              ),
            ),

            // --- NEW: Activity-Specific Suggestions Section ---
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  child: ListView(
                    padding: const EdgeInsets.all(16.0),
                    children: [
                      Text("Activity-Specific Suggestions",
                          style: TextStyle(color: Colors.grey.shade600, fontWeight: FontWeight.bold, fontSize: 18)),
                      const SizedBox(height: 12),
                      if (activitySuggestions.isEmpty)
                        const Text("No activity-specific suggestions available.")
                      else
                        ...activitySuggestions.map((s) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 6.0),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(Icons.analytics_outlined, color: Colors.orange.shade600, size: 20),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(s,
                                  style: const TextStyle(fontSize: 14),
                                ),
                              ),
                            ],
                          ),
                        )),
                    ],
                  ),
                ),
              ),
            ),

            // Prediction and Suggestions Section
            Expanded(
              flex: 3,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : (_error != null
                    ? _buildErrorWidget(_error!)
                    : _buildPredictionWidget(_prediction, _suggestions)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // --- UI Helper Widgets (reverted to English) ---

  Widget _buildErrorWidget(String error) {
    return Card(
      color: Colors.red.shade100,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 40),
            const SizedBox(height: 10),
            const Text("Failed to Load Prediction", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.red)),
            const SizedBox(height: 10),
            Text(error, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionWidget(String? prediction, List<String>? suggestions) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          Text("Prediction Level", style: TextStyle(color: Colors.grey.shade600, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(vertical: 12),
            decoration: BoxDecoration(
              color: Colors.blue.shade50,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              prediction ?? "Not available",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.blue.shade800),
            ),
          ),
          const Divider(height: 24),
          Text("Suggestions for Improvement", style: TextStyle(color: Colors.grey.shade600, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          if (suggestions == null || suggestions.isEmpty)
            const Text("No suggestions available.")
          else
            ...suggestions.map((s) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 6.0),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.check_circle_outline, color: Colors.green.shade600, size: 20),
                  const SizedBox(width: 10),
                  Expanded(child: Text(s, style: const TextStyle(fontSize: 15))),
                ],
              ),
            )),
        ],
      ),
    );
  }
}
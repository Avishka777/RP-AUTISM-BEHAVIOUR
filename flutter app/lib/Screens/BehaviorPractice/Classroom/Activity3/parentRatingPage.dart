import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/Activity3/act6.dart';

class ParentRatingPage extends StatefulWidget {
  final int spentTime;
  final int marks;

  const ParentRatingPage({
    super.key,
    required this.spentTime,
    required this.marks,
  });

  @override
  State<ParentRatingPage> createState() => _ParentRatingPageState();
}

class _ParentRatingPageState extends State<ParentRatingPage> {
  int _rating = 0;
  bool _isLoading = false;

  Future<void> _submitData() async {
    if (_rating == 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Please select a rating before submitting."),
          backgroundColor: Colors.orangeAccent,
        ),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final session = context.read<BehaviorPracticeProvider>();

      session.updateActivity(
        'activity3_5',
        completed: true,
        timeSpentInSeconds: widget.spentTime,
        marks: widget.marks,
        parentSatisfaction: _rating,
      );

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const ClassroomGoodBadActivity6()),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("An unexpected error occurred: $e"),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.lightBlue.shade50,
      appBar: AppBar(
        title: const Text("දෙමාපිය ප්‍රතිචාරය"),
        backgroundColor: Colors.lightBlue.shade200,
        elevation: 0,
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "ක්‍රියාකාරකම් කරන අතරතුර ඔබේ දරුවා පිලිබද ඔබගේ තෘප්තිය ඇතුලත් කරන්න?",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) {
                  return IconButton(
                    icon: Icon(
                      index < _rating
                          ? Icons.star_rounded
                          : Icons.star_border_rounded,
                      size: 52,
                      color: Colors.amber.shade600,
                    ),
                    onPressed: _isLoading
                        ? null
                        : () => setState(() => _rating = index + 1),
                  );
                }),
              ),
              const SizedBox(height: 50),
              _isLoading
                  ? const CircularProgressIndicator()
                  : ElevatedButton(
                onPressed: _submitData,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade600,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                      horizontal: 50, vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  textStyle: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                child: const Text("යවන්න"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
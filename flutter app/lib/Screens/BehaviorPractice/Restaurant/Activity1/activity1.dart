import 'dart:async';
import 'package:flutter/material.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity1/parentRatingPage.dart';
import 'package:ukussa_app/Utils/activityPreferences.dart';
import 'package:audioplayers/audioplayers.dart';

class PlaceRecognitionActivity extends StatefulWidget {
  const PlaceRecognitionActivity({super.key});

  @override
  State<PlaceRecognitionActivity> createState() =>
      _PlaceRecognitionActivityState();
}

class _PlaceRecognitionActivityState
    extends State<PlaceRecognitionActivity> {
  Timer? _timer;
  int _spentTime = 0;
  String? _selectedImage;
  bool _isAnswered = false;

  final String _questionAudio = 'act1.mp3';

  final AudioPlayer _audioPlayer = AudioPlayer();

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  void _startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted) {
        setState(() {
          _spentTime++;
        });
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Plays the audio clip containing the spoken question.
  void _playQuestionAudio() async {
    // Stop any currently playing audio to avoid overlap
    await _audioPlayer.stop();
    // Play the question audio from assets
    await _audioPlayer.play(AssetSource('audio/rs-aquiz/$_questionAudio'));
  }

  void _handleSelection(String selection) {
    if (_isAnswered) return;

    setState(() {
      _selectedImage = selection;
      _isAnswered = true;
      _timer?.cancel();
    });

    final isCorrect = selection == 'act1-1.png';
    final marks = isCorrect ? 100 : 0;

    // Navigate to the parent rating page
    Future.delayed(const Duration(seconds: 1), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ParentRatingPage(
              spentTime: _spentTime,
              marks: marks,
            ),
          ),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    // Get screen dimensions to make image sizes responsive
    final screenWidth = MediaQuery.of(context).size.width;
    // Calculate image size to be a large portion of the screen width
    final imageSize = screenWidth * 0.6;

    return Scaffold(
      backgroundColor: const Color(0xFF87CEEB),
      appBar: AppBar(
        title: const Text("අවන්හල හඳුනාගැනීම"),
        backgroundColor: Colors.blue.shade300,
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                const Icon(Icons.timer),
                const SizedBox(width: 8),
                Text(
                  "$_spentTime s",
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
          )
        ],
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [

                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // The Expanded widget ensures the text wraps nicely
                    // if it's too long for one line.
                    const Expanded(
                      child: Text(
                        "අපි කෑම කන තැන තොරන්න.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    // This is the new audio button
                    IconButton(
                      onPressed: _playQuestionAudio,
                      icon: const Icon(Icons.volume_up_rounded),
                      iconSize: 40,
                      color: Colors.blue.shade900,
                      tooltip: 'Play question audio', // For accessibility
                    ),
                  ],
                ),

                const SizedBox(height: 30),
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildImageOption('act1-1.png', imageSize),
                    const SizedBox(height: 25), // Adjusted space between images
                    _buildImageOption('act1-2.png', imageSize),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageOption(String imageName, double size) {
    final isSelected = _selectedImage == imageName;
    final isCorrect = imageName == 'act1-1.png';
    final showCorrect = _isAnswered && isCorrect;
    final showIncorrect = _isAnswered && isSelected && !isCorrect;

    return GestureDetector(
      onTap: () => _handleSelection(imageName),
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(15),
          border: Border.all(
            color: showCorrect
                ? Colors.green
                : showIncorrect
                ? Colors.red
                : Colors.blue.shade200,
            width: 4,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              spreadRadius: 2,
              blurRadius: 8,
              offset: const Offset(0, 4),
            )
          ],
        ),
        child: Column(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8.0),
              child: Image.asset(
                'assets/images/rs-quiz/$imageName', // Make sure you have these images in your assets folder
                width: size,
                height: size,
                fit: BoxFit.cover,
              ),
            ),
            if (_isAnswered)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Icon(
                  showCorrect ? Icons.check_circle : (showIncorrect ? Icons.cancel : null),
                  color: showCorrect ? Colors.green : Colors.red,
                  size: 30,
                ),
              )
          ],
        ),
      ),
    );
  }
}
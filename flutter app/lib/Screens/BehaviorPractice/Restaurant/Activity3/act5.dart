import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity3/parentRatingPage.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity3/act6.dart';
import 'package:lottie/lottie.dart';
import 'package:audioplayers/audioplayers.dart';

class RestaurantGoodBadActivity5 extends StatefulWidget {
  const RestaurantGoodBadActivity5({super.key});

  @override
  State<RestaurantGoodBadActivity5> createState() =>
      _RestaurantGoodBadActivity5State();
}

class _RestaurantGoodBadActivity5State
    extends State<RestaurantGoodBadActivity5> {
  Timer? _timer;
  int _spentTime = 0;
  String? _selectedImage;
  bool _isAnswered = false;

  bool _isCorrectImageOnTop = false;

  final String _correctAnswer = '3act5-1.png';
  final String _incorrectAnswer = '3act5-2.png';

  final String _questionAudio = '3act5.mp3';

  final AudioPlayer _audioPlayer = AudioPlayer();

  @override
  void initState() {
    super.initState();
    _isCorrectImageOnTop = Random().nextBool();
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

  /// Handles the logic when a user selects an image.
  void _handleSelection(String selection) {
    if (_isAnswered) return;

    setState(() {
      _isAnswered = true;
      _selectedImage = selection;
      _timer?.cancel();
    });

    // REMOVED: Provider update is moved to ParentRatingPage.

    // Wait for a second to show feedback, then navigate.
    Future.delayed(const Duration(seconds: 1), () {
      _navigateToNext();
    });
  }

  /// Navigates to the next screen based on whether the question was answered.
  void _navigateToNext() {
    if (!mounted) return;
    _timer?.cancel();

    if (_isAnswered) {
      // User answered, so go to the Parent Rating Page.
      final isCorrect = _selectedImage == _correctAnswer;
      final marks = isCorrect ? 100 : 0;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => ParentRatingPage(
            spentTime: _spentTime,
            marks: marks,
          ),
        ),
      );
    } else {
      // User skipped, so go directly to Activity 6.
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const RestaurantGoodBadActivity6(),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final imageSize = screenWidth * 0.7;

    return Scaffold(
      backgroundColor: const Color(0xFF87CEEB),
      appBar: AppBar(
        title: const Text("ක්‍රියාකාරකම 3.5"),
        backgroundColor: Colors.blue.shade300,
        elevation: 4,
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                const Icon(Icons.timer_outlined),
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
        child: Column(
          children: [
            Expanded(
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
                            "ඔයාගේ පුටුවේ කවුරුහරි වාඩිවුනොත් මොකද කරන්නේ?",
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
                    if (_isCorrectImageOnTop) ...[
                      _buildImageOption(_correctAnswer, imageSize),
                      const SizedBox(height: 25),
                      _buildImageOption(_incorrectAnswer, imageSize),
                    ] else ...[
                      _buildImageOption(_incorrectAnswer, imageSize),
                      const SizedBox(height: 25),
                      _buildImageOption(_correctAnswer, imageSize),
                    ],
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
              child: ElevatedButton(
                onPressed: _navigateToNext, // This will skip if unanswered.
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange.shade700,
                  foregroundColor: Colors.white,
                  minimumSize: const Size(double.infinity, 50),
                  textStyle: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text("ඊළඟ ක්‍රියාකාරකම"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageOption(String imageName, double size) {
    final isSelected = _selectedImage == imageName;
    final isCorrect = imageName == _correctAnswer;

    Color borderColor = Colors.blue.shade200;
    if (_isAnswered) {
      if (isCorrect) {
        borderColor = Colors.green;
      } else if (isSelected && !isCorrect) {
        borderColor = Colors.red;
      }
    }

    IconData? feedbackIcon;
    Color? iconColor;
    if (_isAnswered && isSelected) {
      feedbackIcon = isCorrect ? Icons.check_circle : Icons.cancel;
      iconColor = isCorrect ? Colors.green : Colors.red;
    }

    return GestureDetector(
      onTap: () => _handleSelection(imageName),
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(15),
          border: Border.all(color: borderColor, width: 4),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              spreadRadius: 2,
              blurRadius: 8,
              offset: const Offset(0, 4),
            )
          ],
        ),
        child: Stack(
          alignment: Alignment.center,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8.0),
              child: Image.asset(
                'assets/images/rs-quiz/$imageName',
                width: size,
                height: size,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) => Container(
                  width: size,
                  height: size,
                  color: Colors.grey[300],
                  child: Center(
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        'Image not found:\nassets/images/rs-quiz/$imageName',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ),
              ),
            ),
            if (feedbackIcon != null)
              Container(
                width: size,
                height: size,
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: Icon(
                  feedbackIcon,
                  color: iconColor,
                  size: 80,
                  shadows: const [Shadow(color: Colors.black54, blurRadius: 15.0)],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

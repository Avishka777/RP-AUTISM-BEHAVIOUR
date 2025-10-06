import 'dart:async';
import 'dart:math'; // Import the math library for Random()
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ukussa_app/Providers/behaviorPracticeProvider.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/Activity3/act5.dart';
import 'package:lottie/lottie.dart';
import 'package:audioplayers/audioplayers.dart';

class RestaurantGoodBadActivity4 extends StatefulWidget {
  const RestaurantGoodBadActivity4({super.key});

  @override
  State<RestaurantGoodBadActivity4> createState() =>
      _RestaurantGoodBadActivity4State();
}

class _RestaurantGoodBadActivity4State
    extends State<RestaurantGoodBadActivity4> {
  Timer? _timer;
  int _spentTime = 0;
  String? _selectedImage;
  bool _isAnswered = false;

  // A boolean to control the random order of images.
  bool _isCorrectImageOnTop = false;

  // The names of the image files in assets.
  final String _correctAnswer = '3act4-1.png';
  final String _incorrectAnswer = '3act4-2.png';

  final String _questionAudio = '3act4.mp3';

  final AudioPlayer _audioPlayer = AudioPlayer();

  @override
  void initState() {
    super.initState();
    // Randomly decide if the correct image should be on top.
    _isCorrectImageOnTop = Random().nextBool();
    _startTimer();
  }

  /// Starts a timer that updates the UI every second.
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
    // Cancel the timer when the widget is removed to prevent memory leaks.
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

  void _showFeedbackDialog(bool isCorrect) {
    showDialog(
      context: context,
      barrierDismissible: false, // Prevents closing by tapping outside
      builder: (BuildContext context) {
        // Automatically close the dialog after 2 seconds
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) {
            Navigator.of(context).pop();
          }
        });

        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Lottie animation widget
                Lottie.asset(
                  isCorrect
                      ? 'assets/animations/correct.json' // Make sure you have this file
                      : 'assets/animations/incorrect.json', // And this one
                  width: 150,
                  height: 150,
                  errorBuilder: (context, error, stackTrace) => const Icon(
                    Icons.error,
                    size: 150,
                    color: Colors.red,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    ).then((_) {
      // This block runs AFTER the dialog is closed.
      // Now, navigate to the next screen.
      _navigateToNext();
    });
  }

  /// Handles the logic when a user selects an image.
  void _handleSelection(String selection) {
    if (_isAnswered) return; // Prevents changing the answer after selection.

    setState(() {
      _isAnswered = true;
      _selectedImage = selection;
      _timer?.cancel(); // Stop the timer once an answer is given.
    });

    final isCorrect = selection == _correctAnswer;

    if (isCorrect) {
      _audioPlayer.play(AssetSource('audio/correctanswer.mp3'));
    } else {
      _audioPlayer.play(AssetSource('audio/wronganswer.mp3'));
    }

    final marks = isCorrect ? 100 : 0;

    // This uses `context.read` as it's a one-time action after the event.
    final session = context.read<BehaviorPracticeProvider>();
    session.updateActivity(
      'activity3_4',
      completed: true,
      timeSpentInSeconds: _spentTime,
      marks: marks,
    );

    // Wait for a second to show feedback, then navigate automatically.
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) {
        _showFeedbackDialog(isCorrect);
      }
    });
  }

  /// Navigates to the next screen. This is called either after answering
  /// or by pressing the "Next" button to skip.
  void _navigateToNext() {
    if (mounted) {
      // Stop the timer if it's still running (i.e., user skipped).
      _timer?.cancel();
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const RestaurantGoodBadActivity5(),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    // Make images responsive, taking up a large portion of the screen width.
    final imageSize = screenWidth * 0.7;

    return Scaffold(
      backgroundColor: const Color(0xFF87CEEB), // Sky blue background
      appBar: AppBar(
        title: const Text("ක්‍රියාකාරකම 3.4"),
        backgroundColor: Colors.blue.shade300,
        elevation: 4,
        actions: [
          // Timer display in the AppBar
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
                            "කෑම කනවිට හොද ලමයෙක් වගේ ඉන්නේ කොහොමද?",
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
                    // Conditionally render images in random order
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
            // Bottom navigation button for skipping
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
              child: ElevatedButton(
                onPressed: _navigateToNext, // Skips the question
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

  /// Builds a tappable image widget with feedback styling.
  Widget _buildImageOption(String imageName, double size) {
    final isSelected = _selectedImage == imageName;
    final isCorrect = imageName == _correctAnswer;

    // Determine border color based on selection state
    Color borderColor = Colors.blue.shade200; // Default color
    if (_isAnswered) {
      if (isCorrect) {
        borderColor = Colors.green; // Always show correct answer in green
      } else if (isSelected && !isCorrect) {
        borderColor = Colors.red; // Show red only for the incorrect selection
      }
    }

    // Determine which icon to show for feedback
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
          border: Border.all(
            color: borderColor,
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
        child: Stack(
          alignment: Alignment.center,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8.0),
              // IMPORTANT: Make sure your images are in this asset path.
              child: Image.asset(
                'assets/images/rs-quiz/$imageName',
                width: size,
                height: size,
                fit: BoxFit.cover,
                // Error builder in case the image fails to load
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
            // Show feedback icon overlay (check or cross)
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
                  shadows: const [
                    Shadow(color: Colors.black54, blurRadius: 15.0)
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

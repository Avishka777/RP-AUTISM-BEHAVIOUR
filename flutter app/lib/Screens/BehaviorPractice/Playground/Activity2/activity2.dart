import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/Activity2/parentRatingPage.dart';
import 'package:audioplayers/audioplayers.dart';

class ObjDetectionActivity extends StatefulWidget {
  const ObjDetectionActivity({super.key});

  @override
  State<ObjDetectionActivity> createState() =>
      _ObjDetectionActivityState();
}

class DraggableItem {
  final String assetPath;
  final bool isCorrect;
  final GlobalKey key;
  bool isVisible;

  DraggableItem({
    required this.assetPath,
    required this.isCorrect,
    required this.key,
    this.isVisible = true,
  });
}

class _ObjDetectionActivityState
    extends State<ObjDetectionActivity> {
  late List<DraggableItem> _items;
  final Set<String> _correctlyDroppedItems = {};

  // Timer state
  late Stopwatch _stopwatch;
  late Timer _timer;

  final String _questionAudio = 'act2.mp3';

  final AudioPlayer _audioPlayer = AudioPlayer();

  // List of correct image names
  final List<String> correctImages = [
    "pg-drag2", "pg-drag3", "pg-drag5"
    //, "pg-drag8"
  ];

  @override
  void initState() {
    super.initState();
    _items = List.generate(5, (index) {
      final imageName = "pg-drag${index + 1}";
      return DraggableItem(
        assetPath: "assets/images/$imageName.png",
        isCorrect: correctImages.contains(imageName),
        key: GlobalKey(),
        isVisible: true,
      );
    });
    _items.shuffle(); // Randomize position

    _stopwatch = Stopwatch()
      ..start();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {});
    });
  }

  @override
  void dispose() {
    _timer.cancel();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Plays the audio clip containing the spoken question.
  void _playQuestionAudio() async {
    // Stop any currently playing audio to avoid overlap
    await _audioPlayer.stop();
    // Play the question audio from assets
    await _audioPlayer.play(AssetSource('audio/pg-aquiz/$_questionAudio'));
  }

  void _onItemDropped(String assetPath, bool isCorrect) {
    if (isCorrect) {
      setState(() {
        _correctlyDroppedItems.add(assetPath);
        // Find the item and make it invisible
        final item = _items.firstWhere((it) => it.assetPath == assetPath);
        item.isVisible = false;
      });

      // Check for win condition
      if (_correctlyDroppedItems.length >=
          (correctImages.length * 0.90).ceil()) {
        _winGame();
      }
    } else {
      // Optional: Add feedback for incorrect drops, e.g., a shake animation or sound
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Try another one!"),
          backgroundColor: Colors.redAccent,
          duration: Duration(seconds: 1),
        ),
      );
    }
  }

  void _winGame() {
    _timer.cancel();
    _stopwatch.stop();

    final timeSpent = _stopwatch.elapsed.inSeconds;

    // Navigate to the rating page, passing the time spent
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) =>
            ParentRatingPage(
              spentTime: timeSpent,
              correctlyPlaced: _correctlyDroppedItems.length,
              totalCorrect: correctImages.length,
            ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final String elapsedTime =
        '${_stopwatch.elapsed.inMinutes.toString().padLeft(
        2, '0')}:${(_stopwatch.elapsed.inSeconds % 60).toString().padLeft(
        2, '0')}';

    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;

    // Calculate safe area for images (avoiding top UI and drop circle)
    final topUIPadding = screenHeight * 0.15;
    final dropCircleRadius = 60.0; // Half of the 120px circle
    final imageSize = 160.0;
    final halfImageSize = imageSize / 2;

    // Define safe boundaries for images
    final minX = 20 + halfImageSize;
    final maxX = screenWidth - 20 - halfImageSize;
    final minY = topUIPadding + 20 + halfImageSize;
    final maxY = screenHeight - 20 - halfImageSize;

    // Center of the drop circle
    final centerX = screenWidth / 2;
    final centerY = screenHeight / 2;

    // Minimum distance from drop circle center to image center
    final minDistanceFromCenter = dropCircleRadius + halfImageSize + 20;

    return Scaffold(
      backgroundColor: const Color(0xFF87CEEB),
      body: SafeArea(
        child: Stack(
          children: [
            // Top UI Elements
            Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Expanded(
                      child: Row(
                        children: [
                          const Expanded(
                            child: Text(
                              "සෙල්ලම් පිටියේ තියෙන දේවල් රවුම තුලට දාමු:",
                              style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white),
                            ),
                          ),
                          IconButton(
                            onPressed: _playQuestionAudio,
                            icon: const Icon(Icons.volume_up_rounded),
                            color: Colors.white,
                            iconSize: 30,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),

                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        "කාලය: $elapsedTime",
                        style: const TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // Central Drop Target
            Center(
              child: DragTarget<String>(
                builder: (context, candidateData, rejectedData) {
                  return Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.5),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.white,
                        width: 3,
                        style: candidateData.isNotEmpty
                            ? BorderStyle.solid
                            : BorderStyle.none,
                      ),
                    ),
                    child: Center(
                      child: _correctlyDroppedItems.isNotEmpty
                          ? Text(
                          "${_correctlyDroppedItems.length} / ${correctImages.length}",
                          style: const TextStyle(
                              fontSize: 22,
                              color: Colors.black54,
                              fontWeight: FontWeight.bold))
                          : null,
                    ),
                  );
                },
                onAccept: (data) {
                  // The data is the asset path. Find the corresponding item to check if it's correct.
                  final item = _items.firstWhere((it) => it.assetPath == data);
                  _onItemDropped(item.assetPath, item.isCorrect);
                },
              ),
            ),

            // Draggable Items arranged around the screen
            ..._items.asMap().entries.map((entry) {
              int index = entry.key;
              DraggableItem item = entry.value;

              // Calculate initial position in a circle
              double angle = (index / _items.length) * 2 * math.pi;
              double radius = math.min(screenWidth, screenHeight) * 0.35;

              double x = radius * math.cos(angle) + centerX;
              double y = radius * math.sin(angle) + centerY;

              // Ensure images stay within screen boundaries
              x = x.clamp(minX, maxX);
              y = y.clamp(minY, maxY);

              // Ensure images don't overlap with drop circle
              final distanceToCenter = math.sqrt(
                  math.pow(x - centerX, 2) + math.pow(y - centerY, 2)
              );

              if (distanceToCenter < minDistanceFromCenter) {
                // Adjust position to maintain minimum distance
                final angleToCenter = math.atan2(y - centerY, x - centerX);
                x = centerX + minDistanceFromCenter * math.cos(angleToCenter);
                y = centerY + minDistanceFromCenter * math.sin(angleToCenter);

                // Re-clamp after adjustment
                x = x.clamp(minX, maxX);
                y = y.clamp(minY, maxY);
              }

              return Positioned(
                left: x - halfImageSize,
                top: y - halfImageSize,
                child: Visibility(
                  visible: item.isVisible,
                  child: Draggable<String>(
                    data: item.assetPath,
                    feedback: Material(
                      color: Colors.transparent,
                      child: Image.asset(item.assetPath, width: 180, height: 180),
                    ),
                    childWhenDragging: Opacity(
                      opacity: 0.5,
                      child: Image.asset(item.assetPath, width: 160, height: 160),
                    ),
                    child: Image.asset(item.assetPath, width: 160, height: 160),
                  ),
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
}
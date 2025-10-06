import 'package:flutter/material.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/instructions.dart';

class CelebrationPage extends StatelessWidget {
  const CelebrationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTap: () {
          // Navigate back to instructions, removing all previous routes from the stack.
          Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (context) => const PlaygroundInstructionScreen()),
                (Route<dynamic> route) => false, // This predicate removes all routes.
          );
        },
        child: Container(
          // Ensure you have 'assets/celebrateBg.png' in your pubspec.yaml
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage("assets/celebrateBg.png"), // Corrected path if needed
              fit: BoxFit.cover,
            ),
          ),
          child: const Center(
            child: Padding(
              padding: EdgeInsets.symmetric(horizontal: 20.0),
              child: Text(
                "ඔබ විශිෂ්ටයි!\nඉදිරියට යාමට ඕනෑම තැනක තට්ටු කරන්න",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 32,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 10.0,
                      color: Colors.black,
                      offset: Offset(2.0, 2.0), // Adjusted shadow for better readability
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
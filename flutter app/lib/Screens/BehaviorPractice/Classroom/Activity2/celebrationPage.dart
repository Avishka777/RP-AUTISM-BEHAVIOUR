import 'package:flutter/material.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/instructions.dart';

class CelebrationPage extends StatelessWidget {
  const CelebrationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTap: () {
          // Navigate back to instructions, removing all previous routes
          Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (context) => const ClassroomInstructionScreen()),
                (Route<dynamic> route) => false,
          );
        },
        child: Container(
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage("assets/celebrateBg.png"),
              fit: BoxFit.cover,
            ),
          ),
          child: const Center(
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
                    offset: Offset(5.0, 5.0),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
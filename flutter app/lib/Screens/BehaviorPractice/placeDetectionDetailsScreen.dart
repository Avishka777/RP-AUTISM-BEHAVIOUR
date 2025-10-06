import 'dart:io';
import 'package:flutter/material.dart';
import 'package:ukussa_app/Models/objectDetectionModel.dart'; // Import the model
import 'package:ukussa_app/Screens/BehaviorPractice/emotionDetectionScreen.dart';

class PlaceDetectionDetailsScreen extends StatelessWidget {
  final File imageFile;
  final String selectedLocation;
  final List<ObjectDetectionResult> detections;

  const PlaceDetectionDetailsScreen({
    super.key,
    required this.imageFile,
    required this.selectedLocation,
    required this.detections,
  });

  @override
  Widget build(BuildContext context) {
    final uniqueDetections = detections.map((d) => d.className).toSet().toList();

    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text('උඩුගත කිරීමේ විස්තර', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black87)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('තෝරාගත් ස්ථානය', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey)),
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
                decoration: BoxDecoration(
                  color: const Color(0xFFE3E7FF),
                  borderRadius: BorderRadius.circular(12.0),
                  border: Border.all(color: const Color(0xFFB3BAF8), width: 1.5),
                ),
                child: Text(selectedLocation, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black87)),
              ),
              const SizedBox(height: 24),
              const Text('උඩුගත කළ පින්තූරය', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey)),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(12.0),
                child: Image.file(imageFile),
              ),
              const SizedBox(height: 24),
              const Text('හඳුනාගත් වස්තූන්', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8.0,
                runSpacing: 8.0,
                children: uniqueDetections.map((className) {
                  return Chip(
                    label: Text(className, style: const TextStyle(fontWeight: FontWeight.w600)),
                    backgroundColor: Colors.blue.shade50,
                    side: BorderSide(color: Colors.blue.shade200),
                  );
                }).toList(),
              ),
              const SizedBox(height: 40),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    // Navigate to the emotion detection screen
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => EmotionDetectionScreen(
                          selectedLocation: selectedLocation,
                          objectDetections: detections,
                        ),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
                  ),
                  child: const Text('ඉදිරියට යන්න', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
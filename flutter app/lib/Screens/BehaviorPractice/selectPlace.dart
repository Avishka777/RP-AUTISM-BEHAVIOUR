import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:ukussa_app/Models/objectDetectionModel.dart'; // Import the model
import 'package:ukussa_app/Screens/BehaviorPractice/placeDetectionDetailsScreen.dart'; // The details screen
//import 'package:ukussa_app/Utils/constValues.dart';
import 'package:ukussa_app/Utils/apiConfig.dart';
import 'package:material_symbols_icons/symbols.dart';

class SelectPlace extends StatefulWidget {
  const SelectPlace({super.key});

  @override
  State<SelectPlace> createState() => _SelectPlaceState();
}

class _SelectPlaceState extends State<SelectPlace> {
  int _selectedLocationIndex = 0;
  File? _selectedImage;
  bool _isLoading = false;

  final List<String> _locations = [
    'ආපන ශාලාව/රෙස්ටුවරන්ට්',
    'සෙල්ලම් පිටිය',
    'පන්තිකාමරය',
  ];
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickAndDetectImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

    if (image == null) return;

    setState(() {
      _selectedImage = File(image.path);
    });

    // If an image is selected, immediately try to upload and detect
    await _uploadAndDetect(File(image.path));
  }

  Future<void> _uploadAndDetect(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    try {

      final String detectObjectsUrl = '${ApiConfig.instance.apiUrl}/detect_objects/';

      var request = http.MultipartRequest('POST', Uri.parse(detectObjectsUrl));
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final decodedBody = json.decode(responseBody);
        final List<dynamic> detectedObjectsJson = decodedBody['detected_objects'];

        final List<ObjectDetectionResult> detections = detectedObjectsJson
            .map((jsonItem) => ObjectDetectionResult.fromJson(jsonItem))
            .toList();

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PlaceDetectionDetailsScreen(
                imageFile: imageFile,
                selectedLocation: _locations[_selectedLocationIndex],
                detections: detections,
              ),
            ),
          );
        }
      } else {
        throw Exception('Failed to detect objects. Status code: ${response.statusCode}');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error: ${e.toString()}'))
        );
      }
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('පින්තූරය උඩුගත කරන්න', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black87)),
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: true,
      ),
      body: Stack(
        children: [
          SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('ස්ථානය තෝරන්න', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
                  const SizedBox(height: 16),
                  _buildSelectableCard(text: _locations[0], index: 0, icon: Icons.restaurant),
                  _buildSelectableCard(text: _locations[1], index: 1, icon: Symbols.playground),
                  _buildSelectableCard(text: _locations[2], index: 2, icon: Icons.school_outlined),
                  const SizedBox(height: 30),
                  const Text('පින්තූරයක් උඩුගත කරන්න', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
                  const SizedBox(height: 16),
                  _buildImageUploader(),
                  const SizedBox(height: 40),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      // The button is only active if an image has been selected.
                      onPressed: _selectedImage == null ? null : () => _uploadAndDetect(_selectedImage!),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFE3E7FF),
                        foregroundColor: Colors.black87,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
                        elevation: 0,
                        disabledBackgroundColor: Colors.grey.shade200,
                      ),
                      child: const Text('ඉදිරියට', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildSelectableCard({required String text, required int index, IconData? icon}) {
    final bool isSelected = _selectedLocationIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _selectedLocationIndex = index),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6.0),
        padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFFE3E7FF) : Colors.white,
          borderRadius: BorderRadius.circular(12.0),
          border: Border.all(color: isSelected ? const Color(0xFFB3BAF8) : Colors.grey.shade300, width: 1.5),
        ),
        child: Row(
          children: [
            if (icon != null) ...[Icon(icon, color: Colors.black87), const SizedBox(width: 12)],
            Expanded(child: Text(text, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black87))),
          ],
        ),
      ),
    );
  }

  Widget _buildImageUploader() {
    return GestureDetector(
      onTap: _pickAndDetectImage,
      child: Container(
        height: 180,
        width: double.infinity,
        decoration: BoxDecoration(
          color: Colors.grey.shade50,
          borderRadius: BorderRadius.circular(12.0),
          border: Border.all(color: Colors.grey.shade300, width: 1.5),
        ),
        child: _selectedImage != null
            ? ClipRRect(borderRadius: BorderRadius.circular(11.0), child: Image.file(_selectedImage!, fit: BoxFit.cover))
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.add_photo_alternate_outlined, size: 50, color: Colors.grey.shade500),
            const SizedBox(height: 12),
            Text('පින්තූරයක් තෝරන්න', textAlign: TextAlign.center, style: TextStyle(fontSize: 15, color: Colors.grey.shade600, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}
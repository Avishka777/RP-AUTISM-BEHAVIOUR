import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';

import 'package:ukussa_app/Models/objectDetectionModel.dart';
import 'package:ukussa_app/Models/emotionDetectionModel.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Classroom/instructions.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Playground/instructions.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/instructions.dart';
import 'package:ukussa_app/Utils/apiConfig.dart';

class EmotionDetectionScreen extends StatefulWidget {
  final String selectedLocation;
  final List<ObjectDetectionResult> objectDetections;

  const EmotionDetectionScreen({
    super.key,
    required this.selectedLocation,
    required this.objectDetections,
  });

  @override
  State<EmotionDetectionScreen> createState() => _EmotionDetectionScreenState();
}

class _EmotionDetectionScreenState extends State<EmotionDetectionScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  int _selectedCameraIndex = 1; // 1 for front
  bool _isProcessing = false;
  bool _isLoading = false;
  Timer? _timer;
  int _captureTime = 3;
  File? _capturedImage;
  EmotionDetectionResult? _emotionResult;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    if (_cameras != null && _cameras!.isNotEmpty) {
      _selectCamera(_selectedCameraIndex);
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("No cameras found on this device."))
        );
      }
    }
  }

  void _selectCamera(int index) {
    if (_cameras == null || _cameras!.isEmpty) return;

    _controller = CameraController(
      _cameras![index],
      ResolutionPreset.medium,
    );

    _controller!.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error initializing camera: ${e.description}'))
        );
      }
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _timer?.cancel();
    super.dispose();
  }

  void _toggleCamera() {
    if (_cameras == null || _cameras!.length < 2) return;

    setState(() {
      _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras!.length;
      _selectCamera(_selectedCameraIndex);
    });
  }

  void _startCaptureCountdown() {
    setState(() {
      _isProcessing = true;
      _captureTime = 3;
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_captureTime > 1) {
        setState(() => _captureTime--);
      } else {
        _timer?.cancel();
        _captureImage();
      }
    });
  }

  Future<void> _captureImage() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      final XFile picture = await _controller!.takePicture();
      final File imageFile = File(picture.path);

      setState(() {
        _capturedImage = imageFile;
      });

      _detectEmotion(imageFile);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error capturing image: ${e.toString()}'))
        );
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<void> _detectEmotion(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    try {
      final String detectEmotionUrl = '${ApiConfig.instance.apiUrl}/detect_emotion/';

      var request = http.MultipartRequest('POST', Uri.parse(detectEmotionUrl));
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final decodedBody = json.decode(responseBody);
        final emotionResult = EmotionDetectionResult.fromJson(decodedBody);

        setState(() {
          _emotionResult = emotionResult;
        });

        _showEmotionResult(emotionResult);
      } else {
        throw Exception('Failed to detect emotion. Status: ${response.statusCode}');
      }

    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error: ${e.toString()}'))
        );
      }
    } finally {
      if(mounted) {
        setState(() {
          _isLoading = false;
          _isProcessing = false;
        });
      }
    }
  }

  void _showEmotionResult(EmotionDetectionResult emotionResult) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('දරුවාගේ හැඟීම: '),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              emotionResult.emotion ?? 'Unknown emotion',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Text(
              'Confidence: ${(emotionResult.confidence! * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 16),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _navigateToInstructionScreen();
            },
            child: const Text('Continue'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              setState(() {
                _capturedImage = null;
                _emotionResult = null;
              });
            },
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }

  void _navigateToInstructionScreen() {
    Widget nextScreen;
    switch (widget.selectedLocation) {
      case 'ආපන ශාලාව/රෙස්ටුවරන්ට්':
        nextScreen = const RestaurantInstructionScreen();
        break;
      case 'සෙල්ලම් පිටිය':
        nextScreen = const PlaygroundInstructionScreen();
        break;
      case 'පන්තිකාමරය':
        nextScreen = const ClassroomInstructionScreen();
        break;
      default:
        nextScreen = const Scaffold(
          body: Center(child: Text("Error: Unknown Location")),
        );
    }

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => nextScreen),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("හැඟීම් හඳුනාගැනීම"),
        actions: [
          if (!_isProcessing && _cameras != null && _cameras!.length > 1)
            IconButton(
              icon: const Icon(Icons.flip_camera_ios_outlined),
              onPressed: _toggleCamera,
            ),
        ],
      ),
      body: Stack(
        children: [
          if (_controller == null || !_controller!.value.isInitialized)
            const Center(child: CircularProgressIndicator())
          else if (_capturedImage != null)
            Image.file(_capturedImage!, fit: BoxFit.cover)
          else
            CameraPreview(_controller!),

          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: const Center(child: CircularProgressIndicator()),
            ),

          if (_isProcessing)
            Container(
              color: Colors.black.withOpacity(0.7),
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Capturing in',
                      style: TextStyle(fontSize: 24, color: Colors.white),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      '$_captureTime',
                      style: const TextStyle(fontSize: 48, color: Colors.white, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),

          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.all(25.0),
              child: _capturedImage != null
                  ? Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  FloatingActionButton(
                    onPressed: () {
                      setState(() {
                        _capturedImage = null;
                        _emotionResult = null;
                      });
                    },
                    child: const Icon(Icons.refresh),
                    heroTag: 'retry',
                  ),
                  FloatingActionButton(
                    onPressed: _navigateToInstructionScreen,
                    child: const Icon(Icons.check),
                    heroTag: 'continue',
                  ),
                ],
              )
                  : FloatingActionButton.large(
                onPressed: _isProcessing ? null : _startCaptureCountdown,
                child: _isProcessing
                    ? Text("$_captureTime", style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold))
                    : const Icon(Icons.camera_alt),
              ),
            ),
          )
        ],
      ),
    );
  }
}

// Update your EmotionDetectionModel to match the image response format
class EmotionDetectionResult {
  final String? emotion;
  final double? confidence;

  EmotionDetectionResult({this.emotion, this.confidence});

  factory EmotionDetectionResult.fromJson(Map<String, dynamic> json) {
    return EmotionDetectionResult(
      emotion: json['emotion'],
      confidence: json['confidence'] != null ? json['confidence'].toDouble() : 0.0,
    );
  }
}
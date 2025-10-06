import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:ukussa_app/Utils/apiConfig.dart';
import 'package:ukussa_app/Models/emotionDetectionModel.dart';
import 'package:ukussa_app/Screens/BehaviorPractice/Restaurant/reportScreen.dart';

class FinalEmotionDetectionScreen extends StatefulWidget {
  const FinalEmotionDetectionScreen({Key? key}) : super(key: key);

  @override
  State<FinalEmotionDetectionScreen> createState() =>
      _FinalEmotionDetectionScreenState();
}

class _FinalEmotionDetectionScreenState
    extends State<FinalEmotionDetectionScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  int _selectedCameraIndex = 1; // Default to front camera
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

  /// Finds available cameras and initializes the controller.
  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    if (_cameras != null && _cameras!.isNotEmpty) {
      // Try to find the front camera, otherwise default to the first one.
      int frontCameraIndex = _cameras!.indexWhere(
              (cam) => cam.lensDirection == CameraLensDirection.front
      );
      if(frontCameraIndex != -1) {
        _selectedCameraIndex = frontCameraIndex;
      } else {
        _selectedCameraIndex = 0;
      }
      // Select the determined camera
      await _selectCamera(_selectedCameraIndex);
    } else if (mounted) {
      _showErrorSnackBar("No cameras found on this device.");
    }
  }

  /// Disposes the old controller and initializes a new one for the selected camera.
  Future<void> _selectCamera(int index) async {
    if (_cameras == null || _cameras!.isEmpty) return;

    // Dispose the old controller before creating a new one to prevent resource leaks.
    await _controller?.dispose();

    final newController = CameraController(
      _cameras![index],
      ResolutionPreset.medium,
    );

    _controller = newController;

    try {
      await _controller!.initialize();
    } on CameraException catch (e) {
      _showErrorSnackBar('Camera error: ${e.description}');
    }

    if (mounted) {
      setState(() {}); // Update the UI to show the new camera preview
    }
  }

  /// Toggles between available cameras.
  Future<void> _toggleCamera() async {
    if (_cameras == null || _cameras!.length < 2) return;
    // Calculate the index of the next camera.
    _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras!.length;
    // Asynchronously select the new camera.
    await _selectCamera(_selectedCameraIndex);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _timer?.cancel();
    super.dispose();
  }

  /// Starts the 3-second countdown before capturing an image.
  void _startCaptureCountdown() {
    if (_controller == null || !_controller!.value.isInitialized || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      _captureTime = 3;
    });

    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (_captureTime > 1) {
        setState(() => _captureTime--);
      } else {
        _timer?.cancel();
        _captureImage();
      }
    });
  }

  /// Captures an image and sends it for emotion detection.
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
        _showErrorSnackBar('Error capturing image: ${e.toString()}');
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  /// Sends the captured image to the emotion detection API.
  Future<void> _detectEmotion(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    try {
      final uri = Uri.parse('${ApiConfig.instance.apiUrl}/detect_emotion/');

      final req = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final streamed = await req.send();
      if (streamed.statusCode != 200) {
        final errorBody = await streamed.stream.bytesToString();
        throw Exception('Server error: ${streamed.statusCode} - $errorBody');
      }

      final body = await streamed.stream.bytesToString();
      final decodedBody = json.decode(body);

      // Create EmotionDetectionResult from the image response format
      final emotionResult = EmotionDetectionResult(
        emotion: decodedBody['emotion'],
        confidence: decodedBody['confidence'] != null
            ? decodedBody['confidence'].toDouble()
            : 0.0,
      );

      setState(() {
        _emotionResult = emotionResult;
      });

      if (mounted) {
        await showDialog(
          context: context,
          barrierDismissible: false,
          builder: (_) => _EmotionResultDialog(
            emotion: emotionResult.emotion ?? 'Unknown',
            confidence: emotionResult.confidence ?? 0.0,
            onContinue: () {
              Navigator.of(context).pop();
              _navigateToReportScreen(emotionResult.emotion);
            },
            onRetry: () {
              Navigator.of(context).pop();
              setState(() {
                _capturedImage = null;
                _emotionResult = null;
                _isProcessing = false;
                _isLoading = false;
              });
            },
          ),
        );
      }
    } catch (e) {
      _showErrorSnackBar('An error occurred: $e');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  /// Navigates to the report screen with the detected emotion.
  void _navigateToReportScreen(String? detectedEmotion) {
    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ReportScreen(detectedEmotion: detectedEmotion),
        ),
      );
    }
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("හැඟීම් හඳුනාගැනීම"),
        centerTitle: true,
        actions: [
          // Only show the toggle button if not processing and there's more than one camera
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
                      style: const TextStyle(
                          fontSize: 48,
                          color: Colors.white,
                          fontWeight: FontWeight.bold
                      ),
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
                    onPressed: () => _navigateToReportScreen(_emotionResult?.emotion),
                    child: const Icon(Icons.check),
                    heroTag: 'continue',
                  ),
                ],
              )
                  : FloatingActionButton.large(
                onPressed: _isProcessing ? null : _startCaptureCountdown,
                child: _isProcessing
                    ? Text("$_captureTime",
                    style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold
                    ))
                    : const Icon(Icons.camera_alt),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// A dialog widget that shows the emotion detection result with options to retry or continue.
class _EmotionResultDialog extends StatelessWidget {
  final String emotion;
  final double confidence;
  final VoidCallback onContinue;
  final VoidCallback onRetry;

  const _EmotionResultDialog({
    required this.emotion,
    required this.confidence,
    required this.onContinue,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('දරුවාගේ හැඟීම:'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            emotion,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          Text(
            'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
            style: const TextStyle(fontSize: 16),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: onContinue,
          child: const Text('Continue'),
        ),
        TextButton(
          onPressed: onRetry,
          child: const Text('Retry'),
        ),
      ],
    );
  }
}

// Updated EmotionDetectionModel to match the image response format
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
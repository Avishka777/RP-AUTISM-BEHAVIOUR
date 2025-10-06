import 'dart:async';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Activity2/instruction2.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/celebratePage.dart';

class DrawingScreen extends StatefulWidget {
  @override
  _DrawingScreenState createState() => _DrawingScreenState();
}

class _DrawingScreenState extends State<DrawingScreen> {
  List<Offset?> points = [];
  double screenHeight = 0;
  double screenWidth = 0;
  int starCount = 0;

  int totalPoints = 0;
  String currentTime = "00:00";

  late Timer _timer;
  int seconds = 0;
  int minutes = 0;
  double erasedPercentage = 0;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  void _startTimer() {
    _timer = Timer.periodic(Duration(seconds: 1), (Timer timer) {
      setState(() {
        seconds++;
        if (seconds == 60) {
          seconds = 0;
          minutes++;
        }
        currentTime =
            "${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}";
      });
    });
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    screenHeight = MediaQuery.of(context).size.height;
    screenWidth = MediaQuery.of(context).size.width;

    if (points.isEmpty) {
      populateFullScreenDrawing();
    }

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }

        NavigationUtils.backNavigation(context, Instruction2());
      },
      child: SafeArea(
        child: Scaffold(
          body: Stack(
            children: [
              Positioned.fill(
                child: Image.asset(
                  'assets/bg2.png',
                  fit: BoxFit.cover,
                ),
              ),
              GestureDetector(
                onPanUpdate: (details) {
                  setState(() {
                    eraseAt(details.localPosition);
                  });
                  checkForFullErasure();
                },
                onPanEnd: (details) {
                  points.add(null);
                },
                child: CustomPaint(
                  painter: DrawingPainter(points),
                  child: Container(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void populateFullScreenDrawing() {
    double step = 5.0;
    for (double y = 0; y < screenHeight; y += step) {
      for (double x = 0; x < screenWidth; x += step) {
        points.add(Offset(x, y));
      }
    }
    totalPoints = points.length;
  }

  void eraseAt(Offset position) {
    for (int i = points.length - 1; i >= 0; i--) {
      if (points[i] != null &&
          (points[i]!.dx - position.dx).abs() < 20.0 &&
          (points[i]!.dy - position.dy).abs() < 20.0) {
        points.removeAt(i);
      }
    }
  }

  Future<void> checkForFullErasure() async {
    setState(() {
      erasedPercentage = ((totalPoints - points.length) / totalPoints) * 100;
    });

    print('Erased: ${erasedPercentage.toStringAsFixed(2)}%');

    if (erasedPercentage >= 98) {
      print((minutes * 60) + seconds);
      if (double.parse(((minutes * 60) + seconds).toString()) < 240) {
        starCount = 3;
      } else if (double.parse(((minutes * 60) + seconds).toString()) > 240 &&
          double.parse(((minutes * 60) + seconds).toString()) < 420) {
        starCount = 2;
      } else if (double.parse(((minutes * 60) + seconds).toString()) > 420) {
        starCount = 1;
      }
      final SharedPreferences prefs = await SharedPreferences.getInstance();

      prefs.setString('a2', starCount.toString());

      if (prefs.getString('done') != null &&
          int.parse(prefs.getString('done')!) < 2) {
        prefs.setString('done', '2');
      }

      NavigationUtils.frontNavigation(
          context,
          CelebratePage(
            text1: 'අපිරිසිදු දෑ පිරිසිදු කරමු.',
            text2: '02',
            starCount: starCount,
          ));
    }
  }

  void showErasureDialog5() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          contentPadding: EdgeInsets.all(0),
          content: Container(
            height: 250,
            width: 100,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/background.jpg'),
                fit: BoxFit.cover,
                colorFilter: ColorFilter.mode(
                    Colors.black.withOpacity(0.3), BlendMode.darken),
              ),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'ඔබ විසින් මූලික තිරය සම්පූර්ණයෙන්ම මකා දැමීමට කටයුතු කර ඇත!',
                  style: TextStyle(fontSize: 16),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        'ජනප්‍රිය කිරීම',
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                    SizedBox(width: 20),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        'නිවර්තනය',
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class DrawingPainter extends CustomPainter {
  final List<Offset?> points;
  DrawingPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Color.fromRGBO(128, 128, 128, 0.5)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 5.0;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, paint);
      } else if (points[i] != null && points[i + 1] == null) {
        canvas.drawCircle(points[i]!, 5.0, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}

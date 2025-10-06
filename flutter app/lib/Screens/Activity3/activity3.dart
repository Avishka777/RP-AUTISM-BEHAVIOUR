import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Activity3/instruction3.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/celebratePage.dart';

class Activity3 extends StatefulWidget {
  final Color pColor;
  const Activity3({super.key, required this.pColor});
  @override
  State<Activity3> createState() => _Activity3State();
}

class _Activity3State extends State<Activity3> {
  final List<List<Offset>> _strokes = [];

  late Path _balloonPath;
  DateTime? _startTime;
  int _paintCount = 0;
  bool _isCompleted = false;
  int starCount = 0;

  @override
  void initState() {
    super.initState();
    _startTime = DateTime.now();
    _balloonPath = Path();
  }

  void _updateBalloonPath(Size size) {
    _balloonPath = Path();

    final bodyRect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height * 0.5),
      width: size.width * 0.75,
      height: size.height * 0.75,
    );
    _balloonPath.addOval(bodyRect);
  }

  double get _completionPercent => (_paintCount / 4500).clamp(0.0, 1.0);

  int get _spentTime {
    if (_startTime == null) return 0;
    final d = DateTime.now().difference(_startTime!);
    return d.inSeconds;
  }

  Future<void> _checkCompletion() async {
    if (!_isCompleted && _completionPercent >= 0.8) {
      _isCompleted = true;
      print("object");
      if (_spentTime < 240) {
        starCount = 3;
      } else if (_spentTime > 240 && _spentTime < 420) {
        starCount = 2;
      } else if (_spentTime > 420) {
        starCount = 1;
      }
      final SharedPreferences prefs = await SharedPreferences.getInstance();

      prefs.setString('a3', starCount.toString());

      if (prefs.getString('done') != null &&
          int.parse(prefs.getString('done')!) < 3) {
        prefs.setString('done', '3');
      }

      NavigationUtils.frontNavigation(
          context,
          CelebratePage(
            text1: 'බැලූනය පාට කරමු.',
            text2: '03',
            starCount: starCount,
          ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }

        NavigationUtils.backNavigation(context, Instruction3());
      },
      child: SafeArea(
        child: Scaffold(
          backgroundColor: AppColors.blue2,
          body: SafeArea(
            child: Center(
              child: GestureDetector(
                onPanStart: (details) {
                  RenderBox box = context.findRenderObject() as RenderBox;
                  Offset pos = box.globalToLocal(details.globalPosition);
                  if (_balloonPath.contains(pos)) {
                    setState(() {
                      _strokes.add([pos]);
                      _paintCount++;
                      _checkCompletion();
                    });
                  }
                },
                onPanUpdate: (details) {
                  RenderBox box = context.findRenderObject() as RenderBox;
                  Offset pos = box.globalToLocal(details.globalPosition);
                  if (_balloonPath.contains(pos)) {
                    setState(() {
                      if (_strokes.isNotEmpty) {
                        _strokes.last.add(pos);
                        _paintCount++;
                        _checkCompletion();
                      }
                    });
                  }
                },
                onPanEnd: (details) {
                  if (_strokes.isNotEmpty && _strokes.last.length < 2) {
                    _strokes.removeLast();
                  }
                },
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    _updateBalloonPath(constraints.biggest);
                    return CustomPaint(
                      painter: OvalBalloonPainter(
                          _strokes, widget.pColor, _balloonPath),
                      child: Container(),
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class OvalBalloonPainter extends CustomPainter {
  final List<List<Offset>> strokes;
  final Color paintColor;
  final Path balloonPath;
  OvalBalloonPainter(this.strokes, this.paintColor, this.balloonPath);

  @override
  void paint(Canvas canvas, Size size) {
    final tiePaint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.fill;
    final tiePath = Path();

    final tieBaseWidth = size.width * 0.1;
    final tieHeight = size.height * 0.1;
    final tieTop = Offset(size.width / 2, balloonPath.getBounds().bottom - 20);

    final tieLeft = Offset(tieTop.dx - tieBaseWidth / 2, tieTop.dy + tieHeight);
    final tieRight =
        Offset(tieTop.dx + tieBaseWidth / 2, tieTop.dy + tieHeight);

    tiePath.moveTo(tieTop.dx, tieTop.dy);
    tiePath.lineTo(tieLeft.dx, tieLeft.dy);
    tiePath.lineTo(tieRight.dx, tieRight.dy);
    tiePath.close();

    canvas.drawPath(tiePath, tiePaint);

    final fillPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    canvas.drawPath(balloonPath, fillPaint);

    final outlinePaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;
    canvas.drawPath(balloonPath, outlinePaint);

    canvas.save();
    canvas.clipPath(balloonPath);

    final strokePaint = Paint()
      ..color = paintColor
      ..strokeWidth = 13
      ..strokeCap = StrokeCap.round;
    for (final stroke in strokes) {
      for (int i = 0; i < stroke.length - 1; i++) {
        canvas.drawLine(stroke[i], stroke[i + 1], strokePaint);
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

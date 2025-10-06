import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Activity4/activity4.dart';
import 'package:ukussa_app/Utils/appColors.dart';

import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/celebratePage.dart';
import 'package:ukussa_app/main.dart';

class Activity4x extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: PopScope(
        canPop: false,
        onPopInvokedWithResult: (didPop, result) async {
          if (didPop) {
            return;
          }

          NavigationUtils.backNavigation(context, Activity4());
        },
        child: SafeArea(
          child: Scaffold(
            backgroundColor: AppColors.blue2,
            body: Center(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: List.generate(
                  4,
                  (index) => Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8.0),
                    child: SwipeFillBox(index: index),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SwipeFillBox extends StatefulWidget {
  final int index;
  SwipeFillBox({required this.index});

  @override
  State<SwipeFillBox> createState() => _SwipeFillBoxState();
}

class _SwipeFillBoxState extends State<SwipeFillBox>
    with SingleTickerProviderStateMixin {
  double fillPercent = 0.0;
  double boxHeight = 0.0;
  bool canSwipe = false;
  bool isAnimatingToFull = false;
  late AnimationController _controller;
  late Animation<double> _animation;
  static int filledContainers = 0;
  static int totalContainers = 4;

  @override
  void initState() {
    super.initState();
    _SwipeFillBoxState.filledContainers = 0;

    _controller =
        AnimationController(vsync: this, duration: Duration(milliseconds: 350));

    _animation = Tween<double>(begin: fillPercent, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    )..addListener(() {
        setState(() {
          fillPercent = _animation.value;
        });
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onPanStart(DragStartDetails details) {
    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset localPosition = box.globalToLocal(details.globalPosition);

    if (localPosition.dy >= 0 && localPosition.dy <= 40) {
      setState(() {
        canSwipe = true;
        isAnimatingToFull = false;
        fillPercent = 0.0;
      });
    } else {
      setState(() {
        canSwipe = false;
      });
    }
  }

  void _onPanUpdate(DragUpdateDetails details) {
    if (!canSwipe || boxHeight == 0.0 || isAnimatingToFull) return;

    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset localPosition = box.globalToLocal(details.globalPosition);

    final double contentPadding = 36;
    final double usableHeight = boxHeight - contentPadding;

    if (localPosition.dx >= 0 &&
        localPosition.dx <= box.size.width &&
        localPosition.dy >= 0 &&
        localPosition.dy <= boxHeight) {
      double calculatedFill = (localPosition.dy) / usableHeight;

      setState(() {
        fillPercent = calculatedFill.clamp(0.0, 1.0);
      });

      if (fillPercent >= 0.65 && !isAnimatingToFull) {
        isAnimatingToFull = true;
        _animation = Tween<double>(begin: fillPercent, end: 1.0).animate(
          CurvedAnimation(parent: _controller, curve: Curves.easeOut),
        );
        _controller.forward(from: 0);
      }
    } else {
      if (!isAnimatingToFull) {
        setState(() {
          fillPercent = 0.0;
          canSwipe = false;
        });
      }
    }
  }

  String _elapsedTime = "0.0";
  int starCount = 0;

  void _stopStopwatch() {
    setState(() {
      MyApp.globalStopwatch.stop();
      _elapsedTime = MyApp.globalStopwatch.elapsed.inSeconds.toString();
    });
  }

  void _resetStopwatch() {
    setState(() {
      MyApp.globalStopwatch.reset();
      _elapsedTime = MyApp.globalStopwatch.elapsed.inSeconds.toString();
    });
  }

  Future<void> _onPanEnd(DragEndDetails details) async {
    if (!isAnimatingToFull) {
      setState(() {
        canSwipe = false;
        fillPercent = 0.0;
      });
    } else {
      filledContainers++;
      if (filledContainers == totalContainers) {
        _stopStopwatch();
        print(_elapsedTime);
        if (double.parse(_elapsedTime) < 240) {
          starCount = 3;
        } else if (double.parse(_elapsedTime) > 240 &&
            double.parse(_elapsedTime) < 420) {
          starCount = 2;
        } else if (double.parse(_elapsedTime) > 420) {
          starCount = 1;
        }
        final SharedPreferences prefs = await SharedPreferences.getInstance();

        prefs.setString('a4', starCount.toString());
        print(_elapsedTime);
        if (prefs.getString('done') != null &&
            int.parse(prefs.getString('done')!) < 4) {
          prefs.setString('done', '4');
        }

        _resetStopwatch();
        NavigationUtils.frontNavigation(
            context,
            CelebratePage(
              text1: 'කඩ ඉරි මත අදිමු.',
              text2: '04',
              starCount: starCount,
            ));

        filledContainers = 0;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      boxHeight = constraints.maxHeight;

      return GestureDetector(
        onPanStart: _onPanStart,
        onPanUpdate: _onPanUpdate,
        onPanEnd: _onPanEnd,
        child: Container(
          width: 70,
          height: MediaQuery.of(context).size.height * 0.7,
          child: Stack(
            children: [
              Container(
                width: double.infinity,
                height: double.infinity,
                decoration: BoxDecoration(
                  color: Color(0xFFFFFFE0),
                  borderRadius: BorderRadius.circular(22),
                  border: Border.all(
                    color: Colors.blueAccent,
                    width: 2,
                  ),
                ),
              ),
              Positioned(
                top: 0,
                left: 0,
                right: 0,
                child: AnimatedContainer(
                  duration: Duration(milliseconds: 80),
                  width: double.infinity,
                  height:
                      MediaQuery.of(context).size.height * 0.7 * fillPercent,
                  decoration: BoxDecoration(
                    color: Colors.yellow,
                    borderRadius: BorderRadius.vertical(
                      top: Radius.circular(22),
                      bottom: Radius.circular(fillPercent == 1.0 ? 22 : 0),
                    ),
                  ),
                ),
              ),
              Container(
                width: double.infinity,
                height: double.infinity,
                padding: EdgeInsets.symmetric(vertical: 18),
                child: Column(
                  children: [
                    Container(
                      width: 16,
                      height: 16,
                      decoration: BoxDecoration(
                        color: Colors.red,
                        shape: BoxShape.circle,
                      ),
                    ),
                    Expanded(
                      child: Center(
                        child: CustomPaint(
                          size: Size(2, double.infinity),
                          painter: VerticalDottedLinePainter(),
                        ),
                      ),
                    ),
                    Icon(
                      Icons.arrow_downward,
                      size: 32,
                      color: Colors.black,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
    });
  }
}

class VerticalDottedLinePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    double dashHeight = 8, dashSpace = 8, startY = 0;
    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 3;

    while (startY < size.height) {
      canvas.drawLine(
        Offset(size.width / 2, startY),
        Offset(size.width / 2, startY + dashHeight),
        paint,
      );
      startY += dashHeight + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

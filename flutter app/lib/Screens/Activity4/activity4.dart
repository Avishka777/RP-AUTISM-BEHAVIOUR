import 'package:flutter/material.dart';
import 'package:ukussa_app/Screens/Activity4/activity4x.dart';
import 'package:ukussa_app/Screens/Activity4/instruction4.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/main.dart';

class Activity4 extends StatefulWidget {
  @override
  State<Activity4> createState() => _Activity4State();
}

class _Activity4State extends State<Activity4> {
  void _stopStopwatch() {
    setState(() {
      MyApp.globalStopwatch.stop();
    });
  }

  void _resetStopwatch() {
    setState(() {
      MyApp.globalStopwatch.reset();
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: PopScope(
        canPop: false,
        onPopInvokedWithResult: (didPop, result) async {
          if (didPop) {
            return;
          }
          _stopStopwatch();
          _resetStopwatch();
          NavigationUtils.backNavigation(context, Instruction4());
        },
        child: SafeArea(
          child: Scaffold(
            backgroundColor: AppColors.blue2,
            body: Center(
              child: Container(
                constraints: BoxConstraints(
                  maxWidth: MediaQuery.of(context).size.width,
                ),
                child: ListView.separated(
                  shrinkWrap: true,
                  physics: NeverScrollableScrollPhysics(),
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: 7,
                  separatorBuilder: (_, __) => SizedBox(height: 32),
                  itemBuilder: (context, index) => SwipeFillBox(index: index),
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
  double boxWidth = 0.0;
  bool canSwipe = false;
  bool isAnimatingToFull = false;
  late AnimationController _controller;
  late Animation<double> _animation;
  static int filledContainers = 0;

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

    if (localPosition.dx >= 0 && localPosition.dx <= 40) {
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
    if (!canSwipe || boxWidth == 0.0 || isAnimatingToFull) return;

    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset localPosition = box.globalToLocal(details.globalPosition);

    if (localPosition.dx >= 0 &&
        localPosition.dx <= boxWidth &&
        localPosition.dy >= 0 &&
        localPosition.dy <= box.size.height) {
      setState(() {
        fillPercent = (localPosition.dx / boxWidth).clamp(0.0, 1.0);
      });

      if (fillPercent >= 0.98 && !isAnimatingToFull) {
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

  void _onPanEnd(DragEndDetails details) {
    if (!isAnimatingToFull) {
      setState(() {
        canSwipe = false;
        fillPercent = 0.0;
      });
    } else {
      filledContainers++;
      if (filledContainers == 7) {
        NavigationUtils.navBarNavigation(context, Activity4x());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      boxWidth = constraints.maxWidth;
      return GestureDetector(
        onPanStart: _onPanStart,
        onPanUpdate: _onPanUpdate,
        onPanEnd: _onPanEnd,
        child: Stack(
          children: [
            Container(
              width: double.infinity,
              height: 70,
              decoration: BoxDecoration(
                color: Color(0xFFFFFFE0),
                borderRadius: BorderRadius.circular(22),
                border: Border.all(
                  color: Colors.blueAccent,
                  width: 2,
                ),
              ),
            ),
            AnimatedContainer(
              duration: Duration(milliseconds: 80),
              width: boxWidth * fillPercent,
              height: 70,
              decoration: BoxDecoration(
                color: Colors.yellow,
                borderRadius: BorderRadius.horizontal(
                  left: Radius.circular(22),
                  right: Radius.circular(fillPercent == 1.0 ? 22 : 0),
                ),
              ),
            ),
            Container(
              width: double.infinity,
              height: 70,
              padding: EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              child: Row(
                children: [
                  Container(
                    width: 16,
                    height: 16,
                    decoration: BoxDecoration(
                      color: Colors.red,
                      shape: BoxShape.circle,
                    ),
                  ),
                  SizedBox(width: 14),
                  Expanded(
                    child: CustomPaint(
                      size: Size(double.infinity, 2),
                      painter: DottedLinePainter(),
                    ),
                  ),
                  SizedBox(width: 14),
                  Icon(
                    Icons.arrow_forward,
                    size: 32,
                    color: Colors.black,
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    });
  }
}

class DottedLinePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    double dashWidth = 8, dashSpace = 8, startX = 0;
    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 3;
    while (startX < size.width) {
      canvas.drawLine(
        Offset(startX, size.height / 2),
        Offset(startX + dashWidth, size.height / 2),
        paint,
      );
      startX += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

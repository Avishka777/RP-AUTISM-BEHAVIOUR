import 'dart:async';
import 'dart:math';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Activity1/instruction1.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/celebratePage.dart';

class BalloonGame extends StatefulWidget {
  @override
  _BalloonGameState createState() => _BalloonGameState();
}

class _BalloonGameState extends State<BalloonGame> {
  List<Balloon> balloons = [];
  int poppedCount = 0;
  String currentTime = "00:00";
  final Random random = Random();
  late Timer _timer;
  int seconds = 0;
  int minutes = 0;
  int starCount = 0;
  late AudioPlayer _audioPlayer;
  bool isPlayingAudio = false;

  @override
  void initState() {
    super.initState();

    _generateBalloons();
    _startTimer();
  }

  void _playBackgroundMusic() async {
    await _audioPlayer.play(AssetSource('audio/balloonpop.mp3'));
    await Future.delayed(Duration(seconds: 1));
    _audioPlayer.stop();
    setState(() {
      isPlayingAudio = false;
    });
  }

  void _generateBalloons() {
    for (int i = 0; i < 4; i++) {
      balloons.add(Balloon(
        imagePath: _getRandomBalloonImage(),
        position: Offset(random.nextDouble() * 300, random.nextDouble() * 500),
        isClicked: false,
      ));
    }
  }

  String _getRandomBalloonImage() {
    List<String> imagePaths = [
      'assets/balloon1.png',
      'assets/balloon2.png',
      'assets/balloon3.png',
      'assets/balloon4.png',
    ];
    return imagePaths[random.nextInt(imagePaths.length)];
  }

  void _popBalloon(Balloon clickedBalloon) {
    setState(() async {
      if (poppedCount < 16) {
        clickedBalloon.position =
            Offset(random.nextDouble() * 300, random.nextDouble() * 500);
      }
      poppedCount++;
      _audioPlayer = AudioPlayer();
      _playBackgroundMusic();

      if (poppedCount >= 16) {
        balloons.forEach((b) {
          b.isClicked = false;
        });
      }

      if (poppedCount >= 17) {
        if (poppedCount == 20) {
          print((minutes * 60) + seconds);
          if (double.parse(((minutes * 60) + seconds).toString()) < 240) {
            starCount = 3;
          } else if (double.parse(((minutes * 60) + seconds).toString()) >
              240 &&
              double.parse(((minutes * 60) + seconds).toString()) < 420) {
            starCount = 2;
          } else if (double.parse(((minutes * 60) + seconds).toString()) >
              420) {
            starCount = 1;
          }
          final SharedPreferences prefs = await SharedPreferences.getInstance();

          prefs.setString('a1', starCount.toString());

          if (prefs.getString('done') != null &&
              int.parse(prefs.getString('done')!) < 1) {
            prefs.setString('done', '1');
          }

          NavigationUtils.frontNavigation(
              context,
              CelebratePage(
                text1: 'බැලූන් පොප් කිරීමට තට්ටු කරන්න',
                text2: '01',
                starCount: starCount,
              ));
        }

        clickedBalloon.isClicked = true;

        balloons = balloons.where((balloon) => !balloon.isClicked).toList();
      }
    });
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
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }

        NavigationUtils.backNavigation(context, Instruction1());
      },
      child: SafeArea(
        child: Scaffold(
          body: Container(
            width: double.infinity,
            height: double.infinity,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/background.png'),
                fit: BoxFit.fill,
              ),
            ),
            child: Stack(
              children: [
                ...balloons.map((balloon) {
                  return Positioned(
                    left: balloon.position.dx,
                    top: balloon.position.dy,
                    child: GestureDetector(
                      onTap: () => _popBalloon(balloon),
                      child: BalloonWidget(balloon: balloon),
                    ),
                  );
                }).toList(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class Balloon {
  final String imagePath;
  Offset position;
  bool isClicked;

  Balloon({
    required this.imagePath,
    required this.position,
    this.isClicked = false,
  });
}

class BalloonWidget extends StatelessWidget {
  final Balloon balloon;

  BalloonWidget({required this.balloon});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width / 2.2,
      height: MediaQuery.of(context).size.height / 5,
      decoration: BoxDecoration(
        image: DecorationImage(
          image: AssetImage(balloon.imagePath),
          fit: BoxFit.fill,
        ),
        borderRadius: BorderRadius.circular(50),
      ),
    );
  }
}
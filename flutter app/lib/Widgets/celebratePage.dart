import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Widgets/label.dart';

class CelebratePage extends StatefulWidget {
  final String text1;
  final String text2;
  final int starCount;
  const CelebratePage(
      {super.key,
      required this.text1,
      required this.text2,
      required this.starCount});

  @override
  State<CelebratePage> createState() => _CelebratePageState();
}

class _CelebratePageState extends State<CelebratePage> {
  late YoutubePlayerController _controller;
  late AudioPlayer _audioPlayer;
  bool isPlayingAudio = false;

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    _playBackgroundMusic();
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
  }

  @override
  void dispose() {
    super.dispose();
    _audioPlayer.stop();
    _controller.dispose();
    _audioPlayer.dispose();
  }

  void _playBackgroundMusic() async {
    await _audioPlayer.play(AssetSource('audio/music.mp3'));
    await Future.delayed(Duration(seconds: 5));
    _audioPlayer.stop();
    setState(() {
      isPlayingAudio = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }
        NavigationUtils.backNavigation(context, MapDashboard());
      },
      child: SafeArea(
        child: Scaffold(
          body: Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/celebrateBg.png'),
                fit: BoxFit.fill,
              ),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 25.0),
                  child: Container(
                    color: AppColors.white1,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Label(
                                hintText: widget.text1,
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font18,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal,
                              ),
                              Label(
                                hintText: widget.text2,
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font18,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal,
                              ),
                            ],
                          ),
                          SizedBox(height: 5.0),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: List.generate(3, (index) {
                              return Icon(
                                Icons.star_rate,
                                size: 35.0,
                                color: widget.starCount > index
                                    ? AppColors.gold2
                                    : AppColors.gray1,
                              );
                            }),
                          )
                        ],
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom: 60.0),
                  child: Container(
                    child: Column(
                      children: [
                        GestureDetector(
                          onTap: () {
                            NavigationUtils.backNavigation(
                                context, MapDashboard());
                          },
                          child: Container(
                            width: MediaQuery.of(context).size.width * 0.8,
                            height: 50,
                            decoration: BoxDecoration(
                              image: DecorationImage(
                                image: AssetImage('assets/nxtBtn.png'),
                                fit: BoxFit.fill,
                              ),
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 40.0),
                          child: GestureDetector(
                            onTap: () {
                              _controller = YoutubePlayerController(
                                initialVideoId: 'NQYR55EiKvs',
                                flags: YoutubePlayerFlags(
                                  autoPlay: true,
                                  mute: false,
                                ),
                              );

                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => VideoPlayerScreen(
                                      controller: _controller),
                                ),
                              );
                            },
                            child: Container(
                              width: MediaQuery.of(context).size.width * 0.8,
                              height: 45,
                              decoration: BoxDecoration(
                                image: DecorationImage(
                                  image: AssetImage('assets/freeBtn.png'),
                                  fit: BoxFit.fill,
                                ),
                                borderRadius: BorderRadius.circular(20),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class VideoPlayerScreen extends StatefulWidget {
  final YoutubePlayerController controller;

  VideoPlayerScreen({required this.controller});

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight]);
  }

  @override
  void dispose() {
    super.dispose();
    widget.controller.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            YoutubePlayerBuilder(
              player: YoutubePlayer(
                controller: widget.controller,
                showVideoProgressIndicator: true,
                progressIndicatorColor: Colors.amber,
                progressColors: const ProgressBarColors(
                  playedColor: Colors.amber,
                  handleColor: Colors.amberAccent,
                ),
                onReady: () {
                  widget.controller.addListener(listener);
                },
              ),
              builder: (context, player) {
                return Column(
                  children: [
                    Expanded(child: player),
                  ],
                );
              },
            ),
            Positioned(
              top: 20.0,
              left: 20.0,
              child: GestureDetector(
                onTap: () {
                  Navigator.pop(context);
                  NavigationUtils.backNavigation(context, MapDashboard());
                },
                child: Container(
                  width: 50.0,
                  height: 50.0,
                  decoration: BoxDecoration(
                    color: AppColors.gold2,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.arrow_back,
                    size: 40.0,
                    color: AppColors.white1,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void listener() {
    if (widget.controller.value.isPlaying) {
      print("Video is playing");
    }
  }
}

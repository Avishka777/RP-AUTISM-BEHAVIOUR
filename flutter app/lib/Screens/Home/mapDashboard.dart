import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Providers/provider.dart';
import 'package:ukussa_app/Screens/Activity1/instruction1.dart';
// import 'package:ukussa_app/Screens/Activity10/instruction10.dart';
// import 'package:ukussa_app/Screens/Activity11/instruction11.dart';
// import 'package:ukussa_app/Screens/Activity12/instruction12.dart';
// import 'package:ukussa_app/Screens/Activity15/instruction15.dart';
// import 'package:ukussa_app/Screens/Activity16/instruction16.dart';
// import 'package:ukussa_app/Screens/Activity17/instruction17.dart';
// import 'package:ukussa_app/Screens/Activity18/instruction18.dart';
// import 'package:ukussa_app/Screens/Activity19/instruction19.dart';
import 'package:ukussa_app/Screens/Activity2/instruction2.dart';
// import 'package:ukussa_app/Screens/Activity29/instruction29.dart';
import 'package:ukussa_app/Screens/Activity3/instruction3.dart';
import 'package:ukussa_app/Screens/Activity4/instruction4.dart';
// import 'package:ukussa_app/Screens/Activity5/instruction5.dart';
// import 'package:ukussa_app/Screens/Activity6/instruction6.dart';
// import 'package:ukussa_app/Screens/Activity7/instruction7.dart';
// import 'package:ukussa_app/Screens/Activity8/instruction8.dart';
// import 'package:ukussa_app/Screens/Activity9/instruction9.dart';
import 'package:ukussa_app/Screens/Home/dashboard.dart';
import 'package:ukussa_app/Screens/Home/ratingScreen.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/constValues.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/levelIcon.dart';
import 'package:ukussa_app/Widgets/snackBar.dart';

class MapDashboard extends StatefulWidget {
  const MapDashboard({super.key});

  @override
  State<MapDashboard> createState() => _MapDashboardState();
}

class _MapDashboardState extends State<MapDashboard> {
  String? a1 = null;

  String? a2 = null;

  String? a3 = null;

  String? a4 = null;

  String? a5 = null;
  String? a6 = null;
  String? a7 = null;
  String? a8 = null;
  String? a9 = null;
  String? a10 = null;
  String? a11 = null;
  String? a12 = null;
  String? a13 = null;
  String? a14 = null;
  String? a15 = null;
  String? a16 = null;
  String? a17 = null;
  String? a18 = null;
  String? a19 = null;
  String? a20 = null;
  String? a21 = null;
  String? a22 = null;
  String? a23 = null;
  String? a24 = null;
  String? a25 = null;
  String? a26 = null;
  String? a27 = null;
  String? a28 = null;
  String? a29 = null;
  String? a30 = null;
  String? a31 = null;

  int? done = null;

  late AudioPlayer _audioPlayer;
  bool isPlayingAudio = false;

  void _playBackgroundMusic() async {
    await _audioPlayer.play(AssetSource('audio/music.mp3'));
    await Future.delayed(Duration(seconds: 5));
    _audioPlayer.stop();
    setState(() {
      isPlayingAudio = false;
    });
  }

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    _playBackgroundMusic();
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    initProcess();
  }

  Future<void> initProcess() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    if (prefs.getString('done') == null) {
      prefs.setString('done', '-2');
    }
    final myModel = Provider.of<MyModel>(context, listen: false);

    a1 = prefs.getString('a1') ?? '0';
    a2 = prefs.getString('a2') ?? '0';
    a3 = prefs.getString('a3') ?? '0';
    a4 = prefs.getString('a4') ?? '0';
    a5 = prefs.getString('a5') ?? '0';
    a6 = prefs.getString('a6') ?? '0';
    a7 = prefs.getString('a7') ?? '0';
    a8 = prefs.getString('a8') ?? '0';
    a9 = prefs.getString('a9') ?? '0';
    a10 = prefs.getString('a10') ?? '0';
    a11 = prefs.getString('a11') ?? '0';
    a12 = prefs.getString('a12') ?? '0';
    a13 = prefs.getString('a13') ?? '0';
    a14 = prefs.getString('a14') ?? '0';
    a15 = prefs.getString('a15') ?? '0';
    a16 = prefs.getString('a16') ?? '0';
    a17 = prefs.getString('a17') ?? '0';
    a18 = prefs.getString('a18') ?? '0';
    a19 = prefs.getString('a19') ?? '0';

    a20 = prefs.getString('a20') ?? '0';
    a21 = prefs.getString('a21') ?? '0';
    a22 = prefs.getString('a22') ?? '0';
    a23 = prefs.getString('a23') ?? '0';
    a24 = prefs.getString('a24') ?? '0';
    a25 = prefs.getString('a25') ?? '0';
    a26 = prefs.getString('a26') ?? '0';
    a27 = prefs.getString('a27') ?? '0';
    a28 = prefs.getString('a28') ?? '0';
    a29 = prefs.getString('a29') ?? '0';
    a30 = prefs.getString('a30') ?? '0';
    a31 = prefs.getString('a31') ?? '0';

    myModel.updateA1(int.parse(a1!));
    myModel.updateA2(int.parse(a2!));
    myModel.updateA3(int.parse(a3!));
    myModel.updateA4(int.parse(a4!));
    myModel.updateA5(int.parse(a5!));
    myModel.updateA6(int.parse(a6!));
    myModel.updateA7(int.parse(a7!));
    myModel.updateA8(int.parse(a8!));
    myModel.updateA9(int.parse(a9!));
    myModel.updateA10(int.parse(a10!));
    myModel.updateA11(int.parse(a11!));
    myModel.updateA12(int.parse(a12!));
    myModel.updateA13(int.parse(a13!));
    myModel.updateA14(int.parse(a14!));
    myModel.updateA15(int.parse(a15!));
    myModel.updateA16(int.parse(a16!));
    myModel.updateA17(int.parse(a17!));
    myModel.updateA18(int.parse(a18!));
    myModel.updateA19(int.parse(a19!));

    myModel.updateA20(int.parse(a20!));
    myModel.updateA21(int.parse(a21!));
    myModel.updateA22(int.parse(a22!));
    myModel.updateA23(int.parse(a23!));
    myModel.updateA24(int.parse(a24!));
    myModel.updateA25(int.parse(a25!));
    myModel.updateA26(int.parse(a26!));
    myModel.updateA27(int.parse(a27!));
    myModel.updateA28(int.parse(a28!));
    myModel.updateA29(int.parse(a29!));
    myModel.updateA30(int.parse(a30!));
    myModel.updateA31(int.parse(a31!));

    done = int.parse(prefs.getString('done') ?? "-2");
    print(a1);
    setState(() {});
  }

  @override
  void dispose() {
    super.dispose();
    _audioPlayer.stop();

    _audioPlayer.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final myModel = Provider.of<MyModel>(context, listen: false);
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }
        NavigationUtils.backNavigation(context, Dashboard());
      },
      child: SafeArea(
        child: Scaffold(
          body: SingleChildScrollView(
            child: Stack(children: [
              Container(
                height: 2750,
                width: MediaQuery.of(context).size.width,
                decoration: BoxDecoration(
                  image: DecorationImage(
                    image: AssetImage('assets/map.png'),
                    fit: BoxFit.fill,
                  ),
                ),
              ),
              Positioned(
                  top: 15.0,
                  right: 15,
                  child: GestureDetector(
                    onTap: () {
                      NavigationUtils.frontNavigation(context, RatingScreen());
                    },
                    child: Container(
                      height: 75,
                      width: 75,
                      decoration: BoxDecoration(
                        image: DecorationImage(
                          image: AssetImage('assets/report.png'),
                        ),
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  )),
              /*
              Positioned(
                  top: 15.0,
                  left: MediaQuery.of(context).size.width * 0.44,
                  child: LevelIcon(
                    level: 31,
                    stars: a31 != null ? int.parse(a31!) : 0,
                    onTap: () {
                      if ((done! + 2) > 31) {
                        myModel.updateCurrentActivity2(31);
                        NavigationUtils.frontNavigation(
                            context, Instruction29());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 31)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 100.0,
                  left: MediaQuery.of(context).size.width * 0.39,
                  child: LevelIcon(
                    level: 30,
                    stars: a30 != null ? int.parse(a30!) : 0,
                    onTap: () {
                      if ((done! + 2) > 30) {
                        myModel.updateCurrentActivity2(30);

                        NavigationUtils.frontNavigation(
                            context, Instruction29());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 30)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 185.0,
                  left: MediaQuery.of(context).size.width * 0.32,
                  child: LevelIcon(
                    level: 29,
                    stars: a29 != null ? int.parse(a29!) : 0,
                    onTap: () {
                      if ((done! + 2) > 29) {
                        myModel.updateCurrentActivity2(29);

                        NavigationUtils.frontNavigation(
                            context, Instruction29());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 29)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 270.0,
                  left: MediaQuery.of(context).size.width * 0.28,
                  child: LevelIcon(
                    level: 28,
                    stars: a28 != null ? int.parse(a28!) : 0,
                    onTap: () {
                      if ((done! + 2) > 28) {
                        myModel.updateCurrentNum(9);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 28)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 355.0,
                  left: MediaQuery.of(context).size.width * 0.3,
                  child: LevelIcon(
                    level: 27,
                    stars: a27 != null ? int.parse(a27!) : 0,
                    onTap: () {
                      if ((done! + 2) > 27) {
                        myModel.updateCurrentNum(8);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 27)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 440.0,
                  left: MediaQuery.of(context).size.width * 0.4,
                  child: LevelIcon(
                    level: 26,
                    stars: a26 != null ? int.parse(a26!) : 0,
                    onTap: () {
                      if ((done! + 2) > 26) {
                        myModel.updateCurrentNum(7);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 26)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 525.0,
                  left: MediaQuery.of(context).size.width * 0.55,
                  child: LevelIcon(
                    level: 25,
                    stars: a25 != null ? int.parse(a25!) : 0,
                    onTap: () {
                      if ((done! + 2) > 25) {
                        myModel.updateCurrentNum(6);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 25)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 610.0,
                  left: MediaQuery.of(context).size.width * 0.62,
                  child: LevelIcon(
                    level: 24,
                    stars: a24 != null ? int.parse(a24!) : 0,
                    onTap: () {
                      if ((done! + 2) > 24) {
                        myModel.updateCurrentNum(5);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 24)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 695.0,
                  left: MediaQuery.of(context).size.width * 0.625,
                  child: LevelIcon(
                    level: 23,
                    stars: a23 != null ? int.parse(a23!) : 0,
                    onTap: () {
                      if ((done! + 2) > 23) {
                        myModel.updateCurrentNum(4);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 23)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 780.0,
                  left: MediaQuery.of(context).size.width * 0.45,
                  child: LevelIcon(
                    level: 22,
                    stars: a22 != null ? int.parse(a22!) : 0,
                    onTap: () {
                      if ((done! + 2) > 22) {
                        myModel.updateCurrentNum(3);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 22)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 810.0,
                  left: MediaQuery.of(context).size.width * 0.2,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 21)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 21,
                    stars: a21 != null ? int.parse(a21!) : 0,
                    onTap: () {
                      if ((done! + 2) > 21) {
                        myModel.updateCurrentNum(2);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 910.0,
                  left: MediaQuery.of(context).size.width * 0.08,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 20)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 20,
                    stars: a20 != null ? int.parse(a20!) : 0,
                    onTap: () {
                      if ((done! + 2) > 20) {
                        myModel.updateCurrentNum(1);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 1020.0,
                  left: MediaQuery.of(context).size.width * 0.1,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 19)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 19,
                    stars: a19 != null ? int.parse(a19!) : 0,
                    onTap: () {
                      if ((done! + 2) > 19) {
                        myModel.updateCurrentNum(0);
                        NavigationUtils.frontNavigation(
                            context, Instruction19());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 1060.0,
                  left: MediaQuery.of(context).size.width * 0.35,
                  child: LevelIcon(
                    level: 18,
                    stars: a18 != null ? int.parse(a18!) : 0,
                    onTap: () {
                      if ((done! + 2) > 18) {
                        NavigationUtils.frontNavigation(
                            context, Instruction18());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 18)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1140.0,
                  left: MediaQuery.of(context).size.width * 0.5,
                  child: LevelIcon(
                    level: 17,
                    stars: a17 != null ? int.parse(a17!) : 0,
                    onTap: () {
                      if ((done! + 2) > 17) {
                        NavigationUtils.frontNavigation(
                            context, Instruction17());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 17)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1250.0,
                  left: MediaQuery.of(context).size.width * 0.53,
                  child: LevelIcon(
                    level: 16,
                    stars: a16 != null ? int.parse(a16!) : 0,
                    onTap: () {
                      if ((done! + 2) > 16) {
                        NavigationUtils.frontNavigation(
                            context, Instruction16());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 16)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1360.0,
                  left: MediaQuery.of(context).size.width * 0.53,
                  child: LevelIcon(
                    level: 15,
                    stars: a15 != null ? int.parse(a15!) : 0,
                    onTap: () {
                      if ((done! + 2) > 15) {
                        NavigationUtils.frontNavigation(
                            context, Instruction15());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 15)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1450.0,
                  left: MediaQuery.of(context).size.width * 0.49,
                  child: LevelIcon(
                    level: 14,
                    stars: a14 != null ? int.parse(a14!) : 0,
                    onTap: () {
                      myModel.updateCurrentActivity(14);
                      NavigationUtils.frontNavigation(context, Instruction12());
                    },
                    bgColor: ((done! + 2) > 14)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1545.0,
                  left: MediaQuery.of(context).size.width * 0.4,
                  child: LevelIcon(
                    level: 13,
                    stars: a13 != null ? int.parse(a13!) : 0,
                    onTap: () {
                      if ((done! + 2) > 13) {
                        myModel.updateCurrentActivity(13);
                        NavigationUtils.frontNavigation(
                            context, Instruction12());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 13)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1640.0,
                  left: MediaQuery.of(context).size.width * 0.43,
                  child: LevelIcon(
                    level: 12,
                    stars: a12 != null ? int.parse(a12!) : 0,
                    onTap: () {
                      if ((done! + 2) > 12) {
                        myModel.updateCurrentActivity(12);
                        NavigationUtils.frontNavigation(
                            context, Instruction12());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 12)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1730.0,
                  left: MediaQuery.of(context).size.width * 0.37,
                  child: LevelIcon(
                    level: 11,
                    stars: a11 != null ? int.parse(a11!) : 0,
                    onTap: () {
                      if ((done! + 2) > 11) {
                        NavigationUtils.frontNavigation(
                            context, Instruction11());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 11)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1815.0,
                  left: MediaQuery.of(context).size.width * 0.27,
                  child: LevelIcon(
                    level: 10,
                    stars: a10 != null ? int.parse(a10!) : 0,
                    onTap: () {
                      if ((done! + 2) > 10) {
                        NavigationUtils.frontNavigation(
                            context, Instruction10());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 10)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1900.0,
                  left: MediaQuery.of(context).size.width * 0.27,
                  child: LevelIcon(
                    level: 9,
                    stars: a9 != null ? int.parse(a9!) : 0,
                    onTap: () {
                      if ((done! + 2) > 9) {
                        NavigationUtils.frontNavigation(
                            context, Instruction9());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                    bgColor: ((done! + 2) > 9)
                        ? AppColors.orange2
                        : AppColors.orange1,
                  )),
              Positioned(
                  top: 1990.0,
                  left: MediaQuery.of(context).size.width * 0.35,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 8)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 8,
                    stars: a8 != null ? int.parse(a8!) : 0,
                    onTap: () {
                      if ((done! + 2) > 8) {
                        NavigationUtils.frontNavigation(
                            context, Instruction8());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2065.0,
                  left: MediaQuery.of(context).size.width * 0.48,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 7)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 7,
                    stars: a7 != null ? int.parse(a7!) : 0,
                    onTap: () {
                      if ((done! + 2) > 7) {
                        NavigationUtils.frontNavigation(
                            context, Instruction7());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2150.0,
                  left: MediaQuery.of(context).size.width * 0.6,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 6)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 6,
                    stars: a6 != null ? int.parse(a6!) : 0,
                    onTap: () {
                      if ((done! + 2) > 6) {
                        NavigationUtils.frontNavigation(
                            context, Instruction6());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2240.0,
                  left: MediaQuery.of(context).size.width * 0.65,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 5)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 5,
                    stars: a5 != null ? int.parse(a5!) : 0,
                    onTap: () {
                      if ((done! + 2) > 5) {
                        NavigationUtils.frontNavigation(
                            context, Instruction5());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  ))
              */
              Positioned(
                  top: 2330.0,
                  left: MediaQuery.of(context).size.width * 0.63,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 4)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 4,
                    stars: a4 != null ? int.parse(a4!) : 0,
                    onTap: () {
                      if ((done! + 2) > 4) {
                        NavigationUtils.frontNavigation(
                            context, Instruction4());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2400.0,
                  left: MediaQuery.of(context).size.width * 0.5,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 3)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 3,
                    stars: a3 != null ? int.parse(a3!) : 0,
                    onTap: () {
                      if ((done! + 2) > 3) {
                        NavigationUtils.frontNavigation(
                            context, Instruction3());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2490.0,
                  left: MediaQuery.of(context).size.width * 0.42,
                  child: LevelIcon(
                    bgColor: ((done! + 2) > 2)
                        ? AppColors.orange2
                        : AppColors.orange1,
                    level: 2,
                    stars: a2 != null ? int.parse(a2!) : 0,
                    onTap: () {
                      if ((done! + 2) > 2) {
                        NavigationUtils.frontNavigation(
                            context, Instruction2());
                      } else {
                        SnackbarUtils.showDefaultSnackBar(
                          context,
                          ConstValues.allRequiredMsg,
                          AppColors.errorcolor,
                        );
                      }
                    },
                  )),
              Positioned(
                  top: 2585.0,
                  left: MediaQuery.of(context).size.width * 0.4,
                  child: LevelIcon(
                    level: 1,
                    stars: a1 != null ? int.parse(a1!) : 0,
                    onTap: () async {
                      NavigationUtils.frontNavigation(context, Instruction1());
                    },
                    bgColor: AppColors.orange2,
                  )),
            ]),
          ),
        ),
      ),
    );
  }
}

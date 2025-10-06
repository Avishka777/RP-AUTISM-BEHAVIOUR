import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:ukussa_app/Screens/Activity4/activity4.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/button.dart';
import 'package:ukussa_app/main.dart';

class Instruction4 extends StatefulWidget {
  const Instruction4({super.key});

  @override
  State<Instruction4> createState() => _Instruction4State();
}

class _Instruction4State extends State<Instruction4> {
  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
  }

  @override
  void dispose() {
    super.dispose();
  }

  void _startStopwatch() {
    setState(() {
      MyApp.globalStopwatch.start();
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
          backgroundColor: AppColors.pintk1,
          body: Padding(
            padding: EdgeInsets.only(top: 30.0),
            child: SingleChildScrollView(
              child: Column(
                children: [
                  Container(
                    width: MediaQuery.of(context).size.width,
                    color: AppColors.pintk2,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            //color: Colors.red,
                            width: MediaQuery.of(context).size.width * 0.85,
                            child: Text(
                              'දෙමාපියන් සඳහා මාර්ගෝපදේශය: "මාර්ගය සොයා ගන්න" ක්‍රියාකාරකම',
                              style: TextStyle(
                                fontSize: AppFonts.font16,
                                color: AppColors.black1,
                              ),
                            ),
                          ),
                          Container(
                            //color: Colors.red,
                            width: MediaQuery.of(context).size.width * 0.1,
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceAround,
                              children: [
                                Text(
                                  "04",
                                  style: TextStyle(
                                    fontSize: AppFonts.font16,
                                    color: AppColors.black1,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 52.0,
                  ),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    color: AppColors.white1,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        'අරමුණ ',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 23.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        'මෙම ක්‍රියාකාරකම් මගින් තිරය මත දී ඇති කඩ ඉර් ඔස්සේ අඟිලි හැසිර වීමෙන් රේඛාව ඔස්සේ නිවැරදිව අත හා ඇඟිලි චලනය කිරීමේ හැකියාව ඇතිවේ. ලිවීම සහ කියවීම වැනි ක්‍රියාකාරකම් සඳහා අවශ්‍යය වමේ සිට දකුණට අතෙහි චලනයන් උද්ධීපනය කරන අතර අවශ්‍ය සියුම් මෝටර් කුසලතා ඇති කරයි.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 23.0,
                  ),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    color: AppColors.white1,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        'මෙම ක්‍රියාකාරකම සංජානන සංවර්ධනයට සහාය වන ආකාරය',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 20.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ සියුම් මෝටර් කුසලතා - වඩා හොඳ අත් අකුරු සහ දෛනික කාර්යයන් සඳහා ඇඟිලි චලනයන් ශක්තිමත් කරයි.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 10.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ අත්-ඇස් සම්බන්ධීකරණය - දරුවන්ට මාර්ගයක් ඔස්සේ ඔවුන්ගේ ඇඟිලි නිවැරදිව මෙහෙය වීමට උපකාරී වේ.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 10.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ අවධානය සහ ඉවසීම - තිරසාර අවධානය සහ කාර්යය සම්පූර්ණ කිරීම දිරිමත් කරයි.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 10.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ අවකාශීය දැනුවත්භාවය - රේඛා තුළ රැඳී සිටීමට සහ චලන රටා තේරුම් ගැනීමට දරුවන්ට උගන්වයි.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 10.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ පූර්ව ලිවීමේ කුසලතා - වමේ සිට දකුණට අත හැසිරවීම දියුණු වේ විවිධ දිශාවන් ඔස්සේ අත චලනය කිරීමේ අභ්‍යාස මගින් අකුරු ලිවීමට අවශය චලන පරාසය අතට ලබා දේ.',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 23.0,
                  ),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    color: AppColors.white1,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        'මෙම ක්‍රියාකාරකම කරන්නේ කෙසේද',
                        style: TextStyle(
                          fontSize: AppFonts.font16,
                          color: AppColors.black1,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 23.0,
                  ),
                  SizedBox(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '01. පරිසරය සකසන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'ස්ටයිලස් හෝ ඇඟිලි පාදක ලුහුබැඳීමක් සහිත ටැබිලටයක් හෝ ස්පර්ශ තිර උපාංගයක් භාවිතා කරන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'දරුවාට අවධානය යොමු කිරීමට උපකාර කිරීම සඳහා නිහඬ, සුව පහසු ඉඩක් තෝරන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 15.0,
                          ),
                          Text(
                            '02. සංකල්පය හඳුන්වා දෙන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'සරල මාර්ගයක් පෙන්වන්න (සෘජු, වක්‍ර සිග්සැග්)',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන්ගේ ඇඟිල්ලෙන් හෝ ස්ටයිලස් සමඟ මාර්ගය දිගේ ලුහුබැඳීමට',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන්ට අවශ්‍ය බව පැහැදිලි කරන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'මාර්ගයේ කුඩා කොටසක් ලුහුබැඳීමෙන් නිරූපණය කරන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 15.0,
                          ),
                          Text(
                            '03. සහභාගීත්වය දිරිමත් කරන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'වාචික දිගැන්වීමක් භාවිතා කරන්න:',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            '"ඔබට ඔබේ ඇඟිල්ලෙන් රේඛාව අනුගමනය කළ හැකිද?"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            '" නියම වැඩක්! අපි එකට තවත් එකක් ලුහුබැඳ යමු!"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන් අරගල කරන්නේ නම්, මාර්ගය දිගේ ඔවුන්ගේ ඇඟිල්ල යොමු කරන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 15.0,
                          ),
                          Text(
                            '04. ඉගෙනීම සහ සංජානන වර්ධනය ශක්තිමත් කරන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'විස්තරාත්මක වචන භාවිතා කරන්න ("ඔබ වක්‍ර මාර්ගය පරිපූර්ණ ලෙස අනුගමනය කළා!").',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'අවබෝධය වැඩි දියුණු කිරීම සඳහා සරල ප්‍රශ්න අසන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            '"ඔබට මෙම සිග්සැග් රේඛාව ලුහුබැඳිය හැකිද?"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            '"මෙම මාර්ගය කුමන හැඩයක් වගේද?"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 15.0,
                          ),
                          Text(
                            '05. ප්‍රගතිය සැමරීම සහ නිරීක්ෂනය කිරීම',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'ධනාත්මක ශක්තිමත් කිරීමක් ලබා දෙන්න (අත්පුඩි, චියර්ස්, අතථ්‍ය ස්ටිකර්).',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  SizedBox(
                    height: 25.0,
                  ),
                  Button(
                    onPressed: () async {
                      _startStopwatch();
                      NavigationUtils.frontNavigation(context, Activity4());
                    },
                    text: "පාඩම ආරම්භ කරන්න",
                    buttonColor: AppColors.green1,
                    height: 50.0,
                    width: MediaQuery.of(context).size.width * 0.75,
                    fontSize: AppFonts.font16,
                    fontWeight: FontWeight.normal,
                    rad: 24.0,
                  ),
                  SizedBox(
                    height: 25.0,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

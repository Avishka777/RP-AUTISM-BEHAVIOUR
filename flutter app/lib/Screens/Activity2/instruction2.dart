import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:ukussa_app/Screens/Activity2/drawingScreen.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/button.dart';

class Instruction2 extends StatefulWidget {
  const Instruction2({super.key});

  @override
  State<Instruction2> createState() => _Instruction2State();
}

class _Instruction2State extends State<Instruction2> {
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
                              'දෙමාපියන් සඳහා මාර්ගෝපදේශය: "පිරිසිදු කිරීමට ස්වයිප් කරන්න" ක්‍රියාකාරකම',
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
                                  "02",
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
                        'මෙම ක්‍රියාකාරකම් දරුවන්ගේ හසුරු කුසලතා සංවර්ධනය සඳහා ඉලක්ක ගත කෙරේ දබරගිල්ලේ සංවේදී තාව හා සියුම් චලන හැකියාවන් වැඩිදියුනු කරයි අතෙහි මොටාර් කුසලතා අභිප්‍රේරණය කිරීම.',
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
                        '✔ සියුම් මෝටර් කුසලතා / විවිධ කාර්යයන් සඳහා අතෙහි චලක පරාසය දියුනු කිරීම',
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
                        '✔ අත් - ඇස් සමායෝජනය, සංවර්ධනය',
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
                        '✔ හේතුව සහ බලපෑම අවබෝධ කර ගැනීම - ස්වයිප් කිරීම සහ පිරිසිදු කිරීම අතර සම්බන්ධතාවය ඉගෙනගනියි.',
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
                        '✔ අවබෝධය සහ අවධානය වැඩිදියුණු කිරීම.',
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
                        '✔ අභ්‍යාසය තුලින් ලබාගන්නා අත්දැකීම් පිරිදිසු කිරීම වැනි සාමාන්‍ය ජීවිතයේ අවස්ථා වලට සම්භන්ධ කිරීම.',
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
                            'ක්‍රියාකාරකම් සඳහා නිහඬ, අවධානය වෙනතකට යොමු නොවන ඉඩක් තෝරන්න.',
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
                            'ඔවුන් නම් ඇඟිල්ලෙන් ස්වයිප් කරන විට, ඔවුන් එය පිරිසිදු කරන බව පැහැදිලි කරන්න! 🧼✨',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන්ට උත්සාහ කිරීමට ඉඩ දීමට පෙර චලනය නිරූපණය කරන්න.',
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
                            'වාචික දි ගැන්වීමක් භාවිතා කරන්න:',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'බලන්න | තිරය අවුල් සහගතය | ඔබට එය පිරිසිදු කිරීමට උදව් කළ හැකිද?"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'වාව්! ඒ කොටස පිරිසිදුයි ! දිගටම කරගෙන යන්න!"',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'ඔවුන් අරගල කරන්නේ නම්, තිරය හරහා ඔවුන්ගේ ඇඟිල්ල මෘදු ලෙස යොමු කරන්න. ',
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
                            'විස්තරාත්මක වචන භාවිතා කරන්න ("ඔබ වම් පැත්ත පිරිසිදු කලා! දැන් දකුණ උත්සාහ කරන්න!")',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'අවබෝධය වැඩි දියුණු කිරීම සඳහා සරල ප්‍රශ්න අසන්න:',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'තවත් කොහෙද අපිරිසිදු',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔබට වේගයෙන් පිරිසිදු කළ හැකිද',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          SizedBox(
                            height: 15.0,
                          ),
                          Text(
                            '05. දරුවාගේ ක්‍රියාවන් ඇගයීම මගින් දරුවා උනන්දු කිරීම',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.red3,
                            ),
                          ),
                          SizedBox(
                            height: 10.0,
                          ),
                          Text(
                            'ධනාත්මක ශක්තිමත් කිරීමක් ලබා දෙන්න (අත්පුඩ්, චියර්ස්, අතථ්‍ය ත්‍යාග). ',
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
                      NavigationUtils.frontNavigation(context, DrawingScreen());
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

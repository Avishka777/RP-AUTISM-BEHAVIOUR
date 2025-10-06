import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:ukussa_app/Screens/Activity1/balloonGame.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/button.dart';

class Instruction1 extends StatefulWidget {
  const Instruction1({super.key});

  @override
  State<Instruction1> createState() => _Instruction1State();
}

class _Instruction1State extends State<Instruction1> {
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
                              'දෙමාපියන් සඳහා මාර්ගෝපදේශය: බැලුන් පොප් කිරීමට තට්ටු කරන්න',
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
                                  "01",
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
                        'මෙම ක්‍රියාකාරකම දරුවන්ගේ ඉන්ද්‍රීය සමායෝජනය සඳහා අභ්‍යාසයකි (ද්‍යුත්- ඇස් සහ මොලය එක්ව ක්‍රියාත්මක වීම), පාවෙන බැලුන් තට්ටු කිරීමෙන් දරුවන් නිරීක්ෂණ කුසලතා සංවර්ධනය - අවධානය යොමු කිරීමට හා ක්ෂනික නිවැරදි තීරණ ගැනීමේ හැකියාව උද්ධීපනය කිරීම. සතුට විනෝදය ආකර්ශනය ඇතිකිරීම.',
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
                        '✔ දෑත් - ඇස් සම්බන්ධීකරණය.',
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
                        '✔ අවබෝධය සහ අවධානය.',
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
                        '✔ හේතුව සහ බලපෑම අවබෝධ කර ගැනීම.',
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
                        '✔ තීරණ ගැනීම හා නිවැරදිව ක්‍රියාත්මක වීමේ හැකියා සංවර්ධනය.',
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
                        '✔ අවධානය - දරුවන් ව්‍යුහගත ක්‍රියාකාරකමක නිරත කරවයි.',
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
                            'ක්‍රියාකාරකම සඳහා නිහඬ, අවධානය වෙනතකට යොමු නොවන ඉඩක් තෝරන්න.',
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
                            'ඔබේ දරුවාට පාවෙන බැලුන් පෙන්වන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන් බැලූනයක් තට්ටු කරන විට, එය පුපුරා යන බව පැහැදිලි කරන්න! 🎈💥',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන්ට උත්සාහ කිරීමට ඉඩ දීමට පෙර එය කරන්නේ කෙසේදැයි නිරූපණය කරන්න.',
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
                            'වාචික දිරි ගැන්වීමක් භාවිතා කරන්න :',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            '(වාව්! බැලුන් දෙස බලන්න! ඔබට එකක් තට්ටු කළ හැකිද? නියම වැඩක්! අපි එකට තවත් එකක් පොප් කරමු!)',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ක්‍රියාකාරකම සඳහා සහභාගී නොවන්නේ නම්, තිරය ස්පර්ශ කිරීමට ඔවුන්ගේ අත මෘදු ලෙස මෙහෙයවන්න.',
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
                            'දරුවාට තේරුම්ගත හැකි පරිදි විධානයන් ලබා දෙන්න ("ඒක රතු බැලූනයක් අපි පොප් කරමු!).',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'තීරණ ගැනීම වැඩි දියුණු කිරීම සඳහා සරල ප්‍රශ්න අසන්න:',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඊළඟට පොප් කිරීමට ඔබට අවශ්‍ය කුමන බැලූනයද',
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
                            'අභිප්‍රේරණය සඳහා අත්පුඩි, චියර්ස්, අතථ්‍ය ත්‍යාග ලබා දෙන්න.',
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
                      NavigationUtils.frontNavigation(context, BalloonGame());
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

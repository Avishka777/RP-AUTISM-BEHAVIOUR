import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ukussa_app/Screens/Activity3/activity3.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/button.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';

class Instruction3 extends StatefulWidget {
  const Instruction3({super.key});

  @override
  State<Instruction3> createState() => _Instruction3State();
}

class _Instruction3State extends State<Instruction3> {
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
                              'දෙමාපියන් සඳහා මාර්ගෝපදේශය: "සීමාව තුල වර්ණ ගැන්වීම" ක්‍රියාකාරකම (🎨)',
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
                                  "03",
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        'මෙම ක්‍රියාකාරකම් දරුවන්ට ඩිපිටල් වර්ණ ගැන්වීමේ නිරත වීමෙන් සියුම් ඇඟිලි චලනයන්,මගින් සීමාව තුල දෑත් ඇස් හැසිරවීමේ කුසලතාව ඇති කිරීම.\nඉහත ක්‍රියාකාරකම විධිමත්ව සිදුකිරීමෙන් තිරයෙහි දකින ප්‍රතිඵලය දරුවාට විනෝදජනක සහ ආකර්ශනීය අත්දැකීම් ලබා දේ. අතෙහි මෝටාර් කුසලතා සංවර්ධනය තවදුරටත් සිදුවීම.',
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ ලේඛනය චිත්‍ර ඇදීම සිවුම් ග්‍රහන හැකියා වැනි මෝටාර් කුසලතා ඇති කිරීම මගින් දෛනික කාර්යයන් සඳහා අවශ්‍ය හස්ථ කෞශලනය ඇතිකිරීම.',
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ සීමාව තුළ නිවැරදිව වර්ණ ගැන්වීමේ හැකියාව ඇති කිරීම.',
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ අවධානය සහ ඉවසීම: කාර්යයක් සම්පූර්ණ කිරීමට තිරසාර අවධානය දිරීමත් කරයි.',
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ ගැලපෙන වර්ණ තෝරාගැනීම සහ නිර්මාණ ශීලීත්වය උද්ධීපනය',
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
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Text(
                        '✔ තීරණ ගැනීම හා ගැලපීම වැනි කුසලතා ඇතිකිරීම. ',
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
                  Container(
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
                            'දරුවාට අවධානය යොමු කිරීමට උපකාර කිරීම සඳහා නිහඬ ඉඩක් තෝරන්න.',
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
                            'තිරය මත කළු - සුදු චිත්‍රයක් පෙන්වන්න.',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන්ට වර්ණ තෝරාගෙන පින්තූරය පුරවා ගත හැකි බව පැහැදිලි කරන්න! 🎨✨',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'වර්ණයක් තෝරා කුඩා ප්‍රදේශයක් පුරවා නිරූපණය කරන්න.',
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
                            'වාචික දිරිගැන්වීම භාවිතා කරන්න',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔබට මුලින්ම භාවිතා කිරීමට අවශ්‍ය වර්ණය කුමක්ද',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'වාව්! ඒක පුදුම සහගතයි. ඔබට ඊළඟ කොටස වර්ණ ගැන්විය හැකිද',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔවුන් අවධානයෙන් බැහැර නම්, අභ්‍යාසය සඳහා යොමු කිරීමට උදව් කිරීමට ඔවුන්ගේ අත මෙහෙයවන්න.',
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
                            'විස්තරාත්මක වචන භාවිතා කරන්න ("ඔබ අහස සඳහා නිල් භාවිතා කළා! හිරු කුමන වර්ණය විය යුතුද?").',
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
                            'ඔබට රතු පැන්සල සොයාගත හැකිද ',
                            style: TextStyle(
                              fontSize: AppFonts.font16,
                              color: AppColors.black1,
                            ),
                          ),
                          Text(
                            'ඔබට සීමාව තුළ පාට කළ හැකිද ',
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
                            'ධනාත්මක ප්‍රතිචාර ලබා දෙන්න (අත්පුඩ්, චියර්ස්, අතථ්‍ය ස්ටිකර්),',
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
                      _showColorSelectionDialog();
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

  Color _selectedColor = Colors.red;

  Future<void> _showColorSelectionDialog() async {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(
            'කැමති පාට තෝරන්න',
            style: TextStyle(
                fontSize: AppFonts.font18, fontWeight: FontWeight.w500),
          ),
          content: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              IconButton(
                icon: Icon(
                  Icons.circle,
                  color: Colors.red,
                  size: 50.0,
                ),
                onPressed: () {
                  setState(() {
                    _selectedColor = Colors.red;
                  });
                  print("Selected Color: Red");
                  Navigator.of(context).pop();
                  _navigateToNextScreen();
                },
              ),
              IconButton(
                icon: Icon(
                  Icons.circle,
                  color: Colors.green,
                  size: 50.0,
                ),
                onPressed: () {
                  setState(() {
                    _selectedColor = Colors.green;
                  });
                  print("Selected Color: Green");
                  Navigator.of(context).pop();
                  _navigateToNextScreen();
                },
              ),
              IconButton(
                icon: Icon(
                  Icons.circle,
                  color: Colors.blue,
                  size: 50.0,
                ),
                onPressed: () {
                  setState(() {
                    _selectedColor = Colors.blue;
                  });
                  print("Selected Color: Blue");
                  Navigator.of(context).pop();
                  _navigateToNextScreen();
                },
              ),
            ],
          ),
        );
      },
    );
  }

  void _navigateToNextScreen() {
    NavigationUtils.frontNavigation(context, Activity3(pColor: _selectedColor));
  }
}

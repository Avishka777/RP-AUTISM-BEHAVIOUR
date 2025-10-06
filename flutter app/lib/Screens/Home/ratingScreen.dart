import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:ukussa_app/Providers/provider.dart';
import 'package:ukussa_app/Screens/Home/mapDashboard.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/label.dart';
import 'package:ukussa_app/Widgets/labelResponsive.dart';

class RatingScreen extends StatefulWidget {
  @override
  State<RatingScreen> createState() => _RatingScreenState();
}

class _RatingScreenState extends State<RatingScreen> {
  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(
      [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown],
    );
  }

  @override
  Widget build(BuildContext context) {
    final myModel = Provider.of<MyModel>(context, listen: false);

    final List<Map<String, dynamic>> items = [
      {
        'number': 1,
        'title': 'බැලූන් පොප් කිරීමට තට්ටු කරන්න',
        'rating': myModel.a1
      },
      {
        'number': 2,
        'title': 'පිරිසිදු කිරීමට ස්වයිප් කරන්න',
        'rating': myModel.a2
      },
      {
        'number': 3,
        'title': 'අන්තර්ක්‍රියාකාරී වර්ණ ගැන්වීම',
        'rating': myModel.a3
      },
      {'number': 4, 'title': 'මාර්ගය සොයා ගන්න', 'rating': myModel.a4},
      {'number': 5, 'title': 'ලොකු සහ කුඩා', 'rating': myModel.a5},
      {'number': 6, 'title': 'දිගු සහ කෙටි', 'rating': myModel.a6},
      {'number': 7, 'title': 'උස සහ කෙටි (උස)', 'rating': myModel.a7},
      {'number': 8, 'title': 'ඉහළ, පහළ', 'rating': myModel.a8},
      {'number': 9, 'title': 'වම, දකුණ සහ මැද', 'rating': myModel.a9},
      {'number': 10, 'title': 'සමාන සහ අසමාන රූප තේරීම', 'rating': myModel.a10},
      {'number': 11, 'title': 'හැඩ හදුනාගැනීම', 'rating': myModel.a11},
      {'number': 12, 'title': 'රවුම ඇඳීම', 'rating': myModel.a12},
      {'number': 13, 'title': 'කොටුව ඇඳීම', 'rating': myModel.a13},
      {'number': 14, 'title': 'ත්‍රිකෝණය ඇඳීම', 'rating': myModel.a14},
      {
        'number': 15,
        'title': 'රටාව හදුනාගැනීම සහ හැඩතල ඇඳීම',
        'rating': myModel.a15
      },
      {'number': 16, 'title': 'බොහෝ සහ ස්වල්ප', 'rating': myModel.a16},
      {'number': 17, 'title': '1-10 හදුනා ගැනීම', 'rating': myModel.a17},
      {'number': 18, 'title': 'ගණන් කර වර්ණ ගැන්වීම', 'rating': myModel.a18},
      {'number': 19, 'title': 'අංක 1 ලිවීම', 'rating': myModel.a19},
      {'number': 20, 'title': 'අංක 2 ලිවීම', 'rating': myModel.a20},
      {'number': 21, 'title': 'අංක 3 ලිවීම', 'rating': myModel.a21},
      {'number': 22, 'title': 'අංක 4 ලිවීම', 'rating': myModel.a22},
      {'number': 23, 'title': 'අංක 5 ලිවීම', 'rating': myModel.a23},
      {'number': 24, 'title': 'අංක 6 ලිවීම', 'rating': myModel.a24},
      {'number': 25, 'title': 'අංක 7 ලිවීම', 'rating': myModel.a25},
      {'number': 26, 'title': 'අංක 8 ලිවීම', 'rating': myModel.a26},
      {'number': 27, 'title': 'අංක 9 ලිවීම', 'rating': myModel.a27},
      {'number': 28, 'title': 'අංක 0 ලිවීම', 'rating': myModel.a28},
      {'number': 29, 'title': 'ගණන් කිරීම සහ ලිවීම', 'rating': myModel.a29},
      {'number': 30, 'title': 'ගණන් කිරීම සහ ලිවීම', 'rating': myModel.a30},
      {'number': 31, 'title': 'ගණන් කිරීම සහ ලිවීම', 'rating': myModel.a31},
    ];

    final lowItems = items.where((item) => item['rating'] <= 1).toList();

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        NavigationUtils.backNavigation(context, MapDashboard());
      },
      child: SafeArea(
        child: Container(
          decoration: BoxDecoration(
            image: DecorationImage(
              image: AssetImage('assets/bgimg.png'),
              fit: BoxFit.fill,
            ),
          ),
          child: Scaffold(
            backgroundColor: Colors.transparent,
            body: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(top: 22.0, bottom: 20),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            color: AppColors.pintk2,
                            child: Padding(
                              padding: const EdgeInsets.all(6.0),
                              child: Label(
                                hintText: "දෙමාපියන් සඳහා මාර්ගෝපදේශය",
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font16,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      color: AppColors.green4,
                      child: Table(
                        columnWidths: {
                          0: FixedColumnWidth(30),
                          1: FlexColumnWidth(),
                          2: FixedColumnWidth(100),
                        },
                        border: TableBorder.symmetric(
                          inside: BorderSide(color: Colors.white, width: 1),
                        ),
                        children: [
                          TableRow(
                            decoration: BoxDecoration(color: Colors.green[100]),
                            children: [
                              SizedBox(height: 8),
                              Text('',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              Text('',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                            ],
                          ),
                          ...items.map((item) => TableRow(
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.all(4.0),
                                    child: Container(
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        color: AppColors.orange2,
                                        border: Border.all(
                                          color: AppColors.black1,
                                          width: 1,
                                        ),
                                      ),
                                      child: Center(
                                        child: Text(
                                          item['number'].toString(),
                                          style: TextStyle(
                                            color: AppColors.black1,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Text(item['title']),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: StarRating(rating: item['rating']),
                                  ),
                                ],
                              )),
                        ],
                      ),
                    ),
                    SizedBox(height: 20),
                    Container(
                      color: AppColors.green5,
                      child: Padding(
                        padding: const EdgeInsets.all(4.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            StarRating(rating: 3),
                            SizedBox(width: 28),
                            Label(
                              hintText: "13/20",
                              textColor: AppColors.black1,
                              fontSize: AppFonts.font16,
                              fontFamily: AppFonts.Lora,
                              fontWeight: FontWeight.normal,
                            ),
                            SizedBox(width: 28),
                          ],
                        ),
                      ),
                    ),
                    if (lowItems.isNotEmpty) ...[
                      SizedBox(height: 20),
                      Row(
                        children: [
                          Container(
                            color: AppColors.pintk2,
                            child: Padding(
                              padding: const EdgeInsets.all(6.0),
                              child: Label(
                                hintText: "වැඩිදියුණු කිරීමේ ක්ෂේත්ර",
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font16,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal,
                              ),
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 15),
                      ...lowItems.map((item) => Padding(
                            padding: const EdgeInsets.symmetric(vertical: 2.0),
                            child: Label(
                              hintText: "* ${item['title']}",
                              textColor: AppColors.black1,
                              fontSize: AppFonts.font16,
                              fontFamily: AppFonts.Lora,
                              fontWeight: FontWeight.w500,
                            ),
                          )),
                      SizedBox(height: 20),
                      Row(
                        children: [
                          Container(
                            color: AppColors.pintk2,
                            child: Padding(
                              padding: const EdgeInsets.all(6.0),
                              child: Label(
                                hintText: "දෙමාපියන් සඳහා උපදෙස්",
                                textColor: AppColors.black1,
                                fontSize: AppFonts.font16,
                                fontFamily: AppFonts.Lora,
                                fontWeight: FontWeight.normal,
                              ),
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 15),
                      LabelResponsive(
                        hintText:
                            '* ඔබේ දරුවාට උදව් අවශ්‍ය ස්ථාන බැලීමට "වැඩිදියුණු කළ යුතු ප්‍රදේශ" කොටස පරීක්ෂා කරන්න.\n* ඔබේ දරුවා අරගල කරන විට කරුණාවන්ත වචන වලින් ඔවුන් දිරිමත් කරන්න.\n* විශ්වාසය සහ කුසලතා ගොඩනැගීම සඳහා ක්‍රියාකාරකම් නැවත කරන්න.\n* සෑම සැසියකටම පසු විවේක මෙවලම් භාවිතා කරන්න.\n* වඩා හොඳ ප්‍රතිඵල සඳහා කෙටි දෛනික සැසි (මිනිත්තු 10–15) උත්සාහ කරන්න.\n* ඔබේ දරුවා අභිප්‍රේරණය කර තබා ගැනීමට කුඩා ජයග්‍රහණ සමරන්න.',
                        textColor: AppColors.black1,
                        fontSize: AppFonts.font16,
                        fontFamily: AppFonts.Lora,
                        fontWeight: FontWeight.w500,
                      ),
                    ],
                    SizedBox(height: 40),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        GestureDetector(
                          onTap: () {},
                          child: Container(
                            height: 35,
                            width: 120,
                            decoration: BoxDecoration(
                              image: DecorationImage(
                                image: AssetImage('assets/btnbg.png'),
                                fit: BoxFit.cover,
                              ),
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class StarRating extends StatelessWidget {
  final int rating;
  final int maxRating;

  const StarRating({
    required this.rating,
    this.maxRating = 3,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(maxRating, (index) {
        return Icon(
          index < rating ? Icons.star : Icons.star_border,
          color: Colors.orange,
          size: 20,
        );
      }),
    );
  }
}

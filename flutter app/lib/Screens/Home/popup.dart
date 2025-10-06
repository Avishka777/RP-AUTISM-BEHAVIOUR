import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ukussa_app/Screens/Home/dashboard.dart';
import 'package:ukussa_app/Utils/appColors.dart';
import 'package:ukussa_app/Utils/appfonts.dart';
import 'package:ukussa_app/Utils/pageNavigations.dart';
import 'package:ukussa_app/Widgets/label.dart';

class MyForm extends StatefulWidget {
  @override
  _MyFormState createState() => _MyFormState();
}

class _MyFormState extends State<MyForm> {
  TextEditingController nameController = TextEditingController();
  TextEditingController ageController = TextEditingController();

  String? selectedGender;

  final _formKey = GlobalKey<FormState>();

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);

    initProcess();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> initProcess() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();

    setState(() {
      nameController.text = prefs.getString('name') ?? '';
      ageController.text = prefs.getString('age') ?? '';
      selectedGender = prefs.getString('gender') ?? '';
    });
  }

  void _submitForm() async {
    if (_formKey.currentState!.validate()) {
      final SharedPreferences prefs = await SharedPreferences.getInstance();
      prefs.setString('name', nameController.text);
      prefs.setString('age', ageController.text);
      prefs.setString('gender', selectedGender!);

      NavigationUtils.frontNavigation(context, Dashboard());
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) {
          return;
        }
        SystemNavigator.pop();
      },
      child: Scaffold(
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Center(
              child: Container(
                decoration: BoxDecoration(
                  color: AppColors.green3,
                  borderRadius: BorderRadius.circular(15),
                  border: Border.all(
                    color: AppColors.red1,
                    width: 1,
                  ),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: SingleChildScrollView(
                    child: Form(
                      key: _formKey,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: <Widget>[
                          Padding(
                            padding: const EdgeInsets.only(bottom: 40.0),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Label(
                                  hintText: "ළමයාගේ තොරතුරු",
                                  textColor: AppColors.red1,
                                  fontSize: AppFonts.font20,
                                  fontFamily: AppFonts.Lora,
                                  fontWeight: FontWeight.normal,
                                ),
                              ],
                            ),
                          ),
                          Text('ළමයාගේ නම:', style: TextStyle(fontSize: 18)),
                          SizedBox(
                            height: 5,
                          ),
                          Container(
                            color: AppColors.white1,
                            child: TextFormField(
                              controller: nameController,
                              decoration: InputDecoration(hintText: ''),
                              validator: (value) {
                                if (value == null || value.isEmpty) {
                                  return 'මෙම ක්ෂේත්‍රය අනිවාර්යයි';
                                }
                                return null;
                              },
                            ),
                          ),
                          SizedBox(height: 16),
                          Text('වයස (අවුරුදු):',
                              style: TextStyle(fontSize: 18)),
                          SizedBox(
                            height: 5,
                          ),
                          Container(
                            color: AppColors.white1,
                            child: TextFormField(
                              controller: ageController,
                              decoration: InputDecoration(hintText: ''),
                              keyboardType: TextInputType.number,
                              validator: (value) {
                                if (value == null || value.isEmpty) {
                                  return 'මෙම ක්ෂේත්‍රය අනිවාර්යයි';
                                }
                                return null;
                              },
                            ),
                          ),
                          SizedBox(height: 16),
                          Text('ස්ත්‍රී / පුරුෂ භාවය : ',
                              style: TextStyle(fontSize: 18)),
                          Row(
                            children: <Widget>[
                              Row(
                                children: <Widget>[
                                  Radio<String>(
                                    activeColor: AppColors.white1,
                                    value: 'පුරුෂ',
                                    groupValue: selectedGender,
                                    onChanged: (String? value) {
                                      setState(() {
                                        selectedGender = value;
                                      });
                                    },
                                  ),
                                  Text('පුරුෂ', style: TextStyle(fontSize: 18)),
                                ],
                              ),
                              Row(
                                children: <Widget>[
                                  Radio<String>(
                                    activeColor: AppColors.white1,
                                    value: 'ස්ත්‍රී',
                                    groupValue: selectedGender,
                                    onChanged: (String? value) {
                                      setState(() {
                                        selectedGender = value;
                                      });
                                    },
                                  ),
                                  Text('ස්ත්‍රී',
                                      style: TextStyle(fontSize: 18)),
                                ],
                              ),
                            ],
                          ),
                          SizedBox(height: 30),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton(
                                onPressed: _submitForm,
                                child: Text(
                                  'යවන්න',
                                  style: TextStyle(color: AppColors.black1),
                                ),
                                style: ButtonStyle(
                                  backgroundColor: MaterialStateProperty.all(
                                      AppColors.green5),
                                  shape: MaterialStateProperty.all(
                                    RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          )
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

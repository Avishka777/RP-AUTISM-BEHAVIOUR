import 'package:flutter/material.dart';
import 'package:ukussa_app/Utils/appColors.dart';

class LevelIcon extends StatefulWidget {
  final int level;
  final int stars;
  final VoidCallback onTap;
  final Color bgColor;
  const LevelIcon(
      {Key? key,
      required this.level,
      required this.stars,
      required this.onTap,
      required this.bgColor})
      : super(key: key);

  @override
  State<LevelIcon> createState() => _LevelIconState();
}

class _LevelIconState extends State<LevelIcon> {
  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 8.0),
          child: GestureDetector(
            onTap: widget.onTap,
            child: Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: widget.bgColor,
                border: Border.all(
                  color: AppColors.black1,
                  width: 2.5,
                ),
              ),
              child: Center(
                child: Text(
                  '${widget.level}',
                  style: TextStyle(
                    fontSize: 30,
                    color: AppColors.black1,
                  ),
                ),
              ),
            ),
          ),
        ),
        Positioned(
          top: 0,
          left: 0,
          child: SizedBox(
            width: 60,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                widget.stars,
                (index) => Icon(
                  Icons.star,
                  color: AppColors.gold1,
                  size: 20,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

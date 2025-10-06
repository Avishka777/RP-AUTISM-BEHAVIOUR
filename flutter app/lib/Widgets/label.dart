import 'package:flutter/material.dart';

class Label extends StatefulWidget {
  final String hintText;
  final Color textColor;
  final double fontSize;
  final String fontFamily;
  final FontWeight fontWeight;
  final bool shouldUnderline;
  final bool isItalic;

  const Label({
    Key? key,
    required this.hintText,
    required this.textColor,
    required this.fontSize,
    required this.fontFamily,
    required this.fontWeight,
    this.shouldUnderline = false,
    this.isItalic = false,
  }) : super(key: key);

  @override
  State<Label> createState() => _LabelState();
}

class _LabelState extends State<Label> {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          widget.hintText,
          style: TextStyle(
            fontSize: widget.fontSize,
            fontFamily: widget.fontFamily,
            color: widget.textColor,
            fontWeight: widget.fontWeight,
            decoration: widget.shouldUnderline
                ? TextDecoration.underline
                : TextDecoration.none,
            fontStyle: widget.isItalic ? FontStyle.italic : FontStyle.normal,
          ),
        ),
      ],
    );
  }
}

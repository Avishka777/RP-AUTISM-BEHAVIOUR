import 'package:flutter/material.dart';

class LabelResponsive extends StatefulWidget {
  final String hintText;
  final Color textColor;
  final double fontSize;
  final String fontFamily;
  final TextAlign textAlign;
  final FontWeight fontWeight;
  final bool shouldUnderline;
  final bool isItalic;

  const LabelResponsive({
    Key? key,
    required this.hintText,
    required this.textColor,
    required this.fontSize,
    required this.fontFamily,
    required this.fontWeight,
    this.shouldUnderline = false,
    this.isItalic = false,
    this.textAlign = TextAlign.left,
  }) : super(key: key);

  @override
  State<LabelResponsive> createState() => _LabelresponsiveState();
}

class _LabelresponsiveState extends State<LabelResponsive> {
  @override
  Widget build(BuildContext context) {
    return Text(
      widget.hintText,
      textAlign: widget.textAlign,
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
    );
  }
}

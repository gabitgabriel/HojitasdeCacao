 
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
 
import 'package:leafscan_mdd/main.dart';
 
void main() {
  testWidgets('LeafScan MDD carga correctamente', (WidgetTester tester) async {
    // Construye la app y dispara un frame.
    await tester.pumpWidget(const LeafScanApp());
 
    // Verifica que el título de la app aparece.
    expect(find.text('LeafScan MDD'), findsOneWidget);
  });
}
 
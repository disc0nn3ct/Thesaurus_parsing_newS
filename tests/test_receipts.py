"""Synthetic receipts only: no saved customer receipts, fiscal IDs or network calls."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from thesaurus import receipts


class ReceiptParserTests(unittest.TestCase):
    def test_multiline_wine_rows_ignore_vat_and_post_total_bonus_discount(self):
        text = 'Кассовый чек / Приход\nООО "ТЕСТ"\nДАТА ВЫДАЧИ 11.10.26 16:31\n'
        values = ['628.86', '817.51', '817.52', '628.86', '440.20', '565.96', '534.53']
        for index, value in enumerate(values):
            text += ('Вино Тест ' + ('Повтор' if index in {1, 2} else str(index)) + '\nкр.сух.0,75\n1 X ' + value
                     + '\nв т.ч. СУММА НДС 22% =\n113.40\n= ' + value
                     + '\nПРИЗНАК СПОСОБА РАСЧЕТА ПОЛНЫЙ РАСЧЕТ\nПОДАКЦИЗНЫЙ ТОВАР\nМера кол-ва предмета расчета шт.\n')
        text += 'ИТОГ 4433.44\nБезналичными 4433.44\nСУММА НДС 22% 799.47\nВаша скидка составила, руб. = 5282.18\nбаллов 290.00'
        result = receipts.parse_receipt_text(text)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['date'], '2026-10-11')
        self.assertEqual(result['merchant'], 'ООО "ТЕСТ"')
        self.assertEqual(result['total'], 4433.44)
        self.assertTrue(result['reconciled'])
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(len(result['items']), 7)
        self.assertEqual([item['amount'] for item in result['items']], [float(value) for value in values])
        self.assertEqual(result['items'][1]['name'], result['items'][2]['name'])
        self.assertTrue(all(item['is_wine'] and item['quantity'] == 1 for item in result['items']))
        self.assertTrue(all(item['volume_ml'] == 750 and item['vintage'] is None for item in result['items']))
        self.assertTrue(all(item['volume_source'] == 'name_decimal_suffix' and item['volume_requires_confirmation'] for item in result['items']))
        self.assertIn('из названия без единиц', ' '.join(result['warnings']))

    def test_inline_and_cyrillic_multipliers_keep_nonwine_in_expenses(self):
        text = ('Дата: 10.10.2026\nВино Test Reserve 2020 750 мл 2 х 1 000,00 = 2 000,00\n'
                'Молоко 1 л\n1 * 99,90\n= 99,90\nИТОГО: 2 099,90 ₽')
        result = receipts.parse_receipt_text(text)
        self.assertTrue(result['reconciled'])
        self.assertEqual(len(result['items']), 2)
        wine, milk = result['items']
        self.assertEqual((wine['quantity'], wine['unit_price'], wine['amount']), (2, 1000, 2000))
        self.assertEqual((wine['vintage'], wine['volume_ml'], wine['winery']), (2020, 750, ''))
        self.assertFalse(milk['is_wine'])
        self.assertEqual(milk['amount'], 99.90)

    def test_line_discount_uses_final_cost_per_unit_without_double_subtraction(self):
        for final in ('\n= 180.00', ''):
            result = receipts.parse_receipt_text('Вино Тест 750 мл\n2 x 100.00\nСкидка 20.00' + final + '\nИТОГ 180.00')
            item = result['items'][0]
            self.assertEqual((item['amount'], item['unit_price'], item['quantity']), (180, 90, 2))
            self.assertTrue(result['reconciled'])

    def test_total_on_next_line_and_decimal_quantity(self):
        result = receipts.parse_receipt_text('Дата 01.10.26\nЯблоки\n0,750 кг x 200,00\n= 150,00\nК ОПЛАТЕ\n150,00')
        self.assertEqual(result['total'], 150)
        self.assertEqual(result['items'][0]['quantity'], .75)
        self.assertEqual(result['items'][0]['unit_price'], 200)
        self.assertFalse(result['items'][0]['is_wine'])
        self.assertTrue(result['reconciled'])

    def test_common_decimal_suffix_suggests_size_but_not_producer_or_vintage(self):
        result = receipts.parse_receipt_text('Вино Мерло кр.сух.0,75\n1 x 100.00\nИТОГ 100.00')
        item = result['items'][0]
        self.assertIsNone(item['vintage'])
        self.assertEqual((item['volume_ml'], item['volume_source'], item['volume_requires_confirmation']),
                         (750, 'name_decimal_suffix', True))
        self.assertEqual(item['winery'], '')
        self.assertIn('вручную', ' '.join(result['warnings']))
        self.assertIn('проверьте объём', ' '.join(result['warnings']))

    def test_supported_decimal_suffixes_include_abbreviations_and_comma(self):
        for suffix, expected in [('кр.сух.0.75', 750), ('0,750', 750), ('(0.375)', 375),
                                 ('0.187', 187), ('0,2', 200), ('0.25', 250), ('0.50', 500),
                                 ('1,5', 1500), ('1.50', 1500), ('13,5% 0,75', 750)]:
            with self.subTest(suffix=suffix):
                volume = receipts.receipt_volume('Вино Мерло '+suffix)
                self.assertEqual(volume, {'volume_ml': expected, 'volume_source': 'name_decimal_suffix',
                                          'volume_requires_confirmation': True})

    def test_explicit_volume_units_are_exact_and_keep_known_vintage(self):
        for suffix, expected in [('750 мл', 750), ('750ml', 750), ('0,75 л', 750), ('0.75L', 750),
                                 ('1 литр', 1000), ('1,5 литра', 1500), ('3 литров', 3000),
                                 ('75cl', 750), ('0.375л', 375), ('750 мл / 0,75 л', 750)]:
            with self.subTest(suffix=suffix):
                result = receipts.parse_receipt_text('Вино Мерло 2020 '+suffix+'\n1 x 628.86\nИТОГ 628.86')
                item = result['items'][0]
                self.assertEqual(item['volume_ml'], expected)
                self.assertEqual(item['volume_source'], 'name_explicit_unit')
                self.assertFalse(item['volume_requires_confirmation'])
                self.assertEqual((item['vintage'], item['winery'], item['amount'], result['total']), (2020, '', 628.86, 628.86))
                self.assertTrue(result['reconciled'])

    def test_year_price_percent_count_and_ambiguous_sizes_are_not_volumes(self):
        names = ['Вино Мерло 2020', 'Вино Мерло 750', 'Вино Мерло 1', 'Вино Мерло 628.86',
                 'Вино Мерло 14.75', 'Вино Мерло 0.75%', 'Вино Мерло цена: 0.75',
                 'Вино Мерло цена 0,75 ₽', 'Вино Мерло количество 0.75', 'Вино Мерло 0.75 шт',
                 'Вино Мерло арт.0.75', 'Вино Мерло №0.75', 'Вино Мерло 0.75 кг',
                 'Вино Мерло 0.75 0.5', 'Вино Мерло 2019.0.75', 'Вино Мерло 0.73',
                 'Вино Мерло -0.75л', 'Вино Мерло 0.7505л', 'Вино Мерло 0.75л 1.5л',
                 'Вино Мерло 750мл 1.5', 'Вино Мерло 999999мл', 'Вино Мерло 0л']
        for name in names:
            with self.subTest(name=name):
                volume = receipts.receipt_volume(name)
                self.assertEqual(volume, {'volume_ml': None, 'volume_source': None,
                                          'volume_requires_confirmation': False})
        self.assertIsNone(receipts.receipt_volume('Соус Wine 0.75', is_wine=False)['volume_ml'])
        self.assertIsNone(receipts.receipt_volume(None)['volume_ml'])
        self.assertIsNone(receipts.receipt_volume('a'*501)['volume_ml'])

    def test_explicit_volume_is_not_mistaken_for_vintage(self):
        item = receipts.parse_receipt_text('Вино Мерло 2020 мл\n1 x 100\nИТОГ 100')['items'][0]
        self.assertEqual(item['volume_ml'], 2020)
        self.assertIsNone(item['vintage'])
        self.assertEqual(item['winery'], '')

    def test_accessory_and_vinegar_are_not_wine(self):
        for name in ('Штопор Wine Special', 'Соус Wine', 'Уксус винный', 'Бокал Wine'):
            with self.subTest(name=name):
                item = receipts.parse_receipt_text(name + '\n1 x 100.00\nИТОГ 100.00')['items'][0]
                self.assertFalse(item['is_wine'])

    def test_missing_or_conflicting_totals_are_explicitly_incomplete(self):
        for suffix in ('', '\nИТОГ 100.00\nИТОГО 101.00'):
            result = receipts.parse_receipt_text('Вино Тест\n1 x 100.00' + suffix)
            self.assertEqual(result['status'], 'candidate')
            self.assertIsNone(result['total'])
            self.assertFalse(result['reconciled'])
        result = receipts.parse_receipt_text('Вино Тест\n1 x 100.00\nИТОГ 150.00')
        self.assertFalse(result['reconciled'])
        self.assertIn('не совпала', ' '.join(result['warnings']))

    def test_refunds_foreign_currency_and_unreadable_text_never_become_new_expenses(self):
        for text in ('Кассовый чек / Возврат прихода\nИТОГ 100.00', 'TOTAL USD 100.00', '', None):
            with self.subTest(text=text):
                result = receipts.parse_receipt_text(text)
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['items'], [])

    def test_fiscal_fingerprint_is_stable_across_text_and_qr_spelling_and_hides_ids(self):
        identifiers = 'ФН 0001111222233334444\nНомер ФД: #00123\nФПД 0000987654'
        one = receipts.parse_receipt_text(identifiers + '\nИТОГ 100.00')
        two = receipts.parse_receipt_text('t=20261011T1230&s=100.00&fn=0001111222233334444&i=00123&fp=0000987654&n=1\nИТОГ 100,00')
        self.assertEqual(one['fingerprint'], two['fingerprint'])
        self.assertRegex(one['fingerprint'], r'^[0-9a-f]{64}$')
        self.assertNotIn('1111222233334444', json.dumps(one))
        other = receipts.parse_receipt_text(identifiers.replace('#00123', '#00124') + '\nИТОГ 100.00')
        self.assertNotEqual(one['fingerprint'], other['fingerprint'])

    def test_header_private_fields_are_not_returned_or_added_to_product_name(self):
        text = ('ООО "ТЕСТ"\nПокупатель: private-person@example.invalid\nТелефон +79990000000\n'
                'Номер карты: 1234567890123456\nВино Тест\n1 x 100.00\nИТОГ 100.00')
        result = receipts.parse_receipt_text(text)
        self.assertEqual(result['items'][0]['name'], 'Вино Тест')
        for private in ('private-person', '79990000000', '1234567890123456'):
            self.assertNotIn(private, json.dumps(result))
        self.assertNotIn('raw_text', result)

    def test_text_item_count_and_numeric_bounds_fail_closed(self):
        self.assertEqual(receipts.parse_receipt_text('x' * (receipts.MAX_TEXT + 1))['status'], 'unavailable')
        text = ''.join('Товар ' + str(index) + '\n1 x 100.00\n' for index in range(201)) + 'ИТОГ 20100.00'
        self.assertEqual(receipts.parse_receipt_text(text)['status'], 'unavailable')
        for quantity in ('0', '100001'):
            result = receipts.parse_receipt_text('Вино Тест\n' + quantity + ' x 100.00\nИТОГ 100.00')
            self.assertEqual(result['items'], [])


class ReceiptRecognitionTests(unittest.TestCase):
    text = 'Дата 11.10.26\nВино Тест 2020 750 мл\n1 x 100.00\nИТОГ 100.00'

    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / '.test-tmp'
        parent.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        temp = patch.object(receipts.tempfile, 'tempdir', str(self.root))
        temp.start()
        self.addCleanup(temp.stop)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))

    def photo(self, name='receipt.png'):
        path = self.root / name
        Image.new('RGB', (120, 100), 'white').save(path)
        return path

    def test_dry_run_never_reads_file_or_calls_ocr_or_pdf(self):
        with patch.object(receipts, '_image') as image, patch.object(receipts, '_ocr_text') as ocr, patch.object(receipts, '_pdf_text') as pdf:
            result = receipts.recognize_receipt('nonexistent', dry_run=True)
        self.assertEqual(result['status'], 'dry_run')
        image.assert_not_called()
        ocr.assert_not_called()
        pdf.assert_not_called()

    def test_photo_uses_local_text_ocr_and_byte_fingerprint_without_fiscal_ids(self):
        path = self.photo()
        with patch.object(receipts, '_ocr_text', return_value=(self.text, 'windows_ocr')) as ocr:
            result = receipts.recognize_receipt(path)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['provider'], 'windows_ocr')
        self.assertEqual(result['fingerprint'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['items'][0]['amount'], 100)
        ocr.assert_called_once()

    def test_unavailable_ocr_and_file_errors_do_not_leak_paths_or_private_text(self):
        with patch.object(receipts, '_ocr_text', side_effect=ValueError('private customer /hidden/path')):
            result = receipts.recognize_receipt(self.photo())
        self.assertEqual(result['status'], 'unavailable')
        self.assertNotIn('private customer', str(result))
        self.assertEqual(receipts.recognize_receipt(self.root / 'missing')['status'], 'unavailable')

    def test_file_size_is_checked_before_processing(self):
        path = self.root / 'oversized.pdf'
        with path.open('wb') as stream:
            stream.truncate(receipts.MAX_PDF_BYTES + 1)
        with patch.object(receipts, '_pdf_text') as pdf, patch.object(receipts, '_image') as image:
            self.assertEqual(receipts.recognize_receipt(path)['status'], 'unavailable')
        pdf.assert_not_called()
        image.assert_not_called()

    def test_ocr_subprocess_has_no_shell_uses_local_russian_language_and_bounded_timeout(self):
        process = SimpleNamespace(returncode=0, stdout=self.text)
        with patch.object(receipts.shutil, 'which', side_effect=lambda name: '/local/tesseract' if name == 'tesseract' else None), patch.object(receipts.subprocess, 'run', return_value=process) as run:
            text, provider = receipts._ocr_text(b'local photo bytes')
        self.assertEqual((text, provider), (self.text, 'tesseract'))
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], '/local/tesseract')
        self.assertIn('rus+eng', argv)
        self.assertFalse(run.call_args.kwargs['shell'])
        self.assertEqual(run.call_args.kwargs['timeout'], 45)

    def test_windows_text_ocr_fallback_does_not_apply_wine_label_field_filter(self):
        process = SimpleNamespace(returncode=0, stdout=self.text)
        with patch.object(receipts.sys, 'platform', 'win32'), patch.object(receipts.shutil, 'which', side_effect=lambda name: 'powershell.exe' if name == 'powershell.exe' else None), patch.object(receipts.subprocess, 'run', return_value=process) as run:
            text, provider = receipts._ocr_text(b'local photo bytes')
        self.assertEqual(provider, 'windows_ocr')
        self.assertIn('ИТОГ 100.00', text)
        self.assertIn('-File', run.call_args.args[0])
        self.assertIn('-LanguageTag', run.call_args.args[0])
        self.assertIn('ru-RU', run.call_args.args[0])
        self.assertNotIn('secret', str(run.call_args))

    def test_missing_russian_ocr_explains_local_backend_and_manual_fallback(self):
        process = SimpleNamespace(returncode=1, stdout='')
        with patch.object(receipts.sys, 'platform', 'win32'), patch.object(receipts.shutil, 'which', side_effect=lambda name: 'powershell.exe' if name == 'powershell.exe' else None), patch.object(receipts.subprocess, 'run', return_value=process):
            result = receipts.recognize_receipt(self.photo())
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('русский язык OCR Windows', str(result['warnings']))
        self.assertIn('вручную', str(result['warnings']))

    def test_ocr_zero_correction_only_touches_money_fractions_and_warns(self):
        text = 'Дата 11.10.26\nВино Тест OOO 2020 750 мл\n2 х 100.oo\n= 200.Oо\nИТОГ 200.00'
        with patch.object(receipts, '_ocr_text', return_value=(text, 'windows_ocr')):
            result = receipts.recognize_receipt(self.photo())
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['items'][0]['amount'], 200)
        self.assertIn('OOO', result['items'][0]['name'])
        self.assertTrue(result['reconciled'])
        self.assertIn('требуют проверки', str(result['warnings']))
        self.assertEqual(receipts._normalize_ocr_money('Wine 200.oo\nФД 200.oo\n1O0.00'), 'Wine 200.oo\nФД 200.oo\n1O0.00')

    def test_ocr_zero_correction_supports_split_total_without_guessing_integer_digits(self):
        self.assertEqual(receipts._normalize_ocr_money('ИТОГО\n250.oo\n3 х 1O0.00'), 'ИТОГО\n250.00\n3 х 1O0.00')

    def test_ocr_output_and_language_are_bounded(self):
        with patch.dict('os.environ', {'RECEIPT_OCR_LANG': 'rus;download'}), self.assertRaises(ValueError):
            receipts._ocr_text(b'image')
        process = SimpleNamespace(returncode=0, stdout='x' * (receipts.MAX_TEXT + 1))
        with patch.object(receipts.shutil, 'which', side_effect=lambda name: '/local/tesseract' if name == 'tesseract' else None), patch.object(receipts.subprocess, 'run', return_value=process), self.assertRaises(ValueError):
            receipts._ocr_text(b'image')


@unittest.skipUnless(importlib.util.find_spec('pypdf'), 'pypdf>=6.10 is required for actual PDF fixtures')
class ReceiptPDFTests(ReceiptRecognitionTests):
    def pdf(self, *, password=None, pages=1, active=False):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        for _ in range(pages):
            page = writer.add_blank_page(width=300, height=600)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                                     NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            content = DecodedStreamObject()
            content.set_data(b'BT /F1 12 Tf 10 550 Td (Wine Test 2020 750 ml) Tj 0 -20 Td (1 x 99.00) Tj 0 -20 Td (= 99.00) Tj ET')
            page[NameObject('/Contents')] = writer._add_object(content)
        if password is not None:
            writer.encrypt(user_password=password, owner_password='synthetic-owner')
        if active:
            writer.add_js("fetch('https://invalid.example/private')")
            writer.add_attachment('private.txt', b'PRIVATE ATTACHMENT NOT RECEIPT TEXT')
        stream = io.BytesIO()
        writer.write(stream)
        path = self.root / 'synthetic.pdf'
        path.write_bytes(stream.getvalue())
        return path

    def test_real_pdf_embedded_text_ignores_javascript_and_attachments(self):
        with patch.object(receipts, '_ocr_text') as ocr, patch.object(receipts, '_render_page') as render:
            result = receipts.recognize_receipt(self.pdf(active=True))
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['provider'], 'pypdf')
        self.assertEqual(len(result['items']), 1)
        self.assertEqual(result['items'][0]['amount'], 99)
        self.assertNotIn('PRIVATE ATTACHMENT', str(result))
        self.assertNotIn('invalid.example', str(result))
        ocr.assert_not_called()
        render.assert_not_called()

    def test_owner_editing_restriction_is_readable_but_user_password_is_rejected(self):
        result = receipts.recognize_receipt(self.pdf(password=''))
        self.assertEqual(result['status'], 'candidate')
        self.assertIn('без пароля', ' '.join(result['warnings']))
        result = receipts.recognize_receipt(self.pdf(password='user-password'))
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('защищён паролем', ' '.join(result['warnings']))
        self.assertNotIn('user-password', str(result))

    def test_page_limit_is_checked_before_ocr_and_stream_limits_are_restored(self):
        from pypdf import filters
        old_limit = filters.ZLIB_MAX_OUTPUT_LENGTH
        with patch.object(receipts, '_ocr_text') as ocr:
            result = receipts.recognize_receipt(self.pdf(pages=11))
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('10 страниц', ' '.join(result['warnings']))
        self.assertEqual(filters.ZLIB_MAX_OUTPUT_LENGTH, old_limit)
        ocr.assert_not_called()

    def test_image_pixel_bomb_is_rejected_before_extraction(self):
        from pypdf.generic import DictionaryObject
        image = DictionaryObject({'/Subtype': '/Image', '/Width': 10000, '/Height': 10000})
        page = Mock()
        page.get.return_value = DictionaryObject({'/XObject': DictionaryObject({'/Im1': image})})
        reader = SimpleNamespace(is_encrypted=False, pages=[page])
        with patch('pypdf.PdfReader', return_value=reader), self.assertRaises(ValueError):
            receipts._pdf_text(b'%PDF-synthetic')
        page.extract_text.assert_not_called()

    def test_scanned_page_is_rendered_once_and_read_locally(self):
        page = Mock()
        page.get.return_value = {}
        page.extract_text.return_value = ''
        reader = SimpleNamespace(is_encrypted=False, pages=[page])
        with patch('pypdf.PdfReader', return_value=reader), patch.object(receipts, '_render_page', return_value=b'rendered local image') as render, patch.object(receipts, '_ocr_text', return_value=(self.text, 'windows_ocr')):
            text, provider, _ = receipts._pdf_text(b'%PDF-synthetic')
        self.assertEqual((text, provider), (self.text, 'windows_ocr'))
        render.assert_called_once_with(b'%PDF-synthetic', 0)

    def test_malformed_pdf_is_explicitly_unavailable(self):
        path = self.root / 'bad.pdf'
        path.write_bytes(b'%PDF-invalid document')
        result = receipts.recognize_receipt(path)
        self.assertEqual(result['status'], 'unavailable')

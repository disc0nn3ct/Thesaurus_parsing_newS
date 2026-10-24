"""Useful request history, redaction, file metadata, task linkage and admin views."""
import csv
import io
import json
import unittest
from types import SimpleNamespace

import test_subscription_access as fixtures

from thesaurus import monitoring
from thesaurus.monitor_content import redact_text, request_details, selected_parameters
from thesaurus.telegram_monitor import MonitoringControls
from thesaurus.telegram_navigation import interaction


class MonitorContentTests(unittest.TestCase):
    setUp = fixtures.SubscriptionAccessTests.setUp
    message = fixtures.SubscriptionAccessTests.message
    view = fixtures.SubscriptionAccessTests.view

    def test_text_and_response_are_saved_with_form_context(self):
        self.bot.market_controls.pending(1,'add_user')
        self.bot._reply(self.client,1,'Введите @username нового пользователя.')
        self.client.get_users.return_value=SimpleNamespace(id=99,first_name='Иван',username='ivan_test')
        self.message('@ivan_test',1)
        row=self.store.one("SELECT * FROM monitor_events WHERE category='interaction' ORDER BY id DESC LIMIT 1")
        details=json.loads(row['details'])
        self.assertEqual(details['request']['text'],'@ivan_test')
        self.assertIn('Введите @username',details['request']['form_prompt'])
        self.assertIn('Доступ разрешён',details['response']['text'])
        self.assertEqual(details['request']['message_id'],99)

    def test_redacts_tokens_fiscal_numbers_authorization_and_url_credentials(self):
        bot='123456789:'+('A'*33)
        token='sk-proj-'+('a'*36)
        text=(f'/sources https://api.telegram.org/bot{bot}/getMe\n{token}\n'
              'api_hash=abcdef0123456789abcdef0123456789\n'
              'Authorization: Bearer someBearerSecret\n'
              'пароль = verySecretPassword\n'
              'ФН: 1234567890123456\n'
              'https://alice:privatepass@example.org/wine?access_token=privatevalue')
        clean=redact_text(text)
        for secret in (bot,token,'abcdef0123456789abcdef0123456789','someBearerSecret','verySecretPassword',
                       '1234567890123456','privatepass','privatevalue'):
            self.assertNotIn(secret,clean)
        self.assertIn('https://example.org/wine',clean)
        self.assertIn('[скрыто]',clean)
        json_secret = 'json-secret-value'
        for text in (json.dumps({'api_key':json_secret}),str({'password':json_secret}),'token='+json_secret,
                     'фискальный документ: 123'):
            with self.subTest(text=text):
                self.assertNotIn('json-secret-value',redact_text(text))
                if 'фискальный' in text:
                    self.assertNotIn('123',redact_text(text))

    def test_secret_form_is_hidden_and_long_text_is_bounded(self):
        request=request_details('handle_message',SimpleNamespace(text='unlabelled-secret',id=3),pending='settings:api_hash')
        self.assertEqual(request['text'],'[Секретный ввод скрыт]')
        self.assertNotIn('unlabelled-secret',str(request))
        self.assertLess(len(redact_text('a'*15000)),4050)

    def test_file_caption_and_dimensions_without_bytes_ids_or_full_paths(self):
        document=SimpleNamespace(file_name=r'C:\private\receipt.pdf',file_size=2048,mime_type='application/pdf',
                                 file_id='private-download-token',file_unique_id='private-id')
        message=SimpleNamespace(id=20,document=document,caption='Покупка вина 12 октября')
        request=request_details('handle_document',message)
        self.assertEqual(request['kind'],'document')
        self.assertEqual(request['file_name'],'receipt.pdf')
        self.assertEqual(request['file_size'],2048)
        self.assertEqual(request['caption'],'Покупка вина 12 октября')
        self.assertNotIn('private',str(request))
        photo=SimpleNamespace(file_size=300,width=1200,height=800)
        request=request_details('handle_wine_photo',SimpleNamespace(photo=photo))
        self.assertEqual((request['width'],request['height']),(1200,800))

    def test_real_clicked_label_or_matching_menu_cache_is_used(self):
        button=SimpleNamespace(text='Прогноз USD на год',callback_data='usd:horizon:12')
        message=SimpleNamespace(id=30,reply_markup=SimpleNamespace(inline_keyboard=[[button]]))
        event=SimpleNamespace(data='usd:horizon:12',message=message)
        self.assertEqual(request_details('',event,callback=True)['button_text'],'Прогноз USD на год')
        event.message=SimpleNamespace(id=40)
        record={'message_id':40,'button_labels':{'usd:horizon:12':'Годовой анализ'}}
        self.assertEqual(request_details('',event,callback=True,record=record)['button_text'],'Годовой анализ')
        record['message_id']=41
        self.assertNotIn('button_text',request_details('',event,callback=True,record=record))

    def test_request_links_queued_task_and_card_shows_current_result_and_parameters(self):
        @interaction()
        def handle_message(bot,client,event):
            job=bot.store.enqueue('dollar_analysis',{'actor_id':2,'config':{
                'forecast_date':'2027-10-01','quote':85.5,'api_key':'hidden-'+'payload-key'}})
            bot._reply(client,2,f'Задание №{job} в очереди.')
        event=SimpleNamespace(text='/dollar',id=91,from_user=SimpleNamespace(id=2),chat=SimpleNamespace(id=2,type='private'))
        handle_message(self.bot,self.client,event)
        row=self.store.one("SELECT * FROM monitor_events WHERE category='interaction' ORDER BY id DESC LIMIT 1")
        details=json.loads(row['details'])
        self.assertEqual(len(details['job_ids']),1)
        job=details['job_ids'][0]
        self.store.execute("UPDATE jobs SET state='done' WHERE id=?",(job,))
        MonitoringControls(self.store,self.bot._reply).event(self.client,1,row['id'])
        text=self.view()[0]
        self.assertIn('/dollar',text)
        self.assertIn('Текущий результат: готово',text)
        self.assertIn('2027-10-01',text)
        self.assertIn('85.5',text)
        self.assertNotIn('hidden-payload-key',text)

    def test_existing_task_reference_never_links_another_users_task(self):
        own=self.store.enqueue('dollar_analysis',{'actor_id':2})
        other=self.store.enqueue('dollar_analysis',{'actor_id':3})
        @interaction()
        def handle_message(bot,client,event):
            bot._reply(client,2,f'Задание №{own} уже в очереди; задание №{other}.')
        handle_message(self.bot,self.client,SimpleNamespace(text='/dollar',id=10,
            from_user=SimpleNamespace(id=2),chat=SimpleNamespace(id=2,type='private')))
        details=json.loads(self.store.one("SELECT details FROM monitor_events WHERE category='interaction' ORDER BY id DESC LIMIT 1")['details'])
        self.assertEqual(details['job_ids'],[own])

    def test_unsupported_voice_has_details_and_explains_result(self):
        message=SimpleNamespace(id=15,voice=SimpleNamespace(file_size=3000,duration=7),
            from_user=SimpleNamespace(id=2),chat=SimpleNamespace(id=2,type='private'))
        self.bot.handle_other_message(self.client,message)
        row=self.store.one("SELECT * FROM monitor_events WHERE category='interaction' ORDER BY id DESC LIMIT 1")
        self.assertEqual(row['outcome'],'unsupported')
        self.assertEqual(row['action'],'upload:voice')
        details=json.loads(row['details'])
        self.assertEqual(details['request']['duration'],7)
        self.assertIn('пока не обрабатывается',details['response']['text'])

    def test_event_detail_is_admin_only_and_csv_contains_redacted_content(self):
        identity=monitoring.safe_record(self.store,'interaction','input:form',actor=2,subject=2,
            details={'request':{'kind':'text','text':redact_text('=SUM(1;2)\npassword=private-pass')},
                     'response':{'text':'Принято'}})
        controls=MonitoringControls(self.store,self.bot._reply)
        with self.assertRaises(PermissionError):
            controls.event(self.client,2,identity)
        controls.feed(self.client,1,'u2',0)
        self.assertIn(f'mon:event:{identity}',self.view()[1])
        controls.export(self.client,1,'u2',0,'all')
        file=self.client.send_document.call_args.args[1]
        content=file.getvalue().decode('utf-8-sig')
        self.assertNotIn('private-pass',content)
        rows=list(csv.DictReader(io.StringIO(content),delimiter=';'))
        self.assertTrue(rows[0]['Текст запроса'].startswith("'=SUM"))
        self.assertEqual(rows[0]['Ответ бота'],'Принято')

    def test_job_parameters_exclude_private_internal_data(self):
        result=selected_parameters({'actor_id':2,'path':'private-file.pdf','file_id':'secret-id',
            'raw_text':'fiscal-data','config':{'forecast_date':'2027-01-01','bank_token':'secret-token',
                'quote':80.5,'selection':['usd','cny']}})
        self.assertNotIn('private',str(result))
        self.assertNotIn('secret',str(result))
        self.assertNotIn('fiscal',str(result))
        self.assertEqual(result['config']['quote'],80.5)

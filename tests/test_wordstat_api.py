"""Offline tests; no credentials, dotenv file or HTTP requests are used."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_api():
    path = Path(__file__).resolve().parents[1] / "functions" / "wordstat_api.py"
    spec = importlib.util.spec_from_file_location("wordstat_api_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    module.WORDSTAT_OAUTH = "test-oauth"
    module.WORDSTAT_CLIENT_ID = "test-client"
    return module


class WordstatApiTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'WORDSTAT_BACKEND': 'legacy'}))
        self.api = load_api()
        self.sleep = self.enterContext(patch.object(self.api.time, "sleep"))
        self.enterContext(patch.object(self.api.requests, "post"))

    def response(self, status, payload=None):
        return Mock(status_code=status, json=Mock(return_value=payload))

    def test_network_failures_retry_and_recover(self):
        self.api.requests.post.side_effect = [
            self.api.requests.exceptions.Timeout("test-oauth"),
            self.api.requests.exceptions.ConnectionError("test-oauth"),
            self.response(200, {"dynamics": []}),
        ]
        self.assertEqual(self.api._wordstat_post("/v1/dynamics", {}), {"dynamics": []})
        self.assertEqual(self.api.requests.post.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_transient_http_statuses_retry(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.api.requests.post.reset_mock()
                self.sleep.reset_mock()
                self.api.requests.post.side_effect = [self.response(status), self.response(200, [])]
                self.assertEqual(self.api._wordstat_post("/v1/getRegionsTree", {}), [])
                self.assertEqual(self.api.requests.post.call_count, 2)
                self.sleep.assert_called_once()

    def test_no_sleep_after_final_attempt(self):
        self.api.requests.post.return_value = self.response(503)
        with self.assertRaisesRegex(RuntimeError, "после 2 попыток"):
            self.api._wordstat_post("/v1/dynamics", {}, retries=2)
        self.assertEqual(self.api.requests.post.call_count, 2)
        self.sleep.assert_called_once_with(1.5)

    def test_permanent_failure_does_not_disclose_response(self):
        response = self.response(401, {"error": "test-oauth"})
        self.api.requests.post.return_value = response
        with self.assertRaises(RuntimeError) as caught:
            self.api._wordstat_post("/v1/dynamics", {})
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-oauth", str(caught.exception))
        response.json.assert_not_called()
        self.sleep.assert_not_called()
        self.assertEqual(self.api.requests.post.call_count, 1)

    def test_invalid_json_has_safe_error(self):
        self.api.requests.post.return_value = Mock(status_code=200, json=Mock(side_effect=ValueError("test-oauth")))
        with self.assertRaisesRegex(RuntimeError, "некорректный JSON") as caught:
            self.api._wordstat_post("/v1/dynamics", {})
        self.assertNotIn("test-oauth", str(caught.exception))

    def test_error_payload_on_http_success_is_not_returned(self):
        self.api.requests.post.return_value = self.response(200, {"error": "test-oauth"})
        with self.assertRaisesRegex(RuntimeError, "API вернул ошибку") as caught:
            self.api._wordstat_post("/v1/dynamics", {})
        self.assertNotIn("test-oauth", str(caught.exception))

    def test_request_does_not_follow_redirects(self):
        self.api.requests.post.return_value = self.response(200, {})
        payload = {"phrase": "пожар"}
        self.api._wordstat_post("/v1/dynamics", payload)
        kwargs = self.api.requests.post.call_args.kwargs
        self.assertEqual(kwargs["json"], payload)
        self.assertEqual(kwargs["timeout"], 30)
        self.assertFalse(kwargs["allow_redirects"])

    def test_invalid_attempt_count_makes_no_request(self):
        with self.assertRaises(ValueError):
            self.api._wordstat_post("/v1/dynamics", {}, retries=0)
        self.api.requests.post.assert_not_called()

    def test_access_failures_have_actionable_safe_codes(self):
        for status, hint in [(401, 'OAuth'), (403, 'Client ID'), (404, 'интерфейс'), (429, 'позже')]:
            with self.subTest(status=status):
                self.api.requests.post.return_value = self.response(status, {'error': 'test-oauth'})
                with self.assertRaises(self.api.WordstatError) as caught:
                    self.api._wordstat_post('/v1/dynamics', {}, retries=1)
                self.assertIn(str(status), caught.exception.safe_message())
                self.assertIn(hint, caught.exception.safe_message())
                self.assertNotIn('test-oauth', str(caught.exception))

    def test_missing_configuration_does_not_reserve_budget_or_call_network(self):
        self.api.WORDSTAT_OAUTH = None
        reserve = Mock()
        with self.assertRaises(self.api.WordstatError) as caught:
            self.api._wordstat_post('/v1/dynamics', {}, before_attempt=reserve)
        self.assertIn('WORDSTAT_OAUTH', caught.exception.safe_message())
        reserve.assert_not_called()
        self.api.requests.post.assert_not_called()

    def test_certificate_failure_is_not_retried_and_never_disables_verification(self):
        self.api.requests.post.side_effect = self.api.requests.exceptions.SSLError('test-oauth')
        reserve = Mock()
        with self.assertRaises(self.api.WordstatError) as caught:
            self.api._wordstat_post('/v1/dynamics', {}, before_attempt=reserve)
        self.assertEqual(caught.exception.kind, 'tls')
        self.assertTrue(caught.exception.stops_collection)
        self.assertIn('SSL/TLS', str(caught.exception))
        self.assertNotIn('test-oauth', str(caught.exception))
        reserve.assert_called_once()
        self.api.requests.post.assert_called_once()
        self.assertNotEqual(self.api.requests.post.call_args.kwargs.get('verify'), False)
        self.sleep.assert_not_called()

    def test_exhausted_network_retry_has_safe_error_and_aborts_collection(self):
        self.api.requests.post.side_effect = self.api.requests.exceptions.Timeout('test-oauth')
        with self.assertRaises(self.api.WordstatError) as caught:
            self.api._wordstat_post('/v1/dynamics', {}, retries=2)
        self.assertEqual(caught.exception.kind, 'network')
        self.assertTrue(caught.exception.stops_collection)
        self.assertNotIn('test-oauth', str(caught.exception))
        self.assertEqual(self.api.requests.post.call_count, 2)

    def cloud_environment(self, **overrides):
        values = {'WORDSTAT_BACKEND': 'cloud', 'WORDSTAT_API_KEY': 'test-api-key',
                  'WORDSTAT_IAM_TOKEN': '', 'WORDSTAT_FOLDER_ID': 'test-folder'}
        values.update(overrides)
        self.enterContext(patch.dict(os.environ, values))

    def test_cloud_dynamics_converts_request_response_and_keeps_budget_hook(self):
        self.cloud_environment()
        self.api.requests.post.return_value = self.response(200, {'results': [
            {'date': '2026-09-21T00:00:00Z', 'count': '120', 'share': '0.002'}]})
        payload = {'phrase': 'рост цен', 'period': 'weekly', 'fromDate': '2026-09-21',
                   'toDate': '2026-09-27', 'regions': [225], 'devices': ['all']}
        reserve = Mock()
        result = self.api._wordstat_post('/v1/dynamics', payload, before_attempt=reserve)
        self.assertEqual(result['dynamics'][0]['count'], 120)
        call = self.api.requests.post.call_args
        self.assertEqual(call.args[0], 'https://searchapi.api.cloud.yandex.net/v2/wordstat/dynamics')
        self.assertEqual(call.kwargs['headers']['Authorization'], 'Api-key test-api-key')
        self.assertNotIn('X-Client-Id', call.kwargs['headers'])
        self.assertEqual(call.kwargs['json'], {'phrase': 'рост цен', 'period': 'PERIOD_WEEKLY',
            'fromDate': '2026-09-21T00:00:00Z', 'toDate': '2026-09-27T00:00:00Z',
            'regions': ['225'], 'devices': ['DEVICE_ALL'], 'folderId': 'test-folder'})
        self.assertFalse(call.kwargs['allow_redirects'])
        self.assertEqual(payload['period'], 'weekly')
        reserve.assert_called_once()

    def test_cloud_iam_regions_and_nested_tree(self):
        self.cloud_environment(WORDSTAT_API_KEY='', WORDSTAT_IAM_TOKEN='test-iam')
        self.api.requests.post.return_value = self.response(200, {'results': [{'region': '213', 'count': '9'}]})
        result = self.api._wordstat_post('/v1/regions', {'phrase': 'пожар', 'regionType': 'regions', 'devices': ['phone']})
        self.assertEqual(result['regions'][0]['regionId'], 213)
        self.assertEqual(self.api.requests.post.call_args.kwargs['json']['region'], 'REGION_REGIONS')
        self.assertEqual(self.api.requests.post.call_args.kwargs['headers']['Authorization'], 'Bearer test-iam')
        self.api.requests.post.return_value = self.response(200, {'regions': [{'id': '225', 'label': 'Россия',
            'children': [{'id': '213', 'label': 'Москва'}]}]})
        result = self.api._wordstat_post('/v1/getRegionsTree', {})
        self.assertEqual(result[0]['children'][0]['value'], 213)

    def test_cloud_configuration_failure_never_sends_old_oauth_to_cloud(self):
        for overrides in [{'WORDSTAT_FOLDER_ID': ''}, {'WORDSTAT_API_KEY': ''}, {'WORDSTAT_IAM_TOKEN': 'test-iam'}]:
            with self.subTest(overrides=overrides):
                self.cloud_environment(**overrides)
                reserve = Mock()
                with self.assertRaises(self.api.WordstatError):
                    self.api._wordstat_post('/v1/dynamics', {}, before_attempt=reserve)
                reserve.assert_not_called()
                self.api.requests.post.assert_not_called()

    def test_cloud_schema_change_is_detected_without_echoing_payload(self):
        self.cloud_environment()
        self.api.requests.post.return_value = self.response(200, {'results': [{'count': 'test-api-key'}]})
        with self.assertRaises(TypeError) as caught:
            self.api._wordstat_post('/v1/getRegionsTree', {})
        self.assertNotIn('test-api-key', str(caught.exception))


if __name__ == "__main__":
    unittest.main()

"""Optional local OCR and SSRF-safe exact retail observations, entirely offline."""
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from thesaurus import wine_tools as tools


class WineToolsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"WINE_PRICE_ALLOWED_HOSTS": "shop.example", "WINE_VISION_MODEL": ""}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        parent = Path(__file__).resolve().parents[1] / ".test-tmp"
        parent.mkdir(exist_ok=True)
        temp = patch.object(tools.tempfile, "tempdir", str(parent))
        temp.start()
        self.addCleanup(temp.stop)
        self.url = "https://shop.example/wine/test"
        self.expected = {"name": "Chateau Test", "winery": "Domaine Test", "vintage": 2020, "volume_ml": 750}

    def product(self, **changes):
        product = {"@context": "https://schema.org", "@type": "Product", "name": "Chateau Test 2020 750 ml",
                   "url": self.url, "brand": {"name": "Domaine Test"},
                   "offers": {"@type": "Offer", "price": "1599.90", "priceCurrency": "RUB",
                              "availability": "https://schema.org/InStock"}}
        product.update(changes)
        return product

    def html(self, product):
        return '<html><script type="application/ld+json">' + json.dumps(product) + '</script></html>'

    def image(self):
        parent = Path(__file__).resolve().parents[1] / ".test-tmp"
        parent.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "label.png"
        Image.new("RGB", (160, 100), "white").save(path)
        return path

    def test_photo_dry_run_never_reads_or_invokes_backend(self):
        with patch.object(tools, "_image") as image, patch.object(tools, "_ocr") as ocr:
            self.assertEqual(tools.recognize_label("missing", dry_run=True)["status"], "dry_run")
        image.assert_not_called()
        ocr.assert_not_called()

    def test_invalid_photo_never_calls_backend(self):
        path = self.image()
        path.write_text("not a photograph", encoding="utf-8")
        with patch.object(tools, "_ocr") as ocr:
            result = tools.recognize_label(path)
        self.assertEqual(result["status"], "invalid_image")
        ocr.assert_not_called()

    def test_photo_has_explicit_manual_fallback(self):
        with patch.object(tools, "_ocr", side_effect=ValueError), patch.object(tools, "_windows_ocr", side_effect=ValueError), patch.object(tools, "_local_vision", side_effect=ValueError):
            result = tools.recognize_label(self.image())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["candidate"], {})
        self.assertIn("вручную", result["warnings"][0])

    def test_ocr_draft_needs_confirmation_and_does_not_guess_winery(self):
        candidate, raw = tools._ocr_candidate("CHATEAU TEST\nMERLOT\n2020\n750 ml")
        self.assertEqual(candidate["name"], "CHATEAU TEST")
        self.assertEqual(candidate["vintage"], 2020)
        self.assertEqual(candidate["winery"], "")
        self.assertEqual(candidate["volume_ml"], None)
        with patch.object(tools, "_ocr", return_value=(candidate, raw)), patch.object(tools, "_local_vision") as vision:
            result = tools.recognize_label(self.image())
        self.assertEqual(result["status"], "candidate")
        self.assertTrue(result["requires_confirmation"])
        vision.assert_not_called()

    def test_two_years_are_ambiguous_not_guessed(self):
        self.assertIsNone(tools._ocr_candidate("WINERY 1890\nMERLOT 2020")[0]["vintage"])
        self.assertEqual(tools._ocr_candidate("WINERY FOUNDED 1890\nMERLOT 2020")[0]["vintage"], 2020)

    def test_tesseract_uses_argv_and_bounded_timeout(self):
        result = SimpleNamespace(returncode=0, stdout="CHATEAU TEST\n2020")
        with patch.object(tools.shutil, "which", return_value="/usr/bin/tesseract"), patch.object(tools.subprocess, "run", return_value=result) as run:
            candidate, _ = tools._ocr(b"synthetic image")
        self.assertEqual(candidate["vintage"], 2020)
        args, kwargs = run.call_args
        self.assertIsInstance(args[0], list)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["timeout"], 45)

    def test_windows_ocr_local_helper_uses_argv(self):
        result = SimpleNamespace(returncode=0, stdout="CHATEAU TEST\n2020")
        with patch.object(tools.sys, "platform", "win32"), patch.object(tools.shutil, "which", return_value="powershell.exe"), patch.object(tools.subprocess, "run", return_value=result) as run:
            candidate, _ = tools._windows_ocr(b"synthetic image")
        self.assertEqual(candidate["name"], "CHATEAU TEST")
        self.assertIn("-File", run.call_args.args[0])
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_vision_is_explicit_local_and_never_uses_text_model(self):
        with patch.dict("os.environ", {"WINE_VISION_MODEL": "", "OLLAMA_VISION_MODEL": "", "OLLAMA_MODEL": "text-only"}), patch.object(tools.requests, "post") as post, self.assertRaises(ValueError):
            tools._local_vision(b"image")
        post.assert_not_called()
        for url, model in (("https://remote.example", "vision"), ("http://127.0.0.1:11434", "vision:cloud"), ("http://localhost:11434/path", "vision")):
            with self.subTest(url=url, model=model), patch.dict("os.environ", {"OLLAMA_URL": url, "WINE_VISION_MODEL": model}), self.assertRaises(ValueError):
                tools._local_vision(b"image")

    def test_vision_output_has_bounds_and_loopback_transport(self):
        data = {"name": "Chateau Test", "winery": "Domaine Test", "vintage": 2020, "volume_ml": 750}
        response = Mock(status_code=200)
        response.iter_content.return_value = [json.dumps({"message": {"content": json.dumps(data)}}).encode()]
        with patch.dict("os.environ", {"OLLAMA_URL": "http://127.0.0.1:11434", "WINE_VISION_MODEL": "local-vision"}), patch.object(tools.requests, "post", return_value=response) as post:
            candidate, _ = tools._local_vision(b"image")
        self.assertEqual(candidate["volume_ml"], 750)
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_thread"], 4)
        self.assertEqual(post.call_args.kwargs["proxies"]["https"], "")
        response.close.assert_called_once()

    def test_malformed_or_oversized_vision_response_is_rejected(self):
        for chunks in ([b'{"message":[]}'], [b"x" * 65537]):
            response = Mock(status_code=200)
            response.iter_content.return_value = chunks
            with patch.dict("os.environ", {"OLLAMA_URL": "http://127.0.0.1:11434", "WINE_VISION_MODEL": "local-vision"}), patch.object(tools.requests, "post", return_value=response), self.assertRaises((TypeError, ValueError)):
                tools._local_vision(b"image")
            response.close.assert_called_once()

    def test_allowed_domains_accept_www_but_not_arbitrary_subdomains(self):
        self.assertEqual(tools.canonical_hosts("WWW.shop.example, other.example"), ["other.example", "shop.example"])
        self.assertEqual(tools.validate_product_url("https://www.shop.example/wine/test#section"), "https://www.shop.example/wine/test")
        with self.assertRaises(ValueError):
            tools.validate_product_url("https://evil.shop.example/wine")
        self.assertEqual(tools.validate_product_url("https://other.example/wine", allowed_hosts=["other.example"]), "https://other.example/wine")
        with self.assertRaises(ValueError):
            tools.validate_product_url(self.url, allowed_hosts=[])

    def test_private_credentials_ports_and_secret_parameters_are_rejected(self):
        values = ["http://shop.example/wine", "https://127.0.0.1/wine", "https://[::1]/wine",
                  "https://localhost/wine", "https://user:password@shop.example/wine", "https://shop.example:8443/wine",
                  "https://shop.example/wine?token=secret", "https://shop.example/wine?password=hidden",
                  "https://shop.example/wine?redirect=https://evil.example", "https://shop.example/wine%0d%0aHeader"]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tools.validate_product_url(value)
        self.assertEqual(tools.validate_product_url(self.url + "?sku=ABC-123"), self.url + "?sku=ABC-123")

    def test_host_settings_reject_urls_ips_and_wildcards(self):
        for value in ("https://shop.example", "127.0.0.1", "*.example", "shop.local", "bad..example"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tools.canonical_hosts(value)

    def test_dns_rejects_any_private_answer_even_when_another_is_public(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in ("8.8.8.8", "127.0.0.1")]
        with patch.object(tools.socket, "getaddrinfo", return_value=records), self.assertRaises(ValueError):
            tools._public_addresses("shop.example")

    def test_redirect_cannot_reach_private_or_non_allowlisted_host(self):
        response = Mock(status=302)
        response.getheader.return_value = "https://127.0.0.1/private"
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(tools, "_public_addresses", return_value=["8.8.8.8"]) as resolve, patch.object(tools, "_rate_limit"), patch.object(tools, "HTTPSConnection", return_value=connection), self.assertRaises(ValueError):
            tools._download(self.url)
        resolve.assert_called_once_with("shop.example")
        connection.close.assert_called_once()

    def test_fetch_pins_socket_address_while_preserving_tls_hostname(self):
        response = Mock(status=200)
        response.getheader.side_effect = lambda key, default=None: {"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "identity"}.get(key, default)
        response.read1.side_effect = [b"<html></html>", b""]
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(tools, "_public_addresses", return_value=["8.8.8.8"]), patch.object(tools, "_rate_limit"), patch.object(tools, "HTTPSConnection", return_value=connection) as factory, patch.object(tools.socket, "create_connection") as connect:
            html, url = tools._download(self.url)
            connection._create_connection(("shop.example", 443), 12)
        self.assertEqual(factory.call_args.args[0], "shop.example")
        self.assertEqual(connect.call_args.args[0], ("8.8.8.8", 443))
        self.assertEqual((html, url), ("<html></html>", self.url))

    def test_schema_exact_offer_has_provenance_and_not_a_sale_value(self):
        result = tools.parse_price(self.html(self.product()), self.url, expected=self.expected)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["amount"], 1599.90)
        self.assertEqual(result["vintage"], 2020)
        self.assertEqual(result["volume_ml"], 750)
        self.assertEqual(result["currency"], "RUB")
        self.assertEqual(result["source"], self.url)
        self.assertTrue(result["as_of"])
        self.assertIn("не цена продажи", result["warnings"][0])

    def test_aggregate_from_prices_unavailable_and_other_currency_are_not_quotes(self):
        for change in ({"@type": "AggregateOffer", "lowPrice": 100}, {"price": "NaN"}, {"price": "0"},
                       {"price": -1}, {"price": "Infinity"}, {"priceCurrency": "USD"}, {"priceCurrency": "RUR"},
                       {"availability": "https://schema.org/OutOfStock"}, {"availability": "https://schema.org/PreOrder"},
                       {"availability": ""}, {"priceValidUntil": "2000-01-01"}):
            product = self.product()
            product["offers"].update(change)
            with self.subTest(change=change):
                self.assertNotEqual(tools.parse_price(self.html(product), self.url)["status"], "ok")

    def test_wrong_vintage_volume_winery_or_variant_rejects_price(self):
        for change in ({"vintage": 2019}, {"volume_ml": 1500}, {"winery": "Other Domaine"}, {"name": "Other Wine"}):
            expected = {**self.expected, **change}
            with self.subTest(change=change):
                result = tools.parse_price(self.html(self.product()), self.url, expected=expected)
                self.assertEqual(result["status"], "mismatch")
        product = self.product(name="Chateau Test 2020")
        self.assertEqual(tools.parse_price(self.html(product), self.url, expected=self.expected)["status"], "mismatch")

    def test_ambiguous_multiple_offers_and_unrelated_products_do_not_choose_minimum(self):
        product = self.product()
        product["offers"] = [product["offers"], {**product["offers"], "price": "1499.90"}]
        self.assertEqual(tools.parse_price(self.html(product), self.url)["status"], "unavailable")
        product = self.product(url="https://shop.example/other-wine")
        self.assertEqual(tools.parse_price(self.html(product), self.url)["status"], "unavailable")

    def test_jsonld_graph_referenced_offer_is_supported(self):
        product = self.product(offers={"@id": "#offer"})
        offer = {"@id": "#offer", "@type": "Offer", "price": 1400, "priceCurrency": "RUB", "availability": "https://schema.org/InStock"}
        result = tools.parse_price(self.html({"@graph": [product, offer]}), self.url, expected=self.expected)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["amount"], 1400)

    def test_fetch_dry_run_never_does_dns_or_network(self):
        with patch.object(tools, "_download") as download, patch.object(tools.socket, "getaddrinfo") as dns:
            self.assertEqual(tools.fetch_price(self.url, dry_run=True)["status"], "dry_run")
        download.assert_not_called()
        dns.assert_not_called()

    def test_fetch_error_is_safe_and_does_not_leak_exception_secrets(self):
        with patch.object(tools, "_download", side_effect=OSError("secret key and password")):
            result = tools.fetch_price(self.url)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("secret", str(result))
        self.assertNotIn("amount", result)

    def test_fetch_uses_explicit_hosts_from_bot_and_expected_bottle(self):
        with patch.object(tools, "_download", return_value=(self.html(self.product()), self.url)) as download:
            result = tools.fetch_price(self.url, expected=self.expected, allowed_hosts=["shop.example"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(download.call_args.kwargs["allowed_hosts"], ["shop.example"])

    def test_founding_year_is_not_used_as_vintage(self):
        product = self.product(name="Chateau Test founded 1890 750 ml")
        self.assertEqual(tools.parse_price(self.html(product), self.url, expected=self.expected)["status"], "mismatch")

    def test_explicit_non_vintage_is_supported_without_guessing_unknown_year(self):
        candidate, _ = tools._ocr_candidate('CHAMPAGNE BRUT NV')
        self.assertEqual(candidate['vintage'], 'NV')
        product = self.product(name='Champagne Brut NV 750 ml')
        expected = {**self.expected, 'name': 'Champagne Brut', 'vintage': 'NV'}
        self.assertEqual(tools.parse_price(self.html(product), self.url, expected=expected)['status'], 'ok')
        unknown = self.product(name='Champagne Brut 750 ml')
        self.assertEqual(tools.parse_price(self.html(unknown), self.url, expected=expected)['status'], 'mismatch')
        self.assertIsNone(tools._ocr_candidate('CHAMPAGNE BRUT')[0]['vintage'])

    def test_rate_limit_waits_between_requests_to_same_host(self):
        with patch.dict(tools._LAST_REQUEST, {"shop.example": 100}, clear=True), patch.object(tools.time, "monotonic", side_effect=[100.25, 101]), patch.object(tools.time, "sleep") as sleep:
            tools._rate_limit("shop.example")
        sleep.assert_called_once_with(.75)

    def test_page_bytes_are_bounded_and_no_compressed_bomb_is_decoded(self):
        response = Mock(status=200)
        response.getheader.side_effect = lambda key, default=None: {"Content-Type": "text/html", "Content-Encoding": "identity"}.get(key, default)
        response.read1.return_value = b"x" * (tools.MAX_PAGE_BYTES + 1)
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(tools, "_public_addresses", return_value=["8.8.8.8"]), patch.object(tools, "_rate_limit"), patch.object(tools, "HTTPSConnection", return_value=connection), self.assertRaises(ValueError):
            tools._download(self.url)
        response.getheader.side_effect = lambda key, default=None: {"Content-Type": "text/html", "Content-Encoding": "gzip"}.get(key, default)
        response.read1.reset_mock()
        with patch.object(tools, "_public_addresses", return_value=["8.8.8.8"]), patch.object(tools, "_rate_limit"), patch.object(tools, "HTTPSConnection", return_value=connection), self.assertRaises(ValueError):
            tools._download(self.url)
        response.read1.assert_not_called()

    def test_image_normalization_removes_metadata_and_bounds_dimensions(self):
        raw = tools._image(self.image())
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(max(image.size), 1600)
            self.assertFalse(image.getexif())

    def test_retail_search_does_not_relax_user_product_url_policy(self):
        search = 'https://decanter.ru/search/?q=Chateau+Test+2020'
        with self.assertRaises(ValueError):
            tools.validate_product_url(search, allowed_hosts=['decanter.ru'])
        self.assertEqual(tools._search_url(search, 'decanter', allowed_hosts=['decanter.ru']), search)
        with self.assertRaises(ValueError):
            tools._search_url(search, 'decanter', allowed_hosts=[])

    def test_retail_search_host_path_query_and_text_are_fixed_and_bounded(self):
        invalid = ['https://evil.example/search/?q=wine', 'https://www.decanter.ru/search/?q=wine',
                   'http://decanter.ru/search/?q=wine', 'https://decanter.ru:443/search/?q=wine',
                   'https://user@decanter.ru/search/?q=wine', 'https://decanter.ru/catalog/?q=wine',
                   'https://decanter.ru/search/?q=wine&token=secret', 'https://decanter.ru/search/?q=wine&q=other',
                   'https://decanter.ru/search/?q=', 'https://decanter.ru/search/?q=%0A',
                   'https://decanter.ru/search/?q=%E2%80%AEwine', 'https://decanter.ru/search/?q=%3Cscript%3E',
                   'https://decanter.ru/search/?q=' + 'x' * 161]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                tools._search_url(url, 'decanter', allowed_hosts=['decanter.ru', 'evil.example'])
        with self.assertRaises(ValueError):
            tools._search_url('https://amwine.ru/search/?q=wine', 'amwine', allowed_hosts=['amwine.ru'])

    def test_search_transport_pins_dns_and_restricts_redirects_to_same_search(self):
        search = 'https://decanter.ru/search/?q=Chateau+Test'
        response = Mock(status=200)
        response.getheader.side_effect = lambda key, default=None: {'Content-Type': 'text/html'}.get(key, default)
        response.read1.side_effect = [b'<html>search results</html>', b'']
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(tools, '_public_addresses', return_value=['8.8.8.8']) as resolve, patch.object(tools, '_rate_limit'), patch.object(tools, 'HTTPSConnection', return_value=connection), patch.object(tools.socket, 'create_connection') as connect:
            html, final = tools._download(search, allowed_hosts=['decanter.ru'], search_shop='decanter')
            connection._create_connection(('decanter.ru', 443), 12)
        self.assertEqual((html, final), ('<html>search results</html>', search))
        resolve.assert_called_once_with('decanter.ru')
        self.assertEqual(connect.call_args.args[0], ('8.8.8.8', 443))
        self.assertEqual(connection.request.call_args.args[:2], ('GET', '/search/?q=Chateau+Test'))
        for location in ('https://evil.example/search/?q=wine', '/product/wine', '/login?q=wine', '/search/?q=wine&token=x',
                         'https://evil.example/product/wine-id1', '/product/wine-id0', '/product/wine-id1?sku=other',
                         '/product/wine-id1#section', '/product/WINE-id1', '/product/wine-id1/../admin'):
            response = Mock(status=302)
            response.getheader.return_value = location
            connection = Mock()
            connection.getresponse.return_value = response
            with self.subTest(location=location), patch.object(tools, '_public_addresses', return_value=['8.8.8.8']) as resolve, patch.object(tools, '_rate_limit'), patch.object(tools, 'HTTPSConnection', return_value=connection), self.assertRaises(ValueError):
                tools._download(search, allowed_hosts=['decanter.ru', 'evil.example'], search_shop='decanter')
            resolve.assert_called_once()
            connection.close.assert_called_once()

    def test_unique_search_redirect_to_same_retailer_product_is_downloaded_once(self):
        search = 'https://decanter.ru/search/?q=Chateau+Minuty+M+Rose+2025'
        target = 'https://decanter.ru/product/chateau-minuty-m-rose-2025-id272350'
        redirect = Mock(status=302)
        redirect.getheader.return_value = target
        page = Mock(status=200)
        page.getheader.side_effect = lambda key, default=None: {'Content-Type': 'text/html'}.get(key, default)
        page.read1.side_effect = [b'<html>exact product</html>', b'']
        connections = [Mock(), Mock()]
        connections[0].getresponse.return_value = redirect
        connections[1].getresponse.return_value = page
        with patch.object(tools, '_public_addresses', return_value=['8.8.8.8']) as resolve, patch.object(tools, '_rate_limit'), patch.object(tools, 'HTTPSConnection', side_effect=connections) as factory:
            html, final = tools._download(search, allowed_hosts=['decanter.ru'], search_shop='decanter')
        self.assertEqual((html, final), ('<html>exact product</html>', target))
        self.assertEqual(factory.call_count, 2)
        self.assertEqual([call.args[0] for call in resolve.call_args_list], ['decanter.ru', 'decanter.ru'])
        self.assertEqual(connections[1].request.call_args.args[:2], ('GET', '/product/chateau-minuty-m-rose-2025-id272350'))
        for connection in connections:
            connection.close.assert_called_once()
        # Only an actual search may start this transport mode.
        with patch.object(tools, '_public_addresses') as resolve, self.assertRaises(ValueError):
            tools._download(target, allowed_hosts=['decanter.ru'], search_shop='decanter')
        resolve.assert_not_called()

    def test_known_retailers_require_their_city_and_stock_adapter(self):
        for host, adapter in (('decanter.ru', 'thesaurus.wine_retail_sources.parse_decanter_product'),
                              ('amwine.ru', 'thesaurus.wine_retail_amwine.parse_amwine_product')):
            url = 'https://' + host + '/wine/test'
            quote = {**self.expected, 'title': self.expected['name'], 'status': 'ok', 'amount': 1500}
            with self.subTest(host=host), patch.object(tools, '_download', return_value=('<html/>', url)), patch(adapter, return_value=quote) as parse, patch.object(tools, 'parse_price') as generic:
                result = tools.fetch_price(url, expected=self.expected, allowed_hosts=[host])
                mismatch = tools.fetch_price(url, expected={**self.expected, 'vintage': 2021}, allowed_hosts=[host])
            self.assertEqual(result['amount'], 1500)
            self.assertEqual(mismatch['status'], 'mismatch')
            self.assertNotIn('amount', mismatch)
            self.assertEqual(parse.call_args.args, ('<html/>', url))
            generic.assert_not_called()

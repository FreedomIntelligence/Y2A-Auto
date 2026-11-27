"""反向代理终止 HTTPS 时的来源校验：浏览器 Origin 是 https://域名，程序看到的是 http://域名。

线上部署是 nginx(HTTPS) → nginx(HTTPS) → 本程序(http)，内层代理传 X-Forwarded-Proto: https。
之前来源校验只比 request.host_url，浏览器里的登录和所有保存操作都被 403「请求来源校验失败」挡住。
"""

import unittest

import app as web_app

TOKEN = "t" * 40
HOST = "y2a.example.com"


class ProxyOriginCheckTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()
        # 会话 cookie 要设在请求用的同一个域名下，否则请求带不上会话，先被 CSRF 校验挡住
        with self.client.session_transaction(base_url=f"http://{HOST}") as sess:
            sess["logged_in"] = True
            sess["csrf_token"] = TOKEN

    def _post(self, origin, forwarded_proto=None):
        headers = {"Origin": origin, "X-CSRF-Token": TOKEN}
        if forwarded_proto is not None:
            headers["X-Forwarded-Proto"] = forwarded_proto
        return self.client.post("/logout", headers=headers, base_url=f"http://{HOST}")

    @staticmethod
    def _message(resp):
        # jsonify 会把中文转成 \uXXXX，按 JSON 解析后再比
        return (resp.get_json(silent=True) or {}).get("message", "")

    def _rejected_by_origin(self, resp):
        return resp.status_code == 403 and "请求来源校验失败" in self._message(resp)

    def test_https_origin_behind_tls_proxy_is_accepted(self):
        resp = self._post(f"https://{HOST}", forwarded_proto="https")
        self.assertFalse(self._rejected_by_origin(resp), resp.status_code)
        self.assertEqual(resp.status_code, 302)

    def test_first_value_of_forwarded_proto_list_is_used(self):
        resp = self._post(f"https://{HOST}", forwarded_proto="https, http")
        self.assertEqual(resp.status_code, 302)

    def test_https_origin_without_proxy_header_is_still_rejected(self):
        self.assertTrue(self._rejected_by_origin(self._post(f"https://{HOST}")))

    def test_other_site_is_rejected_even_with_proxy_header(self):
        self.assertTrue(self._rejected_by_origin(self._post("https://evil.example.com", forwarded_proto="https")))
        self.assertTrue(self._rejected_by_origin(self._post(f"https://{HOST}.evil.com", forwarded_proto="https")))

    def test_unknown_forwarded_proto_is_ignored(self):
        self.assertTrue(self._rejected_by_origin(self._post(f"https://{HOST}", forwarded_proto="javascript")))

    def test_same_scheme_origin_still_works(self):
        self.assertEqual(self._post(f"http://{HOST}").status_code, 302)

    def test_csrf_token_is_still_required(self):
        resp = self.client.post(
            "/logout",
            headers={"Origin": f"https://{HOST}", "X-Forwarded-Proto": "https", "X-CSRF-Token": "wrong" * 8},
            base_url=f"http://{HOST}",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("安全校验失败", self._message(resp))


if __name__ == "__main__":
    unittest.main()

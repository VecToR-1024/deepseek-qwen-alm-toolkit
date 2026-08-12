import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from topk_distill.client import HttpSequenceLogprobClient


class _TeacherHandler(BaseHTTPRequestHandler):
    request_json = None

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        type(self).request_json = json.loads(self.rfile.read(content_length))
        body = json.dumps(
            {
                "logprobs": [[[-1.2, -0.1]]],
                "logprob_token_ids": [[[9, 8]]],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class HttpSequenceLogprobClientTests(unittest.TestCase):
    def test_posts_sequences_and_normalizes_the_provider_response(self):
        server = HTTPServer(("127.0.0.1", 0), _TeacherHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = HttpSequenceLogprobClient(
            f"http://127.0.0.1:{server.server_port}/score",
            timeout=2.0,
        )
        result = client.get_sequence_logprobs(
            sequences=[[1, 2, 3]],
            prompt_lengths=[2],
            top_logprobs=2,
            temperature=1.5,
        )

        self.assertEqual(_TeacherHandler.request_json["sequences"], [[1, 2, 3]])
        self.assertEqual(_TeacherHandler.request_json["top_logprobs"], 2)
        self.assertEqual(result["logprob_token_ids"], [[[8, 9]]])
        self.assertEqual(result["actual_logprobs"], [[[-float("inf")]]])

    def test_rejects_invalid_prompt_lengths_before_making_a_request(self):
        client = HttpSequenceLogprobClient("http://127.0.0.1:1/score")

        with self.assertRaisesRegex(ValueError, "prompt length"):
            client.get_sequence_logprobs(
                sequences=[[1, 2]],
                prompt_lengths=[3],
                top_logprobs=2,
            )


if __name__ == "__main__":
    unittest.main()

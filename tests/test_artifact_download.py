import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch

from fastapi import HTTPException
from fastapi.responses import FileResponse

import app as backend_app


class FakeNotebookLMClient:
    def __init__(self):
        self.artifacts = SimpleNamespace(
            download_slide_deck=AsyncMock(),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class DownloadArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_slide_deck_download_supports_pptx_output_format(self):
        fake_client = FakeNotebookLMClient()
        expected_bytes = b"pptx-bytes"

        async def write_pptx(_notebook_id, output_path, artifact_id=None, output_format="pdf"):
            self.assertEqual(artifact_id, "artifact-123")
            self.assertEqual(output_format, "pptx")
            with open(output_path, "wb") as f:
                f.write(expected_bytes)
            return output_path

        fake_client.artifacts.download_slide_deck.side_effect = write_pptx

        with patch.object(backend_app, "get_client", AsyncMock(return_value=fake_client)):
            response = await backend_app.download_artifact(
                notebook_id="notebook-123",
                type="slide_deck",
                artifact_id="artifact-123",
                output_format="pptx",
            )

        self.assertIsInstance(response, FileResponse)
        self.assertEqual(
            response.media_type,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
        self.assertTrue(response.path.endswith(".pptx"))
        fake_client.artifacts.download_slide_deck.assert_awaited_once_with(
            "notebook-123",
            ANY,
            artifact_id="artifact-123",
            output_format="pptx",
        )

        output_path = fake_client.artifacts.download_slide_deck.await_args.args[1]
        with open(output_path, "rb") as f:
            self.assertEqual(f.read(), expected_bytes)
        if os.path.exists(output_path):
            os.remove(output_path)

    async def test_slide_deck_download_rejects_non_pdf_and_non_pptx_formats(self):
        get_client_mock = AsyncMock()

        with patch.object(backend_app, "get_client", get_client_mock):
            with self.assertRaises(HTTPException) as ctx:
                await backend_app.download_artifact(
                    notebook_id="notebook-123",
                    type="slide_deck",
                    artifact_id=None,
                    output_format="html",
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "output_format for slide_deck must be either pdf or pptx.")
        get_client_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

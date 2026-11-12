"""Bounded HTTP transport with phase reporting and cooperative cancellation."""
import asyncio
import json
import logging
import ssl
import time

import httpx

LOG = logging.getLogger(__name__)


class RequestCancelled(RuntimeError):
    pass


class RequestTimedOut(RuntimeError):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


class ResponseReadError(ValueError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class ChatStream:
    """Accumulate OpenAI-compatible Chat Completions SSE without exposing text."""
    def __init__(self):
        self.parts = []
        self.characters = 0
        self.finish_reason = None
        self.usage = {}
        self.id = None
        self.error = None
        self.done = False

    def event(self, data):
        if data.strip() == "[DONE]":
            self.done = True
            return
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("Неверный JSON-объект в потоке API")
        self.id = chunk.get("id", self.id)
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        if chunk.get("error") is not None:
            self.error = chunk["error"]
            self.done = True
            return
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError("Неверный choices в потоке API")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if choice.get("error") is not None:
                self.error = choice["error"]
                self.done = True
                return
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("Неверный delta в потоке API")
            content = delta.get("content")
            if isinstance(content, str):
                self.parts.append(content)
                self.characters += len(content)

    def response(self):
        result = {"id": self.id, "usage": self.usage,
                  "choices": [{"finish_reason": self.finish_reason or ("stop" if self.done else None),
                               "message": {"content": "".join(self.parts)}}]}
        if self.error is not None:
            result["error"] = self.error
        return result


def post_json(url, api_key, body, *, connect_timeout, request_timeout,
              progress, cancelled, label):
    async def run():
        started = time.monotonic()
        phase = "подключение к API"
        sending_post = False
        stream = ChatStream()
        metadata = {"incomplete": True, "http_status": None, "received_bytes": 0,
                    "content_type": None, "request_started": False}
        prefix = bytearray()

        def snapshot():
            result = stream.response() if stream.characters or stream.id else {}
            result["_transport"] = dict(metadata)
            if prefix:
                result["partial_body"] = bytes(prefix).decode("utf-8", errors="replace")
            return result

        def report():
            progress(f"{label}: {phase} · {time.monotonic()-started:.0f}/{request_timeout:g} с")

        async def trace(event, info):
            nonlocal phase, sending_post
            # Never log trace metadata: it may contain headers or request bodies.
            if event.endswith("connect_tcp.started"):
                phase = "подключение DNS/TCP"
            elif event.endswith("start_tls.started"):
                phase = "защищённое соединение TLS"
            elif event.endswith("send_request_headers.started"):
                if getattr(info.get("request"), "method", None) == b"POST":
                    metadata["request_started"] = True
                return
            elif event.endswith("send_request_body.started"):
                if getattr(info.get("request"), "method", None) != b"POST":
                    return
                metadata["request_started"] = True
                sending_post = True
                phase = "отправка запроса"
            elif event.endswith("send_request_body.complete"):
                # Completion trace events do not repeat the request metadata.
                if not sending_post:
                    return
                sending_post = False
                phase = "запрос отправлен; ожидание HTTP-ответа"
            else:
                return
            LOG.info("%s: %s", label, phase)
            report()

        async def send():
            nonlocal phase
            timeout = httpx.Timeout(connect=connect_timeout, read=request_timeout,
                                    write=min(30, request_timeout), pool=connect_timeout)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                                         verify=ssl.create_default_context()) as client:
                async with client.stream("POST", url, content=body,
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
                         extensions={"trace": trace}) as response:
                    metadata.update(http_status=response.status_code,
                                    content_type=response.headers.get("content-type", ""),
                                    generation_id=response.headers.get("x-generation-id"))
                    phase = f"HTTP {response.status_code}; получение ответа"
                    LOG.info("%s: HTTP %s", label, response.status_code)
                    report()
                    response.raise_for_status()
                    if "text/event-stream" in metadata["content_type"].lower():
                        data = []
                        async for line in response.aiter_lines():
                            metadata["received_bytes"] += len(line.encode("utf-8"))+1
                            if metadata["received_bytes"] > 32*1024*1024:
                                raise ResponseReadError("Ответ API превышает 32 МБ", snapshot())
                            if line.startswith(":"):
                                if not stream.characters:
                                    phase = "HTTP 200; служебный сигнал сервера, генерация ещё не получена"
                                continue
                            if line.startswith("data:"):
                                data.append(line[5:].lstrip())
                            elif not line and data:
                                had_text = bool(stream.characters)
                                try:
                                    stream.event("\n".join(data))
                                except ValueError as exc:
                                    raise ResponseReadError("Некорректное событие SSE: " + str(exc), snapshot()) from exc
                                data.clear()
                                if stream.characters:
                                    phase = f"HTTP 200; генерация: {stream.characters} символов"
                                    if not had_text:
                                        report()
                                if stream.done:
                                    return stream.response()
                        if data:
                            try:
                                stream.event("\n".join(data))
                            except ValueError as exc:
                                raise ResponseReadError("Некорректное событие SSE: " + str(exc), snapshot()) from exc
                        if stream.done or stream.finish_reason:
                            return stream.response()
                        raise ResponseReadError("Поток API оборвался без завершения ответа", snapshot())
                    parts, count = [], 0
                    async for chunk in response.aiter_bytes():
                        count += len(chunk)
                        metadata["received_bytes"] = count
                        prefix.extend(chunk[:max(0, 4096-len(prefix))])
                        if count > 32*1024*1024:
                            raise ValueError("Ответ API превышает 32 МБ")
                        parts.append(chunk)
                        phase = f"HTTP {response.status_code}; получено {count/1024:.1f} КБ"
                    try:
                        return json.loads(b"".join(parts))
                    except ValueError as exc:
                        raise ResponseReadError("API вернул некорректный JSON", snapshot()) from exc

        if cancelled():
            raise RequestCancelled("Smart OCR отменён пользователем")
        report()
        task = asyncio.create_task(send())
        last_report = started
        try:
            # A wall-clock deadline also stops a server sending keep-alive bytes.
            async with asyncio.timeout(request_timeout):
                while not task.done():
                    if cancelled():
                        raise RequestCancelled("Smart OCR отменён пользователем")
                    await asyncio.wait({task}, timeout=0.2)
                    if time.monotonic()-last_report >= 1:
                        report()
                        last_report = time.monotonic()
                if cancelled():
                    raise RequestCancelled("Smart OCR отменён пользователем")
                return await task
        except (TimeoutError, httpx.TimeoutException) as exc:
            kind = "подключения" if isinstance(exc, httpx.ConnectTimeout) else "запроса"
            retry_notice = ("Запрос модели не отправлен; возможен ограниченный повтор подключения."
                            if metadata["request_started"] is False and metadata["http_status"] is None else
                            "Автоматический повтор отключён: предыдущий запрос мог быть принят сервером.")
            raise RequestTimedOut(f"Тайм-аут {kind}: {phase} ({time.monotonic()-started:.0f} с). "
                                  + retry_notice, snapshot()) from exc
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return asyncio.run(run())

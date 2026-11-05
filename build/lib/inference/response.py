"""공통 응답 형식 (docs contracts/api-response.md, Dooray 방식 header/result/totalCount).

성공:  {"header": {"isSuccessful": true,  "resultCode": 0,  "resultMessage": "SUCCESS"}, "result": {...}}
실패:  {"header": {"isSuccessful": false, "resultCode": -1001, "resultMessage": "한국어 문장"}, "result": null}
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ResultCode(Enum):
    # (코드, HTTP 상태, 사용자에게 보여 줄 문장)
    SUCCESS = (0, 200, "SUCCESS")
    ACCEPTED = (0, 202, "SUCCESS")
    INVALID_INPUT = (-1001, 400, "입력값이 올바르지 않습니다.")
    NOT_FOUND = (-1002, 404, "대상을 찾을 수 없습니다.")
    UNAUTHORIZED = (-1004, 401, "인증이 필요합니다.")
    SERVER_ERROR = (-1099, 500, "서버 오류가 발생했습니다.")
    NOT_READY = (-1098, 503, "추론 서버가 준비 중입니다. 잠시 후 다시 시도해 주세요.")
    BUSY = (-4003, 503, "추론 서버가 바쁩니다. 잠시 후 다시 시도해 주세요.")

    @property
    def code(self) -> int:
        return self.value[0]

    @property
    def http(self) -> int:
        return self.value[1]

    @property
    def message(self) -> str:
        return self.value[2]


def _dump(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json", by_alias=True)
    return result


def ok(result: Any = None, code: ResultCode = ResultCode.SUCCESS) -> JSONResponse:
    body = {
        "header": {"isSuccessful": True, "resultCode": 0, "resultMessage": "SUCCESS"},
        "result": _dump(result),
    }
    return JSONResponse(body, status_code=code.http)


def fail(code: ResultCode, result: Any = None, message: str | None = None) -> JSONResponse:
    body = {
        "header": {"isSuccessful": False, "resultCode": code.code, "resultMessage": message or code.message},
        "result": _dump(result),
    }
    return JSONResponse(body, status_code=code.http)

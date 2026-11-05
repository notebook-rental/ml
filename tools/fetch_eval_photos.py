"""실험용 사진 꺼내기: ml-inference 파드 안에서 실행해 최근 반납 검사와 기준 검사 사진을 tar로 내보낸다 (읽기 전용).

저장소(MinIO)에서 **읽기만** 한다. 쓰거나 지우지 않는다. 접속 정보는 파드 환경 변수를 그대로 쓰고 출력하지 않는다.

    kubectl -n notebook-rental exec -i deploy/ml-inference -- python - < tools/fetch_eval_photos.py > photos.tar

고르는 법(환경 변수로 바꿀 수 있다: RETURN_INSPECTION, BASELINE_INSPECTION):
- 반납 검사: originals/inspections/ 아래에서 가장 최근에 사진이 올라온 검사
- 기준 검사: 그 밖에서 가장 최근이면서 컷이 6개 이상인 검사
- 컷마다 가장 최근 사진 1장. 작업본(derived/work/, 비교에 쓰는 사진)이 있으면 그것, 없으면 원본

tar 안: baseline/<shot>.jpg, return/<shot>.jpg, info.json(검사 번호·사진 번호만)
"""

import io
import json
import os
import re
import sys
import tarfile

import boto3
from botocore.config import Config

PAT = re.compile(r"^originals/inspections/\d{4}/\d{2}/(\d+)/([a-z_]+)-(\d+)\.jpg$")


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    bucket = os.environ.get("MINIO_BUCKET", "notebook-rental")
    insp: dict[int, dict[str, tuple]] = {}
    last: dict[int, float] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="originals/inspections/"):
        for o in page.get("Contents", []):
            m = PAT.match(o["Key"])
            if not m:
                continue
            iid, shot, pid = int(m[1]), m[2], int(m[3])
            t = o["LastModified"].timestamp()
            cur = insp.setdefault(iid, {}).get(shot)
            if cur is None or pid > cur[0]:
                insp[iid][shot] = (pid, o["Key"])
            last[iid] = max(last.get(iid, 0), t)
    order = sorted(last, key=last.get, reverse=True)
    for iid in order[:6]:
        print(f"inspection {iid}: shots={sorted(insp[iid])}", file=sys.stderr)
    ret = int(os.environ.get("RETURN_INSPECTION") or order[0])
    base = os.environ.get("BASELINE_INSPECTION")
    base = int(base) if base else next(i for i in order if i != ret and len(insp[i]) >= 6)
    print(f"return={ret} baseline={base}", file=sys.stderr)

    info = {"return": ret, "baseline": base, "photos": {}}
    tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    for folder, iid in (("return", ret), ("baseline", base)):
        for shot, (pid, okey) in sorted(insp[iid].items()):
            data = None
            for key in (f"derived/work/{iid}/{pid}.jpg", okey):
                try:
                    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    break
                except s3.exceptions.NoSuchKey:
                    continue
            if data is None:
                continue
            info["photos"][f"{folder}/{shot}"] = pid
            ti = tarfile.TarInfo(f"{folder}/{shot}.jpg")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    raw = json.dumps(info, indent=1).encode()
    ti = tarfile.TarInfo("info.json")
    ti.size = len(raw)
    tf.addfile(ti, io.BytesIO(raw))
    tf.close()


main()

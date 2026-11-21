FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN npx --yes pnpm@11.25.0 install --frozen-lockfile --ignore-scripts
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE2EDITABLEPPT_PUBLIC_SHARED=1 \
    IMAGE2EDITABLEPPT_RUNTIME_DIR=/srv/imageppt/runtime \
    LOCAL_INPAINT_ENABLED=false \
    INPAINT_PROVIDER=opencv \
    PPT_MASTER_SKILL_DIR=/opt/ppt-master/skills/ppt-master \
    IMAGE2EDITABLEPPT_FONT_DIR=/usr/share/fonts/opentype/noto \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl fontconfig fonts-noto-cjk libgl1 libglib2.0-0 libgomp1 libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/ppt-master \
    && curl --fail --location --silent --show-error --retry 3 \
       https://codeload.github.com/hugohe3/ppt-master/tar.gz/5e8856be0da8e6bc85c4aa28acc614478f7b7137 \
       -o /tmp/ppt-master.tar.gz \
    && tar -xzf /tmp/ppt-master.tar.gz -C /opt/ppt-master --strip-components=1 \
    && rm /tmp/ppt-master.tar.gz

WORKDIR /srv/imageppt/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 'PyYAML>=6' 'XlsxWriter>=3' \
    && python -c 'from pathlib import Path; import os; from rapidocr_onnxruntime import RapidOCR; p = Path(os.environ["PPT_MASTER_SKILL_DIR"]); assert (p / "references/plan-core.md").is_file()' \
    && python /opt/ppt-master/skills/ppt-master/scripts/svg_to_pptx.py --help >/dev/null

COPY backend/app ./app
COPY --from=frontend /build/dist /srv/imageppt/frontend/dist
RUN useradd --create-home --uid 10001 imageppt \
    && mkdir -p /srv/imageppt/runtime \
    && chown imageppt:imageppt /srv/imageppt/runtime
USER imageppt
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn app.shared_web:app --host 0.0.0.0 --port \"${PORT:-10000}\""]

"""Two fresh HTTP analysis requests, preserving existing object-owned scenes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    import httpx
    from app.config import OUTPUTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", help="A test project, not the user's working deck")
    args = parser.parse_args()
    url = f"http://127.0.0.1:8000/api/projects/{args.project_id}"
    report = []
    with httpx.Client(timeout=300) as client:
        baseline = client.get(url + "/slides").raise_for_status().json()
        identities = [{i["id"]: (i.get("owner"), i.get("src")) for i in p["elements"]} for p in baseline]
        for round_index in range(2):
            response = client.post(url + "/analyze", params={"mode": "standard"}).raise_for_status().json()
            for index, slide in enumerate(response["slides"]):
                assert {i["id"]: (i.get("owner"), i.get("src")) for i in slide["elements"]} == identities[index]
                assert slide["metadata"]["preservationQA"]["missingAssetCount"] == 0
            result = {"round": round_index + 1, "aiUsed": response["aiUsed"], "visionProvider": response["visionProvider"], "warnings": response["warnings"], "pages": [{"decision": p["metadata"]["lastRevision"], "qa": {k: v for k, v in p["metadata"]["preservationQA"].items() if k != "objects"}} for p in response["slides"]]}
            report.append(result)
            print(json.dumps({"round": result["round"], "aiUsed": result["aiUsed"], "provider": result["visionProvider"], "decisions": [p["decision"] for p in result["pages"]]}, ensure_ascii=False), flush=True)
    output = OUTPUTS_DIR / args.project_id / "api_revisions.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""
Decode DICOM files fetched from the Google Drive MCP bridge into a staging tree.

Each `download_file_content` call from the Drive MCP server writes a JSON blob
({content: <base64>, title, ...}) to the harness tool-results directory. This
script decodes every such blob and files it under data/dicom_staging/SER<n>/,
routing by the DICOM SeriesNumber. Set TOOL_RESULTS_DIR to the harness
tool-results directory (it is session-specific).
"""
import json, base64, os, glob, sys
import pydicom

TOOL_RESULTS = os.environ.get(
    "TOOL_RESULTS_DIR",
    "/root/.claude/projects/-home-user-brainmodel/f3bb0df5-c8a3-59e2-a01b-a998e22c4fa1/tool-results",
)
STAGING = os.path.join(os.path.dirname(__file__), "..", "data", "dicom_staging")
os.makedirs(STAGING, exist_ok=True)

def harvest():
    files = sorted(glob.glob(os.path.join(TOOL_RESULTS, "mcp-Google_Drive-download_file_content-*.txt")))
    n=0
    for f in files:
        try:
            with open(f) as fh:
                obj = json.load(fh)
        except Exception as e:
            continue
        content = obj.get("content")
        title = obj.get("title","unknown.dcm")
        if not content:
            continue
        try:
            data = base64.b64decode(content)
        except Exception:
            continue
        # temp write, read series number to route
        tmp = os.path.join(STAGING, "_tmp.dcm")
        with open(tmp,"wb") as g: g.write(data)
        try:
            ds = pydicom.dcmread(tmp, stop_before_pixels=True, force=True)
            ser = str(getattr(ds,"SeriesNumber","NA"))
        except Exception:
            ser = "NA"
        d = os.path.join(STAGING, f"SER{ser}")
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, title)
        os.rename(tmp, dest)
        n+=1
    return n

n = harvest()
print(f"harvested {n} files into {STAGING}\n")

# Build manifest grouped by SeriesInstanceUID
series = {}
for f in glob.glob(os.path.join(STAGING,"SER*","*")):
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
    except Exception:
        continue
    uid = getattr(ds,"SeriesInstanceUID","NA")
    if uid not in series:
        series[uid] = {
          "SeriesNumber": getattr(ds,"SeriesNumber","?"),
          "SeriesDescription": getattr(ds,"SeriesDescription","?"),
          "Modality": getattr(ds,"Modality","?"),
          "MRAcquisitionType": getattr(ds,"MRAcquisitionType","?"),
          "SliceThickness": getattr(ds,"SliceThickness","?"),
          "PixelSpacing": list(getattr(ds,"PixelSpacing",[])) or "?",
          "Rows": getattr(ds,"Rows","?"),
          "Columns": getattr(ds,"Columns","?"),
          "ImageType": list(getattr(ds,"ImageType",[])),
          "count_sampled": 0,
        }
    series[uid]["count_sampled"] += 1

print(f"{'SER':>4} {'Acq':>3} {'thick':>6} {'pixspc':>14} {'RxC':>10}  Description")
print("-"*90)
for uid,v in sorted(series.items(), key=lambda kv: int(str(kv[1]['SeriesNumber']).replace('?','0'))):
    ps = v["PixelSpacing"]
    ps = f"{float(ps[0]):.3f}x{float(ps[1]):.3f}" if isinstance(ps,list) and ps else "?"
    print(f"{str(v['SeriesNumber']):>4} {str(v['MRAcquisitionType']):>3} {str(v['SliceThickness']):>6} {ps:>14} {str(v['Rows'])+'x'+str(v['Columns']):>10}  {v['SeriesDescription']}  {v['ImageType'][:3]}")

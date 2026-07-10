import json, base64, sys, os
# Reads a saved MCP download tool-result JSON file, writes the decoded bytes to dest.
src = sys.argv[1]
dest = sys.argv[2]
with open(src) as f:
    obj = json.load(f)
content = obj.get("content") or obj.get("fileContent")
data = base64.b64decode(content)
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "wb") as g:
    g.write(data)
print(f"wrote {len(data)} bytes -> {dest}  (title={obj.get('title')})")

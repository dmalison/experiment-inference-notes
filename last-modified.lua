local cached_date = nil
local did_compute = false

local script = [=[
from datetime import datetime
from pathlib import Path

repo_root = Path.cwd()
source_suffixes = {".css", ".js", ".lua", ".md", ".py", ".qmd", ".yaml", ".yml"}
excluded_dirs = {".git", ".quarto", ".venv", "__pycache__", "experiment_inference_simulation.egg-info"}

source_files = []
for path in repo_root.rglob("*"):
    relative_parts = path.relative_to(repo_root).parts
    if not path.is_file():
        continue
    if any(part in excluded_dirs or part.endswith("_files") for part in relative_parts):
        continue
    if path.suffix in source_suffixes:
        source_files.append(path)

latest_modified = max(path.stat().st_mtime for path in source_files)
formatted_date = datetime.fromtimestamp(latest_modified).strftime("%B %d, %Y").replace(" 0", " ")
print(formatted_date)
]=]

local function latest_modified_date()
  if did_compute then
    return cached_date
  end

  did_compute = true
  local ok, result = pcall(pandoc.pipe, "python3", {"-c", script}, "")
  if ok then
    cached_date = result:gsub("^%s+", ""):gsub("%s+$", "")
  end

  return cached_date
end

function Meta(meta)
  local date = latest_modified_date()
  if date == nil or date == "" then
    return nil
  end

  meta.date = pandoc.MetaInlines({pandoc.Str(date)})
  return meta
end